"""The parts of a windowed run that do not depend on the pipeline's stages.

:meth:`~ob_analytics.pipeline.Pipeline.run_windows` cuts one input into time
windows and runs the depth stages on each in turn, so their peak memory is set
by the largest window rather than by the whole input.  This module holds what
that needs around the stages themselves:

* :func:`window_bounds` turns the caller's cut times into ``(start, end)``
  pairs that cover the whole input.
* :func:`resting_orders` reads the orders left on the book at the end of one
  window, and :func:`seed_rows` writes them back in as rows that come before
  the next window.  That is the carry for the price-level rebuild, which works
  per order: without it each window would start from an empty book.  The depth
  summary needs no rows for it, because its engine holds the book itself and
  carries on from one window to the next.
* :class:`ParquetAppender` writes each window's tables to one Parquet file per
  table as the windows finish, so the depth tables are never held for the
  whole input.
* :class:`~ob_analytics.protocols.WindowedLoader` (defined with the other
  loader contracts) is a loader that can also read its input one window at a
  time, so the events are never held for the whole input either.
* :class:`OrderTypeFiles` decides the order types of such a run from files,
  so the values per order that the rule reads are not held for the whole
  input.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ob_analytics import _engine_frames
from ob_analytics._utils import epoch_ns
from ob_analytics.analytics import (
    _order_types,
    _OrderFacts,
    _warn_on_unknown_orders,
)
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import FeedDeclarations, Level
from ob_analytics.schemas import (
    ORDER_TYPE_DTYPE,
    lot_sizes_from_config,
    tick_sizes_from_config,
)

#: How far before a window's start its seed rows are placed.  Strictly before
#: the start, so every seed row sorts ahead of every row of the window itself,
#: including one stamped exactly at the start.
SEED_OFFSET = pd.Timedelta(1, "ns")


def window_bounds(
    boundaries: Any,
) -> list[tuple[pd.Timestamp | None, pd.Timestamp | None]]:
    """Return the ``(start, end)`` of each window the *boundaries* cut.

    *n* cut times make *n + 1* windows.  The first window has no start and the
    last has no end, so together they cover the whole input and no row is left
    out.  A window holds the rows at or after its start and before its end.

    Parameters
    ----------
    boundaries : time or iterable of times
        Cut times, anything :class:`pandas.Timestamp` accepts.  One time on its
        own, a string included, is one cut.  A time with no zone is read as UTC,
        the zone every canonical frame is on.

    Returns
    -------
    list of tuple
        ``(start, end)`` per window, tz-aware UTC; ``None`` for an open end.

    Raises
    ------
    ConfigError
        If there are no cuts, or the cuts do not strictly increase.
    """
    if isinstance(boundaries, str | bytes | date | np.datetime64):
        boundaries = [boundaries]
    cuts = [_as_utc(b) for b in boundaries]
    if not cuts:
        raise ConfigError(
            "run_windows: give at least one boundary; with none there is one "
            "window, which is what Pipeline.run does"
        )
    for before, after in pairwise(cuts):
        if not before < after:
            raise ConfigError(
                f"run_windows: boundaries must strictly increase; {before} is "
                f"followed by {after}"
            )
    starts: list[pd.Timestamp | None] = [None, *cuts]
    ends: list[pd.Timestamp | None] = [*cuts, None]
    return list(zip(starts, ends, strict=True))


def _as_utc(value: Any, context: str = "run_windows") -> pd.Timestamp:
    """Return *value* as a tz-aware UTC timestamp; no zone is read as UTC.

    *context* names the caller at the start of the error message.
    """
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context}: {value!r} is not a time ({exc})") from exc
    if ts is pd.NaT:
        raise ConfigError(f"{context}: {value!r} is not a time")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def window_positions(
    timestamps: pd.Series,
    windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
) -> list[np.ndarray]:
    """Return, per window, the positions of the rows of *timestamps* it holds.

    One pass over the column, however many windows there are.  The positions
    are in the frame's own order, so a window keeps the rows in the order the
    loader gave them.  A row with no time falls in the first window, so no row
    is lost.
    """
    cuts = epoch_ns([start for start, _ in windows[1:] if start is not None])
    labels = np.searchsorted(cuts, epoch_ns(timestamps), side="right")
    found = pd.Series(labels).groupby(labels).indices
    empty = np.empty(0, dtype="int64")
    return [found.get(k, empty) for k in range(len(windows))]


def resting_orders(events: pd.DataFrame) -> pd.DataFrame:
    """Return the last row of each order still on the price-level book.

    Follows :func:`~ob_analytics.depth.price_level_volume` exactly, because the
    rows returned here are what the next window's price levels start from.  That
    function rebuilds each side of the book separately, so an order is an id on
    one side.  An order counts only when that function puts its volume on a
    level: it has a ``created`` row on its side and is not a ``market`` order.
    The level is its resting price after its last row, the price every
    rebuild places it at, which need not be the price on its last row.  The
    volume is its outstanding size after its last row.

    Parameters
    ----------
    events : pandas.DataFrame
        One window's events with order types, seed rows included.

    Returns
    -------
    pandas.DataFrame
        One row per resting order, with the columns of *events*; ``price`` is
        the price level the order rests on.
    """
    sides = []
    for _, side in events.groupby("direction", observed=True, sort=False):
        on_book = side[
            side["id"].isin(side.loc[side["action"] == "created", "id"])
            & (side["type"] != "market")
        ]
        if on_book.empty:
            continue
        level = _engine_frames.resting_price(on_book)
        last = on_book.assign(price=level).groupby("id", sort=False).tail(1)
        sides.append(last[(last["action"] != "deleted") & (last["volume"] > 0)])
    return pd.concat(sides) if sides else events.iloc[:0]


def seed_rows(resting: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Return *resting* as ``created`` rows that sit just before *start*.

    One row per resting order, with no execution, that puts the previous
    window's orders back on their price levels before the next window begins.
    Each gets a negative ``event_id``, so it cannot collide with a real event
    and sorts before every real one.
    """
    at = start - SEED_OFFSET
    seeds = resting.copy()
    seeds["timestamp"] = at
    seeds["exchange_timestamp"] = at
    seeds.loc[:, "action"] = "created"
    seeds.loc[:, "fill"] = 0
    seeds["event_id"] = -np.arange(1, len(seeds) + 1, dtype="int64")
    return seeds


class ParquetAppender:
    """Write tables one window at a time, as one Parquet file per table.

    Each call to :meth:`write` adds a row group to a working file next to
    ``<dest>/<name>.parquet``.  :meth:`finish` checks that every table the run
    owes was written and only then moves the working files into place, all of
    them or none, the way :class:`~ob_analytics.data.ParquetWriter` does.  So
    a run that fails part-way, or a move that fails, leaves the folder as it
    was rather than a file that reads as complete but holds only the first
    windows.  The files carry the same schema version, tick size, lot size,
    configuration, declarations, level and loader's counts metadata as one
    :class:`~ob_analytics.data.ParquetWriter` writes, so
    :func:`~ob_analytics.data.load_data` reads the folder as it would a single
    run's.

    Parameters
    ----------
    dest : str or Path
        The output folder, created when missing.
    config : PipelineConfig
        The run's configuration, recorded whole, as
        :class:`~ob_analytics.data.ParquetWriter` records it.
    declarations : FeedDeclarations, optional
        The run's declarations, recorded so
        :func:`~ob_analytics.data.load_result` gives them back.
    level : Level, optional
        The run's level, recorded in the same way.
    loader_counts : dict of str to int, optional
        The loader's counts over the whole run, recorded in the ``events``
        file as a single run records them.
    context : str, optional
        The function writing the files, at the start of an error message.
    part : str, optional
        What one call to :meth:`write` holds, in the same message.
    """

    def __init__(
        self,
        dest: str | Path,
        config: Any,
        declarations: FeedDeclarations | None = None,
        loader_counts: dict[str, int] | None = None,
        level: Level | None = None,
        context: str = "run_windows",
        part: str = "window",
    ) -> None:
        self.dest = Path(dest)
        self._context = context
        self._part = part
        self._config = config
        self.dest.mkdir(parents=True, exist_ok=True)
        self._tick_sizes = tick_sizes_from_config(config)
        self._lot_sizes = lot_sizes_from_config(config)
        self._declarations = declarations
        self._level = level
        self._loader_counts = loader_counts
        self._writers: dict[str, pq.ParquetWriter] = {}

    def _working(self, name: str) -> Path:
        return self.dest / f".{name}.parquet.partial"

    def write(
        self, name: str, frame: pd.DataFrame, *, like: pd.DataFrame | None = None
    ) -> None:
        """Append *frame* to the *name* table.

        The first write fixes the file's column types.  A column with no value
        in that first frame has no type of its own (Arrow's ``null``), so pass
        *like*, a frame holding the column's values for the whole run, and the
        type is read from its first value instead.

        The loader's counts on *frame*'s ``attrs`` cover that window only, so
        they are not recorded; the ``events`` file records the whole run's,
        as given to the constructor.
        """
        from ob_analytics.data import ROW_GROUP_ROWS, _to_arrow_table

        table = _to_arrow_table(
            frame,
            tick_sizes=self._tick_sizes,
            lot_sizes=self._lot_sizes,
            config=self._config,
            declarations=self._declarations,
            level=self._level,
            loader_counts=self._loader_counts if name == "events" else None,
        )
        writer = self._writers.get(name)
        if writer is None:
            schema = _typed(table.schema, like)
            writer = pq.ParquetWriter(self._working(name), schema)
            self._writers[name] = writer
        writer.write_table(
            _conform(table, writer.schema, name), row_group_size=ROW_GROUP_ROWS
        )

    def finish(self, names: Iterable[str]) -> None:
        """Close every file and move it into place, once all *names* are written.

        The files are moved all or none: when one move fails, the files
        already moved are put back as they were.

        Raises
        ------
        ConfigError
            If one of *names* was never written; nothing is moved.
        """
        from ob_analytics.data import _install_files

        missing = [name for name in names if name not in self._writers]
        if missing:
            raise ConfigError(
                f"{self._context}: no {self._part} produced rows for "
                f"{missing}, so the output would be incomplete"
            )
        for writer in self._writers.values():
            writer.close()
        _install_files(
            (self._working(name), self.dest / f"{name}.parquet")
            for name in self._writers
        )
        self._writers.clear()

    def abort(self) -> None:
        """Close and delete every working file, leaving the folder as it was."""
        for name, writer in self._writers.items():
            writer.close()
            self._working(name).unlink(missing_ok=True)
        self._writers.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        # A run that finished has already moved its files into place; anything
        # still open here belongs to a run that stopped part-way.
        self.abort()


#: Multiplying an order id by this odd number spreads the ids evenly over all
#: 64-bit values, and no two ids give the same value.  The order-type files
#: are sorted and split by it.
_SPREAD = np.uint64(0x9E3779B97F4A7C15)

#: What :class:`OrderTypeFiles` keeps of each part of :class:`_OrderFacts`,
#: besides the order id.
_FACT_COLUMNS: dict[str, tuple[str, ...]] = {
    "ids": (),
    "created": ("volume",),
    "deleted": ("volume",),
    "changed": (),
    "named": ("event_id",),
}


def _spread(ids: Any) -> np.ndarray:
    """Return the sort key of each order id: the id times :data:`_SPREAD`."""
    with np.errstate(over="ignore"):
        return np.asarray(ids, dtype=np.int64).view(np.uint64) * _SPREAD


class OrderTypeFiles:
    """The order types of a windowed run, worked out from files.

    An order's type is decided over the whole run, from a few values per
    order (:class:`~ob_analytics.analytics._OrderFacts`).  Held in memory,
    those values grow with the input.  This class writes them to *folder* one
    window at a time instead, and then decides the types one group of orders
    at a time, so it never holds more than one group.

    Each window's values are written sorted by a key made from the order id
    (:func:`_spread`), so a group is one range of keys: a slice of each
    window's rows.  The number of groups is chosen once every window is
    written, so that a group holds about :attr:`rows_per_group` rows.  The
    types are written to *folder* too, one file per group, and read back one
    window at a time by :meth:`with_types`.

    Parameters
    ----------
    folder : Path
        An empty working folder.  The files are left there; the caller deletes
        the folder.
    """

    #: About how many rows of order facts one group holds.
    rows_per_group: int = 1 << 20

    def __init__(self, folder: Path) -> None:
        self._folder = folder
        #: Per part of the facts, the number of rows each window wrote.
        self._chunks: dict[str, list[int]] = {name: [] for name in _FACT_COLUMNS}
        #: Bits of the key that name a group; ``None`` until :meth:`decide`.
        self._bits: int | None = None

    def _file(self, name: str) -> Path:
        return self._folder / name

    @property
    def empty(self) -> bool:
        """``True`` when no window held an order."""
        return not sum(self._chunks["ids"])

    def add(self, facts: _OrderFacts) -> None:
        """Write one window's order facts, sorted by key."""
        parts = {
            "ids": {"id": facts.ids},
            "created": facts.created,
            "deleted": facts.deleted,
            "changed": {"id": facts.changed},
            "named": facts.named,
        }
        for name, extra in _FACT_COLUMNS.items():
            frame = parts[name]
            keys = _spread(frame["id"])
            order = np.argsort(keys, kind="stable")
            with self._file(f"{name}.key").open("ab") as out:
                keys[order].tofile(out)
            for column in ("id", *extra):
                values = np.asarray(frame[column], dtype=np.int64)
                with self._file(f"{name}.{column}").open("ab") as out:
                    values[order].tofile(out)
            self._chunks[name].append(len(keys))

    def decide(self, trades: pd.DataFrame) -> None:
        """Decide every order's type, one group of orders at a time.

        *trades* are the whole run's.  The rule is the one
        :func:`~ob_analytics.analytics.set_order_types` applies, and one
        warning gives the number of orders whose type it cannot tell, as a
        single run gives.
        """
        rows = sum(sum(chunks) for chunks in self._chunks.values())
        groups = max(1, -(-rows // self.rows_per_group))
        self._bits = bits = (groups - 1).bit_length()
        # Only the two columns the rule reads.
        named = trades[["maker_event_id", "taker_event_id"]]
        unknown = 0
        for group in range(1 << bits):
            facts = self._group(group)
            types = _order_types(facts, named, warn=False)
            unknown += int((types == "unknown").sum())
            codes = ORDER_TYPE_DTYPE.categories.get_indexer(pd.Index(types))
            keys = _spread(types.index)
            order = np.argsort(keys)
            keys[order].tofile(self._file(f"types-{group}.key"))
            codes.astype(np.int8)[order].tofile(self._file(f"types-{group}.code"))
        for name, extra in _FACT_COLUMNS.items():
            for column in ("key", "id", *extra):
                self._file(f"{name}.{column}").unlink()
        _warn_on_unknown_orders(unknown, stacklevel=4)

    def _bounds(self, group: int) -> tuple[np.uint64 | None, np.uint64 | None]:
        """Return the first key of *group* and of the group after it."""
        assert self._bits is not None
        if not self._bits:
            return None, None
        shift = 64 - self._bits
        low = np.uint64(group << shift) if group else None
        high = np.uint64((group + 1) << shift) if group + 1 < 1 << self._bits else None
        return low, high

    def _group(self, group: int) -> _OrderFacts:
        """Read the order facts of *group* back from every window."""
        low, high = self._bounds(group)
        read: dict[str, dict[str, np.ndarray]] = {}
        for name, extra in _FACT_COLUMNS.items():
            columns = ("id", *extra)
            total = sum(self._chunks[name])
            if not total:
                read[name] = {c: np.empty(0, dtype=np.int64) for c in columns}
                continue
            # Mapped, not read: a search touches only a few pages of a file.
            keys = np.memmap(self._file(f"{name}.key"), dtype=np.uint64, mode="r")
            maps = {
                c: np.memmap(self._file(f"{name}.{c}"), dtype=np.int64, mode="r")
                for c in columns
            }
            slices = []
            start = 0
            for size in self._chunks[name]:
                chunk = keys[start : start + size]
                first = 0 if low is None else int(np.searchsorted(chunk, low))
                last = size if high is None else int(np.searchsorted(chunk, high))
                if last > first:
                    slices.append(slice(start + first, start + last))
                start += size
            read[name] = {
                c: np.concatenate(
                    [np.asarray(m[s]) for s in slices] or [np.empty(0, np.int64)]
                )
                for c, m in maps.items()
            }
            del keys, maps
        return _OrderFacts(
            ids=pd.Index(read["ids"]["id"]).unique(),
            created=pd.DataFrame(read["created"]),
            deleted=pd.DataFrame(read["deleted"]),
            changed=pd.Index(read["changed"]["id"]).unique(),
            named=pd.DataFrame(read["named"])[["event_id", "id"]],
        )

    def with_types(self, events: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of one window's *events* with each row's order type."""
        assert self._bits is not None, "decide() first"
        keys = _spread(events["id"])
        codes = np.full(len(keys), -1, dtype=np.int8)
        groups = keys >> np.uint64(64 - self._bits) if self._bits else None
        for group in np.unique(groups) if groups is not None else [0]:
            rows = slice(None) if groups is None else groups == group
            path = self._file(f"types-{group}.key")
            if not path.stat().st_size:
                continue
            # Mapped, not read: a window looks up only its own orders, so the
            # time per window does not grow with the run.
            found = np.memmap(path, dtype=np.uint64, mode="r")
            group_codes = np.memmap(
                self._file(f"types-{group}.code"), dtype=np.int8, mode="r"
            )
            wanted = keys[rows]
            at = np.minimum(np.searchsorted(found, wanted), len(found) - 1)
            known = found[at] == wanted
            codes[rows] = np.where(known, group_codes[at], -1)
            del found, group_codes
        return events.assign(
            type=pd.Categorical.from_codes(codes, dtype=ORDER_TYPE_DTYPE)
        )


def _typed(schema: pa.Schema, like: pd.DataFrame | None) -> pa.Schema:
    """Give each ``null``-typed field of *schema* the type *like* holds there."""
    if like is None:
        return schema
    for i, field in enumerate(schema):
        if not pa.types.is_null(field.type) or field.name not in like.columns:
            continue
        first = like[field.name].first_valid_index()
        if first is None:
            continue
        value = pa.array([like[field.name].loc[first]]).type
        schema = schema.set(i, field.with_type(value))
    return schema


def _conform(table: pa.Table, schema: pa.Schema, name: str) -> pa.Table:
    """Cast *table* to the *schema* its file was opened with.

    A window can type a column differently from the file when the column holds
    nothing in it: a column of missing values reads as Arrow's ``null`` type.
    The values are the same, so a cast settles it.
    """
    if table.schema.equals(schema, check_metadata=False):
        return table
    try:
        return table.select(schema.names).cast(schema)
    except (KeyError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise ConfigError(
            f"run_windows: the {name} table of one window has columns or types "
            f"the first window's did not, so the windows cannot share one file: "
            f"{exc}"
        ) from exc
