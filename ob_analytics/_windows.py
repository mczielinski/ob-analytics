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
from ob_analytics.exceptions import ConfigError

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


def _as_utc(value: Any) -> pd.Timestamp:
    """Return *value* as a tz-aware UTC timestamp; no zone is read as UTC."""
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        raise ConfigError(f"run_windows: {value!r} is not a time")
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
    cuts = np.array(
        [start.as_unit("ns").value for start, _ in windows[1:] if start is not None],
        dtype="int64",
    )
    index = pd.DatetimeIndex(timestamps).as_unit("ns")
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    times = index.to_numpy().view("int64")
    labels = np.searchsorted(cuts, times, side="right")
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
    owes was written and only then moves the working files into place, so a
    run that fails part-way leaves the folder as it was rather than a file that
    reads as complete but holds only the first windows.  The files carry the
    same schema version, tick size and lot size metadata as one
    :class:`~ob_analytics.data.ParquetWriter` writes, so
    :func:`~ob_analytics.data.load_data` reads the folder as it would a single
    run's.

    Parameters
    ----------
    dest : str or Path
        The output folder, created when missing.
    config : PipelineConfig
        The run's configuration; its tick and lot sizes are recorded.
    """

    def __init__(self, dest: str | Path, config: Any) -> None:
        from ob_analytics.data import _lot_sizes_from_config, _tick_sizes_from_config

        self.dest = Path(dest)
        self.dest.mkdir(parents=True, exist_ok=True)
        self._tick_sizes = _tick_sizes_from_config(config)
        self._lot_sizes = _lot_sizes_from_config(config)
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
        """
        from ob_analytics.data import _to_arrow_table

        table = _to_arrow_table(
            frame, tick_sizes=self._tick_sizes, lot_sizes=self._lot_sizes
        )
        writer = self._writers.get(name)
        if writer is None:
            schema = _typed(table.schema, like)
            writer = pq.ParquetWriter(self._working(name), schema)
            self._writers[name] = writer
        writer.write_table(_conform(table, writer.schema, name))

    def finish(self, names: Iterable[str]) -> None:
        """Close every file and move it into place, once all *names* are written.

        Raises
        ------
        ConfigError
            If one of *names* was never written; nothing is moved.
        """
        missing = [name for name in names if name not in self._writers]
        if missing:
            raise ConfigError(
                f"run_windows: no window produced rows for {missing}, so the "
                "output would be incomplete"
            )
        for name, writer in self._writers.items():
            writer.close()
            self._working(name).replace(self.dest / f"{name}.parquet")
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
