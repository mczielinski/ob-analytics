"""Bitstamp data format: event loading, trade reading, and writing.

Contains the symmetric set of Bitstamp-specific components:

* :class:`BitstampLoader` — loads Bitstamp CSV event data
* :class:`BitstampTradeReader` — reads companion ``trades.csv`` (live capture)
* :class:`BitstampWriter` — writes events back to Bitstamp CSV
* :class:`BitstampSource` — the source descriptor (offline replay + live capture)

Format-agnostic analytics (e.g. :func:`~ob_analytics.analytics.order_aggressiveness`)
live in :mod:`ob_analytics.analytics`.
"""

from __future__ import annotations

import math
import tempfile
import warnings
from collections import deque
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

from ob_analytics._utils import (
    FrameFiles,
    attach_ingest_seq,
    checked_labels,
    checked_times,
    datetime_to_epoch,
    empty_trades,
    epoch_ns,
    finite_floats,
    lots_to_size,
    read_csv_file,
    read_csv_parts,
    require_per_order_events,
    rows_counted_from,
    ticks_to_price,
    to_grid,
    validate_non_empty,
    window_labels,
)
from ob_analytics.capture_files import (
    ORDERS_CSV_COLUMNS,
    ORDERS_NAME,
    TRADES_CSV_COLUMNS,
    TRADES_NAME,
    csv_columns,
    record_format,
)
from ob_analytics.capture_record import CURRENT_FORMAT, FORMAT_TIME_UNITS, read_format
from ob_analytics.config import InstrumentGrid, PipelineConfig, SourceSettings
from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    FeedType,
    Level,
    LoadedWindow,
    RunContext,
    TradeAttribution,
    TradeReader,
    WriteTarget,
)
from ob_analytics.schemas import (
    ACTION_DTYPE,
    BOOK_SIDE_DTYPE,
    ORIGIN_COLUMN,
    SEQUENCE_COLUMN,
    TRADE_DTYPES,
    TRADE_SIDE_DTYPE,
    attach_instrument_identity,
    cast_columns,
    cast_events,
    unsized_order_ids,
    validate_columns,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ob_analytics.live._base import CaptureConfig, EventDict

__all__ = [
    "BitstampLoader",
    "BitstampSource",
    "BitstampTradeReader",
    "BitstampWriter",
]

# ── BitstampLoader ────────────────────────────────────────────────────

#: How many order ids the duplicate-cancellation warning names.
_SHOWN_IDS = 10

#: How many shares :meth:`BitstampLoader.load_windows` splits the orders into
#: by their id.  A group of orders is one share or more.
_ID_SHARES = 256


class BitstampLoader:
    """Load raw limit-order events from a Bitstamp-format CSV.

    Satisfies the :class:`~ob_analytics.protocols.EventLoader` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration: the instrument's ``tick_size`` and
        ``lot_size``.  An order's price that is not a whole number of ticks
        raises ``ConfigError``; a size between two lots is rounded with a
        warning.  A trade's price can lie between two ticks, so it is rounded
        to the nearest tick, not checked.
    venue, symbol : str, optional
        Optional instrument identity.  When either is supplied, the loaded
        frame gains per-row ``venue`` / ``symbol`` columns; ``venue`` falls
        back to ``"bitstamp"`` when only ``symbol`` is given.  Each one not
        supplied is taken from the capture's ``meta.json`` when it records
        one, so a replayed capture names its instrument.  With neither, the
        frame is left untagged.
    """

    #: The source venue used to fill the ``venue`` column when identity tagging
    #: is on but no explicit venue was supplied.
    _VENUE = "bitstamp"

    #: How many rows :meth:`load_windows` reads from the file at a time.
    read_size: int = 250_000
    #: About how many rows one group of orders holds in :meth:`load_windows`.
    rows_per_group: int = 1 << 19

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        venue: str | None = None,
        symbol: str | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._venue = venue
        self._symbol = symbol
        #: The trades of ``trades.csv`` and the events they name, as the last
        #: :meth:`load_windows` matched them, for :class:`BitstampTradeReader`.
        #: ``None`` before its last window is read, and after :meth:`load`.
        self.trade_matches: _TradeMatches | None = None

    def load(self, path: str | Path) -> pd.DataFrame:
        """Read *path* CSV and return a cleaned events DataFrame.

        Parameters
        ----------
        path : str or Path
            Path to a Bitstamp-format CSV with columns ``id``,
            ``timestamp``, ``exchange_timestamp``,
            ``price``, ``volume``, ``action``, ``direction``.

        Returns
        -------
        pandas.DataFrame
            Cleaned events with columns ``id``, ``timestamp``,
            ``exchange_timestamp``, ``price``, ``volume``, ``action``,
            ``direction``, ``event_id``, ``fill``, ``original_number``.

        Notes
        -----
        Each order's rows come in the order they happened: the ``created``
        row first, then the rest by ``timestamp``.  Rows that share an
        instant follow the file's ``sequence`` column when it has one; a
        ``deleted`` row then goes last, and the rest keep their order in the
        file.  A row with no sequence number keeps its place after the row
        before it in the file.  ``event_id`` numbers the rows of all orders
        by the same keys: time, then ``sequence``, then file order.  The
        order ids take no part.  ``fill`` is the drop in size from the order's
        previous row.  A row whose size rises (an order changed to a larger
        amount) has ``fill`` 0.
        """
        self.trade_matches = None
        where = f"BitstampLoader.load({path})"
        capture_format = read_format(path)
        events = self._read_rows(
            read_csv_file(path, where), capture_format.time_unit, where
        )
        if SEQUENCE_COLUMN in events.columns:
            sequence = events[SEQUENCE_COLUMN].ffill()
            events["_sequence"] = sequence.fillna(sequence.min())
        events, removed = _order_lives(events)
        _warn_on_duplicates(removed)
        _number_events(events, first=1)
        return self._finish(events, capture_format)

    def load_windows(
        self,
        path: str | Path,
        windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
        *,
        working: Path | None = None,
    ) -> Iterator[LoadedWindow]:
        """Yield the events of each time window of *path* in turn.

        An order's rows are put in order, and its fills worked out, from all
        of its rows, and an order can last over many windows.  So the rows go
        through files in *working* twice.  The file is read :attr:`read_size`
        rows at a time, and each part's rows are split by order id into
        files.  Then the rows of one group of orders at a time, about
        :attr:`rows_per_group` rows, are read back, put in order, and split
        into one file per window.  Last, each window is read back on its own
        and its events are numbered.  So no more than one part, one group or
        one window is in memory at once, and the file need not be in time
        order.  The events are the rows :meth:`load` gives, cut at the
        windows.

        The trades of the companion ``trades.csv`` are matched to their
        orders one group at a time too, by the rules
        :class:`BitstampTradeReader` applies.  Once the last window is
        yielded, :attr:`trade_matches` holds the trades and the events they
        name, so the trade reader builds the whole run's trades without the
        events.

        Parameters
        ----------
        path : str or Path
            The orders file, as for :meth:`load`.
        windows : list of tuple
            ``(start, end)`` per window, tz-aware UTC, ``None`` for an open
            end, covering the whole file: what
            :func:`~ob_analytics._windows.window_bounds` returns.
        working : Path, optional
            An empty folder for the files, which the caller deletes.  By
            default a temporary folder is made and deleted.

        Yields
        ------
        LoadedWindow
            The window's events, and the ``event_id`` of each of them a trade
            names.

        Raises
        ------
        FileNotFoundError
            If there is no ``trades.csv`` next to *path*.
        """
        with ExitStack() as stack:
            if working is None:
                working = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            yield from self._load_windows(path, windows, working)

    def _load_windows(
        self,
        path: str | Path,
        windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
        working: Path,
    ) -> Iterator[LoadedWindow]:
        """Yield the windows of *path*, sorting its rows through *working*."""
        self.trade_matches = None
        where = f"BitstampLoader.load({path})"
        logger.info("BitstampLoader: reading {} one window at a time", path)
        capture_format = read_format(path)
        unit = capture_format.time_unit

        # First, every part of the file, split by order id.
        shares = FrameFiles(working / "orders")
        dtypes: dict[str, list[Any]] = {}
        template: pd.DataFrame | None = None
        read = kept = 0
        # The sequence number the last part ended on, and the lowest of all.
        carried: Any = None
        lowest: Any = None
        for raw in read_csv_parts(path, where, self.read_size):
            with rows_counted_from(read + 1):
                rows = self._read_rows(raw, unit, where, kept=kept)
            read += len(raw)
            kept += len(rows)
            if SEQUENCE_COLUMN in rows.columns:
                sequence = rows[SEQUENCE_COLUMN].ffill()
                if carried is not None:
                    sequence = sequence.fillna(carried)
                numbered = sequence.dropna()
                if len(numbered):
                    carried = numbered.iloc[-1]
                    least = numbered.min()
                    lowest = least if lowest is None else min(lowest, least)
                rows["_sequence"] = sequence
            for column in rows.columns:
                dtypes.setdefault(column, []).append(rows[column].dtype)
            if template is None:
                template = rows.iloc[:0]
            shares.split(rows, _id_shares(rows["id"]))
        if template is None:
            # No rows: read the header alone, for the error a whole read gives.
            template = self._read_rows(read_csv_file(path, where, nrows=0), unit, where)
        # The type a whole read gives a column that came straight from the
        # file, where the parts read it as different types.
        common = {
            column: np.result_type(*found)
            for column, found in dtypes.items()
            if len(set(found)) > 1 and all(isinstance(d, np.dtype) for d in found)
        }

        # Then each group of orders, put in order and split by window.
        trade_file = _read_trade_file(path, self._config)
        matcher = None if trade_file is None else _TradeMatcher(trade_file)
        by_window = FrameFiles(working / "windows")
        removed = []
        blank = None
        for group in _share_groups(dict(shares.rows), self.rows_per_group):
            events = pd.concat([part for n in group for part in shares.take(n)])
            events, gone = _group_lives(events, lowest)
            removed.append(gone)
            if blank is None:
                blank = events.iloc[:0]
            if matcher is not None:
                matcher.match(events)
            by_window.split(events, window_labels(events["timestamp"], windows))
        if blank is None:
            blank, _ = _group_lives(template, lowest)
        if removed:
            _warn_on_duplicates(pd.concat(removed).sort_values(kind="stable"))

        # Last, each window, numbered as in a whole read.
        named = np.empty(0, np.int64) if matcher is None else matcher.named()
        found: dict[Any, Any] = {}
        first = 1
        for number in range(len(windows)):
            parts = by_window.take(number)
            events = pd.concat(parts) if parts else blank
            # The order of a whole read: by order, then each order's own.
            events = (
                events.sort_values(["id", "_rank"], kind="stable")
                .drop(columns="_rank")
                .reset_index(drop=True)
            )
            for column, dtype in common.items():
                if column in events.columns:
                    events[column] = events[column].astype(dtype)
            _number_events(events, first=first)
            first += len(events)
            events = self._finish(events, capture_format)
            hit = events.loc[events["original_number"].isin(named)]
            found.update(zip(hit["original_number"], hit["event_id"]))
            yield LoadedWindow(events, hit["event_id"].to_numpy())
        self.trade_matches = (
            _TradeMatches(None, None, None, {}, 0)
            if matcher is None
            else matcher.finish(found)
        )

    def _read_rows(
        self, events: pd.DataFrame, unit: str, where: str, *, kept: int = 0
    ) -> pd.DataFrame:
        """Return the rows of the orders file *events* checked and on the grid.

        The rows stay in file order.  *kept* is the number of rows before
        *events* in the file that were kept, for the ingest counter.
        """
        validate_columns(
            events,
            {
                "id",
                "timestamp",
                "exchange_timestamp",
                "price",
                "volume",
                "action",
                "direction",
            },
            "BitstampLoader.load",
        )
        validate_non_empty(events, "BitstampLoader.load")
        # Checked before the filter below, which would drop an empty size and
        # break on a word, so the row numbers are the file's.
        for column, what in (("price", "price"), ("volume", "size")):
            events[column] = finite_floats(events[column], what, where)
        checked_labels(
            events["action"], "action", where, allowed=ACTION_DTYPE.categories
        )
        checked_labels(
            events["direction"], "direction", where, allowed=BOOK_SIDE_DTYPE.categories
        )
        events["timestamp"] = checked_times(
            events["timestamp"], unit, "timestamp", where
        )
        events["exchange_timestamp"] = checked_times(
            events["exchange_timestamp"],
            unit,
            "exchange_timestamp",
            where,
            required=False,
        )

        events = events[events["volume"] >= 0]
        # Which part of a capture wrote each row, when the file says: the clock
        # checks leave out the opening book's rows (see ORIGIN_COLUMN).
        if ORIGIN_COLUMN in events.columns:
            events[ORIGIN_COLUMN] = events[ORIGIN_COLUMN].astype("string")

        # Ordering keys, captured in file (arrival) order before the id-sort
        # reorders the rows (opt-in, so the default frame is unchanged; see
        # ob_analytics.schemas).  The public Bitstamp diff feed publishes a
        # microtimestamp, not a sequence, so ``sequence`` is normally absent and
        # only carried (as nullable Int64) when a capture recorded one.
        if self._config.track_sequence:
            if SEQUENCE_COLUMN in events.columns:
                events[SEQUENCE_COLUMN] = events[SEQUENCE_COLUMN].astype("Int64")
            events = attach_ingest_seq(events, start=kept)

        events = events.reset_index().rename(columns={"index": "original_number"})
        events["original_number"] = events["original_number"] + 1
        # Canonical size is integer lots (issue #226) and canonical price
        # integer ticks (issue #155); the raw CSV carries a base-asset float and
        # a quote-currency float, so both convert on the way in.  A price off
        # the tick grid is refused (issue #500).
        events["volume"] = to_grid(
            events["volume"],
            self._config.lot_size,
            kind="size",
            column="volume",
            where=where,
        )
        events["price"] = to_grid(
            events["price"],
            self._config.tick_size,
            kind="price",
            column="price",
            where=where,
        )

        events["action"] = events["action"].astype(ACTION_DTYPE)
        events["direction"] = events["direction"].astype(BOOK_SIDE_DTYPE)
        return events

    def _finish(self, events: pd.DataFrame, capture_format: Any) -> pd.DataFrame:
        """Return numbered *events* with the instrument identity, cast."""
        events["raw_event_type"] = pd.NA

        # Optional per-row instrument identity (issue #147): the run's, else
        # what the capture recorded (issue #494); a no-op when neither names it.
        events = attach_instrument_identity(
            events,
            venue=self._venue or capture_format.venue,
            symbol=self._symbol or capture_format.symbol,
            default_venue=self._VENUE,
        )

        return cast_events(events)


def _order_lives(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Put each order's rows in order and work out their fills and times.

    *events* holds every row of each of its orders, in file order, with a
    ``_sequence`` column when the file has a sequence.  Returns the rows
    sorted by order, without the duplicate cancellations, and the ids of the
    rows left out.  Each row gets the keys :func:`_number_events` numbers the
    events by, so the rows of a set of orders can be numbered with those of
    other orders later.
    """
    # Each order's rows in the order they happened: the ``created`` row
    # first, then the rest by time.  Size cannot give this order, because
    # some captures in this format raise an order's size.  Rows that share
    # an instant follow the venue's sequence when the file has one; a
    # ``deleted`` row then goes last, since it ends the order, and the
    # rest keep their file order (the sort is stable).  A row with no
    # sequence number (a capture can write a fill it took from the trade
    # tape that way) takes the number of the row before it in the file,
    # so it stays after the message it arrived after.  Rows before the
    # first numbered one take the lowest number, so they stay first.
    order_keys = ["id", "_later", "timestamp"]
    if "_sequence" in events.columns:
        order_keys.append("_sequence")
    order_keys.append("action")
    events = (
        events.assign(_later=events["action"] != "created")
        .sort_values(by=order_keys, kind="stable")
        .drop(columns="_later")
    )

    events, removed = _without_duplicate_deletes(events)

    # A volume drop between consecutive events for the same id is a fill.
    # Bitstamp does not allow order amendments, so a price change between
    # events is the matching engine reporting the fill price (taker
    # orders) or the order walking the book (aggressors), not an in-place
    # modification.  A rise is never a fill: the zero-size ``created`` row
    # of an instant order is followed by one, and some captures in this
    # format raise an order's size.  Integer lots, so the difference is
    # exact and needs no rounding back onto the size grid; ``diff`` still
    # yields a float column because the first row per id is NaN, so cast
    # once it is filled.
    fill_deltas = events.groupby("id")["volume"].diff().fillna(0)
    events["fill"] = (-fill_deltas).clip(lower=0).astype("int64")

    # Sort timestamps within each id.  The rows are already in time order
    # apart from the ``created`` row, which comes first even when a later
    # row is stamped before it; this gives it the order's earliest time.
    # The frame is id-ordered (primary key of the sort above;
    # _without_duplicate_deletes only filters rows), so a stable global (id,
    # timestamp) lexsort gathered back positionally is equivalent to the
    # per-group sort — without the per-group Python lambda, which
    # dominated load time (~19s on the bundled sample).  The lexsort key
    # is the raw UTC-ns int64; the tz-aware column is reordered through
    # its ExtensionArray so the zone and unit survive (issue #154).
    ts_i8 = epoch_ns(events["timestamp"])
    ids = events["id"].to_numpy()
    order = np.lexsort((ts_i8, ids))
    events["timestamp"] = events["timestamp"].array[order]

    # The keys after time that number the events across orders: the venue's
    # sequence when the file has one, then file order.  Order ids say nothing
    # about time (Independent Reserve writes this format with UUIDs), so they
    # take no part.  Each order's rows, in the order set above, take that
    # order's sequence numbers and file positions in rising order, gathered
    # back the same way as the timestamps.  Within an order every key then
    # rises, so its own order is kept.
    def per_order(values: np.ndarray) -> np.ndarray:
        return values[np.lexsort((values, ids))]

    events["_tie_number"] = per_order(events["original_number"].to_numpy())
    if "_sequence" in events.columns:
        sequence = events.pop("_sequence")
        if pd.api.types.is_numeric_dtype(sequence):
            values = sequence.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            values = sequence.to_numpy(dtype=object)
        events["_tie_sequence"] = per_order(values)
    return events, removed


def _group_lives(events: pd.DataFrame, lowest: Any) -> tuple[pd.DataFrame, pd.Series]:
    """Run :func:`_order_lives` on one group of orders read a window at a time.

    The rows before the first numbered one in the file take *lowest*, the
    lowest sequence number of the whole file.  Each row is given its place in
    the group as ``_rank``: the rows of one order rise in it, in their order.
    """
    if "_sequence" in events.columns and lowest is not None:
        events = events.assign(_sequence=events["_sequence"].fillna(lowest))
    events, removed = _order_lives(events)
    events["_rank"] = np.arange(len(events), dtype=np.int64)
    return events, removed


def _number_events(events: pd.DataFrame, *, first: int) -> None:
    """Number *events* in place by time, then the keys :func:`_order_lives` gave.

    The keys are taken off the frame.  The numbers count from *first*, the
    number of the earliest of *events*.
    """
    keys = [events.pop("_tie_number").to_numpy()]
    if "_tie_sequence" in events.columns:
        keys.append(events.pop("_tie_sequence").to_numpy())
    by_time = np.lexsort((*keys, epoch_ns(events["timestamp"])))
    event_id = np.empty(len(events), dtype=np.int64)
    event_id[by_time] = np.arange(first, first + len(events))
    events["event_id"] = event_id


def _without_duplicate_deletes(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop redundant 'deleted' events that repeat a cancellation for one order id.

    Returns the events kept and the order id of each row dropped.
    """
    deletes = events[events["action"] == "deleted"].sort_values(
        by=["id", "volume"], kind="stable"
    )
    dup_ids = deletes.loc[deletes["id"].duplicated(), "id"]
    duplicate_deletes = deletes[deletes["id"].isin(dup_ids)]
    duplicate_rows = duplicate_deletes.loc[
        duplicate_deletes["id"].duplicated(), "original_number"
    ]
    dropped = events["original_number"].isin(duplicate_rows)
    return events[~dropped], events.loc[dropped, "id"]


def _warn_on_duplicates(removed_ids: pd.Series) -> None:
    """Warn that the rows of *removed_ids* repeated a cancellation, when any did."""
    rem_dup = len(removed_ids)
    if rem_dup > 0:
        removed_ids = removed_ids.astype(str)
        shown = " ".join(removed_ids.iloc[:_SHOWN_IDS])
        more = " ..." if len(removed_ids) > _SHOWN_IDS else ""
        warnings.warn(
            f"BitstampLoader: removed {rem_dup} duplicate order "
            f"cancellations, which repeat a cancellation already in the "
            f"file (order ids {shown}{more})",
            stacklevel=2,
        )


def _id_shares(ids: pd.Series) -> np.ndarray:
    """Return the share of the orders each of *ids* falls in, from 0."""
    hashed = pd.util.hash_pandas_object(ids, index=False).to_numpy()
    return (hashed % np.uint64(_ID_SHARES)).astype(np.int64)


def _share_groups(rows: dict[int, int], most: int) -> Iterator[list[int]]:
    """Yield the shares of the orders a group at a time.

    *rows* gives the number of rows of each share.  A group is a run of
    shares that together hold no more than *most* rows, or one share that
    holds more.
    """
    group: list[int] = []
    held = 0
    for share in sorted(rows):
        if group and held + rows[share] > most:
            yield group
            group, held = [], 0
        group.append(share)
        held += rows[share]
    if group:
        yield group


# ── BitstampTradeReader ───────────────────────────────────────────────

#: The unit the writer saves times in: that of the current capture format.
_WRITE_UNIT = FORMAT_TIME_UNITS[CURRENT_FORMAT]


@dataclass(frozen=True)
class _TradeFile:
    """The rows of a ``trades.csv`` file, read and checked.

    Under the current column names, with the times on the UTC clock, the
    sizes in lots, and the orders each trade names as maker and taker.
    """

    where: str
    raw: pd.DataFrame
    received: pd.Series
    venue: pd.Series
    amounts: pd.Series
    maker_id: np.ndarray
    taker_id: np.ndarray


def _trades_path(path: Any) -> Path:
    """Return the ``trades.csv`` next to *path*.

    Raises
    ------
    FileNotFoundError
        If there is none.
    """
    p = Path(path)
    if p.is_file():
        p = p.parent
    candidate = p / TRADES_NAME
    if not candidate.exists():
        raise FileNotFoundError(
            f"BitstampTradeReader: no trades.csv next to {path!r} "
            f"(looked at {candidate}). Capture trades alongside orders "
            f"with `ob-analytics capture bitstamp`."
        )
    return candidate


def _read_trade_file(path: Any, config: PipelineConfig) -> _TradeFile | None:
    """Read the ``trades.csv`` next to *path*; ``None`` when it has no rows."""
    file = _trades_path(path)
    where = f"BitstampTradeReader.load({file})"
    capture_format = read_format(file)
    raw = read_csv_file(file, where)

    if raw.empty:
        return None

    columns = csv_columns(TRADES_NAME, capture_format.version)
    validate_columns(raw, set(columns), "BitstampTradeReader.load")
    # Read a format version 1 file under the current names.
    raw = raw.rename(columns=dict(zip(columns, TRADES_CSV_COLUMNS, strict=True)))
    for column, what in (("price", "price"), ("volume", "size")):
        raw[column] = finite_floats(raw[column], what, where)
    # A trade with no known side or order (as this format's writer saves
    # one from a feed that does not name them) leaves the cell empty.
    checked_labels(
        raw["direction"],
        "direction",
        where,
        allowed=TRADE_SIDE_DTYPE.categories,
        required=False,
    )
    received = checked_times(
        raw["timestamp"], capture_format.time_unit, "timestamp", where
    )
    venue = checked_times(
        raw["exchange_timestamp"],
        capture_format.time_unit,
        "exchange_timestamp",
        where,
        required=False,
    )

    # `direction` is the taker side: "buy" => taker is the buyer.
    taker_buys = raw["direction"].astype(str) == "buy"
    taker_id = np.where(taker_buys, raw["buy_order_id"], raw["sell_order_id"])
    maker_id = np.where(taker_buys, raw["sell_order_id"], raw["buy_order_id"])

    # Trade sizes share the events' integer-lot grid (issue #226), so the
    # fill matching compares exact integers rather than rounded floats.
    amounts = pd.Series(
        to_grid(
            raw["volume"],
            config.lot_size,
            kind="size",
            column="volume",
            where=where,
        ),
        index=raw.index,
    )
    return _TradeFile(where, raw, received, venue, amounts, maker_id, taker_id)


@dataclass(frozen=True)
class _TradeMatches:
    """The trades of a run read a window at a time, and the events they name.

    What :meth:`BitstampLoader.load_windows` leaves for
    :class:`BitstampTradeReader`.  ``trades`` is ``None`` when the file has
    no trades.
    """

    trades: _TradeFile | None
    maker_event_id: np.ndarray | None
    taker_event_id: np.ndarray | None
    #: The ``original_number`` of each event a trade names, by ``event_id``.
    id_to_og: dict[Any, Any]
    #: How many trades were matched to an instant order by its id.
    by_order: int


class _TradeMatcher:
    """Match the trades to their orders one group of orders at a time.

    A trade names its maker and taker orders, and the matching rules of
    :class:`BitstampTradeReader` read only the rows of the order named.  So
    the trades of a run can be matched one group of orders at a time.  An
    event's ``event_id`` is not known until its window is read, so a match is
    kept as the event's ``original_number`` until then.
    """

    def __init__(self, trades: _TradeFile) -> None:
        self._trades = trades
        n = len(trades.amounts)
        #: The ``original_number`` of the event each trade names, or -1.
        self._maker = np.full(n, -1, dtype=np.int64)
        self._taker = np.full(n, -1, dtype=np.int64)
        self._by_order = 0

    def match(self, events: pd.DataFrame) -> None:
        """Match the trades that name an order of *events*, every row of them.

        *events* is what :func:`_group_lives` returns.  Its ``_rank`` stands
        in for the ``event_id`` the rules read, since it rises with it within
        an order.
        """
        trades = self._trades
        stand_in = events[["id", "fill", "action", "volume"]].assign(
            event_id=events["_rank"].to_numpy()
        )
        lookup = BitstampTradeReader._build_lookup(stand_in)
        maker = BitstampTradeReader._resolve_event_ids(
            trades.maker_id, trades.amounts, lookup
        )
        taker = BitstampTradeReader._resolve_event_ids(
            trades.taker_id, trades.amounts, lookup
        )
        self._by_order += BitstampTradeReader._match_quote_amount_takers(
            stand_in, trades.taker_id, taker
        )
        number = events["original_number"].to_numpy()
        for ranks, out in ((maker, self._maker), (taker, self._taker)):
            hit = ~pd.isna(ranks)
            out[hit] = number[ranks[hit].astype(np.int64)]

    def named(self) -> np.ndarray:
        """Return the ``original_number`` of every event a trade names."""
        both = np.concatenate([self._maker, self._taker])
        return np.unique(both[both >= 0])

    def finish(self, event_ids: dict[Any, Any]) -> _TradeMatches:
        """Return the matches, given the ``event_id`` of each event named.

        *event_ids* maps the ``original_number`` of each event a trade names
        to its ``event_id``.
        """

        def as_event_ids(numbers: np.ndarray) -> np.ndarray:
            # As the trade reader builds them: an int per match, NaN for none.
            return np.array(
                [int(event_ids[n]) if n >= 0 else float("nan") for n in numbers],
                dtype=object,
            )

        return _TradeMatches(
            self._trades,
            as_event_ids(self._maker),
            as_event_ids(self._taker),
            {eid: og for og, eid in event_ids.items()},
            self._by_order,
        )


class BitstampTradeReader:
    """Build trades from the companion ``trades.csv`` that
    ``ob-analytics capture bitstamp`` writes next to ``orders.csv``.

    Reads each row from ``<source_dir>/trades.csv`` and projects it into
    the canonical trades schema, joining against the events frame to
    resolve ``buy_order_id`` / ``sell_order_id`` into ``event_id`` and
    ``original_number`` references.  It keeps the venue's time of each trade
    as ``exchange_timestamp``, empty where the file has none.

    A trade is matched to a row of the order it names whose ``fill`` is the
    trade's size, the earliest such row first.  Bitstamp does not always
    report that size: an instant order bought by quote amount is created
    with size 0 and then shows a size of 0 or one lot, never what it
    bought.  When no row shows the trade's size and the taker the trade
    names is such an order (one created with size 0), the trade is matched
    to that order's ``deleted`` row, else its latest row, and every such
    trade of that order names the same row.  The order and its type are
    then right, but its rows show less fill than its trades.
    :func:`~ob_analytics.quality.data_quality_summary` counts these trades
    as ``trades_matched_by_order``.

    Every other trade is matched on size only.  A trade stays unresolved
    (NaN) when no row of its order shows its size: the events have no row
    of that order, or a message was lost.  The same rules apply to captures
    from other venues read through this reader.

    Satisfies the :class:`~ob_analytics.protocols.TradeReader` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration: the instrument's ``tick_size`` and
        ``lot_size``.
    loader : BitstampLoader, optional
        The loader that read the events.  When it read them one window at a
        time (:meth:`BitstampLoader.load_windows`), it has matched the trades
        already, and :meth:`load`, given an empty events frame, builds the
        trades from what it kept.  :class:`BitstampSource` passes its loader.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        loader: BitstampLoader | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._loader = loader

    def load(self, events: pd.DataFrame, path: Any) -> pd.DataFrame:
        matches = None if self._loader is None else self._loader.trade_matches
        if matches is not None and events.empty:
            if matches.trades is None:
                return empty_trades()
            assert matches.maker_event_id is not None
            assert matches.taker_event_id is not None
            return self._frame(
                matches.trades,
                matches.maker_event_id,
                matches.taker_event_id,
                matches.id_to_og,
                matches.by_order,
            )

        trade_file = _read_trade_file(path, self._config)
        if trade_file is None:
            return empty_trades()
        ev_lookup = self._build_lookup(events)
        maker_event_id = self._resolve_event_ids(
            trade_file.maker_id, trade_file.amounts, ev_lookup
        )
        taker_event_id = self._resolve_event_ids(
            trade_file.taker_id, trade_file.amounts, ev_lookup
        )
        by_order = self._match_quote_amount_takers(
            events, trade_file.taker_id, taker_event_id
        )
        # A Series keyed by event_id, so the lookup is one vectorised index
        # probe.  A dict built over every event cost about 1.8 s at 5M events
        # to serve a few thousand trades (#203).
        id_to_og = pd.Series(
            events["original_number"].to_numpy(), index=events["event_id"].to_numpy()
        )
        return self._frame(
            trade_file, maker_event_id, taker_event_id, id_to_og, by_order
        )

    def _frame(
        self,
        trade_file: _TradeFile,
        maker_event_id: np.ndarray,
        taker_event_id: np.ndarray,
        id_to_og: dict[Any, Any] | pd.Series,
        by_order: int,
    ) -> pd.DataFrame:
        """Return the trades frame of *trade_file* with the events matched."""
        raw = trade_file.raw
        maker_og = pd.Series(maker_event_id).map(id_to_og)
        taker_og = pd.Series(taker_event_id).map(id_to_og)

        trades = pd.DataFrame(
            {
                # ``.array`` keeps the tz-aware UTC dtype (``.values`` would
                # drop the zone to a naive numpy datetime64).
                "timestamp": trade_file.received.array,
                "exchange_timestamp": trade_file.venue.array,
                # Trade prints share the events' integer-tick price grid (#155).
                # An execution can print between two ticks, so its price is
                # rounded to the nearest tick, not checked (issue #500).
                "price": to_grid(
                    raw["price"],
                    self._config.tick_size,
                    kind="price",
                    column="price",
                    where=trade_file.where,
                    checked=np.zeros(len(raw), dtype=bool),
                ),
                "volume": trade_file.amounts.values,
                "direction": pd.Categorical(
                    raw["direction"].astype(str), dtype=TRADE_SIDE_DTYPE
                ),
                "maker_event_id": maker_event_id,
                "taker_event_id": taker_event_id,
                "maker": trade_file.maker_id,
                "taker": trade_file.taker_id,
                "maker_og": maker_og.values,
                "taker_og": taker_og.values,
            }
        )

        trades = cast_columns(
            trades.sort_values("timestamp", kind="stable").reset_index(drop=True),
            TRADE_DTYPES,
        )

        logger.info(
            "BitstampTradeReader: {} trades ({} maker_event_id resolved, "
            "{} taker_event_id resolved, {} of them instant orders matched by "
            "order id)",
            len(trades),
            trades["maker_event_id"].notna().sum(),
            trades["taker_event_id"].notna().sum(),
            by_order,
        )
        return trades

    @staticmethod
    def _order_key(oid: Any) -> Any | None:
        """Hashable lookup key for an order id, or ``None`` when there is none.

        Bitstamp publishes integer order ids, but this reader also serves
        captures from other venues replayed through the same schema: some
        publish opaque ids (cryptofeed's independent_reserve UUIDs), and a
        public trade tape carries no maker/taker id at all.  Integers keep
        their exact previous behaviour; anything else falls back to its string
        form, and a missing id returns ``None`` so the trade resolves to NaN
        rather than raising.
        """
        if oid is None:
            return None
        if isinstance(oid, float) and math.isnan(oid):
            return None
        try:
            return int(oid)
        except (TypeError, ValueError):
            text = str(oid).strip()
            return text or None

    @classmethod
    def _build_lookup(
        cls,
        events: pd.DataFrame,
    ) -> dict[Any, list[tuple[int, int]]]:
        out: dict[Any, list[tuple[int, int]]] = {}
        non_zero = events[events["fill"] > 0]
        for oid, fill, eid in zip(
            non_zero["id"], non_zero["fill"], non_zero["event_id"]
        ):
            key = cls._order_key(oid)
            if key is None:
                continue
            out.setdefault(key, []).append((int(fill), int(eid)))
        return out

    @classmethod
    def _resolve_event_ids(
        cls,
        order_ids: np.ndarray,
        amounts: pd.Series,
        ev_lookup: dict[Any, list[tuple[int, int]]],
    ) -> np.ndarray:
        # Bucket each order's candidate fills by size, preserving candidate
        # order within each bucket: the earliest unconsumed match wins, at O(1)
        # per trade.  The index is built fresh per call so the maker and taker
        # passes consume independently.
        #
        # The bucket key is an exact lot count (issue #226).  While sizes were
        # floats both sides had to be rounded to ``volume_decimals`` first, so
        # that a fill and the trade print reporting it landed in the same
        # bucket; integers match or they do not.
        index: dict[Any, dict[int, deque[int]]] = {}
        for oid, cand in ev_lookup.items():
            by_fill: dict[int, deque[int]] = {}
            for fill, eid in cand:
                by_fill.setdefault(fill, deque()).append(eid)
            index[oid] = by_fill

        result: list[int | float] = []
        for oid, amt in zip(order_ids, amounts):
            picked: int | float = float("nan")
            key = cls._order_key(oid)
            bucket = None if key is None else index.get(key)
            if bucket is not None:
                matches = bucket.get(int(amt))
                if matches:
                    picked = matches.popleft()
            result.append(picked)

        return np.array(result, dtype=object)

    @classmethod
    def _match_quote_amount_takers(
        cls,
        events: pd.DataFrame,
        taker_ids: np.ndarray,
        taker_event_id: np.ndarray,
    ) -> int:
        """Match the unresolved takers that are instant orders by order id.

        Fills *taker_event_id* in place and returns how many trades it
        matched.
        """
        # Issue #352: Bitstamp never reports what an instant order bought by
        # quote amount executed.  Such an order is created with size 0, the
        # mark this step keys on, and no row of it shows the trade's size.  The
        # trade still names the order, so the taker is its `deleted` row, else
        # its latest row.  Any other order with no row of the trade's size has
        # lost a message, and its trade stays unmatched for `unmatched_trades`
        # to report.
        instant_ids = unsized_order_ids(events)
        if instant_ids.empty:
            return 0
        rows = events.loc[
            events["id"].isin(instant_ids), ["id", "event_id", "action"]
        ].assign(_deleted=lambda f: f["action"] == "deleted")
        # Sorted so each order's `deleted` row, else its latest row, is last.
        chosen = (
            rows.sort_values(["_deleted", "event_id"], kind="stable")
            .groupby("id", sort=False, observed=True)["event_id"]
            .last()
        )
        row_of = {cls._order_key(oid): int(eid) for oid, eid in chosen.items()}

        matched = 0
        for pos in np.flatnonzero(pd.isna(taker_event_id)):
            key = cls._order_key(taker_ids[pos])
            if key in row_of:
                taker_event_id[pos] = row_of[key]
                matched += 1
        return matched


# ── BitstampWriter ────────────────────────────────────────────────────


class BitstampWriter:
    """Write pipeline events back to Bitstamp-format CSV.

    Writes a folder laid out as ``ob-analytics capture bitstamp`` lays one
    out: ``orders.csv``, and ``trades.csv`` when the run has trades.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol.
    """

    target = WriteTarget.FOLDER

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def write(
        self,
        data: dict[str, pd.DataFrame],
        dest: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Write the events to ``<dest>/orders.csv`` in Bitstamp's format.

        When *data* also carries a ``"trades"`` frame, a ``trades.csv`` is
        written next to it in capture format, so that a subsequent
        :class:`BitstampTradeReader` can reconstruct trades when the
        ``orders.csv`` is re-read.  The events' ``sequence`` and ``origin``
        columns are written too when the events carry them, so a capture
        read and written again keeps them.

        The files are in the current capture format (times in nanoseconds),
        and the folder's ``meta.json`` records its version: one is written
        when there is none, and the other fields of one already there are
        kept.

        Parameters
        ----------
        data : dict of str to DataFrame
            Must contain an ``"events"`` key.  An optional ``"trades"`` key
            triggers a ``trades.csv`` in the same folder.
        dest : str or Path
            Output folder, created when missing.

        Returns
        -------
        Path
            The written ``orders.csv`` path.

        Raises
        ------
        ConfigError
            If the events have no rows: an L2 run has no per-order events.
        """
        events = data["events"]
        require_per_order_events(events, "BitstampWriter")
        p = Path(dest) / ORDERS_NAME
        columns: dict[str, Any] = {
            "id": events["id"],
            "timestamp": datetime_to_epoch(events["timestamp"], _WRITE_UNIT),
            "exchange_timestamp": datetime_to_epoch(
                events["exchange_timestamp"], _WRITE_UNIT
            ),
            # Restore the quote-currency float from integer ticks (#155).
            "price": ticks_to_price(
                events["price"],
                self._config.tick_size,
                decimals=self._config.price_decimals,
            ),
            # Restore the base-asset float from integer lots (#226).
            "volume": lots_to_size(
                events["volume"],
                self._config.lot_size,
                decimals=self._config.volume_decimals,
            ),
            "action": events["action"].astype(str),
            "direction": events["direction"].astype(str),
        }
        # The capture's own columns, when the events carry them: the venue
        # sequence, and which part of the capture wrote each row.
        if SEQUENCE_COLUMN in events.columns:
            columns[SEQUENCE_COLUMN] = events[SEQUENCE_COLUMN].astype("Int64")
        if ORIGIN_COLUMN in events.columns:
            columns[ORIGIN_COLUMN] = events[ORIGIN_COLUMN]
        out = pd.DataFrame(columns)[[c for c in ORDERS_CSV_COLUMNS if c in columns]]
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)

        trades = data.get("trades")
        if trades is not None:
            self._write_companion_trades(trades, p.parent / TRADES_NAME)
        elif (p.parent / TRADES_NAME).is_file() and (
            read_format(p.parent).version != CURRENT_FORMAT
        ):
            logger.warning(
                "BitstampWriter: {} is left as it was; this folder is now read "
                "in capture format {}, so a trades.csv from an older capture "
                "will not load",
                p.parent / TRADES_NAME,
                CURRENT_FORMAT,
            )
        record_format(p.parent)
        return p

    def _write_companion_trades(self, trades: pd.DataFrame, dest: Path) -> None:
        """Write a capture-style ``trades.csv`` next to the events CSV.

        Reconstructs the live-capture trade schema from the pipeline's
        ``maker``/``taker`` fields so that :class:`BitstampTradeReader` can
        re-read it.  An empty *trades* frame yields a header-only file.
        """
        if trades.empty:
            pd.DataFrame(columns=list(TRADES_CSV_COLUMNS)).to_csv(dest, index=False)
            return
        side = trades["direction"].astype(str)
        buy_order_id = np.where(side == "buy", trades["taker"], trades["maker"])
        sell_order_id = np.where(side == "buy", trades["maker"], trades["taker"])
        # Go through the canonical inverse so a tz-aware UTC column
        # round-trips cleanly.
        ts = datetime_to_epoch(trades["timestamp"], _WRITE_UNIT)
        venue = ts
        if "exchange_timestamp" in trades.columns:
            venue = datetime_to_epoch(
                trades["exchange_timestamp"].fillna(trades["timestamp"]), _WRITE_UNIT
            )
        out = pd.DataFrame(
            {
                "trade_id": np.arange(1, len(trades) + 1, dtype=np.int64),
                "timestamp": ts,
                "exchange_timestamp": venue,
                "price": ticks_to_price(
                    trades["price"],
                    self._config.tick_size,
                    decimals=self._config.price_decimals,
                ),
                "volume": lots_to_size(
                    trades["volume"],
                    self._config.lot_size,
                    decimals=self._config.volume_decimals,
                ),
                "buy_order_id": buy_order_id,
                "sell_order_id": sell_order_id,
                "direction": side.to_numpy(),
            }
        )
        out.to_csv(dest, index=False)


# ── BitstampSource descriptor ─────────────────────────────────────────


class BitstampSource:
    """The Bitstamp source: offline CSV replay **and** live WebSocket capture.

    One descriptor for both ways to get Bitstamp data.  It satisfies
    :class:`~ob_analytics.protocols.OfflineSource` (the CSV loader / trade
    reader / writer, inline and dependency-light) and
    :class:`~ob_analytics.live._base.LiveSource` (snapshot / stream / shutdown,
    delegated to a WebSocket engine imported lazily from
    :mod:`ob_analytics.live.bitstamp`).  The lazy import is what keeps the
    offline path free of the ``[live]`` extra: importing this module never
    imports ``websockets``; only starting a live capture does.

    Conforms structurally to the source protocols — no inheritance required.
    """

    name = "bitstamp"
    # Per-order (market-by-order) feed — the full reconstruction model.
    level = Level.L3
    # The Bitstamp public feed is a placement/cancellation diff stream, not a
    # matched-engine view, so it can carry genuinely crossed resting orders.
    feed_type = FeedType.DIFF_FEED
    # The live_orders channel reports every order, takers included, and each
    # trade names both orders, so a trade's maker and taker can both be found.
    trade_attribution = TradeAttribution.BOTH
    # BTC/USD: a cent tick and satoshi lots.
    grid = InstrumentGrid()

    def __init__(self, settings: SourceSettings | None = None) -> None:
        # Bitstamp needs no per-source knobs; the empty typed settings keep the
        # construction signature uniform with sources that do (e.g. ccxt).
        self.settings = settings or SourceSettings()
        # The live WebSocket engine, built on first live use so snapshot →
        # stream → shutdown share one connection and order-book state.
        self._engine: Any = None

    # -- offline capability -------------------------------------------------

    def create_loader(self, config: PipelineConfig, ctx: RunContext) -> EventLoader:
        return BitstampLoader(config, venue=ctx.venue, symbol=ctx.symbol)

    def create_trade_reader(
        self, config: PipelineConfig, ctx: RunContext, loader: Any
    ) -> TradeReader:
        # A loader that read its input one window at a time has matched the
        # trades already, so the reader takes the run's loader.
        return BitstampTradeReader(
            config, loader=loader if isinstance(loader, BitstampLoader) else None
        )

    def create_writer(
        self, config: PipelineConfig | None, ctx: RunContext
    ) -> DataWriter:
        return BitstampWriter(config)

    # -- live capability (delegated to the WebSocket engine, lazily) --------

    def _live_engine(self) -> Any:
        """Return the cached WebSocket engine, importing it on first use.

        Raises :class:`ImportError` with an install hint when the ``[live]``
        extra (``websockets``) is not installed.
        """
        if self._engine is None:
            try:
                from ob_analytics.live.bitstamp import BitstampCapturer
            except ImportError as exc:  # pragma: no cover - only without [live]
                raise ImportError(
                    'The bitstamp live source requires the "live" extra: '
                    'pip install "ob-analytics[live]"'
                ) from exc
            self._engine = BitstampCapturer()
        return self._engine

    def preflight(self) -> None:
        """Raise the install hint now if the ``[live]`` extra is missing (SupportsPreflight)."""
        self._live_engine()

    def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        return self._live_engine().snapshot(config)

    def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        return self._live_engine().stream(config)

    def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        return self._live_engine().shutdown_synthetic_events()

    def diagnostics(self) -> dict[str, Any]:
        """Per-run counters from the live engine (SupportsDiagnostics)."""
        return self._live_engine().diagnostics()


# ── Register this source ──────────────────────────────────────────────
# Registration runs at the bottom, after ``BitstampSource`` is defined.
from ob_analytics.sources import register_source

register_source("bitstamp", BitstampSource)
