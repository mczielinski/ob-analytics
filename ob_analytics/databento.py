"""Databento DBN support: the market-by-order (MBO) schema as L3 events.

Databento publishes normalized market data for many venues in its own binary
encoding, DBN.  Its **MBO** schema is a per-order feed — every record carries
an ``order_id``, a ``price``, a ``size`` and a ``side`` — which is the shape
ob-analytics reconstructs from, so a DBN MBO file replays through the standard
:class:`~ob_analytics.pipeline.Pipeline` (issue #100).

The ``action`` field says what a record does.  Databento documents the seven
values as follows, and only the first four change the book:

==========  ======  ====================================================
``action``  name    meaning
==========  ======  ====================================================
``A``       add     insert a new order into the book
``M``       modify  change an order's price and/or size
``C``       cancel  remove some or all of an order's size
``R``       clear   remove every resting order for the instrument
``T``       trade   an aggressing order traded; the book does not change
``F``       fill    a resting order was filled; the book does not change
``N``       none    flags only; the book does not change
==========  ======  ====================================================

``size`` means a different thing per action, which is what makes the mapping to
the canonical schema (see :mod:`ob_analytics.schemas`) work out:

* on ``A`` and ``M`` it is the order's **new total size**, which is the
  canonical ``volume`` (outstanding size after the event) directly;
* on ``C`` it is the **amount removed**, so the outstanding size is carried
  per order and reduced by it — a partial cancel becomes a ``changed`` event
  and one that empties the order becomes a ``deleted`` event;
* on ``F`` it is the **amount executed**, which becomes the canonical ``fill``
  on the book event that follows it for the same order, and one trade row.

A venue reports an execution as a fill and then a separate cancel or modify
that takes the size off the book, so the two are paired by order: every ``F``
is charged to the next ``A``/``M``/``C`` record for the same ``order_id``.
That is what tells a cancel that was really an execution apart from a cancel
the trader asked for.  When the cancel ends the order, its ``deleted`` row
keeps the executed part as ``fill`` and only the rest as ``volume``, so an
order filled in full ends with ``volume`` 0 and leaves its price level once.
When a cancel or modify leaves the order resting but also takes off more than
the fill, moves the order or makes it bigger, the loader writes two rows for
it: one with the fill and one with the rest, since a ``changed`` row is either
an execution or a reduction, never both.

Timestamps
----------
A DBN record carries two clocks.  ``ts_recv`` is Databento's own receive time,
which they guarantee to be monotonic per instrument, so it becomes the
canonical ``timestamp`` the pipeline orders events by.  ``ts_event`` is the
venue matching engine's own time, which can go backwards across gateways, so it
becomes ``exchange_timestamp``.  Both are UTC nanoseconds already (issue #154).

Scope
-----
This module reads stored DBN files.  Databento's Live client is not wired up
here; see issue #100 for that half.

A modify has no canonical event of its own.  Databento's ``M`` can move an
order to another price or make it bigger, and both lose queue priority, so both
are really a new queue entry.  The shared schema has ``created``, ``changed``
and ``deleted`` and nothing for a move, so the loader records the new price and
size on a ``changed`` event.  The price-level rebuild
(:func:`~ob_analytics.depth.price_level_volume`) reads a ``changed`` row that
reports no execution as the order's new level and size, so the depth follows
the order.  The loss of queue priority is not modelled.

What is refused and what is dropped
-----------------------------------
A feed this adapter does not understand is refused: a publisher that only sends
top-of-book or price-level data, a file holding a price-level schema, a file
covering more than one book, an ``action`` outside DBN's own alphabet, and an
``order_id`` too big for the shared schema's signed 64-bit id.

A malformed record inside a feed it does understand is dropped and counted: a
book action, trade or fill with no price (:data:`UNDEF_PRICE`), and a book
action with no side.  A clear is exempt from the price check, because its price
is never read.
Either would otherwise land somewhere wrong — a level at nine billion, or an
order on neither side of the book — and refusing a whole session over a handful
of them would be worse than saying how many went.

``databento`` is an optional dependency (``pip install
"ob-analytics[databento]"``).  It is imported lazily, so importing this module —
and listing sources — never requires it.
"""

from __future__ import annotations

import warnings
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger

from ob_analytics import _engine_frames
from ob_analytics._utils import (
    UTC_NS_DTYPE,
    attach_ingest_seq,
    empty_trades,
    epoch_ns,
    lots_to_size,
    off_tick_grid,
    price_to_ticks,
    size_to_lots,
    ticks_to_price,
    to_utc_ns,
)
from ob_analytics._windows import LoadedWindow
from ob_analytics.config import InstrumentGrid, PipelineConfig, SourceSettings
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    FeedType,
    Level,
    RunContext,
    SequenceKind,
    TradeAttribution,
    TradeSource,
    WriteTarget,
)
from ob_analytics.schemas import (
    MAYBE_BAD_BOOK_ATTR,
    SEQUENCE_COLUMN,
    TRADE_DTYPES,
    TRADE_SIDE_DTYPE,
    attach_instrument_identity,
    cast_columns,
    cast_events,
    validate_columns,
)

# ── DBN constants ─────────────────────────────────────────────────────
#
# Spelled out here rather than imported from ``databento_dbn`` so this module
# imports, and the source registers, without the optional dependency.

#: Databento fixed-point price scale: one raw unit is 1e-9 of the quote
#: currency, so the raw integer divides by this to reach the quote currency.
#: This is the feed's encoding, not the instrument's tick.
DBN_PRICE_DIVISOR = 1_000_000_000

#: The grid a US equity is on: a cent tick and whole shares.  DBN sizes are
#: whole shares or contracts, so one lot is one unit and the canonical integer
#: size is the venue's own count.
DATABENTO_GRID = InstrumentGrid(tick_size=0.01, lot_size=1.0)


#: The raw price that means "no price" (``INT64_MAX``).
UNDEF_PRICE = 9_223_372_036_854_775_807

#: ``flags`` bits.  The loader reads three of them.  ``F_TOB`` and ``F_MBP``
#: mark records that are *not* individual orders — a publisher that only sends
#: top-of-book or price-level data, normalized into MBO records whose
#: ``order_id`` means nothing — and the loader refuses them.
#: ``F_MAYBE_BAD_BOOK`` marks a record after a gap in the channel that
#: Databento could not repair; the loader counts those records and warns.
#: ``F_SNAPSHOT``, which marks a record of a replayed book, is not read.
F_MAYBE_BAD_BOOK = 1 << 2
F_MBP = 1 << 4
F_SNAPSHOT = 1 << 5
F_TOB = 1 << 6

ACTION_ADD = "A"
ACTION_MODIFY = "M"
ACTION_CANCEL = "C"
ACTION_CLEAR = "R"
ACTION_TRADE = "T"
ACTION_FILL = "F"
ACTION_NONE = "N"

#: The actions that change the resting book, in the order they are handled.
BOOK_ACTIONS: frozenset[str] = frozenset({ACTION_ADD, ACTION_MODIFY, ACTION_CANCEL})

#: The actions whose ``price`` is read: the book actions place or move an
#: order, and a trade or fill prices a trade row.  A clear's price is never read.
PRICED_ACTIONS: frozenset[str] = BOOK_ACTIONS | {ACTION_TRADE, ACTION_FILL}

#: Every action DBN defines.  A record carrying anything else is not a feed
#: this adapter understands, so it is refused rather than dropped.
KNOWN_ACTIONS: frozenset[str] = BOOK_ACTIONS | {
    ACTION_CLEAR,
    ACTION_TRADE,
    ACTION_FILL,
    ACTION_NONE,
}

#: The largest order id the shared schema's ``int64`` id column can hold.
#: DBN's ``order_id`` is unsigned 64-bit, so a bigger one has no faithful
#: representation and is refused instead of wrapped to a negative number.
MAX_ORDER_ID = 2**63 - 1

#: Resting side of an ``A`` / ``M`` / ``C`` record.
_SIDE_TO_DIRECTION: dict[str, str] = {"B": "bid", "A": "ask"}
_DIRECTION_TO_SIDE: dict[str, str] = {"bid": "B", "ask": "A"}

#: Aggressor side of a ``T`` record: Databento's ``side`` on a trade is the
#: side the aggressor took.
_TRADE_SIDE_TO_DIRECTION: dict[str, str] = {"B": "buy", "A": "sell"}

#: Aggressor side of an ``F`` record: Databento's ``side`` on a fill is the
#: side of the *resting* order, so the aggressor took the other one.
_FILL_SIDE_TO_DIRECTION: dict[str, str] = {"A": "buy", "B": "sell"}

#: The raw MBO fields this loader reads.  ``symbol`` is added by
#: ``DBNStore.to_df`` when the file carries a symbol mapping.
_REQUIRED_MBO_COLUMNS: tuple[str, ...] = (
    "ts_recv",
    "ts_event",
    "action",
    "side",
    "price",
    "size",
    "order_id",
    "flags",
)


class DatabentoSettings(SourceSettings):
    """Typed settings for :class:`DatabentoSource`.

    A DBN file can hold several instruments and, for a consolidated dataset
    such as ``DBEQ.BASIC``, several publishers of the same instrument.  Each
    (instrument, publisher) pair is a book of its own, so a run covers exactly
    one of them: name it here, or the loader raises and lists what the file
    holds.  A file with one instrument and one publisher needs no setting.

    Attributes
    ----------
    instrument_id : int, optional
        Keep only records with this ``instrument_id``.
    publisher_id : int, optional
        Keep only records from this publisher (one venue of a consolidated
        dataset).
    raw_symbol : str, optional
        Keep only records whose mapped ``symbol`` is this.  Works only on a
        file that carries Databento's symbol mapping; use *instrument_id*
        otherwise.
    trades_from : {"fills", "prints"}
        Which records the trades frame is built from.  ``"fills"`` (the
        default) uses the ``F`` records, one row per resting order a trade hit,
        each naming that order — which is what order classification and queue
        analysis read.  A file with no fills at all falls back to its prints.
        ``"prints"`` uses the ``T`` records: the venue's whole tape, including
        the auction, non-displayed and off-exchange trades that come with no
        fill, but with no maker named on any row.  Pick ``"prints"`` when the
        question is about the tape — VWAP, bars, flow toxicity, costs — and the
        loader has warned that fills leave part of it out.
    dataset : str
        The dataset id :class:`DatabentoWriter` stamps into the DBN metadata it
        writes.  Read only on the way out; the loader takes the file's own.
    """

    instrument_id: int | None = None
    publisher_id: int | None = None
    raw_symbol: str | None = None
    trades_from: Literal["fills", "prints"] = "fills"
    dataset: str = "OB.ANALYTICS"


# ── Reading a DBN file into its raw record frame ──────────────────────


def read_mbo_frame(source: Any) -> pd.DataFrame:
    """Return the raw MBO records of *source* as a DataFrame, in file order.

    Parameters
    ----------
    source : str, Path, DBNStore, or pandas.DataFrame
        A ``.dbn`` / ``.dbn.zst`` path, an already-open
        ``databento.DBNStore``, or a frame of MBO records the caller read
        itself (``store.to_df(price_type="fixed")``).  A frame is taken as is,
        which is how a caller who already sliced or filtered the records hands
        them straight to the loader.

    Returns
    -------
    pandas.DataFrame
        One row per record with at least :data:`_REQUIRED_MBO_COLUMNS`.
        Prices are Databento's raw fixed-point integers, not decimals.

    Raises
    ------
    ConfigError
        If the file holds a schema other than ``mbo``, or a frame is missing
        required columns.
    ImportError
        If a path or store was given and ``databento`` is not installed.
    """
    if isinstance(source, pd.DataFrame):
        frame = source.reset_index() if source.index.name else source.copy()
    else:
        with _mbo_store(source) as store:
            # ``price_type="fixed"`` keeps Databento's raw integer prices,
            # which convert to ticks exactly; the decimal default would
            # round-trip them through a float first.
            frame = store.to_df(price_type="fixed").reset_index()

    validate_columns(frame, _REQUIRED_MBO_COLUMNS, "DatabentoLoader: MBO frame")
    return frame


def _mbo_parts(source: Any, rows: int) -> Iterator[pd.DataFrame]:
    """Yield the raw MBO records of *source* in file order, *rows* at a time.

    Takes what :func:`read_mbo_frame` takes, and reads a file or store a part
    at a time, so the whole file is never in memory.
    """
    if isinstance(source, pd.DataFrame):
        frame = source.reset_index() if source.index.name else source
        validate_columns(frame, _REQUIRED_MBO_COLUMNS, "DatabentoLoader: MBO frame")
        for start in range(0, len(frame), rows):
            yield frame.iloc[start : start + rows]
        return
    with _mbo_store(source) as store:
        for part in store.to_df(price_type="fixed", count=rows):
            part = part.reset_index()
            validate_columns(part, _REQUIRED_MBO_COLUMNS, "DatabentoLoader: MBO frame")
            yield part


@contextmanager
def _mbo_store(source: Any) -> Generator[Any]:
    """Give the open ``DBNStore`` of a path or store, refusing a bad file.

    A store opened here from a path is closed on the way out; a store the
    caller passed is left open.  A file that is not DBN, holds a schema other
    than MBO, or ends part way through its metadata or a record raises
    :class:`ConfigError`, whether it fails on opening or while the records are
    read inside the ``with`` block.  ``databento`` only warns about a cut
    record and reads the records before it, so that warning is an error here:
    a frame missing the end of the file is a wrong answer.
    """
    db = _databento()
    from databento_dbn import DBNError

    bad_file = (db.BentoError, db.common.error.BentoWarning, DBNError, ValueError)
    opened = not hasattr(source, "to_df")
    store = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message=".*truncated", category=db.common.error.BentoWarning
            )
            store = db.DBNStore.from_file(Path(source)) if opened else source
            _check_mbo_schema(store)
            yield store
    except bad_file as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(
            f"DatabentoLoader: {source} is not a DBN file this reader can read: {exc}"
        ) from exc
    finally:
        if opened and store is not None:
            _close_dbn(store)


def _check_mbo_schema(store: Any) -> None:
    """Raise unless an open ``DBNStore`` holds the MBO schema."""
    schema = str(getattr(store.metadata, "schema", "") or "")
    if schema != "mbo":
        raise ConfigError(
            f"DatabentoLoader: file holds the {schema!r} schema, which is "
            "not per-order data. Request the 'mbo' schema for order-book "
            "reconstruction; the price-level schemas (mbp-1, mbp-10, "
            "bbo, tbbo) carry no order ids."
        )


def _databento() -> Any:
    """Return the ``databento`` module, imported on first use."""
    try:
        import databento
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ImportError(
            "Reading DBN files needs the databento package, which is an "
            'optional dependency: pip install "ob-analytics[databento]".'
        ) from exc
    return databento


def _close_dbn(store: Any) -> None:
    """Close the file a ``DBNStore`` from :func:`_mbo_store` holds open.

    A store has no close of its own. Closing its ``reader`` is not enough for
    a compressed file: from Python 3.14 databento decompresses with the
    standard library's ``ZstdFile``, which leaves the file it wraps open. So
    close the file itself, which the store's data source holds. If a databento
    version has no such attribute, fall back to closing the reader.
    """
    data_source = getattr(store, "_data_source", None)
    if data_source is None:
        store.reader.close()
        return
    try:
        # The data source seeks its file to the start before returning it,
        # which raises on a file that something else has already closed.
        data_file = data_source.reader
    except ValueError:
        return
    data_file.close()


# ── DatabentoLoader ───────────────────────────────────────────────────


class DatabentoLoader:
    """Load Databento MBO records into the canonical per-order events frame.

    Satisfies the :class:`~ob_analytics.protocols.EventLoader` protocol.

    The fills the loader pairs with their book events are kept on
    :attr:`trade_records` for :class:`DatabentoTradeReader` to project into the
    trades frame — the same arrangement :class:`~ob_analytics.lobster.LobsterSource`
    uses for the companion orderbook file, and for the same reason: the second
    table is a by-product of reading the first, and reading the file twice to
    get it would be wasted work on a multi-million-record window.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration: the instrument's ``tick_size`` and
        ``lot_size``.  A grid field it does not set comes from
        :data:`DATABENTO_GRID`, a cent tick and whole shares.  The raw prices are in Databento's fixed-point encoding
        (:data:`DBN_PRICE_DIVISOR`), which is part of the format, not a
        setting.
    settings : DatabentoSettings, optional
        Which instrument and publisher of a multi-instrument file to read.
    venue, symbol : str, optional
        Optional instrument identity (issue #147).  When either is supplied the
        loaded frame gains per-row ``venue`` / ``symbol`` columns; ``venue``
        falls back to ``"databento"``.
    """

    #: Fills the venue reported no passive side for.
    _VENUE = "databento"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        settings: DatabentoSettings | None = None,
        venue: str | None = None,
        symbol: str | None = None,
    ) -> None:
        self._config = DATABENTO_GRID.config(config)
        self._settings = settings or DatabentoSettings()
        self._venue = venue
        self._symbol = symbol
        #: The ``F`` (fill) and ``T`` (trade) records of the last
        #: :meth:`load`, each already carrying the ``maker_event_id`` of the
        #: book event it was charged to.  ``None`` before the first load.
        self.trade_records: pd.DataFrame | None = None
        #: How many records of the last :meth:`load` carry
        #: ``F_MAYBE_BAD_BOOK``: Databento's mark for a record after a gap it
        #: could not repair.  ``0`` before the first load.
        self.maybe_bad_book_records: int = 0

    #: How many records :meth:`load_windows` reads from the file at a time.
    read_size: int = 250_000

    def load(self, source: Any) -> pd.DataFrame:
        """Load MBO records from *source* and return the events frame.

        Parameters
        ----------
        source : str, Path, DBNStore, or pandas.DataFrame
            Anything :func:`read_mbo_frame` accepts.

        Returns
        -------
        pandas.DataFrame
            Canonical events: ``id``, ``timestamp``, ``exchange_timestamp``,
            ``price`` (integer ticks), ``volume`` and ``fill`` (integer lots),
            ``action``, ``direction``, ``event_id``, ``original_number``,
            ``raw_event_type`` (the DBN action letter) and ``raw_size`` (the
            record's own ``size`` field).  ``attrs[MAYBE_BAD_BOOK_ATTR]``
            holds :attr:`maybe_bad_book_records`.
        """
        logger.info("DatabentoLoader: reading {}", source)
        book = _BookFilter(self._settings)
        raw = book.narrow(read_mbo_frame(source))
        book.finish()
        reading = _Reading()
        events, records = self._read_part(raw, reading, last=True)
        self.maybe_bad_book_records = _warn_on_maybe_bad_book(reading)
        self.trade_records = _in_record_order([records])
        return events

    def load_windows(
        self,
        source: Any,
        windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
    ) -> Iterator[LoadedWindow]:
        """Yield the events of each time window of *source* in turn.

        The file is read :attr:`read_size` records at a time, and only one
        window's records and events are in memory at once.  The state an order
        carries from one window into the next (its size, its price, a fill
        whose book record has not arrived yet) is kept between windows, so the
        events are the rows :meth:`load` gives, cut at the windows.  Their
        ``event_id`` and ``original_number`` count across the whole file.

        A window holds the records received (``ts_recv``) at or after its
        start and before its end.  Once the last window is yielded,
        :attr:`trade_records` holds the trade records of the whole file, so
        :class:`DatabentoTradeReader` builds the whole run's trades.

        Parameters
        ----------
        source : str, Path, DBNStore, or pandas.DataFrame
            Anything :func:`read_mbo_frame` accepts.
        windows : list of tuple
            ``(start, end)`` per window, tz-aware UTC, ``None`` for an open
            end, covering the whole file: what
            :func:`~ob_analytics._windows.window_bounds` returns.

        Yields
        ------
        LoadedWindow
            The window's events, and the ``event_id`` of each of them a trade
            names.

        Raises
        ------
        ConfigError
            If a record was received before the start of the window it falls
            in: the records are not in receive-time order, which Databento
            guarantees for one instrument.
        """
        logger.info("DatabentoLoader: reading {} one window at a time", source)
        self.trade_records = None
        book = _BookFilter(self._settings)
        parts = (book.narrow(raw) for raw in _mbo_parts(source, self.read_size))
        reading = _Reading()
        records = []
        cut = _window_records(parts, windows)
        for number, raw in enumerate(cut, 1):
            last = number == len(windows)
            if last:
                book.finish()
            events, part_records = self._read_part(raw, reading, last=last)
            records.append(part_records)
            yield LoadedWindow(events, part_records["maker_event_id"].dropna().unique())
        self.maybe_bad_book_records = _warn_on_maybe_bad_book(reading)
        self.trade_records = _in_record_order(records)

    def _read_part(
        self, raw: pd.DataFrame, reading: _Reading, *, last: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the events and trade records of the next part of the file.

        *reading* holds what the parts before left behind, and is updated for
        the part after.  *last* marks the part that ends the file.
        """
        _reject_aggregated_records(raw)
        flagged = _count_maybe_bad_book(raw)
        reading.flagged += flagged

        work = self._to_working_frame(raw, first=reading.records + 1)
        first_event = reading.events
        events, records = _book_events_and_fills(work, reading, last=last)
        reading.records += len(raw)

        if self._config.track_sequence:
            # The venue's own sequence number, where the publisher sends one
            # (0 when it does not), plus the local arrival counter.  Both are
            # taken while the frame is still in file order.
            if "sequence" in work.columns:
                seq_by_record = pd.Series(
                    work["sequence"].to_numpy(),
                    index=work["original_number"].to_numpy(),
                )
                events[SEQUENCE_COLUMN] = (
                    events["original_number"].map(seq_by_record).astype("Int64")
                )
            events = attach_ingest_seq(events, start=first_event)

        events = events.drop(columns=["source_pos"])

        logger.info(
            "DatabentoLoader: {} book events ({} created, {} changed, "
            "{} deleted), {} fills, {} trade prints",
            len(events),
            int((events["action"] == "created").sum()),
            int((events["action"] == "changed").sum()),
            int((events["action"] == "deleted").sum()),
            int((records["raw_event_type"] == ACTION_FILL).sum()),
            int((records["raw_event_type"] == ACTION_TRADE).sum()),
        )

        events = attach_instrument_identity(
            events,
            venue=self._venue,
            symbol=self._symbol,
            default_venue=self._VENUE,
        )
        events = cast_events(events)
        events.attrs[MAYBE_BAD_BOOK_ATTR] = flagged
        return events, records

    def _to_working_frame(self, raw: pd.DataFrame, *, first: int) -> pd.DataFrame:
        """Return the columns the mapping needs, typed and in file order.

        Keeps the file's own row order: a fill is charged to the book event
        that follows it, so re-sorting here would break the pairing.  The
        1-based file position is kept as ``original_number``; *raw*'s first
        record is at *first*.
        """
        cfg = self._config
        action = raw["action"].astype(str).str.strip()
        side = raw["side"].astype(str).str.strip()
        order_id = _order_ids(raw["order_id"])

        work = pd.DataFrame(
            {
                "original_number": np.arange(first, first + len(raw), dtype="int64"),
                "action": action.to_numpy(),
                "side": side.to_numpy(),
                "id": order_id,
                "raw_price": raw["price"].astype("int64").to_numpy(),
                "raw_size": raw["size"].astype("int64").to_numpy(),
                "timestamp": to_utc_ns(raw["ts_recv"]),
                "exchange_timestamp": to_utc_ns(raw["ts_event"]),
            }
        )
        if "sequence" in raw.columns:
            work["sequence"] = raw["sequence"].astype("int64").to_numpy()

        # Canonical price is integer ticks (issue #155) and canonical size
        # integer lots (issue #226).  The raw price is Databento's fixed-point
        # integer, so it divides by the feed's encoding scale to reach the
        # quote currency before it is quantised onto the instrument's tick.
        work["price"] = price_to_ticks(
            work["raw_price"].to_numpy() / DBN_PRICE_DIVISOR, cfg.tick_size
        )
        work["size_lots"] = size_to_lots(work["raw_size"].to_numpy(), cfg.lot_size)
        work["direction"] = work["side"].map(_SIDE_TO_DIRECTION)
        return _usable_records(work)


class _BookFilter:
    """Narrow the records to the single book a run covers, part by part.

    Applies whichever of the :class:`DatabentoSettings` filters were set, and
    insists that what is left is one instrument from one publisher: two
    publishers of the same instrument are two books, and merging them would
    reconstruct a book that never existed.  A file read in parts is checked
    across all of them: :meth:`finish` raises if a filter kept nothing.
    """

    def __init__(self, settings: DatabentoSettings) -> None:
        self._filters = [
            (column, wanted)
            for column, wanted in (
                ("instrument_id", settings.instrument_id),
                ("publisher_id", settings.publisher_id),
                ("symbol", settings.raw_symbol),
            )
            if wanted is not None
        ]
        self._kept = dict.fromkeys((column for column, _ in self._filters), 0)
        self._present: dict[str, set[Any]] = {
            "instrument_id": set(),
            "publisher_id": set(),
        }

    def narrow(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return the records of *raw* that belong to the run's book."""
        for column, wanted in self._filters:
            if column not in raw.columns:
                raise ConfigError(
                    f"DatabentoLoader: cannot filter on {column!r} — the "
                    f"records carry no such column. Present: {list(raw.columns)}"
                )
            raw = raw[raw[column] == wanted]
            self._kept[column] += len(raw)

        for column, setting in (
            ("instrument_id", "instrument_id"),
            ("publisher_id", "publisher_id"),
        ):
            if column not in raw.columns:
                continue
            present = self._present[column]
            present.update(pd.unique(raw[column]).tolist())
            if len(present) > 1:
                raise ConfigError(
                    f"DatabentoLoader: the records cover {len(present)} values "
                    f"of {column} ({sorted(present)[:10]}). One run "
                    "reconstructs one book, so pick one with "
                    f"DatabentoSettings({setting}=...)."
                )

        return raw.reset_index(drop=True)

    def finish(self) -> None:
        """Raise if one of the filters has kept no record of the file."""
        for column, wanted in self._filters:
            if not self._kept[column]:
                raise ConfigError(
                    f"DatabentoLoader: no records left after {column}={wanted!r}."
                )


@dataclass
class _Reading:
    """What one part of a file leaves for the next.

    A file read whole is one part.  Read a window at a time, each part starts
    where the one before stopped: the orders still on the book, the fills
    whose book record has not come yet, and the counts the next part's
    numbers continue from.
    """

    #: Records read so far; the next part's ``original_number`` follows on.
    records: int = 0
    #: Events numbered so far; the next part's ``event_id`` follows on.
    events: int = 0
    #: The orders resting at the end of the part: ``id``, ``price`` (the
    #: price of the order's last row), ``outstanding`` and ``direction``.
    book: pd.DataFrame | None = None
    #: The orders that have left the book since the last clear.  An id seen
    #: again before the next clear is the same order, as in a whole read.
    gone: pd.Index = field(default_factory=lambda: pd.Index([], dtype="int64"))
    #: Fill records not yet charged to a book record: the window ended
    #: between a fill and the cancel or modify that takes its size off.
    fills: pd.DataFrame | None = None
    #: Records carrying ``F_MAYBE_BAD_BOOK``, for the warning once the file
    #: is done.
    flagged: int = 0


def _window_records(
    parts: Iterable[pd.DataFrame],
    windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
) -> Iterator[pd.DataFrame]:
    """Yield the raw records of each of *windows* in turn, from *parts*.

    A window holds the records received at or after its start and before its
    end.  The records are taken in file order, so one received before the
    start of the window it would fall in is out of receive-time order, and
    raises.
    """
    source = iter(parts)
    # The part not yet taken, and when each of its records was received.
    rest: pd.DataFrame | None = None
    received = np.empty(0, dtype="int64")
    template: pd.DataFrame | None = None
    for start, end in windows:
        taken, taken_at = [], []
        end_ns = None if end is None else epoch_ns([end])[0]
        while True:
            if rest is None:
                rest = next(source, None)
                if rest is None:
                    break
                if template is None:
                    template = rest.iloc[:0]
                received = epoch_ns(to_utc_ns(rest["ts_recv"]))
            if end_ns is None:
                n = len(rest)
            else:
                later = received >= end_ns
                n = int(np.argmax(later)) if later.any() else len(rest)
            taken.append(rest.iloc[:n])
            taken_at.append(received[:n])
            if n < len(rest):
                rest, received = rest.iloc[n:], received[n:]
                break
            rest = None
        if template is None:
            template = _no_mbo_records()
        window = pd.concat(taken, ignore_index=True) if taken else template
        if start is not None and len(window):
            early = np.concatenate(taken_at) < epoch_ns([start])[0]
            if early.any():
                raise ConfigError(
                    f"DatabentoLoader: {int(early.sum())} records received "
                    f"before {start} come after a record received at or after "
                    "it, so the file is not in receive-time order and cannot "
                    "be read one window at a time. Run it whole with "
                    "Pipeline.run."
                )
        yield window


def _no_mbo_records() -> pd.DataFrame:
    """Return a zero-row frame of the MBO fields the loader reads."""
    dtypes = {
        "ts_recv": UTC_NS_DTYPE,
        "ts_event": UTC_NS_DTYPE,
        "action": "object",
        "side": "object",
        "price": "int64",
        "size": "int64",
        "order_id": "uint64",
        "flags": "uint8",
    }
    return pd.DataFrame({k: pd.Series([], dtype=v) for k, v in dtypes.items()})


def _order_ids(order_id: pd.Series) -> np.ndarray:
    """Return *order_id* as ``int64``, refusing one the schema cannot hold.

    DBN's ``order_id`` is unsigned 64-bit and the shared schema's ``id`` is
    signed, so a plain cast wraps an id above :data:`MAX_ORDER_ID` to a
    negative number — silently, and in a way that can collapse two orders onto
    one key in the per-order state machine.  Refuse it instead.
    """
    as_uint = pd.to_numeric(order_id, errors="raise")
    too_big = as_uint > MAX_ORDER_ID
    if bool(np.asarray(too_big).any()):
        biggest = int(as_uint[too_big].max())
        raise ConfigError(
            f"DatabentoLoader: {int(np.asarray(too_big).sum())} records carry "
            f"an order id above {MAX_ORDER_ID} (largest {biggest}). The shared "
            "schema holds order ids as signed 64-bit, so these cannot be "
            "represented without wrapping to a negative id and merging "
            "distinct orders."
        )
    return as_uint.to_numpy(dtype="int64")


def _usable_records(work: pd.DataFrame) -> pd.DataFrame:
    """Return the records this loader can read, refusing or dropping the rest.

    Two different problems, handled two different ways.

    An ``action`` outside :data:`KNOWN_ACTIONS` means the records are not the
    DBN market-by-order data this adapter was written for — a newer DBN action,
    or a frame whose action column was spelled some other way.  There is no
    safe reading of a record whose meaning is unknown, so it raises, the same
    as an aggregated publisher or a price-level schema.

    A record with no price (:data:`UNDEF_PRICE`) or, on a book action, no side
    is a single malformed record inside a feed that is otherwise understood.
    Those are dropped and counted: keeping them would put a level at nine
    billion, or an order on neither side of the book, and refusing the whole
    file over a handful of them would be worse than saying how many went.
    """
    action = work["action"].to_numpy()
    unknown = ~np.isin(action, list(KNOWN_ACTIONS))
    if unknown.any():
        seen = sorted({str(a) for a in action[unknown]})[:10]
        raise ConfigError(
            f"DatabentoLoader: {int(unknown.sum())} records carry an action "
            f"this loader does not know ({seen}). DBN's actions are "
            f"{sorted(KNOWN_ACTIONS)}; a lower-case or renamed column has to "
            "be put back into DBN's own spelling first."
        )

    # Only the records whose price is read can be spoiled by a missing one: the
    # book actions and the two trade records.  A clear's price means nothing and
    # is never read, and dropping a clear over it would leave every order it
    # removed resting for the rest of the window.
    priced = np.isin(action, list(PRICED_ACTIONS))
    no_price = priced & (work["raw_price"].to_numpy() == UNDEF_PRICE)
    no_side = np.isin(action, list(BOOK_ACTIONS)) & work["direction"].isna().to_numpy()

    for mask, why, effect in (
        (
            no_price,
            "no price (UNDEF_PRICE)",
            "there is no level to put them on",
        ),
        (
            no_side,
            "no side on a book action",
            "there is no side of the book to put them on",
        ),
    ):
        count = int(mask.sum())
        if count:
            # Which way the book is off depends on what was dropped: a missing
            # add leaves it short, a missing cancel leaves an order resting that
            # should have gone.  Say that it is off, not which way.
            logger.warning(
                "DatabentoLoader: dropped {} of {} records with {} — {}. The "
                "book will not reflect them: a dropped add leaves it short, a "
                "dropped cancel leaves an order resting that should have gone",
                count,
                len(work),
                why,
                effect,
            )

    keep = ~(no_price | no_side)
    if keep.all():
        return work
    return work[keep].reset_index(drop=True)


def _reject_aggregated_records(raw: pd.DataFrame) -> None:
    """Raise unless every record is a real per-order record.

    Some Databento publishers only send top-of-book or price-level data, and
    Databento normalizes that into MBO records with ``F_TOB`` or ``F_MBP`` set
    and a meaningless ``order_id``.  Reconstructing per-order state from those
    would invent order identity the feed never had, which is the thing the L2
    path exists to avoid — so say so instead.
    """
    flags = raw["flags"].to_numpy(dtype="int64", na_value=0)
    aggregated = flags & (F_TOB | F_MBP)
    if aggregated.any():
        n_tob = int((flags & F_TOB).astype(bool).sum())
        n_mbp = int((flags & F_MBP).astype(bool).sum())
        raise ConfigError(
            f"DatabentoLoader: {int(aggregated.astype(bool).sum())} of "
            f"{len(raw)} records are aggregated, not per-order "
            f"({n_tob} top-of-book, {n_mbp} price-level). This publisher does "
            "not send order-level data, so its order ids mean nothing. Read "
            "its price-level schema through the L2 path instead (see "
            "ob_analytics.depth_l2)."
        )


def _count_maybe_bad_book(raw: pd.DataFrame) -> int:
    """Count the records carrying ``F_MAYBE_BAD_BOOK``."""
    flags = raw["flags"].to_numpy(dtype="int64", na_value=0)
    return int((flags & F_MAYBE_BAD_BOOK).astype(bool).sum())


def _warn_on_maybe_bad_book(reading: _Reading) -> int:
    """Warn when records carry ``F_MAYBE_BAD_BOOK``, and return how many do.

    Databento sets the flag on the records after a gap in the channel that it
    could not repair.  The records are kept: the book rebuilt from them may be
    wrong from the gap on, but dropping them would not make it right.
    """
    if reading.flagged:
        warnings.warn(
            f"DatabentoLoader: {reading.flagged} of {reading.records} records "
            "carry F_MAYBE_BAD_BOOK. Databento found a gap in the feed that it "
            "could not repair, so the book may be wrong from the first of "
            "these records on",
            stacklevel=3,
        )
    return reading.flagged


# ── The action mapping ────────────────────────────────────────────────


def _book_events_and_fills(
    work: pd.DataFrame,
    reading: _Reading | None = None,
    *,
    last: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split MBO records into the canonical events frame and the trade records.

    Returns ``(events, trade_records)``.  ``events`` holds the A/M/C records
    plus a synthetic ``deleted`` row per order still resting when an ``R``
    (clear) arrives.  ``trade_records`` holds the F and T records, each with
    the ``maker_event_id`` of the book event its fill was charged to.

    *work* is one part of the file, and *reading* what the parts before it
    left (see :class:`_Reading`); it is updated for the next part.  The
    orders the earlier parts left are put back in front of *work* as rows of
    their own, the seeds, so this part's records find each order as it was.
    The seeds' own rows are dropped once the fills are charged.  Unless the
    part is the *last*, a fill with no book record after it is kept for the
    next part rather than reported as having none.
    """
    reading = _Reading() if reading is None else reading
    lead = _carried_records(work, reading)
    if len(lead):
        work = pd.concat([lead, work], ignore_index=True)
    action = work["action"].to_numpy()
    is_book = np.isin(action, list(BOOK_ACTIONS))
    is_fill = action == ACTION_FILL
    is_clear = action == ACTION_CLEAR

    # An ``R`` wipes the book, so an order id before it and the same id after
    # it are different orders.  Counting clears gives each stretch between them
    # its own number, and grouping by (epoch, id) keeps the two apart without
    # splitting the frame up.
    work = work.assign(epoch=np.cumsum(is_clear))

    # Rank of each book record within its order, counted over the A/M/C records
    # only.  A fill sits between two of them, so its rank says which book
    # record it is charged to: the next one, rank + 1.
    order_key = [work["epoch"].to_numpy(), work["id"].to_numpy()]
    book_rank = (
        pd.Series(is_book.astype("int64")).groupby(order_key).cumsum().to_numpy()
    )
    work = work.assign(book_rank=book_rank, is_book=is_book)

    events = _events_from_book_records(work[is_book])
    events = _add_clear_deletes(events, work[is_clear])
    events = events.sort_values("source_pos", kind="stable").reset_index(drop=True)
    _keep_book(reading, events, epoch=int(work["epoch"].iloc[-1]) if len(work) else 0)

    fills = work[is_fill]
    return _charge_fills_to_events(events, fills, work, reading, last=last)


def _carried_records(work: pd.DataFrame, reading: _Reading) -> pd.DataFrame:
    """Return the records the parts before *work* leave in front of it.

    First a seed per order: an add of its outstanding size at the price of
    its last row, for each order resting, and for each order that has left
    the book and whose id *work* uses again before a clear.  The seed's price
    is also where the order rests, because a Databento cancel carries the
    price of the order it cancels.  Then the fills that wait for their
    book record.  The seeds are numbered at or below ``reading.records``, so
    their rows sort ahead of the part's own and are told apart by that.
    """
    pieces = []
    seeds = reading.book
    if len(reading.gone):
        action = work["action"].to_numpy()
        before_clear = np.cumsum(action == ACTION_CLEAR) == 0
        book_ids = work.loc[before_clear & np.isin(action, list(BOOK_ACTIONS)), "id"]
        gone = reading.gone[reading.gone.isin(book_ids)]
        if len(gone):
            # Gone orders have nothing resting.  Only the id matters: a modify
            # of it is a change, not a new order, and a cancel removes nothing.
            gone_seeds = pd.DataFrame(
                {
                    "id": gone.to_numpy(dtype="int64"),
                    "price": np.zeros(len(gone), dtype="int64"),
                    "outstanding": np.zeros(len(gone), dtype="int64"),
                    "direction": None,
                }
            )
            seeds = gone_seeds if seeds is None else pd.concat([seeds, gone_seeds])
    if seeds is not None and len(seeds):
        n = len(seeds)
        pieces.append(
            pd.DataFrame(
                {
                    "original_number": np.arange(
                        reading.records - n + 1, reading.records + 1, dtype="int64"
                    ),
                    "action": ACTION_ADD,
                    "id": seeds["id"].to_numpy(),
                    "raw_price": 0,
                    "raw_size": 0,
                    "timestamp": pd.Series([pd.NaT] * n, dtype=UTC_NS_DTYPE).array,
                    "exchange_timestamp": pd.Series(
                        [pd.NaT] * n, dtype=UTC_NS_DTYPE
                    ).array,
                    "price": seeds["price"].to_numpy(),
                    "size_lots": seeds["outstanding"].to_numpy(),
                    "direction": seeds["direction"].to_numpy(),
                }
            )
        )
    if reading.fills is not None and len(reading.fills):
        pieces.append(reading.fills)
    if not pieces:
        return work.iloc[:0]
    return pd.concat(pieces, ignore_index=True)


def _keep_book(reading: _Reading, events: pd.DataFrame, *, epoch: int) -> None:
    """Keep the orders *events* leave on the book, for the next part.

    *events* are the part's rows before the fills are charged, seeds
    included, in event order.  *epoch* is the part's last: a clear ends every
    order before it, so only the orders after the part's last clear carry on.
    """
    in_epoch = events[events["epoch"] == epoch]
    last_rows = in_epoch.drop_duplicates(subset=["id"], keep="last")
    resting = last_rows["action"] != "deleted"
    reading.book = last_rows.loc[
        resting, ["id", "price", "outstanding", "direction"]
    ].reset_index(drop=True)
    gone = pd.Index(last_rows.loc[~resting, "id"], dtype="int64")
    if epoch == 0:
        gone = reading.gone.append(gone).unique()
    reading.gone = gone[~gone.isin(reading.book["id"])]


def _events_from_book_records(book: pd.DataFrame) -> pd.DataFrame:
    """Map the A / M / C records onto the canonical action and volume.

    ``volume`` is the order's outstanding size after the event, except on a
    ``deleted`` row where it is, for now, the whole size the cancel removed.
    ``outstanding`` is the size after the event on every row, ``0`` on a
    ``deleted`` one.  Once the fills are charged, :func:`_split_deleted_volume`
    and :func:`_split_mixed_changes` use them to make every row follow the
    schema's rule (see :mod:`ob_analytics.schemas`).
    """
    if book.empty:
        return _empty_book_events()

    action = book["action"].to_numpy()
    size = book["size_lots"].to_numpy()
    epoch = book["epoch"].to_numpy()
    order_id = book["id"].to_numpy()
    keys = [epoch, order_id]

    # ``A`` and ``M`` both state the order's new total size, so they set the
    # outstanding size outright; ``C`` subtracts from whatever it was.  Running
    # the subtractions from the last set record is the whole of the state
    # machine, and it vectorises: number the stretches between set records and
    # take a cumulative sum of the cancels inside each one.
    is_set = np.isin(action, [ACTION_ADD, ACTION_MODIFY])
    is_cancel = action == ACTION_CANCEL
    set_run = pd.Series(is_set.astype("int64")).groupby(keys).cumsum().to_numpy()

    base = pd.Series(np.where(is_set, size, np.nan)).groupby(keys).ffill().to_numpy()
    removed = (
        pd.Series(np.where(is_cancel, size, 0))
        .groupby([epoch, order_id, set_run])
        .cumsum()
        .to_numpy()
    )
    outstanding = base - removed

    # A modify for an order this window never saw added is an add: it is the
    # first record that puts the order on the book, which is how Databento's
    # own reference book builder treats it.
    first_of_order = ~pd.MultiIndex.from_arrays([epoch, order_id]).duplicated()
    created = (action == ACTION_ADD) | (is_set & first_of_order)

    # A cancel with no add before it belongs to an order that was already
    # resting when the window opened; its outstanding size is unknowable, so
    # the row reports the size removed and ends the order, which is the
    # conservative reading.
    orphan_cancel = is_cancel & np.isnan(base)
    n_orphan = int(orphan_cancel.sum())
    if n_orphan:
        logger.warning(
            "DatabentoLoader: {} cancels are for orders with no add in this "
            "window (a book that was already resting when it opened); their "
            "outstanding size is unknown, so they are recorded as deletes of "
            "the size removed",
            n_orphan,
        )

    oversized = is_cancel & (outstanding < 0)
    if oversized.any():
        logger.warning(
            "DatabentoLoader: {} cancels remove more than the order's "
            "outstanding size — the records are inconsistent; treating them "
            "as full deletes",
            int(oversized.sum()),
        )

    emptied = is_cancel & (orphan_cancel | ~(outstanding > 0))
    canonical = np.where(created, "created", np.where(emptied, "deleted", "changed"))
    # A cancel that empties the order removes what it had resting just before:
    # its own size, or less when it asks to remove more than there was, and
    # nothing when the order was already gone.  An orphan's resting size is
    # unknown, so it removes its own size.
    resting_before = np.where(orphan_cancel, size, np.clip(outstanding + size, 0, size))
    volume = np.where(is_set, size, np.where(emptied, resting_before, outstanding))

    return pd.DataFrame(
        {
            "id": order_id,
            "timestamp": book["timestamp"].array,
            "exchange_timestamp": book["exchange_timestamp"].array,
            "price": book["price"].to_numpy(),
            "volume": volume.astype("int64"),
            "outstanding": np.where(emptied, 0, volume).astype("int64"),
            "action": canonical,
            "direction": book["direction"].to_numpy(),
            "original_number": book["original_number"].to_numpy(),
            "raw_event_type": action,
            "raw_size": book["raw_size"].to_numpy(),
            "epoch": epoch,
            "book_rank": book["book_rank"].to_numpy(),
            "source_pos": book["original_number"].to_numpy().astype("float64"),
        }
    )


def _empty_book_events() -> pd.DataFrame:
    """Return the zero-row frame :func:`_events_from_book_records` produces."""
    dtypes = {
        "id": "int64",
        "timestamp": UTC_NS_DTYPE,
        "exchange_timestamp": UTC_NS_DTYPE,
        "price": "int64",
        "volume": "int64",
        "outstanding": "int64",
        "action": "object",
        "direction": "object",
        "original_number": "int64",
        "raw_event_type": "object",
        "raw_size": "int64",
        "epoch": "int64",
        "book_rank": "int64",
        "source_pos": "float64",
    }
    return pd.DataFrame({k: pd.Series([], dtype=v) for k, v in dtypes.items()})


def _add_clear_deletes(events: pd.DataFrame, clears: pd.DataFrame) -> pd.DataFrame:
    """Append a ``deleted`` row per order still resting at each ``R`` record.

    A clear removes every resting order at once, and the feed says nothing
    more about them.  Without a row each, those orders would look like they
    rested for the whole window.  The synthetic rows take the clear's own
    timestamp and sort immediately before it.
    """
    if clears.empty or events.empty:
        return events

    pieces = [events]
    for _, clear in clears.iterrows():
        resting = _resting_at(events, epoch=int(clear["epoch"]) - 1)
        if resting.empty:
            continue
        n = len(resting)
        pieces.append(
            pd.DataFrame(
                {
                    "id": resting["id"].to_numpy(),
                    "timestamp": _repeat_ts(clear["timestamp"], n),
                    "exchange_timestamp": _repeat_ts(clear["exchange_timestamp"], n),
                    "price": resting["price"].to_numpy(),
                    "volume": resting["volume"].to_numpy(),
                    "outstanding": 0,
                    "action": "deleted",
                    "direction": resting["direction"].to_numpy(),
                    "original_number": int(clear["original_number"]),
                    "raw_event_type": ACTION_CLEAR,
                    "raw_size": resting["volume"].to_numpy(),
                    "epoch": int(clear["epoch"]) - 1,
                    # No book record of its own, so no slot a fill can be
                    # charged to.
                    "book_rank": -1,
                    # Half a position before the clear, so a stable sort puts
                    # these rows between the last real record and it.
                    "source_pos": float(clear["original_number"]) - 0.5,
                }
            )
        )
        logger.info(
            "DatabentoLoader: book clear at record {} removed {} resting orders",
            int(clear["original_number"]),
            len(resting),
        )

    return pd.concat(pieces, ignore_index=True)


def _repeat_ts(value: pd.Timestamp, n: int) -> pd.api.extensions.ExtensionArray:
    """Return *value* repeated *n* times, still on the canonical UTC-ns clock.

    ``numpy.repeat`` would hand back an object array of Timestamps, which
    concatenates into an object column and loses the zone.
    """
    return pd.Series([value] * n).astype(UTC_NS_DTYPE).array


def _resting_at(events: pd.DataFrame, *, epoch: int) -> pd.DataFrame:
    """Return the last event of each order of *epoch* that is still on the book."""
    in_epoch = events[events["epoch"] == epoch]
    if in_epoch.empty:
        return in_epoch
    last = in_epoch.sort_values("source_pos", kind="stable").drop_duplicates(
        subset=["id"], keep="last"
    )
    return last[last["action"] != "deleted"]


def _charge_fills_to_events(
    events: pd.DataFrame,
    fills: pd.DataFrame,
    work: pd.DataFrame,
    reading: _Reading,
    *,
    last: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach ``fill`` to the events and ``maker_event_id`` to the fills.

    A fill is charged to the next book record for the same order, which is the
    record that takes the executed size off the book.  Several fills can land
    on one book record when a sweep takes an order out in pieces, so the
    ``fill`` on an event is the sum of them.  The rows are numbered once the
    fills have made them final, so ``event_id`` runs 1, 2, 3, ... in event
    order, from where *reading* left off.  Seed rows (see
    :func:`_carried_records`) are dropped before that.
    """
    slot = ["epoch", "id", "book_rank"]
    events = events.copy()
    events["fill"] = 0

    if not fills.empty:
        target = fills.assign(book_rank=fills["book_rank"] + 1)
        charged = target.groupby(slot, sort=False)["size_lots"].sum()
        key = pd.MultiIndex.from_frame(events[slot])
        events["fill"] = charged.reindex(key).fillna(0).to_numpy().astype("int64")

    events = _split_deleted_volume(events)
    events = _split_mixed_changes(events)
    events = events[events["source_pos"].to_numpy() > reading.records]
    events = events.reset_index(drop=True)
    events["event_id"] = np.arange(
        reading.events + 1, reading.events + len(events) + 1, dtype="int64"
    )
    reading.events += len(events)
    reading.fills = None

    if not fills.empty:
        made = events.set_index(pd.MultiIndex.from_frame(events[slot]))[
            ["event_id", "original_number"]
        ]
        # An order id can repeat across epochs, and a book record is one slot,
        # so the index is unique by construction; guard anyway because a
        # malformed file could repeat a rank.
        made = made[~made.index.duplicated()]
        maker = made.reindex(pd.MultiIndex.from_frame(target[slot]))
        maker_event_id = maker["event_id"].to_numpy()
        maker_og = maker["original_number"].to_numpy()
        uncharged = pd.isna(maker_event_id)
        if not last:
            # A fill after the part's last clear may yet meet its book record
            # in the next part.
            waiting = uncharged & (fills["epoch"].to_numpy() == work["epoch"].iloc[-1])
            if waiting.any():
                reading.fills = fills[waiting].drop(
                    columns=["epoch", "book_rank", "is_book"]
                )
                keep = ~waiting
                fills, uncharged = fills[keep], uncharged[keep]
                maker_event_id, maker_og = maker_event_id[keep], maker_og[keep]
        orphan = int(uncharged.sum())
        if orphan:
            logger.warning(
                "DatabentoLoader: {} fills have no book record after them in "
                "this window (the window ends mid-execution); their trades "
                "carry no maker event",
                orphan,
            )
    else:
        maker_event_id = maker_og = np.empty(0)

    trade_records = _trade_records(fills, work, maker_event_id, maker_og)
    events = events.drop(columns=["epoch", "book_rank", "outstanding"])
    return events, trade_records


def _split_deleted_volume(events: pd.DataFrame) -> pd.DataFrame:
    """Leave on each ``deleted`` row only the size it removes without trading.

    Until the fills are charged, a ``deleted`` row's ``volume`` is the whole
    size its cancel took off the book.  Part of that size may be the fill the
    row now carries.  The schema counts that part once, as ``fill``, so
    ``volume`` keeps the rest: ``0`` when the order was executed in full.

    A fill larger than the size the cancel removed is inconsistent, and the
    row then keeps ``0`` with a warning.
    """
    deleted = (events["action"] == "deleted").to_numpy()
    rest = events["volume"].to_numpy() - events["fill"].to_numpy()
    overfilled = int((deleted & (rest < 0)).sum())
    if overfilled:
        warnings.warn(
            f"DatabentoLoader: {overfilled} cancels that end an order remove "
            "less than the fill charged to them; the records are inconsistent, "
            "so those rows record the fill and no cancelled size",
            stacklevel=2,
        )
    volume = np.where(deleted, np.maximum(rest, 0), events["volume"].to_numpy())
    return events.assign(volume=volume.astype("int64"))


def _split_mixed_changes(events: pd.DataFrame) -> pd.DataFrame:
    """Split each ``changed`` row that both executes and reduces or moves.

    The schema lets a ``changed`` row be an execution (its fill comes off and
    nothing else changes) or a reduction or move with no fill, never both.  A
    Databento ``C`` or ``M`` that follows a fill can be both: a fill of 3 and
    then a cancel of 5 executes 3 and cancels 2, and a fill and then a modify
    to a new price executes and moves the order.  Such a row becomes two:

    * an execution row with the fill, the order's previous outstanding size
      less the fill, and the price the order rests at;
    * then a row with no fill and the record's own price and size, which takes
      off the rest, adds what the order gained, or moves the order.

    The second row shares the record's ``original_number`` and has no slot a
    fill can be charged to.  A fill larger than what the order had resting is
    capped at that size on the execution row.  Rows are in event order and
    carry ``epoch``, ``book_rank`` and ``outstanding``.
    """
    before = (
        events.groupby(["epoch", "id"], sort=False)["outstanding"]
        .shift(fill_value=0)
        .to_numpy()
    )
    fill = events["fill"].to_numpy()
    volume = events["volume"].to_numpy()
    # A row after the order's delete has nothing resting to execute against,
    # so it is left alone.
    candidate = (events["action"] == "changed").to_numpy() & (fill > 0) & (before > 0)
    if not candidate.any():
        return events

    executed = np.minimum(fill, before)
    left = before - executed
    is_modify = (events["raw_event_type"] == ACTION_MODIFY).to_numpy()
    # A modify may have moved the order, which is only known once the order's
    # resting price is worked out, so every modify with a fill is split here
    # and the second row is dropped below when it changes nothing.
    split = candidate & (is_modify | (left != volume) | (executed != fill))

    inconsistent = int(
        (split & (((left < volume) & ~is_modify) | (executed != fill))).sum()
    )
    if inconsistent:
        warnings.warn(
            f"DatabentoLoader: {inconsistent} cancels or modifies disagree "
            "with the fill charged to them (a cancel removes less than the "
            "fill, or the fill is larger than the order had resting); the "
            "book keeps the size the cancel or modify records leave",
            stacklevel=2,
        )
    if not split.any():
        return events

    part = np.zeros(len(events), dtype="int64")
    execution = events.assign(
        fill=np.where(split, executed, fill),
        volume=np.where(split, left, volume),
        part=np.where(split, 1, part),
    )
    rest = events[split].assign(
        fill=0,
        part=2,
        book_rank=-1,
        source_pos=events.loc[split, "source_pos"] + 0.25,
    )
    out = pd.concat([execution, rest], ignore_index=True)
    out = out.sort_values("source_pos", kind="stable").reset_index(drop=True)

    # Only the orders with a split row need their resting price worked out.
    affected = out["id"].isin(events.loc[split, "id"]).to_numpy()
    execution_row = affected & (out["part"] == 1).to_numpy()
    resting = _engine_frames.resting_price(out[affected]).to_numpy()
    price = out["price"].to_numpy().copy()
    price[execution_row] = resting[execution_row[affected]]
    out["price"] = price
    noop = (
        (out["part"] == 2)
        & (out["volume"] == out["volume"].shift())
        & (out["price"] == out["price"].shift())
    )
    return out[~noop].drop(columns=["part"]).reset_index(drop=True)


def _trade_records(
    fills: pd.DataFrame,
    work: pd.DataFrame,
    maker_event_id: np.ndarray,
    maker_og: np.ndarray,
) -> pd.DataFrame:
    """Return the F and T records the trade reader projects into trades.

    ``maker_og`` is the ``original_number`` of the book event a fill was
    charged to, and ``record`` the record's own.
    """
    prints = work[work["action"] == ACTION_TRADE]

    fill_rows = pd.DataFrame(
        {
            "timestamp": fills["timestamp"].array,
            "price": fills["price"].to_numpy(),
            "volume": fills["size_lots"].to_numpy(),
            "direction": pd.Categorical(
                fills["side"].map(_FILL_SIDE_TO_DIRECTION), dtype=TRADE_SIDE_DTYPE
            ),
            "maker": fills["id"].to_numpy(),
            "maker_event_id": pd.array(maker_event_id, dtype="Int64"),
            "maker_og": pd.array(maker_og, dtype="Int64"),
            "record": fills["original_number"].to_numpy(),
            "raw_event_type": ACTION_FILL,
        }
    )
    print_rows = pd.DataFrame(
        {
            "timestamp": prints["timestamp"].array,
            "price": prints["price"].to_numpy(),
            "volume": prints["size_lots"].to_numpy(),
            "direction": pd.Categorical(
                prints["side"].map(_TRADE_SIDE_TO_DIRECTION), dtype=TRADE_SIDE_DTYPE
            ),
            "maker": pd.NA,
            "maker_event_id": pd.array([pd.NA] * len(prints), dtype="Int64"),
            "maker_og": pd.array([pd.NA] * len(prints), dtype="Int64"),
            "record": prints["original_number"].to_numpy(),
            "raw_event_type": ACTION_TRADE,
        }
    )
    return pd.concat([fill_rows, print_rows], ignore_index=True)


def _in_record_order(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Join the trade records of a file's parts, fills first, in file order.

    That is the order one part on its own gives.  A fill whose book record was
    in a later part comes back into its place among the fills.
    """
    records = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    is_print = (records["raw_event_type"] == ACTION_TRADE).to_numpy()
    order = np.lexsort((records["record"].to_numpy(), is_print))
    return records.iloc[order].reset_index(drop=True)


# ── DatabentoTradeReader ──────────────────────────────────────────────


class DatabentoTradeReader:
    """Build trades from the fill and trade records of a Databento MBO file.

    Satisfies the :class:`~ob_analytics.protocols.TradeSource` protocol.

    Which record makes a trade is :attr:`DatabentoSettings.trades_from`:

    * **Fills (``F``), the default.** A fill names the resting order, so each
      one becomes a trade row with a ``maker`` and a ``maker_event_id``, one row
      per resting order a sweep took out — the same per-maker-leg shape the
      LOBSTER and Bitstamp readers produce.  A file with no fills falls back to
      its prints.  The catch: a trade the publisher sent no fill for — an
      auction, a trade against a non-displayed order, an off-exchange print —
      has nothing to become, so it is not in the frame.  The reader warns with
      the volume left out.
    * **Trade prints (``T``).** The venue's whole tape, one row per print, but
      no maker on any row.

    The two are not mixed.  A print and the fills behind it describe one
    execution, and nothing in a DBN record ties them together reliably — a
    fill and the modify it causes can carry different receive times — so
    joining them would risk counting the same volume twice.

    Where the venue states the aggressor, ``direction`` is its answer rather
    than an estimate.  Where it does not, the pipeline classifies it against
    the reconstructed quotes.

    The taker's own order is **not** identified.  A DBN trade record does not
    reliably carry the aggressing order's id, so ``taker`` and
    ``taker_event_id`` are NA.  That is worth knowing before reading
    :func:`~ob_analytics.analytics.set_order_types`: with no taker ids it
    labels executed resting orders ``resting-limit`` and never ``market`` or
    ``market-limit``.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.
    loader : DatabentoLoader
        The loader that read the events frame.  The fills come from it, so it
        must be the same instance, already used for the run
        (:class:`DatabentoSource` wires this up).
    settings : DatabentoSettings, optional
        Supplies :attr:`~DatabentoSettings.trades_from`.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        loader: DatabentoLoader,
        settings: DatabentoSettings | None = None,
    ) -> None:
        self._config = DATABENTO_GRID.config(config)
        self._loader = loader
        self._settings = settings or DatabentoSettings()

    def load(self, events: pd.DataFrame, source: Any) -> pd.DataFrame:
        """Build the trades DataFrame for the run.

        Parameters
        ----------
        events : pandas.DataFrame
            Unused; each trade record already names its maker event.
        source
            Unused; the trade records came off the file with the events.

        Returns
        -------
        pandas.DataFrame
        """
        records = self._loader.trade_records
        if records is None:
            raise ConfigError(
                "DatabentoTradeReader: the loader has not read a file yet. "
                "Pass the same DatabentoLoader instance that loaded the "
                "events (Pipeline(source=DatabentoSource()) does this)."
            )
        if records.empty:
            return empty_trades()

        kind = self._record_kind(records)
        chosen = records[records["raw_event_type"] == kind]
        if kind == ACTION_FILL:
            _warn_on_volume_gap(records)
        if chosen.empty:
            return empty_trades()

        n = len(chosen)
        trades = pd.DataFrame(
            {
                "timestamp": chosen["timestamp"].array,
                "price": chosen["price"].to_numpy(),
                "volume": chosen["volume"].to_numpy(),
                "direction": pd.Categorical(
                    chosen["direction"].to_numpy(), dtype=TRADE_SIDE_DTYPE
                ),
                "maker_event_id": chosen["maker_event_id"].to_numpy(),
                # Databento does not identify the aggressing order (see the
                # class docstring), so the taker side is unattributed.
                "taker_event_id": pd.array([pd.NA] * n, dtype="Int64"),
                "maker": chosen["maker"].to_numpy(),
                "taker": pd.array([pd.NA] * n, dtype="object"),
                "maker_og": chosen["maker_og"].to_numpy(),
                "taker_og": pd.array([pd.NA] * n, dtype="object"),
            }
        )
        trades = cast_columns(
            trades.sort_values("timestamp", kind="stable").reset_index(drop=True),
            TRADE_DTYPES,
        )

        logger.info(
            "DatabentoTradeReader: {} trades from {} records ({} with a maker event)",
            len(trades),
            "fill" if kind == ACTION_FILL else "trade-print",
            int(trades["maker_event_id"].notna().sum()),
        )
        return trades

    def _record_kind(self, records: pd.DataFrame) -> str:
        """Return the action the trades are built from for *records*."""
        kinds = records["raw_event_type"]
        if self._settings.trades_from == "prints":
            if not (kinds == ACTION_TRADE).any():
                logger.warning(
                    "DatabentoTradeReader: trades_from='prints' but the file "
                    "has no trade prints; the trades frame is empty"
                )
            return ACTION_TRADE
        return ACTION_FILL if (kinds == ACTION_FILL).any() else ACTION_TRADE


def _warn_on_volume_gap(records: pd.DataFrame) -> None:
    """Warn when the fills do not account for all the traded volume.

    A publisher that sends both fills and trade prints sends one print per
    execution and one fill per resting order it took out, so the two volumes
    should agree.  When they do not, some trades — an opening auction, say, or
    a print against a non-displayed order — have no fills behind them and are
    not in the trades frame.
    """
    by_kind = records.groupby("raw_event_type", observed=True)["volume"].sum()
    filled = int(by_kind.get(ACTION_FILL, 0))
    printed = int(by_kind.get(ACTION_TRADE, 0))
    if printed and filled != printed:
        logger.warning(
            "DatabentoTradeReader: fills account for {} of {} printed trade "
            "volume ({} unaccounted). Trades with no fill behind them — "
            "auctions, non-displayed orders, off-exchange prints — are not in "
            "the trades frame; DatabentoSettings(trades_from='prints') builds "
            "it from the whole tape instead, without makers",
            filled,
            printed,
            printed - filled,
        )


# ── DatabentoWriter ───────────────────────────────────────────────────


class DatabentoWriter:
    """Write an events frame back out as a DBN file of MBO records.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol, and
    inverts :class:`DatabentoLoader` row for row: a ``created`` row becomes
    ``A``, a ``changed`` row ``M``, and a ``deleted`` row ``C``.  The file
    reads back to the same events, but not always to the same records: a
    cancel the loader wrote as two ``changed`` rows comes back as two ``M``.
    The
    ``C`` removes what the order had resting before the row: its ``volume``
    plus its ``fill`` (see :mod:`ob_analytics.schemas`).  A fill the order
    took while resting becomes an ``F`` record immediately before the row
    that carries it.

    DBN shows an order only while it rests.  A fill the order took as the
    aggressor is not written as an ``F``, and its record keeps the order's own
    price.  An order that took liquidity when it arrived first appears at its
    last such fill, as an ``A`` for the size it had left, after the other
    records of that instant; if nothing was left, it is not written at all.
    The aggressor's fills are the fills of the orders classed ``market`` and,
    when ``data`` has a ``"trades"`` table, the events a trade names as
    ``taker_event_id``.  Each trade then comes back once, from its resting
    order's ``F``.

    DBN sizes are whole units (shares or contracts), so a size that is not a
    whole number of units at the config's ``lot_size`` is refused.

    The point is a round trip — reading a window, working on it, and writing
    something another DBN reader can open — not re-creating a vendor file byte
    for byte.  The metadata says ``OB.ANALYTICS`` (or whatever
    :attr:`DatabentoSettings.dataset` says) rather than claiming to be
    Databento's own data.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration; its ``tick_size`` and ``lot_size`` invert the
        canonical ticks and lots, and prices are written in Databento's
        fixed-point encoding (:data:`DBN_PRICE_DIVISOR`).  It is the config
        of the run being written; it defaults to :data:`DATABENTO_GRID`.
    settings : DatabentoSettings, optional
        ``dataset``, ``instrument_id`` and ``publisher_id`` to stamp on the
        records (defaulting to ``1`` and ``1`` when unset).
    """

    target = WriteTarget.FILE

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        settings: DatabentoSettings | None = None,
    ) -> None:
        self._config = config if config is not None else DATABENTO_GRID.config()
        self._settings = settings or DatabentoSettings()

    def write(
        self,
        data: dict[str, pd.DataFrame],
        dest: str | Path,
        *,
        symbol: str = "SYMBOL",
        **kwargs: Any,
    ) -> Path:
        """Write ``data["events"]`` to *dest* as a DBN file.

        Parameters
        ----------
        data : dict of str to DataFrame
            Must contain ``"events"``.  A ``"trades"`` table, when present,
            names the fills each aggressor took, which are not written.
        dest : str or Path
            Output file, such as ``events.dbn``.
        symbol : str
            The raw symbol to record in the file's metadata.

        Returns
        -------
        pathlib.Path
            The file written.

        Raises
        ------
        ValueError
            If a size is not a whole number of units at the config's
            ``lot_size``.
        """
        try:
            from databento_dbn import Action, MBOMsg, Metadata, Schema, Side, SType
        except ImportError as exc:  # pragma: no cover - exercised by the extra
            raise ImportError(
                "Writing DBN files needs the databento package, which is an "
                'optional dependency: pip install "ob-analytics[databento]".'
            ) from exc

        events = data["events"]
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        records = self._records(events, data.get("trades"))
        start = int(records[0][0]) if records else 0
        # The enums are built with ``from_str`` rather than named as members:
        # databento-dbn's own type stubs declare the members as ``str``, so
        # ``SType.RAW_SYMBOL`` does not type-check against a parameter that
        # wants an ``SType``, while ``from_str`` does.
        meta = Metadata(
            dataset=self._settings.dataset,
            start=start,
            stype_in=SType.from_str("raw_symbol"),
            stype_out=SType.from_str("instrument_id"),
            schema=Schema.from_str("mbo"),
            symbols=[symbol],
        )

        publisher_id = self._settings.publisher_id or 1
        instrument_id = self._settings.instrument_id or 1
        payload = bytearray(meta.encode())
        for ts_recv, ts_event, order_id, price, size, action, side in records:
            payload += bytes(
                MBOMsg(
                    publisher_id=publisher_id,
                    instrument_id=instrument_id,
                    ts_event=int(ts_event),
                    order_id=int(order_id),
                    price=int(price),
                    size=int(size),
                    action=Action.from_str(action),
                    side=Side.from_str(side),
                    ts_recv=int(ts_recv),
                )
            )
        dest.write_bytes(bytes(payload))

        logger.info("DatabentoWriter: wrote {} records to {}", len(records), dest)
        return dest

    def _records(
        self, events: pd.DataFrame, trades: pd.DataFrame | None = None
    ) -> list[tuple[Any, ...]]:
        """Return the DBN records for *events*, fills before the rows they sit on."""
        cfg = self._config
        events, fill = _resting_rows(events, trades)

        # Back from integer ticks and lots to the feed's own encoding: the
        # quote-currency price re-scaled by Databento's fixed-point divisor,
        # and the size as the venue's own whole quantity.  A ``deleted`` row's
        # ``C`` takes off everything the order had resting: the part removed
        # without trading and the part executed.
        price_raw = np.round(
            ticks_to_price(events["price"], cfg.tick_size, decimals=cfg.price_decimals)
            * DBN_PRICE_DIVISOR
        ).astype("int64")
        action = events["action"].astype(str).to_numpy()
        removed = np.where(
            action == "deleted",
            events["volume"].to_numpy() + events["fill"].to_numpy(),
            events["volume"].to_numpy(),
        )
        size_raw = _whole_units(removed, cfg.lot_size, "volume")
        fill_raw = _whole_units(fill, cfg.lot_size, "fill")

        ts_recv = epoch_ns(events["timestamp"])
        ts_event = epoch_ns(events["exchange_timestamp"])
        order_id = events["id"].to_numpy()
        side = events["direction"].astype(str).map(_DIRECTION_TO_SIDE).to_numpy()

        out: list[tuple[Any, ...]] = []
        for i in range(len(events)):
            if fill_raw[i] > 0:
                # The fill the loader charged to this event, put back where it
                # came from: just before the record that takes the size off.
                out.append(
                    (
                        ts_recv[i],
                        ts_event[i],
                        order_id[i],
                        price_raw[i],
                        fill_raw[i],
                        ACTION_FILL,
                        side[i],
                    )
                )
            out.append(
                (
                    ts_recv[i],
                    ts_event[i],
                    order_id[i],
                    price_raw[i],
                    size_raw[i],
                    _CANONICAL_TO_ACTION[action[i]],
                    side[i],
                )
            )
        return out


def _resting_rows(
    events: pd.DataFrame, trades: pd.DataFrame | None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the rows to write, and the size of the ``F`` before each one.

    A market-by-order feed shows an order only while it rests.  So the rows
    of an order that first takes liquidity, up to its last fill as the
    aggressor, are not written.  If something is left after that fill, that
    row becomes the order's ``A``, with the size left, at the order's own
    price.  If nothing is left, the order is not written at all.

    A fill the order took as the aggressor is never written as an ``F``, and
    its row keeps the order's own price: a venue can report it at the price
    it traded at, and a record at that price would read back as the order
    moving there.  A fill the order took while resting is written as an
    ``F``.
    """
    fill = events.get("fill", pd.Series(0, index=events.index)).to_numpy()
    taker = _aggressor_fills(events, trades)
    if not taker.any():
        # A table the Databento loader made names no aggressor: nothing to do.
        return events.assign(fill=fill), fill
    action = events["action"].astype(str).to_numpy()
    order = events["id"].to_numpy()
    by_order = pd.Series(order)

    # The leading run of each order: its first row and the aggressor fills
    # straight after it.
    first = ~by_order.duplicated().to_numpy()
    lead = pd.Series(first | taker).astype("int64").groupby(order).cumprod()
    in_lead = lead.to_numpy() == 1
    next_in_lead = (
        pd.Series(in_lead).groupby(order).shift(-1).fillna(False).to_numpy(bool)
    )
    last_lead = in_lead & ~next_in_lead
    took_first = pd.Series(in_lead & taker).groupby(order).transform("any")
    left_nothing = pd.Series(
        last_lead & ((action == "deleted") | (events["volume"].to_numpy() == 0))
    )
    gone = (took_first & left_nothing.groupby(order).transform("any")).to_numpy()
    took_first = took_first.to_numpy()
    joins = took_first & last_lead & ~gone
    keep = ~gone & ~(took_first & in_lead & ~last_lead)

    # The price the order rests at, which every rebuild places it at.
    resting = _engine_frames.resting_price(events).to_numpy()
    price = np.where(taker | joins, resting, events["price"].to_numpy())

    rows = events.assign(
        price=price,
        action=np.where(joins, "created", action),
        fill=np.where(joins, 0, fill),
    ).astype({"price": events["price"].dtype})
    written_fill = np.where(taker, 0, fill)
    rows, written_fill, joins = rows[keep], written_fill[keep], joins[keep]
    # In time order, with each joining aggressor's rows at its instant after
    # the other orders' rows there: the makers it took must leave the book
    # before it rests.  Its own rows keep their order, so a cancel of what it
    # had left still follows its ``A``.
    when = epoch_ns(rows["timestamp"])
    kept_order = rows["id"].to_numpy()
    joined_at = (
        pd.Series(np.where(joins, when, np.nan)).groupby(kept_order).transform("max")
    ).to_numpy()
    trails = when == joined_at
    in_order = np.lexsort((trails, when))
    return rows.iloc[in_order], written_fill[in_order]


def _aggressor_fills(events: pd.DataFrame, trades: pd.DataFrame | None) -> np.ndarray:
    """Return which rows of *events* carry a fill their order took as the aggressor.

    Those are the fills of the orders
    :func:`~ob_analytics.analytics.set_order_types` classed ``market``, which
    only ever took liquidity, and the rows a trade names as its
    ``taker_event_id``.
    """
    if "fill" not in events.columns:
        return np.zeros(len(events), dtype=bool)
    filled = (events["fill"] > 0).to_numpy()
    taker = np.zeros(len(events), dtype=bool)
    if "type" in events.columns:
        taker |= filled & (events["type"].astype(str) == "market").to_numpy()
    if trades is not None and not trades.empty:
        if "taker_event_id" in trades.columns and "event_id" in events.columns:
            # A row a trade also names as its maker took that fill resting.
            taker |= filled & _named(events, trades, "taker_event_id")
            taker &= ~_named(events, trades, "maker_event_id")
        else:
            warnings.warn(
                "DatabentoWriter: the trades have no taker_event_id, or the "
                "events no event_id, so the trades cannot say which fills an "
                "aggressor took; only the market orders' fills are left out",
                stacklevel=5,
            )
    return taker


def _named(events: pd.DataFrame, trades: pd.DataFrame, column: str) -> np.ndarray:
    """Return which rows of *events* a trade names in its *column*."""
    if column not in trades.columns:
        return np.zeros(len(events), dtype=bool)
    named = pd.to_numeric(trades[column], errors="coerce").dropna()
    return events["event_id"].isin(named.astype("int64")).to_numpy()


def _whole_units(lots: Any, lot_size: float, column: str) -> np.ndarray:
    """Return *lots* as whole units of the instrument, or raise.

    DBN carries sizes as whole shares or contracts.  A size that is a fraction
    of a unit has no DBN record, and rounding it would change the book.
    """
    size = lots_to_size(lots, lot_size)
    not_whole = off_tick_grid(size, 1.0)
    if not_whole.any():
        raise ValueError(
            f"DatabentoWriter: {int(not_whole.sum())} {column} value(s) are not a "
            f"whole number of units at lot_size={lot_size:g}, and DBN sizes are "
            "whole shares or contracts.  Pass the config the events were made "
            "with; a size finer than one unit cannot be written to DBN."
        )
    return np.round(size).astype("int64")


#: How a canonical action is written back out.
_CANONICAL_TO_ACTION: dict[str, str] = {
    "created": ACTION_ADD,
    "changed": ACTION_MODIFY,
    "deleted": ACTION_CANCEL,
}


# ── DatabentoSource descriptor ────────────────────────────────────────


@dataclass
class DatabentoSource:
    """The Databento source: offline replay of DBN market-by-order files (L3).

    Offline only — Databento's Live client is not wired up here — so it
    satisfies :class:`~ob_analytics.protocols.OfflineSource` and has no live
    capability.  Use it as::

        from ob_analytics import Pipeline
        from ob_analytics.databento import DatabentoSource, DatabentoSettings

        source = DatabentoSource(settings=DatabentoSettings(raw_symbol="AAPL"))
        result = Pipeline(source=source).run("xnas-itch-20240403.mbo.dbn.zst")

    The defaults suit a US equity: a one-cent tick and whole shares.  Another
    instrument needs its own, e.g. ``PipelineConfig(tick_size=0.25,
    price_decimals=2)`` for the E-mini S&P 500 future.
    """

    name: str = field(default="databento", init=False, repr=False)
    # Databento normalizes a venue's own matching-engine feed, so a bid can
    # never rest above an ask and the reconstructed book is never crossed.
    feed_type: FeedType = field(default=FeedType.MATCHED_BOOK, init=False, repr=False)
    # A fill names the resting order it was charged to; a DBN trade record
    # does not reliably carry the aggressor's order id.
    trade_attribution: TradeAttribution = field(
        default=TradeAttribution.MAKER_ONLY, init=False, repr=False
    )
    # ``sequence`` numbers every message on the venue's channel, so it skips
    # the messages about other instruments, and the trade and fill records'
    # numbers leave with them when they become trades.  Only a step back is a
    # fault.
    sequence_kind: SequenceKind = field(
        default=SequenceKind.MONOTONIC, init=False, repr=False
    )
    # Market-by-order: one record per resting order, with stable identity.
    level: Level = field(default=Level.L3, init=False, repr=False)
    # Declared as the base type the ``Source`` protocol states, the way every
    # other source declares it; :meth:`dbn_settings` narrows it back.
    settings: SourceSettings = field(default_factory=DatabentoSettings)
    # A cent tick and whole shares: a US equity.
    grid: InstrumentGrid = field(default=DATABENTO_GRID, init=False, repr=False)

    def dbn_settings(self) -> DatabentoSettings:
        """Return :attr:`settings` as :class:`DatabentoSettings`.

        Raises
        ------
        ConfigError
            If the source was built with another source's settings.
        """
        if not isinstance(self.settings, DatabentoSettings):
            raise ConfigError(
                "DatabentoSource: settings must be DatabentoSettings, got "
                f"{type(self.settings).__name__}."
            )
        return self.settings

    def create_loader(self, config: PipelineConfig, ctx: RunContext) -> EventLoader:
        settings = self.dbn_settings()
        return DatabentoLoader(
            config,
            settings=settings,
            venue=ctx.venue,
            symbol=ctx.symbol or settings.raw_symbol,
        )

    def create_trade_source(
        self, config: PipelineConfig, ctx: RunContext, loader: Any
    ) -> TradeSource:
        # The trades come off the same read as the events, so the reader takes
        # the run's loader rather than re-reading the file.
        if not isinstance(loader, DatabentoLoader):
            raise ConfigError(
                "DatabentoSource: the trades come from the DatabentoLoader that "
                f"reads the events, and this run's loader is a "
                f"{type(loader).__name__}.  Pass a DatabentoLoader, or a trade "
                "source of your own."
            )
        return DatabentoTradeReader(config, loader=loader, settings=self.dbn_settings())

    def create_writer(
        self, config: PipelineConfig | None, ctx: RunContext
    ) -> DataWriter:
        return DatabentoWriter(config, settings=self.dbn_settings())


# ── Register this source ──────────────────────────────────────────────
# Registration runs at the bottom, after ``DatabentoSource`` is defined.
from ob_analytics.sources import register_source

register_source("databento", DatabentoSource)
