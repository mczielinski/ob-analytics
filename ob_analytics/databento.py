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
the trader asked for.

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
the order.  The loss of queue priority is not modelled.  A modify that also
carries a fill is read as an execution report; if it also moves the order or
changes its size by more than the fill, the loader says in a warning how many
rows the depth will be off by.

What is refused and what is dropped
-----------------------------------
A feed this adapter does not understand is refused: a publisher that only sends
top-of-book or price-level data, a file holding a price-level schema, a file
covering more than one book, an ``action`` outside DBN's own alphabet, and an
``order_id`` too big for the shared schema's signed 64-bit id.

A malformed record inside a feed it does understand is dropped and counted: one
with no price (:data:`UNDEF_PRICE`), and one with no side on a book action.
Either would otherwise land somewhere wrong — a level at nine billion, or an
order on neither side of the book — and refusing a whole session over a handful
of them would be worse than saying how many went.

``databento`` is an optional dependency (``pip install
"ob-analytics[databento]"``).  It is imported lazily, so importing this module —
and listing sources — never requires it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ob_analytics._utils import (
    UTC_NS_DTYPE,
    attach_ingest_seq,
    empty_trades,
    epoch_to_datetime,
    lots_to_size,
    price_to_ticks,
    size_to_lots,
    ticks_to_price,
)
from ob_analytics.config import PipelineConfig, SourceSettings
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    FeedType,
    Level,
    RunContext,
    TradeSource,
)
from ob_analytics.schemas import SEQUENCE_COLUMN, attach_instrument_identity

# ── DBN constants ─────────────────────────────────────────────────────
#
# Spelled out here rather than imported from ``databento_dbn`` so this module
# imports, and the source registers, without the optional dependency.

#: Databento fixed-point price scale: one raw unit is 1e-9 of the quote
#: currency, so the raw integer divides by this to reach the quote currency.
#: This is the feed's encoding, not the instrument's tick (see
#: :attr:`~ob_analytics.config.PipelineConfig.price_divisor`).
DBN_PRICE_DIVISOR = 1_000_000_000

#: The raw price that means "no price" (``INT64_MAX``).
UNDEF_PRICE = 9_223_372_036_854_775_807

#: ``flags`` bits this loader reads.  ``F_TOB`` and ``F_MBP`` mark records that
#: are *not* individual orders — a publisher that only sends top-of-book or
#: price-level data, normalized into MBO records whose ``order_id`` means
#: nothing.  ``F_MAYBE_BAD_BOOK`` marks a gap in the channel.
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
    dataset : str
        The dataset id :class:`DatabentoWriter` stamps into the DBN metadata it
        writes.  Read only on the way out; the loader takes the file's own.
    """

    instrument_id: int | None = None
    publisher_id: int | None = None
    raw_symbol: str | None = None
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
        store = source if hasattr(source, "to_df") else _open_dbn(source)
        schema = str(getattr(store.metadata, "schema", "") or "")
        if schema != "mbo":
            raise ConfigError(
                f"DatabentoLoader: file holds the {schema!r} schema, which is "
                "not per-order data. Request the 'mbo' schema for order-book "
                "reconstruction; the price-level schemas (mbp-1, mbp-10, "
                "bbo, tbbo) carry no order ids."
            )
        # ``price_type="fixed"`` keeps Databento's raw integer prices, which
        # convert to ticks exactly; the decimal default would round-trip them
        # through a float first.
        frame = store.to_df(price_type="fixed").reset_index()

    missing = [c for c in _REQUIRED_MBO_COLUMNS if c not in frame.columns]
    if missing:
        raise ConfigError(
            f"DatabentoLoader: MBO frame is missing required columns {missing}. "
            f"Present: {list(frame.columns)}"
        )
    return frame


def _open_dbn(path: Any) -> Any:
    """Return a ``databento.DBNStore`` for *path* (lazy import)."""
    try:
        import databento as db
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ImportError(
            "Reading DBN files needs the databento package, which is an "
            'optional dependency: pip install "ob-analytics[databento]".'
        ) from exc

    return db.DBNStore.from_file(Path(path))


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
        Pipeline configuration.  ``price_divisor`` is Databento's fixed-point
        scale (:data:`DBN_PRICE_DIVISOR`) and ``lot_size`` is normally ``1``,
        both supplied by :meth:`DatabentoSource.config_defaults`.
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
        self._config = config or PipelineConfig()
        self._settings = settings or DatabentoSettings()
        self._venue = venue
        self._symbol = symbol
        #: The ``F`` (fill) and ``T`` (trade) records of the last
        #: :meth:`load`, each already carrying the ``maker_event_id`` of the
        #: book event it was charged to.  ``None`` before the first load.
        self.trade_records: pd.DataFrame | None = None

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
            record's own ``size`` field).
        """
        logger.info("DatabentoLoader: reading {}", source)
        raw = read_mbo_frame(source)
        raw = self._select_instrument(raw)
        _reject_aggregated_records(raw)

        work = self._to_working_frame(raw)
        events, fills = _book_events_and_fills(work)

        self.trade_records = fills

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
            events = attach_ingest_seq(events)

        events = events.drop(columns=["source_pos"])
        events["action"] = pd.Categorical(
            events["action"], categories=["created", "changed", "deleted"], ordered=True
        )
        events["direction"] = pd.Categorical(
            events["direction"], categories=["bid", "ask"], ordered=True
        )

        logger.info(
            "DatabentoLoader: {} book events ({} created, {} changed, "
            "{} deleted), {} fills, {} trade prints",
            len(events),
            int((events["action"] == "created").sum()),
            int((events["action"] == "changed").sum()),
            int((events["action"] == "deleted").sum()),
            int((fills["raw_event_type"] == ACTION_FILL).sum()),
            int((fills["raw_event_type"] == ACTION_TRADE).sum()),
        )

        events = attach_instrument_identity(
            events,
            venue=self._venue,
            symbol=self._symbol,
            default_venue=self._VENUE,
        )
        return events

    # ── input narrowing ───────────────────────────────────────────────

    def _select_instrument(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Narrow *raw* to the single book this run covers.

        Applies whichever of the :class:`DatabentoSettings` filters were set,
        then insists that what is left is one instrument from one publisher:
        two publishers of the same instrument are two books, and merging them
        would reconstruct a book that never existed.
        """
        s = self._settings
        for column, wanted in (
            ("instrument_id", s.instrument_id),
            ("publisher_id", s.publisher_id),
            ("symbol", s.raw_symbol),
        ):
            if wanted is None:
                continue
            if column not in raw.columns:
                raise ConfigError(
                    f"DatabentoLoader: cannot filter on {column!r} — the "
                    f"records carry no such column. Present: {list(raw.columns)}"
                )
            raw = raw[raw[column] == wanted]
            if raw.empty:
                raise ConfigError(
                    f"DatabentoLoader: no records left after {column}={wanted!r}."
                )

        for column, setting in (
            ("instrument_id", "instrument_id"),
            ("publisher_id", "publisher_id"),
        ):
            if column not in raw.columns:
                continue
            present = pd.unique(raw[column])
            if len(present) > 1:
                raise ConfigError(
                    f"DatabentoLoader: the records cover {len(present)} values "
                    f"of {column} ({sorted(present.tolist())[:10]}). One run "
                    "reconstructs one book, so pick one with "
                    f"DatabentoSettings({setting}=...)."
                )

        return raw.reset_index(drop=True)

    def _to_working_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return the columns the mapping needs, typed and in file order.

        Keeps the file's own row order: a fill is charged to the book event
        that follows it, so re-sorting here would break the pairing.  The
        1-based file position is kept as ``original_number``.
        """
        cfg = self._config
        action = raw["action"].astype(str).str.strip()
        side = raw["side"].astype(str).str.strip()
        order_id = _order_ids(raw["order_id"])

        work = pd.DataFrame(
            {
                "original_number": np.arange(1, len(raw) + 1, dtype="int64"),
                "action": action.to_numpy(),
                "side": side.to_numpy(),
                "id": order_id,
                "raw_price": raw["price"].astype("int64").to_numpy(),
                "raw_size": raw["size"].astype("int64").to_numpy(),
                "timestamp": _as_utc_ns(raw["ts_recv"]),
                "exchange_timestamp": _as_utc_ns(raw["ts_event"]),
            }
        )
        if "sequence" in raw.columns:
            work["sequence"] = raw["sequence"].astype("int64").to_numpy()

        # Canonical price is integer ticks (issue #155) and canonical size
        # integer lots (issue #226).  The raw price is Databento's fixed-point
        # integer, so it divides by the feed's encoding scale to reach the
        # quote currency before it is quantised onto the instrument's tick.
        work["price"] = price_to_ticks(
            work["raw_price"].to_numpy() / cfg.price_divisor, cfg.tick_size
        )
        work["size_lots"] = size_to_lots(work["raw_size"].to_numpy(), cfg.lot_size)
        work["direction"] = work["side"].map(_SIDE_TO_DIRECTION)
        return _usable_records(work)


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

    no_price = work["raw_price"].to_numpy() == UNDEF_PRICE
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
            logger.warning(
                "DatabentoLoader: dropped {} of {} records with {} — {}. The "
                "book is missing whatever liquidity they carried",
                count,
                len(work),
                why,
                effect,
            )

    keep = ~(no_price | no_side)
    if keep.all():
        return work
    return work[keep].reset_index(drop=True)


def _as_utc_ns(series: pd.Series) -> pd.Series:
    """Return *series* as the canonical tz-aware UTC nanosecond clock.

    ``DBNStore.to_df`` already hands back tz-aware datetimes; a frame built
    from CSV or JSON carries the same instants as integer nanoseconds.
    """
    if pd.api.types.is_integer_dtype(series):
        return epoch_to_datetime(series, "ns")
    return pd.to_datetime(series, utc=True).astype(UTC_NS_DTYPE)


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

    gaps = int((flags & F_MAYBE_BAD_BOOK).astype(bool).sum())
    if gaps:
        logger.warning(
            "DatabentoLoader: {} records carry F_MAYBE_BAD_BOOK — the feed "
            "reported an unrecoverable gap, so the book may be wrong from "
            "there on",
            gaps,
        )


# ── The action mapping ────────────────────────────────────────────────


def _book_events_and_fills(
    work: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split MBO records into the canonical events frame and the trade records.

    Returns ``(events, trade_records)``.  ``events`` holds the A/M/C records
    plus a synthetic ``deleted`` row per order still resting when an ``R``
    (clear) arrives.  ``trade_records`` holds the F and T records, each with
    the ``maker_event_id`` of the book event its fill was charged to.
    """
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
    events["event_id"] = np.arange(1, len(events) + 1, dtype="int64")

    fills = work[is_fill]
    events, trade_records = _charge_fills_to_events(events, fills, work)
    _warn_on_amend_with_fill(events)
    return events, trade_records


def _events_from_book_records(book: pd.DataFrame) -> pd.DataFrame:
    """Map the A / M / C records onto the canonical action and volume.

    ``volume`` is the order's outstanding size after the event, except on a
    ``deleted`` row where it is the size removed — the convention the Bitstamp
    and LOBSTER loaders already follow (see :mod:`ob_analytics.schemas`).
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
    volume = np.where(is_set | emptied, size, outstanding)

    return pd.DataFrame(
        {
            "id": order_id,
            "timestamp": book["timestamp"].array,
            "exchange_timestamp": book["exchange_timestamp"].array,
            "price": book["price"].to_numpy(),
            "volume": volume.astype("int64"),
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


def _warn_on_amend_with_fill(events: pd.DataFrame) -> None:
    """Warn about the one modify the price-level rebuild cannot represent.

    Databento's ``M`` can move an order to another price or make it bigger.
    Both are recorded on a ``changed`` event with the new price and size, and
    :func:`~ob_analytics.depth.price_level_volume` follows either: a
    ``changed`` row that reports no execution moves the order's volume to the
    price it carries and adds any size it gained.

    A modify that also carries a fill is read as an execution report instead,
    because some venues report an execution at the price it traded at, not the
    price the order rests at.  The depth then takes only the fill off, at the
    level the order was resting on.  If the same modify also moved the order,
    or left it with a size other than its previous size less the fill, the
    depth is off by the difference.  The per-order tables — events, lifetimes,
    trades — are right either way.
    """
    order = events.groupby("id", sort=False)
    prev_price = order["price"].shift()
    prev_volume = order["volume"].shift()
    amended = (
        (events["raw_event_type"] == ACTION_MODIFY)
        & (events["action"] == "changed")
        & (events["fill"] > 0)
        & (
            (events["price"] != prev_price)
            | (events["volume"] != prev_volume - events["fill"])
        )
    )
    count = int(amended.sum())
    if count:
        logger.warning(
            "DatabentoLoader: {} modifies carry a fill and also move the "
            "order or change its size by more than the fill — the price-level "
            "depth takes off only the fill, at the price the order was resting "
            "at, so it is off by the rest. The events, order lifetimes and "
            "trades are unaffected",
            count,
        )


def _empty_book_events() -> pd.DataFrame:
    """Return the zero-row frame :func:`_events_from_book_records` produces."""
    dtypes = {
        "id": "int64",
        "timestamp": UTC_NS_DTYPE,
        "exchange_timestamp": UTC_NS_DTYPE,
        "price": "int64",
        "volume": "int64",
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach ``fill`` to the events and ``maker_event_id`` to the fills.

    A fill is charged to the next book record for the same order, which is the
    record that takes the executed size off the book.  Several fills can land
    on one book record when a sweep takes an order out in pieces, so the
    ``fill`` on an event is the sum of them.
    """
    slot = ["epoch", "id", "book_rank"]
    events = events.copy()
    events["fill"] = 0

    if not fills.empty:
        target = fills.assign(book_rank=fills["book_rank"] + 1)
        charged = target.groupby(slot, sort=False)["size_lots"].sum()
        key = pd.MultiIndex.from_frame(events[slot])
        events["fill"] = charged.reindex(key).fillna(0).to_numpy().astype("int64")

        event_ids = pd.Series(
            events["event_id"].to_numpy(),
            index=pd.MultiIndex.from_frame(events[slot]),
        )
        # An order id can repeat across epochs, and a book record is one slot,
        # so the index is unique by construction; guard anyway because a
        # malformed file could repeat a rank.
        event_ids = event_ids[~event_ids.index.duplicated()]
        maker_event_id = event_ids.reindex(
            pd.MultiIndex.from_frame(target[slot])
        ).to_numpy()
        orphan = int(pd.isna(maker_event_id).sum())
        if orphan:
            logger.warning(
                "DatabentoLoader: {} fills have no book record after them in "
                "this window (the window ends mid-execution); their trades "
                "carry no maker event",
                orphan,
            )
    else:
        maker_event_id = np.empty(0)

    trade_records = _trade_records(fills, work, maker_event_id)
    events = events.drop(columns=["epoch", "book_rank"])
    return events, trade_records


def _trade_records(
    fills: pd.DataFrame, work: pd.DataFrame, maker_event_id: np.ndarray
) -> pd.DataFrame:
    """Return the F and T records the trade reader projects into trades."""
    prints = work[work["action"] == ACTION_TRADE]

    fill_rows = pd.DataFrame(
        {
            "timestamp": fills["timestamp"].array,
            "price": fills["price"].to_numpy(),
            "volume": fills["size_lots"].to_numpy(),
            "direction": fills["side"].map(_FILL_SIDE_TO_DIRECTION).to_numpy(),
            "maker": fills["id"].to_numpy(),
            "maker_event_id": pd.array(maker_event_id, dtype="Int64"),
            "maker_og": fills["original_number"].to_numpy(),
            "raw_event_type": ACTION_FILL,
        }
    )
    print_rows = pd.DataFrame(
        {
            "timestamp": prints["timestamp"].array,
            "price": prints["price"].to_numpy(),
            "volume": prints["size_lots"].to_numpy(),
            "direction": prints["side"].map(_TRADE_SIDE_TO_DIRECTION).to_numpy(),
            "maker": pd.NA,
            "maker_event_id": pd.array([pd.NA] * len(prints), dtype="Int64"),
            "maker_og": prints["original_number"].to_numpy(),
            "raw_event_type": ACTION_TRADE,
        }
    )
    return pd.concat([fill_rows, print_rows], ignore_index=True)


# ── DatabentoTradeReader ──────────────────────────────────────────────


class DatabentoTradeReader:
    """Build trades from the fill and trade records of a Databento MBO file.

    Satisfies the :class:`~ob_analytics.protocols.TradeSource` protocol.

    Which record makes a trade depends on what the publisher sends:

    * **Fills (``F``) when the file has them.** A fill names the resting order,
      so each one becomes a trade row with a ``maker`` and a ``maker_event_id``,
      one row per resting order a sweep took out — the same per-maker-leg shape
      the LOBSTER and Bitstamp readers produce.
    * **Trade prints (``T``) otherwise.** Some publishers report no passive
      side at all; then the print is all there is, and the row carries the
      volume and the aggressor's side but no maker.

    Either way the aggressor's side comes from the venue, not from a
    classifier: Databento states it on both records, so ``direction`` is the
    real taker side rather than an estimate.

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
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        loader: DatabentoLoader,
    ) -> None:
        self._config = config or PipelineConfig()
        self._loader = loader

    def load(self, events: pd.DataFrame, source: Any) -> pd.DataFrame:
        """Build the trades DataFrame for the run.

        Parameters
        ----------
        events : pandas.DataFrame
            The loaded events frame, used to map a maker event back to its
            ``original_number``.
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

        has_fills = (records["raw_event_type"] == ACTION_FILL).any()
        chosen = records[
            records["raw_event_type"] == (ACTION_FILL if has_fills else ACTION_TRADE)
        ]
        if has_fills:
            _warn_on_volume_gap(records)
        if chosen.empty:
            return empty_trades()

        og_by_event = pd.Series(
            events["original_number"].to_numpy(), index=events["event_id"].to_numpy()
        )
        maker_og = chosen["maker_event_id"].map(og_by_event)

        n = len(chosen)
        trades = pd.DataFrame(
            {
                "timestamp": chosen["timestamp"].array,
                "price": chosen["price"].to_numpy(),
                "volume": chosen["volume"].to_numpy(),
                "direction": pd.Categorical(
                    chosen["direction"].to_numpy(),
                    categories=["buy", "sell"],
                    ordered=True,
                ),
                "maker_event_id": chosen["maker_event_id"].to_numpy(),
                # Databento does not identify the aggressing order (see the
                # class docstring), so the taker side is unattributed.
                "taker_event_id": pd.array([pd.NA] * n, dtype="Int64"),
                "maker": chosen["maker"].to_numpy(),
                "taker": pd.array([pd.NA] * n, dtype="object"),
                "maker_og": maker_og.astype("Int64").to_numpy(),
                "taker_og": pd.array([pd.NA] * n, dtype="object"),
            }
        )
        trades = trades.sort_values("timestamp", kind="stable").reset_index(drop=True)

        logger.info(
            "DatabentoTradeReader: {} trades from {} records ({} with a maker event)",
            len(trades),
            "fill" if has_fills else "trade-print",
            int(trades["maker_event_id"].notna().sum()),
        )
        return trades


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
            "the trades frame",
            filled,
            printed,
            printed - filled,
        )


# ── DatabentoWriter ───────────────────────────────────────────────────


class DatabentoWriter:
    """Write an events frame back out as a DBN file of MBO records.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol, and
    inverts :class:`DatabentoLoader` record for record: a ``created`` row
    becomes ``A``, a ``changed`` row ``M``, a ``deleted`` row ``C`` removing
    the outstanding size, and a non-zero ``fill`` becomes an ``F`` record
    immediately before the row that carries it.

    The point is a round trip — reading a window, working on it, and writing
    something another DBN reader can open — not re-creating a vendor file byte
    for byte.  The metadata says ``OB.ANALYTICS`` (or whatever
    :attr:`DatabentoSettings.dataset` says) rather than claiming to be
    Databento's own data.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration; its ``price_divisor``, ``tick_size`` and
        ``lot_size`` invert the canonical ticks and lots.
    settings : DatabentoSettings, optional
        ``dataset``, ``instrument_id`` and ``publisher_id`` to stamp on the
        records (defaulting to ``1`` and ``1`` when unset).
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        settings: DatabentoSettings | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
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
            Must contain ``"events"``.
        dest : str or Path
            Output file.  A directory is filled with ``events.dbn``.
        symbol : str
            The raw symbol to record in the file's metadata.

        Returns
        -------
        pathlib.Path
            The file written.
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
        if dest.is_dir():
            dest = dest / "events.dbn"
        dest.parent.mkdir(parents=True, exist_ok=True)

        records = self._records(events)
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

    def _records(self, events: pd.DataFrame) -> list[tuple[Any, ...]]:
        """Return the DBN records for *events*, fills before the rows they sit on."""
        cfg = self._config
        # Back from integer ticks and lots to the feed's own encoding: the
        # quote-currency price re-scaled by Databento's fixed-point divisor,
        # and the size as the venue's own whole quantity.
        price_raw = np.round(
            ticks_to_price(events["price"], cfg.tick_size, decimals=cfg.price_decimals)
            * cfg.price_divisor
        ).astype("int64")
        size_raw = np.round(
            lots_to_size(events["volume"], cfg.lot_size, decimals=cfg.volume_decimals)
        ).astype("int64")
        fill_raw = np.round(
            lots_to_size(
                events.get("fill", pd.Series(0, index=events.index)),
                cfg.lot_size,
                decimals=cfg.volume_decimals,
            )
        ).astype("int64")

        ts_recv = events["timestamp"].astype("int64").to_numpy()
        ts_event = events["exchange_timestamp"].astype("int64").to_numpy()
        order_id = events["id"].to_numpy()
        action = events["action"].astype(str).to_numpy()
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


#: How a canonical action is written back out.  A ``deleted`` row reports the
#: size removed, which is exactly what a ``C`` record carries.
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
    # Market-by-order: one record per resting order, with stable identity.
    level: Level = field(default=Level.L3, init=False, repr=False)
    # Declared as the base type the ``Source`` protocol states, the way every
    # other source declares it; :meth:`dbn_settings` narrows it back.
    settings: SourceSettings = field(default_factory=DatabentoSettings)

    _loader: DatabentoLoader | None = field(default=None, repr=False, init=False)

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

    def _make_loader(self, config: PipelineConfig, ctx: RunContext) -> DatabentoLoader:
        """Build the loader for this run and remember it."""
        settings = self.dbn_settings()
        self._loader = DatabentoLoader(
            config,
            settings=settings,
            venue=ctx.venue,
            symbol=ctx.symbol or settings.raw_symbol,
        )
        return self._loader

    def create_loader(self, config: PipelineConfig, ctx: RunContext) -> EventLoader:
        return self._make_loader(config, ctx)

    def create_trade_source(
        self, config: PipelineConfig, ctx: RunContext
    ) -> TradeSource:
        # The trades come off the same read as the events, so the reader takes
        # the loader itself rather than re-reading the file.
        loader = self._loader or self._make_loader(config, ctx)
        return DatabentoTradeReader(config, loader=loader)

    def create_writer(
        self, config: PipelineConfig | None, ctx: RunContext
    ) -> DataWriter:
        return DatabentoWriter(config, settings=self.dbn_settings())

    def compute_depth(
        self,
        events: pd.DataFrame,
        config: Any,
        source: Any,
        ctx: RunContext,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        # MBO is the book itself: there is no companion depth file to prefer
        # over the reconstruction, so the standard price-level path runs.
        return None

    def config_defaults(self) -> dict[str, Any]:
        return {
            # Databento's fixed-point encoding: one raw unit is 1e-9.
            "price_divisor": DBN_PRICE_DIVISOR,
            "tick_size": 0.01,
            "price_decimals": 2,
            # Sizes are whole shares or contracts, so one lot is one unit and
            # the canonical integer size (issue #226) is the venue's own count.
            "lot_size": 1.0,
            "volume_decimals": 0,
            "timestamp_unit": "ns",
        }

    def required_context(self) -> list[str]:
        # DBN records carry absolute UTC timestamps, so there is no date to
        # supply.
        return []


# ── Register this source ──────────────────────────────────────────────
# Registration runs at the bottom, after ``DatabentoSource`` is defined.
from ob_analytics.sources import register_source

register_source("databento", DatabentoSource)
