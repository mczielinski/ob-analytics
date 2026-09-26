"""cryptofeed L2/L3 live capturer -- the native-per-order complement to CCXT.

Wraps cryptofeed (``bmoscon/cryptofeed``): a library built to stream
normalised market data over websockets and maintain the order book for you.
Where :mod:`ob_analytics.live.ccxt_source` gives the widest **L2** venue list
plus prediction markets, cryptofeed is the source that can also deliver
**L3** (market-by-order) on the venues that publish per-order data -- the feed
the reconstruction engine was built for.

Level is **discovered from the venue, not hardcoded**: a cryptofeed exchange
class declares its channels in ``websocket_channels``, so a venue offering
``l3_book`` is captured at :attr:`~ob_analytics.protocols.Level.L3` and every
other venue at :attr:`~ob_analytics.protocols.Level.L2`.  This tracks
cryptofeed's own churn -- across 2.4 and 2.5 the L3 venues are bitstamp,
bitfinex, blockchain and independent_reserve; Coinbase publishes L2 only now.
``level=`` overrides the choice, but asking for L3 on a venue that does not
publish it raises rather than faking per-order IDs.

Order ids are written exactly as the venue publishes them -- integers on
bitstamp, bitfinex and blockchain, UUID strings on independent_reserve.  The
shared schema keys orders by identity, not by integer, so nothing is
re-labelled on the way through.

``cryptofeed`` is an optional dependency (``pip install
"ob-analytics[cryptofeed]"``), imported lazily so importing this module -- and
listing sources -- never requires it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, TypeGuard

import pandas as pd
from loguru import logger

from ob_analytics.config import SourceSettings
from ob_analytics.live._base import CaptureConfig, EventDict
from ob_analytics.protocols import FeedType, Level, TradeAttribution

#: cryptofeed's channel names (``cryptofeed.defines``), spelled out so this
#: module imports without cryptofeed installed.
_L3_CHANNEL = "l3_book"
_L2_CHANNEL = "l2_book"
_TRADES_CHANNEL = "trades"

#: Default depth of the callback -> iterator buffer.
_DEFAULT_QUEUE_SIZE = 10_000

#: How many orders per side a venue's L3 book shows, for venues whose book is
#: a window on the full book rather than all of it.  Keyed by cryptofeed's
#: exchange id.  Bitstamp's ``detail_order_book`` channel sends the top 100
#: bids and the top 100 asks on every message, so an order that drops below
#: the 100th place has left the view, not the book.
_L3_WINDOW: dict[str, int] = {"BITSTAMP": 100}

#: How long, in seconds of venue time, a book's report that an order is gone
#: is held for a trade that filled it (see ``_fill_events``).  The book and the
#: trades arrive on separate channels; on Bitstamp the book usually lands a few
#: milliseconds first, so without the wait a filled order reads as cancelled.
_FILL_GRACE_S = 2.0


def _epoch_s_to_ts(seconds: Any) -> pd.Timestamp:
    """cryptofeed timestamps are float epoch *seconds*; ``None`` uses the time now.

    Both branches land on the canonical tz-aware UTC nanosecond clock.
    """
    if seconds is None:
        return pd.Timestamp.now(tz="UTC").as_unit("ns")
    return pd.Timestamp(float(seconds), unit="s", tz="UTC").as_unit("ns")


def _trade_order_ids(trade: Any) -> tuple[Any, Any]:
    """Return ``(buy_order_id, sell_order_id)`` from a trade's raw frame.

    cryptofeed's ``Trade`` has no field for the orders on each side of a
    print, but some venues send them: Bitstamp's ``live_trades`` frame carries
    ``buy_order_id`` and ``sell_order_id`` in its ``data``.  They are read from
    the raw frame so the trade can be matched to the orders in the book.  A
    venue that does not send them gets empty strings, never invented ids.
    """
    raw = getattr(trade, "raw", None)
    data = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(data, dict):
        return "", ""
    buy, sell = data.get("buy_order_id"), data.get("sell_order_id")
    return ("" if buy is None else buy), ("" if sell is None else sell)


def _has_entries(delta: Any) -> TypeGuard[dict[str, Any]]:
    """Is this a delta that actually says something changed?

    cryptofeed's L3 venues disagree on how they signal "no delta here": some
    pass ``None`` (bitstamp on every message, blockchain and
    independent_reserve on their opening book) and bitfinex passes a populated
    dict whose sides are both empty.  Both mean the same thing -- the callback
    carries only the maintained book -- so both must fall through to the
    full-book diff.  Checking truthiness alone would take the empty dict for a
    real delta and silently emit nothing.
    """
    if not delta:
        return False
    return any(delta.get(side) for side in ("bid", "ask"))


def _book_levels(book: Any) -> dict[str, dict[float, Any]]:
    """Return ``{"bid": {price: value}, "ask": ...}`` from a cryptofeed book.

    ``value`` is the level size at L2 and a ``{order_id: size}`` mapping at
    L3 -- cryptofeed uses the same container for both.  Prices come back as
    ``Decimal``; they are floated here so the two levels compare and
    serialise the same way everywhere downstream.
    """
    inner = getattr(book, "book", None)
    raw = inner.to_dict() if inner is not None else {}
    out: dict[str, dict[float, Any]] = {}
    for side in ("bid", "ask"):
        side_levels = raw.get(side) or {}
        out[side] = {
            float(price): (
                {oid: float(size) for oid, size in value.items()}
                if isinstance(value, dict)
                else float(value)
            )
            for price, value in side_levels.items()
        }
    return out


class CryptofeedSettings(SourceSettings):
    """Typed settings for :class:`CryptofeedSource`.

    Attributes
    ----------
    exchange : str or object
        The cryptofeed venue id (e.g. ``"bitstamp"``, ``"binance"``), or a
        pre-built exchange class for tests / advanced callers.  Empty by
        default so a source can be constructed before the venue is known.
    level : Level, optional
        Force the capture resolution.  ``None`` (default) discovers it from the
        venue's declared channels.
    queue_size : int
        Depth of the buffer between cryptofeed's callbacks and the capturer's
        iterator.  Bounded, so a consumer that falls behind slows the feed
        rather than growing memory without limit.
    feed_handler : object, optional
        A pre-built ``cryptofeed.FeedHandler`` for tests / advanced callers
        (multiple feeds, custom config).  ``None`` builds one per run.
    """

    exchange: Any = ""
    level: Level | None = None
    queue_size: int = _DEFAULT_QUEUE_SIZE
    feed_handler: Any = None


def _import_feed_handler() -> Any:
    """Return cryptofeed's ``FeedHandler`` class, or raise the install hint."""
    try:
        from cryptofeed import FeedHandler
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            "The cryptofeed source requires the 'cryptofeed' extra: "
            'pip install "ob-analytics[cryptofeed]"'
        ) from exc
    return FeedHandler


def _venue_channels(exchange: Any) -> dict[str, str]:
    """Return the venue's ``websocket_channels`` mapping (empty if unknown)."""
    return getattr(exchange, "websocket_channels", None) or {}


def _resolve_level(exchange: Any, requested: Level | None) -> Level:
    """Pick the capture level from the venue's declared channels.

    A venue offering cryptofeed's ``l3_book`` channel is captured per-order;
    everything else is price-level.  *requested* overrides the discovered
    value, except that L3 cannot be forced onto a venue that does not publish
    it -- that would mean inventing order IDs, the anti-pattern the L2 path
    exists to avoid.
    """
    publishes_l3 = _L3_CHANNEL in _venue_channels(exchange)
    if requested is None:
        return Level.L3 if publishes_l3 else Level.L2
    if requested is Level.L3 and not publishes_l3:
        raise ValueError(
            f"cryptofeed venue {getattr(exchange, 'id', exchange)!r} does not "
            f"publish an L3 (per-order) book; it offers "
            f"{sorted(_venue_channels(exchange))}. Capture it at L2 instead -- "
            "synthesising order IDs from price-level data is not supported."
        )
    return requested


class CryptofeedSource:
    """Live-capture a cryptofeed venue as an L3 order stream or L2 depth stream."""

    name = "cryptofeed"
    # cryptofeed maintains the venue's own book (applying its snapshot +
    # deltas), so bids never rest above asks in the reconstructed book.
    feed_type = FeedType.MATCHED_BOOK

    def __init__(self, settings: SourceSettings | None = None) -> None:
        self.settings: SourceSettings = settings or CryptofeedSettings()
        self._exchange: Any = getattr(self.settings, "exchange", "")
        self._level: Level | None = None
        # Diff baselines for venues that resend a whole book with no delta.
        self._last_l2: dict[str, dict[float, float]] = {"bid": {}, "ask": {}}
        # Every order currently resting: order_id -> (price, direction, volume).
        self._open_orders: dict[Any, tuple[float, str, float]] = {}
        # Venue time (epoch seconds) of the last book that showed each order,
        # and of the last trade applied to it (see ``_fill_events``).  The
        # book and the trades arrive on separate channels, so these say which
        # of the two already accounts for a fill.
        self._shown_at: dict[Any, float] = {}
        self._filled_at: dict[Any, float] = {}
        # Orders a book left out while in view, held for a late trade:
        # order_id -> (the ``deleted`` event, venue time it is due out).  Only
        # once the trade tape has named an order (``_tape_names_orders``).
        self._held_deletes: dict[Any, tuple[EventDict, float]] = {}
        self._tape_names_orders = False
        # Identity for events synthesised at shutdown, when no book is in hand.
        self._symbol = ""
        self.synthetic_deleted = 0

        # Venue sequence continuity (see ``note_sequence``).
        self._last_sequence: int | None = None

        # Diagnostics (surfaced in meta.json via SupportsDiagnostics).
        self.sequence_gaps = 0
        self.sequence_missing = 0
        self.book_updates = 0
        self.order_events = 0
        self.depth_rows = 0
        self.trade_events = 0
        self.tape_fills = 0
        self.tape_fills_already_shown = 0
        self.errors = 0

    @property
    def level(self) -> Level:
        """The capture resolution, discovered from the venue's channels.

        Resolved on first access rather than in ``__init__`` because a venue
        given as a string id has to be looked up in cryptofeed's exchange map
        before its declared channels can be read -- and that import is
        deliberately deferred so this module loads without the extra.  Once
        resolved the answer is cached, so a run cannot change resolution
        midway.

        Reading this attribute never raises: it is what ``isinstance(src,
        LiveSource)`` touches, and a protocol check must not explode.  When the
        venue cannot be resolved yet -- no venue chosen, or the extra not
        installed -- an explicitly requested level stands and anything else
        answers L2 *provisionally*, without caching.  Starting a capture then
        resolves the venue for real and raises the install hint.
        """
        if self._level is not None:
            return self._level
        requested: Level | None = getattr(self.settings, "level", None)
        try:
            exchange = self._exchange_class()
        except (ImportError, ValueError):
            return requested or Level.L2
        self._level = _resolve_level(exchange, requested)
        return self._level

    @level.setter
    def level(self, value: Level) -> None:
        """Pin the resolution, bypassing discovery.

        The :class:`~ob_analytics.protocols.Source` contract declares ``level``
        as a plain attribute, so it has to be settable; prefer
        ``CryptofeedSettings(level=...)``, which is validated against what the
        venue actually publishes.
        """
        self._level = value

    @property
    def trade_attribution(self) -> TradeAttribution:
        """Which orders of a trade this capture's order events can name.

        cryptofeed's L3 channels carry the book, and a book holds resting
        orders only: a taker trades on arrival and never appears.  So an L3
        capture names the maker only, and an L2 capture, with no order
        identity, names neither.  Until the venue resolves, the answer is the
        L3 one unless L2 was asked for: an L2 run ignores attribution anyway.
        """
        if getattr(self.settings, "level", None) is Level.L2:
            return TradeAttribution.NONE
        try:
            self._exchange_class()
        except (ImportError, ValueError):
            return TradeAttribution.MAKER_ONLY
        if self.level is Level.L2:
            return TradeAttribution.NONE
        return TradeAttribution.MAKER_ONLY

    @property
    def _venue(self) -> str:
        """The venue id to report, matching what cryptofeed stamps on payloads.

        cryptofeed's exchange classes carry an upper-case ``id`` ("BITSTAMP")
        and put it on every book and trade, while the settings hold whatever
        the caller typed.  Reporting the resolved class's id keeps
        ``meta.json`` and the per-row ``venue`` in agreement; before the venue
        resolves (no extra installed, none chosen) the configured string is the
        best answer available.
        """
        configured = str(getattr(self._exchange, "id", self._exchange) or "")
        try:
            resolved = self._exchange_class()
        except (ImportError, ValueError):
            return configured
        return str(getattr(resolved, "id", "") or configured)

    # -- identity / time ----------------------------------------------------

    def _configured_identity(self) -> dict[str, str]:
        """Identity for events with no cryptofeed object to read it from.

        Used by the synthetic close-outs at shutdown, which come from local
        state rather than from a book.
        """
        return {"venue": self._venue, "symbol": self._symbol}

    def _payload_identity(self, payload: Any) -> dict[str, str]:
        """Venue + symbol read off a cryptofeed book or trade.

        Taken from the object itself rather than from settings, so a
        multi-symbol feed tags each row with the symbol it actually came from.
        ``venue`` falls back to the resolved venue id when the payload omits
        one, so a row's venue never disagrees with ``meta.json``.
        """
        return {
            "venue": str(getattr(payload, "exchange", "") or "") or self._venue,
            "symbol": str(getattr(payload, "symbol", "") or "") or self._symbol,
        }

    def _book_common(
        self, book: Any, receipt_timestamp: float | None = None
    ) -> dict[str, Any]:
        """The fields every row of one book callback shares.

        Both resolutions stamp the same clocks, venue sequence, and instrument
        identity onto each row they emit, so the two translators build it here
        rather than each spelling it out.  ``timestamp`` is when the capture
        received the message (cryptofeed's receipt time) and
        ``exchange_timestamp`` is the venue's own stamp; without a receipt time
        the venue stamp stands in for both.
        """
        venue_ts = _epoch_s_to_ts(getattr(book, "timestamp", None))
        return {
            "timestamp": (
                venue_ts
                if receipt_timestamp is None
                else _epoch_s_to_ts(receipt_timestamp)
            ),
            "exchange_timestamp": venue_ts,
            "sequence": getattr(book, "sequence_number", None),
            **self._payload_identity(book),
        }

    # -- sequence continuity ------------------------------------------------

    def note_sequence(self, book: Any) -> None:
        """Record one book's venue sequence and score it for continuity.

        cryptofeed owns reconnection: it re-establishes a dropped connection
        internally, so a capture cannot count reconnects directly.  What it can
        observe is the discontinuity a reconnect (or a dropped message) leaves
        in the venue's own sequence, which is what
        :func:`~ob_analytics.analytics.detect_sequence_gaps` scores after the
        fact.  Counting it live too means ``meta.json`` says whether a run is
        clean without re-reading the capture.  Venues that publish no sequence
        are simply never scored.
        """
        sequence = getattr(book, "sequence_number", None)
        if sequence is None:
            return
        try:
            current = int(sequence)
        except (TypeError, ValueError):
            return
        previous = self._last_sequence
        self._last_sequence = current
        if previous is None:
            return
        skipped = current - previous - 1
        if skipped > 0:
            self.sequence_gaps += 1
            self.sequence_missing += skipped

    # -- L2 translation (pure) ----------------------------------------------

    def _l2_rows(
        self, book: Any, receipt_timestamp: float | None = None
    ) -> list[EventDict]:
        """Return the depth rows for one cryptofeed L2 book callback.

        cryptofeed's L2 delta entries are ``(price, new_absolute_size)`` with
        ``0`` meaning the level was removed -- already the shape the L2 sink
        wants, so a delta is emitted directly.  When the venue supplies no
        delta (its opening book, or a venue that resends the whole book), the
        maintained book is diffed against the last one so unchanged levels stay
        silent and vanished levels emit a ``0``.
        """
        common = self._book_common(book, receipt_timestamp)
        delta = getattr(book, "delta", None)
        if _has_entries(delta):
            rows = [
                {"side": side, "price": float(price), "volume": float(size), **common}
                for side in ("bid", "ask")
                for price, size in delta.get(side) or ()
            ]
            # Keep the diff baseline current so a later delta-less book is
            # compared against what the deltas already reported.  Only the
            # delta path is cheap enough to take on every update; the whole
            # book is converted here and nowhere else on this branch.
            self._last_l2 = _book_levels(book)
            return rows
        return self._l2_rows_from_full_book(_book_levels(book), common)

    def _l2_rows_from_full_book(
        self, levels: dict[str, dict[float, float]], common: dict[str, Any]
    ) -> list[EventDict]:
        """Diff a whole L2 book against the previous one into depth rows."""
        rows: list[EventDict] = []
        for side in ("bid", "ask"):
            current = levels.get(side, {})
            previous = self._last_l2.get(side, {})
            for price, size in current.items():
                if previous.get(price) != size:
                    rows.append(
                        {"side": side, "price": price, "volume": size, **common}
                    )
            for price in previous:
                if price not in current:
                    rows.append({"side": side, "price": price, "volume": 0.0, **common})
        self._last_l2 = levels
        return rows

    # -- L3 translation (pure) ----------------------------------------------

    def _l3_events(
        self, book: Any, receipt_timestamp: float | None = None
    ) -> list[EventDict]:
        """Return the per-order events for one cryptofeed L3 book callback.

        Two paths, because cryptofeed's L3 venues do not agree on shape:

        * **A populated delta** is the venue's own statement of what changed,
          so it is mapped entry by entry.  Its ``(order_id, price, quantity)``
          triples carry real IDs, and a venue that models a price move as a
          removal followed by an add (bitfinex) keeps that meaning -- a move
          loses queue priority, so it must not read as one order silently
          changing price.
        * **No delta, or an empty one**, means the callback carries only the
          maintained book: bitstamp resends it whole every message, and
          blockchain, independent_reserve and bitfinex all open that way.  The
          book is diffed against the tracked orders to recover the same
          created / changed / deleted vocabulary.  Where the book is only the
          top of the venue's book (bitstamp, see ``_L3_WINDOW``), an order
          that drops out past the edge of the window is out of view, not gone.

        Either way the IDs are the venue's own.  Nothing here invents one.
        """
        common = self._book_common(book, receipt_timestamp)
        delta = getattr(book, "delta", None)
        if _has_entries(delta):
            return self._l3_events_from_delta(delta, common)
        shown_at = getattr(book, "timestamp", None)
        return self._l3_events_from_full_book(
            _book_levels(book),
            common,
            shown_at=None if shown_at is None else float(shown_at),
        )

    def _l3_events_from_delta(
        self, delta: dict[str, Any], common: dict[str, Any]
    ) -> list[EventDict]:
        """Map ``(order_id, price, quantity)`` triples to order events."""
        # Applied to a copy and committed only once the whole delta has
        # parsed.  A malformed entry half-way through would otherwise leave the
        # tracked orders reflecting changes whose events were discarded, and
        # every later diff would be wrong in a way nothing reports.
        staged = dict(self._open_orders)
        events: list[EventDict] = []
        for direction in ("bid", "ask"):
            for entry in delta.get(direction) or ():
                order_id, price, quantity = entry
                price = float(price)
                quantity = float(quantity)
                if quantity <= 0:
                    # A removal: report the size the order last rested at,
                    # since the delta only carries the zero.
                    known = staged.pop(order_id, None)
                    volume = known[2] if known is not None else 0.0
                    action = "deleted"
                else:
                    action = "changed" if order_id in staged else "created"
                    volume = quantity
                    staged[order_id] = (price, direction, quantity)
                events.append(
                    {
                        "id": order_id,
                        "price": price,
                        "volume": volume,
                        "action": action,
                        "direction": direction,
                        **common,
                    }
                )
        self._open_orders = staged
        return events

    def _window_edges(
        self, levels: dict[str, dict[float, Any]]
    ) -> dict[str, float | None]:
        """Return the worst price each side shows, if that side is cut off.

        A side is cut off when the venue declares a window (``_L3_WINDOW``) and
        the side shows that many orders or more: the venue's book may go on past
        the last order shown.  Past that edge, and at the edge price itself,
        where the level may be cut in two, an order the message leaves out is
        out of view rather than gone.  ``None`` means the side is the whole
        book, so every order it leaves out is gone.
        """
        window = _L3_WINDOW.get(self._venue.upper())
        edges: dict[str, float | None] = {"bid": None, "ask": None}
        if window is None:
            return edges
        for direction in ("bid", "ask"):
            side = levels.get(direction, {})
            if sum(len(orders) for orders in side.values()) >= window:
                edges[direction] = min(side) if direction == "bid" else max(side)
        return edges

    def _l3_events_from_full_book(
        self,
        levels: dict[str, dict[float, Any]],
        common: dict[str, Any],
        *,
        shown_at: float | None = None,
    ) -> list[EventDict]:
        """Diff a whole per-order book against the tracked orders.

        An order the book leaves out is ``deleted`` only if the book shows its
        price.  One past the edge of a cut-off side (see ``_window_edges``) is
        kept as it last was, with no event: when it comes back into view it is
        the same order, and it is only reported deleted once the window covers
        its price and it is not there.

        *shown_at* is the venue time of the book.  An order that a trade has
        filled since then (see ``_fill_events``) is already more up to date
        than this book, so the book says nothing about it and it is left as
        tracked.
        """
        current: dict[Any, tuple[float, str, float]] = {}
        for direction in ("bid", "ask"):
            for price, orders in levels.get(direction, {}).items():
                for order_id, size in orders.items():
                    current[order_id] = (price, direction, float(size))
        edges = self._window_edges(levels)

        def out_of_view(price: float, direction: str) -> bool:
            edge = edges[direction]
            if edge is None:
                return False
            return price <= edge if direction == "bid" else price >= edge

        def filled_since(order_id: Any) -> bool:
            filled_at = self._filled_at.get(order_id)
            return (
                shown_at is not None and filled_at is not None and filled_at > shown_at
            )

        events: list[EventDict] = []
        kept: dict[Any, tuple[float, str, float]] = {}
        for order_id, (price, direction, volume) in list(current.items()):
            previous = self._open_orders.get(order_id)
            held = self._held_deletes.pop(order_id, None)
            if previous is None and held is not None:
                # Left out of one book and back in the next: not gone after
                # all, so the held ``deleted`` is dropped, not reported.
                event = held[0]
                previous = (event["price"], event["direction"], event["volume"])
            if previous is not None and filled_since(order_id):
                kept[order_id] = previous
                del current[order_id]
                continue
            if shown_at is not None:
                self._shown_at[order_id] = shown_at
            if previous == (price, direction, volume):
                continue
            events.append(
                {
                    "id": order_id,
                    "price": price,
                    "volume": volume,
                    "action": "created" if previous is None else "changed",
                    "direction": direction,
                    **common,
                }
            )
        for order_id, (price, direction, volume) in self._open_orders.items():
            if order_id in current or order_id in kept:
                continue
            if out_of_view(price, direction) or filled_since(order_id):
                kept[order_id] = (price, direction, volume)
                continue
            self._shown_at.pop(order_id, None)
            self._filled_at.pop(order_id, None)
            deleted = {
                "id": order_id,
                "price": price,
                "volume": volume,
                "action": "deleted",
                "direction": direction,
                **common,
            }
            if self._tape_names_orders and shown_at is not None:
                self._held_deletes[order_id] = (deleted, shown_at + _FILL_GRACE_S)
            else:
                events.append(deleted)
        self._open_orders = {**kept, **current}
        if shown_at is not None:
            events.extend(self._release_held_deletes(shown_at))
        return events

    def _release_held_deletes(self, now: float | None = None) -> list[EventDict]:
        """Return the held ``deleted`` events due by venue time *now* (all if ``None``)."""
        due = [
            order_id
            for order_id, (_, due_at) in self._held_deletes.items()
            if now is None or due_at <= now
        ]
        return [self._held_deletes.pop(order_id)[0] for order_id in due]

    # -- fills from the trade tape (pure) -----------------------------------

    def _tracked_id(self, order_id: Any) -> Any:
        """The tracked order a trade's order id names, or ``None``.

        A venue can spell one id two ways: Bitstamp's book lists order ids as
        strings and its trades as integers.
        """
        if order_id in ("", None):
            return None
        if order_id in self._open_orders:
            return order_id
        spelled = str(order_id)
        return spelled if spelled in self._open_orders else None

    def _held_id(self, order_id: Any) -> Any:
        """The held-for-deletion order a trade's order id names, or ``None``."""
        if order_id in self._held_deletes:
            return order_id
        spelled = str(order_id)
        return spelled if spelled in self._held_deletes else None

    def _fill_held(
        self, order_id: Any, amount: float, traded_at: float | None
    ) -> EventDict:
        """Report a fill of an order a book has already left out.

        The book that left the order out came first, so the fill happened
        before it: the ``changed`` event is stamped with that book's receive
        time and the trade's venue time, and the held ``deleted`` now reports
        the size left after the fill, 0 for a full fill.
        """
        deleted, due_at = self._held_deletes[order_id]
        remaining = max(round(deleted["volume"] - amount, 12), 0.0)
        self._held_deletes[order_id] = ({**deleted, "volume": remaining}, due_at)
        self.tape_fills += 1
        return {
            **deleted,
            "exchange_timestamp": (
                deleted["exchange_timestamp"]
                if traded_at is None
                else _epoch_s_to_ts(traded_at)
            ),
            "volume": remaining,
            "action": "changed",
        }

    def _fill_events(
        self, trade: Any, receipt_timestamp: float | None = None
    ) -> list[EventDict]:
        """Return a ``changed`` event for each tracked order a trade filled.

        A book that shows only resting orders learns of a fill from the next
        book: a size that dropped, or an order that is gone.  A full fill
        then reads like a cancel, and several fills between two books merge
        into one drop that matches no single trade.  Where the venue's trade
        names its orders (Bitstamp does; see ``_trade_order_ids``), each fill
        is reported as the trade arrives instead: the resting order's size
        drops by the trade's amount.

        The order is never deleted here.  A fully filled order drops to size
        0 and stays tracked until a book leaves it out; that book reports it
        ``deleted`` at size 0, which is how the native Bitstamp feed reports a
        fill.  When the book comes first and has already left the order out,
        its ``deleted`` is held for ``_FILL_GRACE_S`` of venue time (once the
        tape is known to name orders), and the late trade reports its fill
        against it (see ``_fill_held``).

        Books and trades arrive on separate channels, in either order, so a
        trade is applied only when it is later, on the venue's clock, than the
        last book that showed the order.  A book as late as the trade already
        includes it, and applying it again would count the fill twice; the
        book's own size change stands for it instead.
        """
        traded_at = getattr(trade, "timestamp", None)
        traded_at = None if traded_at is None else float(traded_at)
        amount = float(trade.amount)
        events: list[EventDict] = []
        for named in _trade_order_ids(trade):
            if named in ("", None):
                continue
            self._tape_names_orders = True
            held_id = self._held_id(named)
            if held_id is not None:
                events.append(self._fill_held(held_id, amount, traded_at))
                continue
            order_id = self._tracked_id(named)
            if order_id is None:
                continue
            shown_at = self._shown_at.get(order_id)
            if traded_at is not None and shown_at is not None and traded_at <= shown_at:
                self.tape_fills_already_shown += 1
                continue
            price, direction, volume = self._open_orders[order_id]
            # Sizes are exact decimals on the venue; rounding keeps the
            # difference on the same float a book would report it as.
            remaining = max(round(volume - amount, 12), 0.0)
            self._open_orders[order_id] = (price, direction, remaining)
            if traded_at is not None:
                self._filled_at[order_id] = traded_at
            self.tape_fills += 1
            events.append(
                {
                    "id": order_id,
                    "timestamp": _epoch_s_to_ts(receipt_timestamp),
                    "exchange_timestamp": _epoch_s_to_ts(traded_at),
                    "sequence": None,
                    "price": price,
                    "volume": remaining,
                    "action": "changed",
                    "direction": direction,
                    **self._payload_identity(trade),
                }
            )
        return events

    # -- trade translation (pure) -------------------------------------------

    def _map_trade(
        self, trade: Any, receipt_timestamp: float | None = None
    ) -> EventDict:
        """Map a cryptofeed ``Trade`` to the universal trade-event shape.

        cryptofeed reports price and amount as ``Decimal``; they are floated
        for the shared schema.  The buy and sell order IDs come from the raw
        frame where the venue sends them (see ``_trade_order_ids``) and stay
        empty where it does not.  ``side`` is the taker side.  ``timestamp`` is
        the receipt time (the time now, without one).
        """
        buy_order_id, sell_order_id = _trade_order_ids(trade)
        return {
            "trade_id": getattr(trade, "id", "") or "",
            "timestamp": _epoch_s_to_ts(receipt_timestamp),
            "exchange_timestamp": _epoch_s_to_ts(getattr(trade, "timestamp", None)),
            "price": float(trade.price),
            "amount": float(trade.amount),
            "buy_order_id": buy_order_id,
            "sell_order_id": sell_order_id,
            "side": getattr(trade, "side", "") or "",
            **self._payload_identity(trade),
        }

    # -- shutdown -----------------------------------------------------------

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        """Close out everything still resting, so every ``id`` has a lifecycle.

        L2 price levels have no lifecycle to close, so an L2 capture yields
        nothing here.
        """
        if self.level is not Level.L2:
            for held in self._release_held_deletes():
                yield held
            ts = pd.Timestamp.now(tz="UTC").as_unit("ns")
            for order_id, (price, direction, volume) in list(self._open_orders.items()):
                self.synthetic_deleted += 1
                yield {
                    "id": order_id,
                    "timestamp": ts,
                    "exchange_timestamp": ts,
                    "price": price,
                    "volume": volume,
                    "action": "deleted",
                    "direction": direction,
                    **self._configured_identity(),
                }
            self._open_orders.clear()

    # -- diagnostics --------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """Per-run counters for meta.json (SupportsDiagnostics)."""
        return {
            "exchange": self._venue,
            "level": str(self.level),
            "book_updates": self.book_updates,
            "order_events": self.order_events,
            "depth_rows": self.depth_rows,
            "trade_events": self.trade_events,
            "tape_fills": self.tape_fills,
            "tape_fills_already_shown": self.tape_fills_already_shown,
            "sequence_gaps": self.sequence_gaps,
            "sequence_missing": self.sequence_missing,
            "synthetic_deleted": self.synthetic_deleted,
            "errors": self.errors,
        }

    # -- the cryptofeed bridge ----------------------------------------------

    def preflight(self) -> None:
        """Raise now if the capture cannot start (SupportsPreflight).

        Resolves the venue, which raises the install hint when the extra is
        missing and :class:`ValueError` for no venue or an unknown one. A
        supplied ``feed_handler`` and exchange class need no import.
        """
        self._exchange_class()
        if getattr(self.settings, "feed_handler", None) is None:
            _import_feed_handler()

    def _build_feed_handler(self) -> Any:
        """Return the ``FeedHandler`` to run (lazy import of ``cryptofeed``)."""
        supplied = getattr(self.settings, "feed_handler", None)
        if supplied is not None:
            return supplied
        return _import_feed_handler()()

    def _exchange_class(self) -> Any:
        """Resolve the settings' venue to a cryptofeed exchange class.

        A string is looked up in cryptofeed's ``EXCHANGE_MAP``; anything else
        is taken to be an exchange class already (tests / advanced callers).
        """
        exchange = self._exchange
        if not isinstance(exchange, str):
            return exchange
        if not exchange:
            raise ValueError(
                "cryptofeed source needs CryptofeedSettings(exchange='<venue id>') "
                "(e.g. 'bitstamp')."
            )
        try:
            from cryptofeed.exchanges import EXCHANGE_MAP
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "The cryptofeed source requires the 'cryptofeed' extra: "
                'pip install "ob-analytics[cryptofeed]"'
            ) from exc
        try:
            return EXCHANGE_MAP[exchange.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown cryptofeed exchange {exchange!r}; expected one of "
                f"{len(EXCHANGE_MAP)} venues."
            ) from None

    async def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        """Yield nothing: cryptofeed delivers the opening book as its first callback.

        Unlike a REST-seeded source, cryptofeed maintains the book itself and
        hands over the whole opening state in the first ``l2_book`` /
        ``l3_book`` callback.  That callback is translated by the same code as
        every later one -- an L3 opening book has no delta, so it diffs to one
        ``created`` per resting order, and an L2 opening book emits every level
        at its absolute size.  Synthesising a second opening book here would
        duplicate it.
        """
        self._resolve_symbol(config)
        for _ in ():
            yield {}

    def _resolve_symbol(self, config: CaptureConfig) -> None:
        """Record the run's symbol for events not derived from a book."""
        self._symbol = config.pair

    async def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        """Bridge cryptofeed's callbacks into the runner's async iterator.

        cryptofeed pushes data by invoking async callbacks from its own tasks,
        while the runner pulls from an iterator.  A **bounded** queue joins the
        two: callbacks await a slot, so a consumer that falls behind applies
        backpressure to the feed instead of dropping events (an L3 book cannot
        survive a gap) or growing without limit.

        The feed handler runs on the caller's event loop
        (``run(start_loop=False)``) and never installs signal handlers -- the
        capture runner owns Ctrl-C.  Shutdown goes through ``stop_async``, not
        ``close``, which would stop the loop out from under the runner.
        """
        self._resolve_symbol(config)
        queue: asyncio.Queue[tuple[str, EventDict, Any]] = asyncio.Queue(
            maxsize=max(
                1, int(getattr(self.settings, "queue_size", _DEFAULT_QUEUE_SIZE))
            )
        )
        handler = self._build_feed_handler()
        exchange_cls = self._exchange_class()
        book_channel = _L3_CHANNEL if self.level is Level.L3 else _L2_CHANNEL

        async def on_book(book: Any, receipt_timestamp: float) -> None:
            self.book_updates += 1
            self.note_sequence(book)
            try:
                if self.level is Level.L3:
                    events = self._l3_events(book, receipt_timestamp)
                    self.order_events += len(events)
                    kind = "order"
                else:
                    events = self._l2_rows(book, receipt_timestamp)
                    self.depth_rows += len(events)
                    kind = "depth"
            except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the run
                self.errors += 1
                logger.warning("[cryptofeed] dropped a book update: {!r}", exc)
                return
            # The raw frame is archived once per update, not once per event.
            raw = getattr(book, "raw", None)
            for event in events:
                await queue.put((kind, event, raw))
                raw = None

        async def on_trade(trade: Any, receipt_timestamp: float) -> None:
            try:
                event = self._map_trade(trade, receipt_timestamp)
                fills = (
                    self._fill_events(trade, receipt_timestamp)
                    if self.level is Level.L3
                    else []
                )
            except Exception as exc:  # noqa: BLE001
                self.errors += 1
                logger.warning("[cryptofeed] dropped a trade: {!r}", exc)
                return
            for fill in fills:
                await queue.put(("order", fill, None))
            self.order_events += len(fills)
            self.trade_events += 1
            await queue.put(("trade", event, getattr(trade, "raw", None)))

        handler.add_feed(
            exchange_cls(
                symbols=[config.pair],
                channels=[book_channel, _TRADES_CHANNEL],
                callbacks={book_channel: on_book, _TRADES_CHANNEL: on_trade},
            )
        )
        logger.info(
            "[cryptofeed] {} {} at {} (channels: {}, {})",
            self._venue,
            config.pair,
            self.level,
            book_channel,
            _TRADES_CHANNEL,
        )

        deadline = time.monotonic() + config.minutes * 60.0
        handler.run(start_loop=False, install_signal_handlers=False)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                # cryptofeed clears ``running`` once it has shut its feeds
                # down.  Draining what is left and returning beats spinning
                # out the rest of the window against a dead handler.
                if not getattr(handler, "running", True) and queue.empty():
                    return
                try:
                    yield await asyncio.wait_for(
                        queue.get(), timeout=min(remaining, 0.5)
                    )
                except TimeoutError:
                    continue
        finally:
            await self._stop_handler(handler)

    async def _stop_handler(self, handler: Any) -> None:
        """Shut the feed handler down without touching the running loop."""
        stop_async = getattr(handler, "stop_async", None)
        if stop_async is None:
            return
        try:
            await stop_async(loop=asyncio.get_running_loop())
        except TypeError:
            await stop_async()
        except Exception as exc:  # noqa: BLE001 - shutdown must not mask the run
            logger.debug("[cryptofeed] feed handler stop error: {!r}", exc)


# ── Register this source ──────────────────────────────────────────────
# Registered unconditionally: importing this module never imports cryptofeed
# (it is imported lazily when a capture starts), so the source is discoverable
# with or without the extra, and a capture without it raises an install hint.
from ob_analytics.sources import register_source

register_source("cryptofeed", CryptofeedSource)
