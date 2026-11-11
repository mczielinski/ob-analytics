"""Protocol, data shapes, and sink contract for live order-book capture.

A :class:`LiveSource` implementation translates a venue's native message
stream into the universal ob-analytics events schema. Each source is
responsible only for *parsing*; persistence, raw-frame archival, and
shutdown sequencing are handled generically by the runner so every venue
benefits.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ob_analytics.exceptions import ObAnalyticsError
from ob_analytics.protocols import Clocks, Source

# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


class UnmappedMessageError(ObAnalyticsError, ValueError):
    """An L3 message a live source could not turn into order events.

    An L3 source raises it from :meth:`LiveSource.snapshot` or
    :meth:`LiveSource.stream` instead of leaving the message out: without
    it the capture's book would no longer match the venue's.  The runner
    records it as :attr:`CaptureResult.capture_error`, and
    :func:`~ob_analytics.live.run_capture` starts the next segment from a
    new opening book.  A message that does not change the book (a trade on
    a feed whose order events carry the fills) can be left out instead, and
    counted under ``dropped`` (see :class:`SupportsDiagnostics`).
    """


@dataclass(frozen=True)
class CaptureConfig:
    """User-facing capture-run parameters.

    Run-level knobs shared by every venue — *what* symbol, *where* to write,
    for *how long*.  Per-venue settings (e.g. the CCXT exchange id) are **not**
    here; they are typed :class:`~ob_analytics.config.SourceSettings` carried by
    the source itself (``CcxtSource(settings=CcxtSettings(...))``), which is
    what replaced the former untyped ``extras`` dict.
    """

    pair: str  # venue-specific symbol, e.g. "btcusd"
    out_dir: Path
    minutes: float = 10.0
    keep_raw: bool = True  # write raw.jsonl alongside parsed CSVs
    #: Start a new segment after the current one has been current for this
    #: many minutes (``None``: never on time).
    #: Used by :func:`~ob_analytics.live.run_capture` only.
    roll_minutes: float | None = None
    #: Start a new segment once the current one has written this many
    #: megabytes since its first live event, so the opening snapshot does not
    #: count (``None``: never on size).  Both limits must be above 0.  Used by
    #: :func:`~ob_analytics.live.run_capture` only.
    roll_mb: float | None = None


@dataclass(frozen=True)
class CaptureResult:
    """Summary returned to the caller once the run finishes."""

    out_dir: Path
    n_order_events: int
    n_trade_events: int
    n_raw_frames: int
    started: pd.Timestamp
    ended: pd.Timestamp
    extras: dict[str, Any] = field(default_factory=dict)
    n_depth_events: int = 0
    #: L3 only: orders in the opening book that no later order event or trade
    #: mentioned. Most are far from the touch and simply never traded; one at
    #: the touch is an order the venue's snapshot listed after it had gone.
    #: ``None`` on an L2 run, where a price level has no id to confirm.
    n_snapshot_unconfirmed: int | None = None
    #: ``repr`` of the first exception raised once output existed, or ``None``
    #: for a run that ended normally or was stopped by one signal (a second
    #: one, which stops the run at once, is recorded here too). The rows
    #: written before the error stay on disk, so a failed run is still readable.
    capture_error: str | None = None
    #: Where :attr:`capture_error` was raised: ``"snapshot"``, ``"stream"``,
    #: ``"shutdown"``, or ``"close"`` for a file that failed to close; ``None``
    #: when there was no error.
    capture_error_phase: str | None = None
    #: When the first live event arrived, and when the stream stopped: the
    #: stretch of time the run covers.  ``stream_started`` is ``None`` when the
    #: stream delivered nothing.
    stream_started: pd.Timestamp | None = None
    stream_ended: pd.Timestamp | None = None


# Capturers yield these one at a time; the runner buffers and writes them.
EventDict = dict[str, Any]
"""One event a live source yields: a dict of column name to value.

The required keys are:

* an order event (L3): ``id``, ``timestamp``, ``exchange_timestamp``,
  ``price``, ``volume``, ``action``, ``direction``.
* a depth event (L2): ``timestamp``, ``exchange_timestamp``, ``side``,
  ``price``, ``volume``.  ``volume`` is the new size at that price; ``0``
  removes the level.
* a trade event: ``trade_id``, ``timestamp``, ``exchange_timestamp``,
  ``price``, ``amount``, ``buy_order_id``, ``sell_order_id``, ``side``.
"""


@dataclass
class VenueClockCount:
    """How many of a capture's books carried the venue's own time.

    A venue that sends no time with its book gives a capture one clock, the
    receive time, and ``exchange_timestamp`` copies it.  Some venues send a
    time on every message but none on the opening book (cryptofeed's
    Independent Reserve and Coinbase); those books count as without, and the
    capture still has two clocks.  :attr:`clocks` says which the capture had.
    """

    with_venue_time: int = 0
    without_venue_time: int = 0

    def note(self, venue_time: Any) -> None:
        """Count one book, given the venue time it carried (``None`` for none)."""
        if venue_time is None:
            self.without_venue_time += 1
        else:
            self.with_venue_time += 1

    @property
    def clocks(self) -> Clocks:
        """:attr:`Clocks.RECEIVE_ONLY` when books came and none had a venue time.

        Before any book, and when any book had one, the capture has both
        clocks.
        """
        if self.without_venue_time and not self.with_venue_time:
            return Clocks.RECEIVE_ONLY
        return Clocks.BOTH


# ---------------------------------------------------------------------------
# Book state shared by the live sources
# ---------------------------------------------------------------------------


class OpenOrders(dict[Any, tuple[float, str, float]]):
    """The orders resting on an L3 book: order id -> ``(price, direction, volume)``.

    An L3 live source records each order it reports as created or changed,
    and removes it when it is deleted.  At shutdown :meth:`close` gives every
    order still resting a ``deleted`` event, so each id in ``orders.csv`` has
    a complete ``created -> ... -> deleted`` lifecycle.
    """

    def close(self, at: pd.Timestamp, **fields: Any) -> list[EventDict]:
        """Return a ``deleted`` event at *at* for each open order, and forget them.

        Each event carries the order's last price, direction and volume, and
        *fields* (the instrument's ``venue`` and ``symbol``, say).
        """
        events: list[EventDict] = [
            {
                "id": order_id,
                "timestamp": at,
                "exchange_timestamp": at,
                "price": price,
                "volume": volume,
                "action": "deleted",
                "direction": direction,
                **fields,
            }
            for order_id, (price, direction, volume) in self.items()
        ]
        self.clear()
        return events


def diff_levels(
    previous: dict[float, float], levels: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Update one side of an L2 book in place, and return what changed.

    *previous* maps each price to its size, as last reported.  *levels* is
    the side's whole book now, as ``(price, size)`` pairs with no price
    repeated.  Returns ``(price, size)`` for each level whose size changed or
    that is new, then ``(price, 0.0)`` for each level that is gone, and makes
    *previous* match *levels*.

    *previous* is changed in place rather than rebuilt: a deep book side can
    hold tens of thousands of levels, and this runs on every update of a
    capture that can last for days.
    """
    changes: list[tuple[float, float]] = []
    seen: set[float] = set()
    for price, size in levels:
        seen.add(price)
        if previous.get(price) != size:
            changes.append((price, size))
            previous[price] = size
    gone = [price for price in previous if price not in seen]
    for price in gone:
        changes.append((price, 0.0))
        del previous[price]
    return changes


# ---------------------------------------------------------------------------
# Sink: how the runner persists what a capturer yields
# ---------------------------------------------------------------------------


@runtime_checkable
class CaptureSink(Protocol):
    """Write target for a capture run.

    The default implementation,
    :class:`~ob_analytics.capture_files.FileCaptureSink`, writes the capture
    folder :mod:`ob_analytics.capture_files` describes: ``orders.csv`` for an
    L3 source or ``depth.csv`` for an L2 one (see
    :attr:`~ob_analytics.protocols.Source.level`), ``trades.csv``,
    ``raw.jsonl`` and ``meta.json``.
    """

    def write_order(self, event: EventDict) -> None: ...
    def write_depth(self, event: EventDict) -> None: ...
    def write_trade(self, event: EventDict) -> None: ...
    def write_raw(self, frame: Any) -> None: ...
    def finalize(self, result: CaptureResult) -> None: ...


# ---------------------------------------------------------------------------
# The capturer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LiveSource(Source, Protocol):
    """The live-capture capability of a :class:`~ob_analytics.protocols.Source`.

    A live source is an async iterator of order/trade (or depth) events, plus
    two bookend methods for snapshot + shutdown synthesis.  It inherits the
    source coordinates from :class:`~ob_analytics.protocols.Source` — ``name``,
    ``level``, ``feed_type``, and typed ``settings`` — so a venue that both
    replays files and captures live agrees with itself on those.

    :attr:`~ob_analytics.protocols.Source.level` routes the book events the
    runner writes: :attr:`~ob_analytics.protocols.Level.L3` → per-order events
    to ``orders.csv``; :attr:`~ob_analytics.protocols.Level.L2` → price-level
    depth updates to ``depth.csv``.

    Implementors only worry about parsing. Persistence, raw-frame archival,
    and signal handling live in ``ob_analytics.live._runner``.

    A source does not reconnect on its own.  When it loses its connection it
    raises from :meth:`stream`: a book carried across a disconnect misses
    every change made while it was down.  :func:`~ob_analytics.live.run_capture`
    then closes the segment, waits, and starts a new one from a fresh snapshot,
    recording the time between them as a gap in ``manifest.json``.

    A live source MAY additionally implement :class:`SupportsDiagnostics` to
    surface per-run counters in ``meta.json``; that hook is a separate,
    optional protocol so it is never required to conform to this one.
    """

    def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        """Yield the opening book: one event per resting order (L3) or level (L2).

        Called once at startup before :meth:`stream`. **L3:** each event MUST
        have ``action="created"``; the runner writes them to ``orders.csv`` so
        later ``changed`` / ``deleted`` events have matching creates. **L2:**
        each event is an absolute-size depth row (``side`` / ``price`` /
        ``volume``) written to ``depth.csv``.
        """
        ...

    def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        """Yield ``(kind, event, raw_frame)`` for every live event.

        ``kind`` is ``"order"`` (L3), ``"depth"`` (L2), or ``"trade"``.
        ``raw_frame`` is the original JSON-decoded WebSocket payload (or
        ``None``); the runner writes it to raw.jsonl iff ``config.keep_raw``.

        Implementations should self-terminate after ``config.minutes`` of
        wall-clock time. The runner *also* enforces this externally, so
        cancellation must be cooperative.
        """
        ...

    def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        """Yield synthetic close-out events for everything still on the book.

        Called once at shutdown. **L3:** each event MUST have
        ``action="deleted"``, giving every ``id`` in ``orders.csv`` a complete
        ``created -> ... -> deleted`` lifecycle. **L2:** price levels have no
        lifecycle to close, so an L2 capturer typically yields nothing here.
        """
        ...


@runtime_checkable
class SupportsDiagnostics(Protocol):
    """Optional source capability: per-run counters for ``meta.json``.

    Kept deliberately separate from :class:`LiveSource` so that writing a live
    source never forces a ``diagnostics()`` method onto it. A source that
    wants to expose counters (dropped frames, reconnects, synthetic-event
    tallies, ...) just defines this method; the runner detects it structurally
    (``isinstance(source, SupportsDiagnostics)``) and merges the returned
    mapping into :attr:`CaptureResult.extras` (and thus ``meta.json``).

    Count the messages the source received but could not use, and left out,
    under ``dropped``: the capture manifest totals that key, and
    ``ob-analytics audit`` fails its ``dropped_messages`` check when it is
    above zero.  An error that ends the stream is not counted there; the
    runner records it as :attr:`CaptureResult.capture_error`.
    """

    def diagnostics(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of per-run counters."""
        ...


@runtime_checkable
class SupportsPreflight(Protocol):
    """Optional source capability: check the source can run before it starts.

    A source whose venue library is an optional extra imports it lazily, so a
    missing extra would otherwise surface only once the stream starts, after
    the output files exist. The runner calls :meth:`preflight` before it
    creates any output, so such a run stops with the install hint instead.
    """

    def preflight(self) -> None:
        """Raise :class:`ImportError` (with the install hint) if the source cannot run.

        May also raise :class:`~ob_analytics.exceptions.ConfigError` for
        settings that cannot work (no venue chosen), or
        :class:`~ob_analytics.exceptions.UnknownNameError` for an unknown
        venue id.
        """
        ...
