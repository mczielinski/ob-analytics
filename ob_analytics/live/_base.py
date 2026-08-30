"""Protocol, data shapes, and sink contract for live order-book capture.

A :class:`LiveSource` implementation translates a venue's native message
stream into the universal ob-analytics events schema. Each source is
responsible only for *parsing*; persistence, raw-frame archival, and
shutdown sequencing are handled generically by the runner so every venue
benefits.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ob_analytics.protocols import Source

# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


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


# Single canonical event dict shape, mirroring BitstampLoader's CSV columns.
# Capturers yield these one at a time; the runner buffers/writes.
EventDict = dict[str, Any]
# Required keys for an order event (L3):  id, timestamp, exchange_timestamp,
#                                    price, volume, action, direction
# Required keys for a depth event (L2):   timestamp, exchange_timestamp,
#                                    side, price, volume  (volume = the new
#                                    absolute size at that price; 0 removes it)
# Required keys for a trade event:   trade_id, timestamp, exchange_timestamp,
#                                    price, amount, buy_order_id,
#                                    sell_order_id, side


# ---------------------------------------------------------------------------
# Sink: how the runner persists what a capturer yields
# ---------------------------------------------------------------------------


@runtime_checkable
class CaptureSink(Protocol):
    """Write target for a capture run.

    The default implementation (in ``_runner.py``) writes, keyed on the
    source's :attr:`~ob_analytics.protocols.Source.level`:
    - orders.csv  -- L3 only: append-only, BitstampLoader-compatible schema
    - depth.csv   -- L2 only: append-only, L2DepthLoader-compatible schema
    - trades.csv  -- append-only (both resolutions)
    - raw.jsonl   -- every raw frame, one JSON object per line (if keep_raw)
    - meta.json   -- finalised at shutdown
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
    rate-limiting reconnects, and signal handling all live in
    ``ob_analytics.live._runner``.

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
    """

    def diagnostics(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of per-run counters."""
        ...
