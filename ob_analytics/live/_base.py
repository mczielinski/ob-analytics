"""Protocol, data shapes, and sink contract for live order-book capture.

A :class:`LiveCapturer` implementation translates a venue's native message
stream into the universal ob-analytics events schema. Each capturer is
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


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureConfig:
    """User-facing capture parameters."""

    pair: str  # venue-specific symbol, e.g. "btcusd"
    out_dir: Path
    minutes: float = 10.0
    keep_raw: bool = True  # write raw.jsonl alongside parsed CSVs
    extras: dict[str, Any] = field(default_factory=dict)


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


# Single canonical event dict shape, mirroring BitstampLoader's CSV columns.
# Capturers yield these one at a time; the runner buffers/writes.
EventDict = dict[str, Any]
# Required keys for an order event:  id, timestamp, exchange_timestamp,
#                                    price, volume, action, direction
# Required keys for a trade event:   trade_id, timestamp, exchange_timestamp,
#                                    price, amount, buy_order_id,
#                                    sell_order_id, side


# ---------------------------------------------------------------------------
# Sink: how the runner persists what a capturer yields
# ---------------------------------------------------------------------------


@runtime_checkable
class CaptureSink(Protocol):
    """Write target for a capture run.

    The default implementation (in ``_runner.py``) writes:
    - orders.csv  -- append-only, BitstampLoader-compatible schema
    - trades.csv  -- append-only
    - raw.jsonl   -- every raw frame, one JSON object per line (if keep_raw)
    - meta.json   -- finalised at shutdown
    """

    def write_order(self, event: EventDict) -> None: ...
    def write_trade(self, event: EventDict) -> None: ...
    def write_raw(self, frame: Any) -> None: ...
    def finalize(self, result: CaptureResult) -> None: ...


# ---------------------------------------------------------------------------
# The capturer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LiveCapturer(Protocol):
    """Translate a venue's live feed into universal-schema events.

    Implementations are async iterators of order/trade events, plus two
    bookend methods for snapshot + shutdown synthesis.

    Implementors only worry about parsing. Persistence, raw-frame archival,
    rate-limiting reconnects, and signal handling all live in
    ``ob_analytics.live._runner``.
    """

    name: str
    """Stable lowercase venue identifier, e.g. ``"bitstamp"``."""

    def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        """Yield synthetic ``created`` events reconstructing the initial book.

        Called once at startup before :meth:`stream`. Each yielded event MUST
        have ``action="created"``. The runner writes these to orders.csv so
        subsequent ``changed`` / ``deleted`` events have matching creates.
        """
        ...

    def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        """Yield ``(kind, event, raw_frame)`` for every live event.

        ``kind`` is ``"order"`` or ``"trade"``. ``raw_frame`` is the original
        JSON-decoded WebSocket payload (or ``None``); the runner writes it
        to raw.jsonl iff ``config.keep_raw``.

        Implementations should self-terminate after ``config.minutes`` of
        wall-clock time. The runner *also* enforces this externally, so
        cancellation must be cooperative.
        """
        ...

    def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        """Yield synthetic ``deleted`` events for everything still on the book.

        Called once at shutdown. Each yielded event MUST have
        ``action="deleted"``. Gives every ``id`` in orders.csv a complete
        ``created -> ... -> deleted`` lifecycle.
        """
        ...
