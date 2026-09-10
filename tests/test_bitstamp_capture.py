"""BitstampCapturer's opening snapshot against a fake WS and REST book -- no network.

The snapshot must overlap the stream: a REST book older than every live order
event leaves a gap in which an order can vanish unreported (issue #237).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("websockets")

from ob_analytics.live import bitstamp as live_bitstamp
from ob_analytics.live._base import CaptureConfig
from ob_analytics.live.bitstamp import BitstampCapturer

_ORDERS = "live_orders_btcusd"
_CONFIG = CaptureConfig(pair="btcusd", out_dir=Path("unused"), minutes=0.001)


def _order_frame(order_id: int, microtimestamp: int) -> str:
    return json.dumps(
        {
            "channel": _ORDERS,
            "event": "order_created",
            "data": {
                "id": order_id,
                "price": "100.0",
                "amount": "1.0",
                "order_type": 0,
                "microtimestamp": str(microtimestamp),
            },
        }
    )


def _book(microtimestamp: int, *order_ids: int) -> dict[str, Any]:
    return {
        "microtimestamp": str(microtimestamp),
        "timestamp": str(microtimestamp // 1_000_000),
        "bids": [["100.0", "1.0", str(oid)] for oid in order_ids],
        "asks": [],
    }


class _FakeWs:
    def __init__(self, frames: list[str]) -> None:
        self._frames = list(frames)

    async def recv(self) -> str:
        await asyncio.sleep(0.001)
        if self._frames:
            return self._frames.pop(0)
        raise TimeoutError


def _capturer(
    monkeypatch, frames: list[str], books: list[dict[str, Any]], fetch_delay=0.0
) -> BitstampCapturer:
    """A capturer whose WS replays *frames* and whose REST fetches return
    *books* in turn, repeating the last one."""
    queue = list(books)

    def fetch(pair: str) -> dict[str, Any]:
        time.sleep(fetch_delay)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    async def open_ws(self, channels: list[str]) -> None:
        self._ws = _FakeWs(frames)

    monkeypatch.setattr(live_bitstamp, "_fetch_book_snapshot", fetch)
    monkeypatch.setattr(live_bitstamp, "SNAPSHOT_RETRY_SECONDS", 0.05)
    monkeypatch.setattr(BitstampCapturer, "_open_ws", open_ws)
    return BitstampCapturer()


def _snapshot_ids(cap: BitstampCapturer) -> list[int]:
    async def collect() -> list[int]:
        return [ev["id"] async for ev in cap.snapshot(_CONFIG)]

    return asyncio.run(collect())


class TestSnapshotOverlap:
    def test_a_snapshot_older_than_the_stream_is_fetched_again(self, monkeypatch):
        # The first book (t=500) predates the only live event (t=1000), so an
        # order it lists may already be gone: order 7 is, and the second book
        # (t=2000) no longer has it.
        cap = _capturer(
            monkeypatch,
            frames=[_order_frame(9, 1_000)],
            books=[_book(500, 7, 8), _book(2_000, 8)],
        )
        assert _snapshot_ids(cap) == [8]

        diag = cap.diagnostics()
        assert diag["snapshot_fetches"] == 2
        assert diag["snapshot_overlap"] is True
        assert diag["snapshot_microtimestamp"] == 2_000

    def test_a_snapshot_that_overlaps_is_used_at_once(self, monkeypatch):
        # A slow fetch lets the live event at t=1000 arrive first, and the
        # book at t=2000 already covers it.
        cap = _capturer(
            monkeypatch,
            frames=[_order_frame(9, 1_000)],
            books=[_book(2_000, 8)],
            fetch_delay=0.05,
        )
        assert _snapshot_ids(cap) == [8]
        assert cap.diagnostics()["snapshot_fetches"] == 1
        assert cap.diagnostics()["snapshot_overlap"] is True

    def test_gives_up_after_the_limit_and_says_so(self, monkeypatch):
        monkeypatch.setattr(live_bitstamp, "SNAPSHOT_MAX_FETCHES", 3)
        cap = _capturer(monkeypatch, frames=[], books=[_book(500, 7)])

        # No live event ever proves overlap: the last book is used anyway.
        assert _snapshot_ids(cap) == [7]
        diag = cap.diagnostics()
        assert diag["snapshot_fetches"] == 3
        assert diag["snapshot_overlap"] is False

    def test_live_events_the_snapshot_covers_are_skipped(self, monkeypatch):
        cap = _capturer(
            monkeypatch,
            frames=[_order_frame(9, 1_000)],
            books=[_book(500, 8), _book(2_000, 8, 9)],
        )
        assert _snapshot_ids(cap) == [8, 9]

        async def replay() -> list[tuple[str, Any, Any]]:
            return [item async for item in cap.stream(_CONFIG)]

        # Order 9's created (t=1000) is already in the t=2000 book.
        assert [kind for kind, _, _ in asyncio.run(replay())] == []
        assert cap.diagnostics()["pre_snapshot_skipped"] == 1
