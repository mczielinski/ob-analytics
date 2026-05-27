"""Tests for the live-capture protocol and runner -- no network."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pandas as pd
import pytest

from ob_analytics.live import (
    CaptureConfig,
    LiveCapturer,
    get_capturer,
    list_capturers,
    register_capturer,
)
from ob_analytics.live._base import EventDict
from ob_analytics.live._runner import run_capturer


# ---------------------------------------------------------------------------
# A deterministic, no-network capturer
# ---------------------------------------------------------------------------


class _FakeCapturer:
    name = "fake"

    def __init__(self) -> None:
        self._open: dict[int, dict[str, Any]] = {}

    async def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        ts = pd.Timestamp("2025-01-01", tz="UTC")
        for i, (price, side) in enumerate([(100.0, "bid"), (101.0, "ask")], start=1):
            ev: EventDict = {
                "id": i,
                "timestamp": ts,
                "exchange_timestamp": ts,
                "price": price,
                "volume": 1.0,
                "action": "created",
                "direction": side,
            }
            self._open[i] = {"price": price, "direction": side}
            yield ev

    async def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        ts = pd.Timestamp("2025-01-01 00:00:01", tz="UTC")
        yield (
            "order",
            {
                "id": 3,
                "timestamp": ts,
                "exchange_timestamp": ts,
                "price": 100.5,
                "volume": 0.5,
                "action": "created",
                "direction": "bid",
            },
            {"raw": "frame-1"},
        )
        self._open[3] = {"price": 100.5, "direction": "bid"}
        yield (
            "trade",
            {
                "trade_id": 1,
                "timestamp": ts,
                "exchange_timestamp": ts,
                "price": 100.5,
                "amount": 0.5,
                "buy_order_id": 3,
                "sell_order_id": 4,
                "side": "buy",
            },
            {"raw": "frame-2"},
        )

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        ts = pd.Timestamp("2025-01-01 00:00:02", tz="UTC")
        for oid, last in self._open.items():
            yield {
                "id": oid,
                "timestamp": ts,
                "exchange_timestamp": ts,
                "price": last["price"],
                "volume": 0.0,
                "action": "deleted",
                "direction": last["direction"],
            }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_get(self):
        register_capturer("fake", _FakeCapturer)
        assert "fake" in list_capturers()
        assert get_capturer("fake") is _FakeCapturer

    def test_case_insensitive_lookup(self):
        register_capturer("Fake2", _FakeCapturer)
        assert get_capturer("fake2") is _FakeCapturer
        assert get_capturer("FAKE2") is _FakeCapturer

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown capturer"):
            get_capturer("nonexistent-venue")


class TestProtocolConformance:
    def test_fake_is_a_livecapturer(self):
        # runtime_checkable Protocol check
        assert isinstance(_FakeCapturer(), LiveCapturer)


class TestRunner:
    def test_runs_and_writes_files(self, tmp_path):
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="btcusd", out_dir=out, minutes=0.001, keep_raw=True)
        cap = _FakeCapturer()
        result = asyncio.run(run_capturer(cap, cfg))

        assert (out / "orders.csv").exists()
        assert (out / "trades.csv").exists()
        assert (out / "raw.jsonl").exists()
        assert (out / "meta.json").exists()

        # Snapshot (2) + stream-order (1) + shutdown-delete (3) = 6 orders
        assert result.n_order_events == 6
        assert result.n_trade_events == 1
        assert result.n_raw_frames == 2

    def test_disables_raw(self, tmp_path):
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="btcusd", out_dir=out, minutes=0.001, keep_raw=False)
        asyncio.run(run_capturer(_FakeCapturer(), cfg))
        assert not (out / "raw.jsonl").exists()

    def test_output_is_loader_compatible(self, tmp_path):
        """The captured orders.csv must be loadable by BitstampLoader."""
        from ob_analytics.bitstamp import BitstampLoader

        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="btcusd", out_dir=out, minutes=0.001)
        asyncio.run(run_capturer(_FakeCapturer(), cfg))

        events = BitstampLoader().load(out / "orders.csv")
        # Loader applied without error and returned a non-empty frame
        assert len(events) > 0
        assert "direction" in events.columns
        assert "action" in events.columns
