"""Tests for the live-capture protocol and runner -- no network."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pandas as pd
import pytest

from ob_analytics.config import SourceSettings
from ob_analytics.live import CaptureConfig, LiveSource, SupportsDiagnostics
from ob_analytics.live._base import EventDict
from ob_analytics.live._runner import run_capturer
from ob_analytics.protocols import FeedType, Level
from ob_analytics.sources import get_source, list_sources, register_source

# ---------------------------------------------------------------------------
# A deterministic, no-network capturer
# ---------------------------------------------------------------------------


class _FakeCapturer:
    name = "fake"
    level = Level.L3
    feed_type = FeedType.DIFF_FEED
    settings = SourceSettings()

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


class _DiagCapturer(_FakeCapturer):
    """A capturer that also implements the optional diagnostics() hook."""

    name = "diag"

    def diagnostics(self) -> dict[str, Any]:
        return {"dropped": 7, "reconnects": 2}


class _FakeL2Capturer:
    """A deterministic L2 (price-level) capturer -- no network, no order IDs."""

    name = "fake-l2"
    level = Level.L2
    feed_type = FeedType.MATCHED_BOOK
    settings = SourceSettings()

    async def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        ts = pd.Timestamp("2025-01-01", tz="UTC")
        for price, side, volume in [(100.0, "bid", 5.0), (101.0, "ask", 3.0)]:
            yield {
                "timestamp": ts,
                "exchange_timestamp": ts,
                "side": side,
                "price": price,
                "volume": volume,
            }

    async def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        ts = pd.Timestamp("2025-01-01 00:00:01", tz="UTC")
        # bid 100.0 grows 5 -> 7 (absolute size)
        yield (
            "depth",
            {
                "timestamp": ts,
                "exchange_timestamp": ts,
                "side": "bid",
                "price": 100.0,
                "volume": 7.0,
            },
            {"raw": "d1"},
        )
        yield (
            "trade",
            {
                "trade_id": 1,
                "timestamp": ts,
                "exchange_timestamp": ts,
                "price": 101.0,
                "amount": 1.0,
                "buy_order_id": 0,
                "sell_order_id": 0,
                "side": "buy",
            },
            {"raw": "t1"},
        )
        # ask 101.0 emptied -> 0 (level removed)
        yield (
            "depth",
            {
                "timestamp": ts,
                "exchange_timestamp": ts,
                "side": "ask",
                "price": 101.0,
                "volume": 0.0,
            },
            {"raw": "d2"},
        )

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        # L2 price levels have no lifecycle to close: emit nothing.
        for _ in ():
            yield {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_get(self):
        register_source("fake", _FakeCapturer)
        assert "fake" in list_sources()
        assert get_source("fake") is _FakeCapturer

    def test_case_insensitive_lookup(self):
        register_source("Fake2", _FakeCapturer)
        assert get_source("fake2") is _FakeCapturer
        assert get_source("FAKE2") is _FakeCapturer

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown source"):
            get_source("nonexistent-venue")


class TestProtocolConformance:
    def test_fake_is_a_livesource(self):
        # runtime_checkable Protocol check
        assert isinstance(_FakeCapturer(), LiveSource)


class TestDiagnosticsProtocol:
    """diagnostics() is an *optional* capability, not part of LiveSource."""

    def test_diagnostics_is_optional_for_livesource(self):
        # A plain live source with no diagnostics() still conforms to
        # LiveSource but is NOT a SupportsDiagnostics.
        cap = _FakeCapturer()
        assert isinstance(cap, LiveSource)
        assert not isinstance(cap, SupportsDiagnostics)

    def test_source_with_diagnostics_conforms(self):
        cap = _DiagCapturer()
        assert isinstance(cap, LiveSource)
        assert isinstance(cap, SupportsDiagnostics)

    def test_runner_merges_diagnostics_into_extras(self, tmp_path):
        cfg = CaptureConfig(pair="btcusd", out_dir=tmp_path / "cap", minutes=0.001)
        result = asyncio.run(run_capturer(_DiagCapturer(), cfg))
        assert result.extras["dropped"] == 7
        assert result.extras["reconnects"] == 2

    def test_runner_no_diagnostics_leaves_extras_empty(self, tmp_path):
        cfg = CaptureConfig(pair="btcusd", out_dir=tmp_path / "cap", minutes=0.001)
        result = asyncio.run(run_capturer(_FakeCapturer(), cfg))
        assert result.extras == {}


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


class TestL2Runner:
    """An L2 capturer writes depth.csv (not orders.csv) and replays through
    the L2 depth path -- no faked per-order IDs."""

    def test_l2_capturer_conforms(self):
        cap = _FakeL2Capturer()
        assert isinstance(cap, LiveSource)
        assert cap.level is Level.L2

    def test_l2_writes_depth_not_orders(self, tmp_path):
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="x", out_dir=out, minutes=0.001, keep_raw=True)
        result = asyncio.run(run_capturer(_FakeL2Capturer(), cfg))

        assert (out / "depth.csv").exists()
        assert (out / "trades.csv").exists()
        assert not (out / "orders.csv").exists()

        # snapshot (2) + stream depth (2) = 4 depth events; 1 trade; 0 orders
        assert result.n_depth_events == 4
        assert result.n_trade_events == 1
        assert result.n_order_events == 0

    def test_l2_meta_reports_depth(self, tmp_path):
        import json

        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="x", out_dir=out, minutes=0.001)
        asyncio.run(run_capturer(_FakeL2Capturer(), cfg))

        meta = json.loads((out / "meta.json").read_text())
        assert meta["n_depth_events"] == 4
        assert meta["n_order_events"] == 0

    def test_l2_output_replays_through_depth_path(self, tmp_path):
        """The captured depth.csv must load via the L2 depth path."""
        from ob_analytics.depth_l2 import L2DepthLoader

        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="x", out_dir=out, minutes=0.001)
        asyncio.run(run_capturer(_FakeL2Capturer(), cfg))

        depth = L2DepthLoader().load(out / "depth.csv")
        # 4 absolute-size price-level rows (the 0-size removal is kept)
        assert len(depth) == 4
        assert set(depth["direction"].dropna().unique()) <= {"bid", "ask"}
