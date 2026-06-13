"""Phase-2 correctness batch: config/format merge, mutation hygiene,
pre-existing classification, and capture counters under cancellation."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from ob_analytics.config import PipelineConfig
from ob_analytics.pipeline import Pipeline
from ob_analytics.protocols import RunContext


class TestConfigFormatMerge:
    """WS-1.2: explicit config fields overlay format defaults, never erase them."""

    def test_lobster_price_divisor_survives_explicit_config(self):
        from ob_analytics.lobster import LobsterFormat

        p = Pipeline(
            config=PipelineConfig(depth_bps=50),
            format=LobsterFormat(),
            ctx=RunContext(trading_date="2012-06-21"),
        )
        assert p.config.price_divisor == 10_000  # was silently reset to 1
        assert p.config.depth_bps == 50  # the explicit field wins
        assert p.config.volume_decimals == 0  # other defaults intact

    def test_explicit_field_overrides_format_default(self):
        from ob_analytics.lobster import LobsterFormat

        p = Pipeline(
            config=PipelineConfig(price_decimals=4),
            format=LobsterFormat(),
            ctx=RunContext(trading_date="2012-06-21"),
        )
        assert p.config.price_decimals == 4  # explicit beats the format's 2


class TestMutationHygiene:
    """WS-1.5: analytics functions must not mutate their input frames."""

    def test_set_order_types_leaves_input_unchanged(self, tiny_events):
        trades = pd.DataFrame({"maker_event_id": [1], "taker_event_id": [2]})
        from ob_analytics.analytics import set_order_types

        before_cols = list(tiny_events.columns)
        out = set_order_types(tiny_events, trades)
        assert list(tiny_events.columns) == before_cols
        assert "type" not in tiny_events.columns
        assert "type" in out.columns

    def test_order_aggressiveness_leaves_input_unchanged(self):
        from ob_analytics.analytics import order_aggressiveness

        ts = pd.Timestamp("2024-01-01")
        events = pd.DataFrame(
            {
                "direction": pd.Categorical(["bid"]),
                "action": pd.Categorical(["created"]),
                "type": pd.Categorical(["resting-limit"]),
                "timestamp": [ts],
                "event_id": [1],
                "price": [100.0],
            }
        )
        summary = pd.DataFrame(
            {
                "timestamp": [ts],
                "event_id": [0],
                "best_bid_price": [99.0],
                "best_ask_price": [101.0],
            }
        )
        before_cols = list(events.columns)
        out = order_aggressiveness(events, summary)
        assert list(events.columns) == before_cols
        assert "aggressiveness_bps" not in events.columns
        assert "aggressiveness_bps" in out.columns


class TestPreExistingClassification:
    """WS-8.2: orders without a created row get an explicit class."""

    def test_no_created_row_is_pre_existing_not_unknown(self):
        from ob_analytics.analytics import set_order_types

        ts = pd.Timestamp("2024-01-01")
        events = pd.DataFrame(
            {
                "event_id": [1, 2],
                "id": [7, 8],
                "timestamp": [ts, ts],
                "price": [100.0, 101.0],
                "volume": [5.0, 5.0],
                "direction": pd.Categorical(["bid", "ask"]),
                # id 7 first appears as a changed row (pre-existing);
                # id 8 has a normal created row.
                "action": pd.Categorical(
                    ["changed", "created"],
                    categories=["created", "changed", "deleted"],
                    ordered=True,
                ),
                "fill": [2.0, 0.0],
            }
        )
        trades = pd.DataFrame({"maker_event_id": [np.nan], "taker_event_id": [np.nan]})
        out = set_order_types(events, trades)
        assert (out.loc[out["id"] == 7, "type"] == "pre-existing").all()
        assert (out.loc[out["id"] == 8, "type"] != "pre-existing").all()

    def test_trade_derived_labels_take_precedence(self):
        from ob_analytics.analytics import set_order_types

        ts = pd.Timestamp("2024-01-01")
        events = pd.DataFrame(
            {
                "event_id": [1],
                "id": [0],  # hidden-exec style: no created row...
                "timestamp": [ts],
                "price": [100.0],
                "volume": [5.0],
                "direction": pd.Categorical(["ask"]),
                "action": pd.Categorical(
                    ["changed"],
                    categories=["created", "changed", "deleted"],
                    ordered=True,
                ),
                "fill": [5.0],
            }
        )
        # ...but it is a trade maker, so the maker-derived label wins.
        trades = pd.DataFrame({"maker_event_id": [1], "taker_event_id": [np.nan]})
        out = set_order_types(events, trades)
        assert (out["type"] == "resting-limit").all()


class TestStreamCountsSurviveCancellation:
    """WS-1.6: meta.json counters must include rows streamed before SIGINT."""

    def test_cancelled_stream_keeps_counts(self):
        from ob_analytics.live._runner import _stream

        class _Sink:
            def __init__(self):
                self.orders = 0
                self.trades = 0
                self.raw = 0

            def write_order(self, ev):
                self.orders += 1

            def write_trade(self, ev):
                self.trades += 1

            def write_raw(self, frame):
                self.raw += 1

            def finalize(self, result):
                pass

        class _Capturer:
            name = "fake"

            async def stream(self, config):
                for i in range(1000):
                    yield "order", {"i": i}, {"f": i}
                    if i == 4:
                        await asyncio.sleep(10)  # park so cancellation lands

            async def snapshot(self, config):
                return
                yield  # pragma: no cover

            async def shutdown_synthetic_events(self):
                return
                yield  # pragma: no cover

        async def run() -> dict[str, int]:
            sink = _Sink()
            counts = {"order": 0, "trade": 0, "raw": 0}
            task = asyncio.create_task(_stream(_Capturer(), None, sink, counts))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return counts

        counts = asyncio.run(run())
        # Five orders were streamed before the park; the cancelled task's
        # counts must reflect them (previously they were discarded).
        assert counts["order"] == 5
        assert counts["raw"] == 5
