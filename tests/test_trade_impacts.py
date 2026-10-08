"""Tests for ob_analytics.analytics.trade_impacts (WS-4.4)."""

from __future__ import annotations

import pandas as pd
import pytest

from ob_analytics.analytics import trade_impacts

TS = pd.Timestamp("2015-05-01 00:00:00")


def _trades(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": TS + pd.Timedelta(seconds=r["t"]),
                "price": r["price"],
                "volume": r["vol"],
                "direction": r["dir"],
                "taker": r["taker"],
            }
            for r in rows
        ]
    )


class TestTradeImpacts:
    def test_aggregates_sweep_per_taker(self) -> None:
        # Taker 1 sweeps two levels; taker 2 a single fill.
        out = trade_impacts(
            _trades(
                [
                    {"t": 0, "price": 100.0, "vol": 10.0, "dir": "buy", "taker": 1},
                    {"t": 1, "price": 101.0, "vol": 5.0, "dir": "buy", "taker": 1},
                    {"t": 2, "price": 99.0, "vol": 8.0, "dir": "sell", "taker": 2},
                ]
            )
        )
        assert len(out) == 2
        one = out[out["id"] == 1].iloc[0]
        assert one["min_price"] == 100.0
        assert one["max_price"] == 101.0
        assert one["hits"] == 2
        assert one["vol"] == 15.0
        # VWAP = (100*10 + 101*5) / 15
        assert one["vwap"] == pytest.approx(1505.0 / 15.0)
        assert one["dir"] == "buy"
        assert one["start_time"] == TS
        assert one["end_time"] == TS + pd.Timedelta(seconds=1)

    def test_missing_columns_raises(self) -> None:
        bad = pd.DataFrame({"price": [1.0], "volume": [1.0]})
        with pytest.raises(Exception, match="trade_impacts"):
            trade_impacts(bad)

    def test_vwap_does_not_wrap_on_a_large_sweep(self) -> None:
        """Price in ticks times size in lots can pass int64."""
        trades = pd.DataFrame(
            {
                "timestamp": [TS, TS + pd.Timedelta(seconds=1)],
                "price": [7_800_000, 7_800_100],
                "volume": [800_000_000_000, 800_000_000_000],
                "direction": ["buy", "buy"],
                "taker": [1, 1],
            }
        )
        out = trade_impacts(trades)
        assert out["vwap"].iloc[0] == pytest.approx(7_800_050)
