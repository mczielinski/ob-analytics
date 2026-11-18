"""Tests for ob_analytics.depth.depth_bands."""

from __future__ import annotations

import pandas as pd
import pytest

from ob_analytics.depth import depth_bands
from ob_analytics.exceptions import ConfigError

TS = pd.Timestamp("2015-05-01 00:00:00", tz="UTC")


def _summary() -> pd.DataFrame:
    # Bands are listed far-first on purpose: the result puts the touch first.
    return pd.DataFrame(
        {
            "timestamp": [TS + pd.Timedelta(seconds=s) for s in (0.2, 0.2, 0.7, 1.5)],
            "best_bid_vol": [1.0, 1.0, 1.0, 1.0],
            "bid_vol100bps": [9.0, 8.0, 6.0, 4.0],
            "bid_vol50bps": [3.0, 2.0, 4.0, 1.0],
            "ask_vol100bps": [7.0, 7.0, 5.0, 3.0],
            "ask_vol50bps": [1.0, 1.0, 3.0, 2.0],
        }
    )


class TestDepthBands:
    def test_one_row_per_timestamp_touch_first(self) -> None:
        out = depth_bands(_summary())
        assert out.columns.tolist() == [
            "timestamp",
            "bid_vol50bps",
            "bid_vol100bps",
            "ask_vol50bps",
            "ask_vol100bps",
        ]
        # The two rows at 0.2 s keep the last one: the book after both.
        assert len(out) == 3
        assert out["bid_vol50bps"].tolist() == [2.0, 4.0, 1.0]
        assert out["bid_vol100bps"].tolist() == [8.0, 6.0, 4.0]

    def test_interval_averages_and_stamps_each_interval_at_its_end(self) -> None:
        out = depth_bands(_summary(), interval="1s")
        assert out["timestamp"].tolist() == [
            TS + pd.Timedelta(seconds=1),
            TS + pd.Timedelta(seconds=2),
        ]
        assert out["bid_vol50bps"].tolist() == pytest.approx([3.0, 1.0])
        assert out["ask_vol100bps"].tolist() == pytest.approx([6.0, 3.0])

    def test_integer_lots_are_reported_in_the_instruments_units(self) -> None:
        summary = _summary()
        bands = [c for c in summary.columns if c.endswith("bps")]
        summary[bands] = (summary[bands] * 1000).astype("int64")
        summary.attrs["lot_size"] = 0.001
        out = depth_bands(summary)
        assert out["ask_vol50bps"].tolist() == pytest.approx([1.0, 3.0, 2.0])

    def test_a_summary_without_bands_raises(self) -> None:
        with pytest.raises(ConfigError, match="depth_bands"):
            depth_bands(_summary()[["timestamp", "best_bid_vol"]])

    def test_mismatched_bid_and_ask_bands_raise(self) -> None:
        with pytest.raises(ConfigError, match="differ"):
            depth_bands(_summary().drop(columns="ask_vol100bps"))
