"""Tests for ob_analytics.analytics.maker_resting_times."""

from __future__ import annotations

import pandas as pd
import pytest

from ob_analytics.analytics import maker_resting_times
from ob_analytics.exceptions import ConfigError

TS = pd.Timestamp("2015-05-01 00:00:00", tz="UTC")


def _events() -> pd.DataFrame:
    rows = [
        # (event_id, order id, seconds, action)
        (1, 10, 0.0, "created"),
        (2, 10, 2.0, "changed"),
        (3, 20, 1.0, "created"),
        (4, 20, 5.0, "changed"),
        # Order 30 was in the book before the data starts: no created row.
        (5, 30, 3.0, "changed"),
    ]
    return pd.DataFrame(
        {
            "event_id": [r[0] for r in rows],
            "id": [r[1] for r in rows],
            "timestamp": [TS + pd.Timedelta(seconds=r[2]) for r in rows],
            "action": [r[3] for r in rows],
        }
    )


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [TS + pd.Timedelta(seconds=s) for s in (2.0, 3.0, 5.0)],
            "price": [100.0, 100.5, 101.0],
            "volume": [1.0, 2.0, 3.0],
            "direction": ["sell", "buy", "buy"],
            "maker_event_id": [2, 5, 4],
        }
    )


class TestMakerRestingTimes:
    def test_resting_time_runs_from_the_makers_placement(self) -> None:
        out = maker_resting_times(_events(), _trades())
        # The trade against order 30 is left out: its placement is unknown.
        assert out["id"].tolist() == [10, 20]
        assert out["created_ts"].tolist() == [TS, TS + pd.Timedelta(seconds=1)]
        assert out["resting_s"].tolist() == pytest.approx([2.0, 4.0])
        assert out["direction"].tolist() == ["sell", "buy"]

    def test_integer_ticks_and_lots_are_reported_in_the_instruments_units(
        self,
    ) -> None:
        trades = _trades()
        trades["price"] = (trades["price"] * 100).round().astype("int64")
        trades["volume"] = (trades["volume"] * 1000).round().astype("int64")
        trades.attrs.update(tick_size=0.01, lot_size=0.001)
        out = maker_resting_times(_events(), trades)
        assert out["price"].tolist() == pytest.approx([100.0, 101.0])
        assert out["volume"].tolist() == pytest.approx([1.0, 3.0])

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(ConfigError, match="maker_resting_times"):
            maker_resting_times(_events(), _trades().drop(columns="maker_event_id"))
