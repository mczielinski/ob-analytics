"""Tests for ob_analytics.analytics.cancelled_orders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics.analytics import cancelled_orders
from ob_analytics.exceptions import ConfigError

TS = pd.Timestamp("2015-05-01 00:00:00", tz="UTC")


def _row(oid, t, action, *, price=100.0, volume=10.0, fill=0.0, kind, bps=np.nan):
    return {
        "id": oid,
        "timestamp": TS + pd.Timedelta(seconds=t),
        "price": price,
        "volume": volume,
        "fill": fill,
        "direction": "bid" if oid % 2 else "ask",
        "action": action,
        "type": kind,
        "aggressiveness_bps": bps,
    }


def _events() -> pd.DataFrame:
    rows = [
        # Order 1: placed 2 bps behind the best bid, cancelled 1.5 s later.
        _row(1, 0.0, "created", kind="flashed-limit", bps=-2.0),
        _row(1, 1.5, "deleted", kind="flashed-limit"),
        # Order 2: placed with no best ask to measure from, cancelled 3 s later.
        _row(2, 1.0, "created", price=101.0, volume=4.0, kind="flashed-limit"),
        _row(2, 4.0, "deleted", price=101.0, volume=4.0, kind="flashed-limit"),
        # Order 3: a resting order that was filled, not cancelled.
        _row(3, 0.5, "created", kind="resting-limit", bps=0.0),
        _row(3, 2.0, "changed", volume=0.0, fill=10.0, kind="resting-limit"),
        # Order 5: a flashed order still resting when the data ends.
        _row(5, 3.0, "created", kind="flashed-limit", bps=1.0),
    ]
    events = pd.DataFrame(rows)
    events["action"] = pd.Categorical(
        events["action"], categories=["created", "changed", "deleted"], ordered=True
    )
    return events


class TestCancelledOrders:
    def test_one_row_per_cancelled_order(self) -> None:
        out = cancelled_orders(_events())
        assert out["id"].tolist() == [1, 2]
        first = out.iloc[0]
        assert first["placed_ts"] == TS
        assert first["timestamp"] == TS + pd.Timedelta(seconds=1.5)
        assert first["age_s"] == pytest.approx(1.5)
        assert first["distance_bps"] == pytest.approx(-2.0)
        assert first["volume"] == pytest.approx(10.0)
        assert first["direction"] == "bid"
        assert out.iloc[1]["age_s"] == pytest.approx(3.0)
        assert np.isnan(out.iloc[1]["distance_bps"])

    def test_integer_ticks_and_lots_are_reported_in_the_instruments_units(
        self,
    ) -> None:
        events = _events()
        events["price"] = (events["price"] * 100).round().astype("int64")
        events["volume"] = (events["volume"] * 1000).round().astype("int64")
        events["fill"] = (events["fill"] * 1000).round().astype("int64")
        events.attrs.update(tick_size=0.01, lot_size=0.001)
        out = cancelled_orders(events)
        assert out["price"].tolist() == pytest.approx([100.0, 101.0])
        assert out["volume"].tolist() == pytest.approx([10.0, 4.0])

    def test_unclassified_events_raise(self) -> None:
        with pytest.raises(ConfigError, match="cancelled_orders"):
            cancelled_orders(_events().drop(columns="type"))
