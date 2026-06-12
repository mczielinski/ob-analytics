"""Tests for analytics.order_lifecycles — the canonical lifecycle table."""

from __future__ import annotations

import pandas as pd
import pytest

from ob_analytics.analytics import order_lifecycles

_TS = pd.Timestamp("2012-06-21 09:30:00")


def _ev(rows: list[tuple]) -> pd.DataFrame:
    """(id, secs, price, volume, action, fill, direction) -> events frame."""
    df = pd.DataFrame(
        rows,
        columns=["id", "secs", "price", "volume", "action", "fill", "direction"],
    )
    df["timestamp"] = _TS + pd.to_timedelta(df.pop("secs"), unit="s")
    return df


class TestOutcomes:
    def test_fill_exhaustion_terminates_without_delete(self):
        # The LOBSTER teal-wall regression: a fully-executed order never
        # emits a delete; exhaustion (outstanding -> 0) must end it.
        life = order_lifecycles(
            _ev(
                [
                    (1, 0, 100.0, 50.0, "created", 0.0, "bid"),
                    (1, 5, 100.0, 0.0, "changed", 50.0, "bid"),
                ]
            )
        )
        row = life.set_index("id").loc[1]
        assert row["outcome"] == "filled"
        assert row["end_ts"] == _TS + pd.Timedelta(seconds=5)
        assert row["filled_vol"] == 50.0

    def test_partial_then_cancel(self):
        life = order_lifecycles(
            _ev(
                [
                    (2, 0, 100.0, 50.0, "created", 0.0, "ask"),
                    (2, 3, 100.0, 30.0, "changed", 20.0, "ask"),
                    (2, 9, 100.0, 30.0, "deleted", 0.0, "ask"),
                ]
            )
        )
        row = life.set_index("id").loc[2]
        assert row["outcome"] == "partial"
        assert row["end_ts"] == _TS + pd.Timedelta(seconds=9)
        assert row["filled_vol"] == 20.0

    def test_cancelled_without_execution(self):
        life = order_lifecycles(
            _ev(
                [
                    (3, 0, 100.0, 50.0, "created", 0.0, "bid"),
                    (3, 7, 100.0, 50.0, "deleted", 0.0, "bid"),
                ]
            )
        )
        row = life.set_index("id").loc[3]
        assert row["outcome"] == "cancelled"
        assert row["filled_vol"] == 0.0

    def test_resting_has_nat_end(self):
        life = order_lifecycles(_ev([(4, 0, 100.0, 50.0, "created", 0.0, "ask")]))
        row = life.set_index("id").loc[4]
        assert row["outcome"] == "resting"
        assert pd.isna(row["end_ts"])

    def test_outcomes_partition_all_orders(self):
        life = order_lifecycles(
            _ev(
                [
                    (1, 0, 100.0, 50.0, "created", 0.0, "bid"),
                    (1, 5, 100.0, 0.0, "changed", 50.0, "bid"),
                    (2, 0, 100.0, 50.0, "created", 0.0, "ask"),
                    (2, 9, 100.0, 50.0, "deleted", 0.0, "ask"),
                    (3, 0, 100.0, 50.0, "created", 0.0, "bid"),
                ]
            )
        )
        assert len(life) == 3
        assert life["outcome"].isin(["filled", "partial", "cancelled", "resting"]).all()


class TestEdgeCases:
    def test_pre_existing_orders_excluded(self):
        # No created row (pre-existing book / hidden id=0): no lifecycle.
        life = order_lifecycles(
            _ev(
                [
                    (9, 1, 100.0, 30.0, "changed", 20.0, "bid"),
                    (8, 0, 100.0, 50.0, "created", 0.0, "ask"),
                ]
            )
        )
        assert list(life["id"]) == [8]

    def test_zero_size_placement_does_not_self_terminate(self):
        # Degenerate volume-0 creations (Bitstamp market placeholders) must
        # not read as exhausted-at-birth.
        life = order_lifecycles(_ev([(7, 0, 100.0, 0.0, "created", 0.0, "bid")]))
        row = life.set_index("id").loc[7]
        assert row["outcome"] == "resting"
        assert pd.isna(row["end_ts"])

    def test_carries_type_and_aggressiveness_when_present(self):
        ev = _ev(
            [
                (5, 0, 100.0, 50.0, "created", 0.0, "bid"),
                (5, 2, 100.0, 50.0, "deleted", 0.0, "bid"),
            ]
        )
        ev["type"] = "flashed-limit"
        ev["aggressiveness_bps"] = [4.2, float("nan")]
        life = order_lifecycles(ev)
        row = life.set_index("id").loc[5]
        assert row["type"] == "flashed-limit"
        assert row["aggressiveness_bps"] == pytest.approx(4.2)
