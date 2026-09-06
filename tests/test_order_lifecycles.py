"""Tests for analytics.order_lifecycles — the canonical lifecycle table."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ob_analytics.analytics import order_lifecycles
from ob_analytics.engine._lifecycles import DEFAULT_FILL_TOLERANCE, _grouped_sum

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


class TestBaseAssetFloatSizes:
    """Sizes that are floats in the base asset, not whole lot counts.

    Canonical sizes are integer lots, but a float frame is a first-class input:
    it is what ``display_result`` hands the L3 faces, and what a pre-4.0 file or
    a hand-built frame carries.  Such sizes are routinely below 1 -- a fill of
    0.121 BTC -- where a total that is cast to ``int64`` truncates to zero and
    the order reads as though it never executed.

    Every other case in this module places whole numbers, which is exactly why
    that cast passed the suite: 50.0 truncates to 50 and nothing looks wrong.
    """

    def test_fully_executed_fractional_order_is_filled(self):
        life = order_lifecycles(
            _ev(
                [
                    (1, 0, 78319.0, 0.121, "created", 0.0, "bid"),
                    (1, 5, 78319.0, 0.0, "changed", 0.121, "bid"),
                ]
            )
        )
        row = life.set_index("id").loc[1]
        assert row["outcome"] == "filled"
        assert row["filled_vol"] == pytest.approx(0.121)

    def test_partially_executed_fractional_order_is_partial(self):
        life = order_lifecycles(
            _ev(
                [
                    (2, 0, 78319.0, 0.5, "created", 0.0, "ask"),
                    (2, 3, 78319.0, 0.44, "changed", 0.06, "ask"),
                    (2, 9, 78319.0, 0.44, "deleted", 0.0, "ask"),
                ]
            )
        )
        row = life.set_index("id").loc[2]
        assert row["outcome"] == "partial"
        assert row["filled_vol"] == pytest.approx(0.06)

    def test_filled_vol_keeps_the_units_it_was_given(self):
        # A float frame in, a float total out; an int64 total here is the
        # truncation that turned every filled order into a cancelled one.
        life = order_lifecycles(
            _ev(
                [
                    (4, 0, 78319.0, 0.25, "created", 0.0, "bid"),
                    (4, 1, 78319.0, 0.1, "changed", 0.15, "bid"),
                    (4, 2, 78319.0, 0.1, "deleted", 0.0, "bid"),
                ]
            )
        )
        assert life["filled_vol"].dtype == "float64"
        assert life.set_index("id").loc[4, "filled_vol"] == pytest.approx(0.15)


class TestGroupedSum:
    """``_grouped_sum`` sums each order's fills without changing their units."""

    def test_integer_lots_sum_exactly(self):
        slot = np.array([0, 0, 1])
        # Lot counts far past float64's exact-integer range, where a sum
        # routed through float64 would round to a neighbouring even value.
        values = np.array([2**53 + 1, 2, 7], dtype=np.int64)
        total = _grouped_sum(slot, values, 2)
        assert total.dtype == np.int64
        assert list(total) == [2**53 + 3, 7]

    def test_float_sizes_are_not_truncated_to_whole_units(self):
        # The regression: base-asset sizes are mostly below 1, so an int64
        # cast sent every fill to zero and every filled order to "cancelled".
        slot = np.array([0, 0, 1])
        values = np.array([0.121, 0.063841, 0.06])
        total = _grouped_sum(slot, values, 2)
        assert total.dtype == np.float64
        assert total == pytest.approx([0.184841, 0.06])

    def test_small_fills_against_a_large_total_are_compensated(self):
        # Many sub-lot fills sweeping one large order: a plain accumulation
        # loses the small ones in the running total's low bits and lands
        # short by far more than DEFAULT_FILL_TOLERANCE, which reads a
        # fully-executed order as merely partial.  The compensated sum does
        # not, so the total matches an exact one.
        rng = np.random.default_rng(0)
        fills = np.r_[1e8, np.full(2769, 1e-9)]
        rng.shuffle(fills)
        exact = math.fsum(fills)

        total = _grouped_sum(np.zeros(len(fills), dtype=np.int64), fills, 1)[0]
        assert total >= exact - DEFAULT_FILL_TOLERANCE

        plain = float(np.sum(fills))
        assert plain < exact - DEFAULT_FILL_TOLERANCE, (
            "fixture no longer exercises the compensation"
        )
