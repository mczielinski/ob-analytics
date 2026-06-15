"""Tests for the FIFO queue-position engine (WS-4.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics.queue import QUEUE_COLUMNS, queue_age_grid, queue_positions

TS = pd.Timestamp("2015-05-01 00:00:00")


def _events(rows: list[dict]) -> pd.DataFrame:
    """Build a canonical events frame from compact (id, t, price, vol, dir, action)."""
    recs = []
    for i, r in enumerate(rows):
        recs.append(
            {
                "event_id": i + 1,
                "id": r["id"],
                "timestamp": TS + pd.Timedelta(seconds=r["t"]),
                "price": r["price"],
                "volume": r["vol"],
                "direction": r.get("dir", "bid"),
                "action": r["action"],
            }
        )
    return pd.DataFrame(
        recs,
        columns=[
            "event_id",
            "id",
            "timestamp",
            "price",
            "volume",
            "direction",
            "action",
        ],
    )


def _rows_for(out: pd.DataFrame, oid: int) -> pd.DataFrame:
    return out[out["id"] == oid].reset_index(drop=True)


# A three-order bid queue at price 100 that drains front-to-back.
SCENARIO = [
    {"id": 1, "t": 0, "price": 100.0, "vol": 10.0, "action": "created"},  # A
    {"id": 2, "t": 1, "price": 100.0, "vol": 5.0, "action": "created"},  # B
    {"id": 3, "t": 2, "price": 100.0, "vol": 3.0, "action": "created"},  # C
    {"id": 1, "t": 3, "price": 100.0, "vol": 4.0, "action": "changed"},  # A partial
    {"id": 2, "t": 4, "price": 100.0, "vol": 5.0, "action": "deleted"},  # B cancels
    {"id": 3, "t": 5, "price": 100.0, "vol": 3.0, "action": "changed"},  # C touched
    {"id": 1, "t": 6, "price": 100.0, "vol": 4.0, "action": "deleted"},  # A cancels
    {"id": 3, "t": 7, "price": 100.0, "vol": 3.0, "action": "changed"},  # C touched
]


class TestQueuePositions:
    def test_columns(self) -> None:
        out = queue_positions(_events(SCENARIO), levels="all")
        assert list(out.columns) == list(QUEUE_COLUMNS)

    def test_initial_fifo_ranks_and_ahead_volume(self) -> None:
        out = queue_positions(_events(SCENARIO), levels="all")
        # On creation: A->rank1/ahead0, B->rank2/ahead10, C->rank3/ahead15.
        a0 = _rows_for(out, 1).iloc[0]
        b0 = _rows_for(out, 2).iloc[0]
        c0 = _rows_for(out, 3).iloc[0]
        assert (a0["rank"], a0["ahead_volume"], a0["queue_len"]) == (1, 0.0, 1)
        assert (b0["rank"], b0["ahead_volume"], b0["queue_len"]) == (2, 10.0, 2)
        assert (c0["rank"], c0["ahead_volume"], c0["queue_len"]) == (3, 15.0, 3)

    def test_partial_fill_keeps_place_reduces_remaining(self) -> None:
        out = queue_positions(_events(SCENARIO), levels="all")
        a = _rows_for(out, 1)
        changed = a[a["action"] == "changed"].iloc[0]
        assert changed["rank"] == 1  # A keeps the front
        assert changed["remaining"] == 4.0  # 10 -> 4

    def test_deletion_ahead_promotes_followers(self) -> None:
        out = queue_positions(_events(SCENARIO), levels="all")
        c = _rows_for(out, 3)
        # C: rank 3 at creation, 2 after B leaves (t5), 1 after A leaves (t7).
        assert list(c["rank"]) == [3, 2, 1]
        # After B(deleted) and A(reduced to 4), C's ahead = A's remaining only.
        assert c[c["age_s"] == 3.0]["ahead_volume"].iloc[0] == 4.0  # t5 row

    def test_rank_monotone_nonincreasing_over_lifetime(self) -> None:
        out = queue_positions(_events(SCENARIO), levels="all")
        for oid in (1, 2, 3):
            ranks = _rows_for(out, oid)["rank"].to_numpy()
            assert (ranks[1:] <= ranks[:-1]).all(), f"id {oid} rank rose: {ranks}"

    def test_deleted_row_emitted_at_last_position(self) -> None:
        out = queue_positions(_events(SCENARIO), levels="all")
        b = _rows_for(out, 2)
        assert b.iloc[-1]["action"] == "deleted"
        assert b.iloc[-1]["rank"] == 2  # B was 2nd when cancelled

    def test_age_seconds(self) -> None:
        out = queue_positions(_events(SCENARIO), levels="all")
        a = _rows_for(out, 1)
        assert a.iloc[0]["age_s"] == 0.0  # created
        assert a[a["action"] == "changed"].iloc[0]["age_s"] == 3.0  # t3 - t0


class TestTouchFilter:
    def _two_levels(self) -> pd.DataFrame:
        # Best bid 100 (A); a deeper bid 99 (D) that never becomes the touch.
        return _events(
            [
                {"id": 1, "t": 0, "price": 100.0, "vol": 10.0, "action": "created"},
                {"id": 9, "t": 1, "price": 99.0, "vol": 7.0, "action": "created"},
                {"id": 1, "t": 2, "price": 100.0, "vol": 6.0, "action": "changed"},
                {"id": 9, "t": 3, "price": 99.0, "vol": 7.0, "action": "changed"},
            ]
        )

    def test_touch_excludes_deeper_levels(self) -> None:
        out = queue_positions(self._two_levels(), levels="touch")
        assert set(out["id"]) == {1}  # only the best-bid order
        assert (out["price"] == 100.0).all()

    def test_all_keeps_deeper_levels(self) -> None:
        out = queue_positions(self._two_levels(), levels="all")
        assert set(out["id"]) == {1, 9}

    def test_promoted_level_becomes_touch(self) -> None:
        # When the best bid empties, the next level is the touch.
        ev = _events(
            [
                {"id": 1, "t": 0, "price": 100.0, "vol": 10.0, "action": "created"},
                {"id": 9, "t": 1, "price": 99.0, "vol": 7.0, "action": "created"},
                {"id": 1, "t": 2, "price": 100.0, "vol": 10.0, "action": "deleted"},
                {"id": 9, "t": 3, "price": 99.0, "vol": 7.0, "action": "changed"},
            ]
        )
        out = queue_positions(ev, levels="touch")
        # id 9 only counts as touch once 100 is gone (its t3 changed row).
        nine = _rows_for(out, 9)
        assert list(nine["age_s"]) == [2.0]  # only the post-promotion row


class TestEdgeCases:
    def test_hidden_orders_excluded(self) -> None:
        ev = _events(
            [
                {"id": 0, "t": 0, "price": 100.0, "vol": 5.0, "action": "created"},
                {"id": 1, "t": 1, "price": 100.0, "vol": 5.0, "action": "created"},
            ]
        )
        out = queue_positions(ev, levels="all")
        assert set(out["id"]) == {1}

    def test_uncreated_order_skipped(self) -> None:
        # A changed/deleted whose creation we never saw is ignored.
        ev = _events(
            [{"id": 7, "t": 0, "price": 100.0, "vol": 5.0, "action": "deleted"}]
        )
        assert queue_positions(ev, levels="all").empty

    def test_empty_events(self) -> None:
        ev = _events([])
        out = queue_positions(ev, levels="touch")
        assert out.empty
        assert list(out.columns) == list(QUEUE_COLUMNS)

    def test_invalid_levels_raises(self) -> None:
        with pytest.raises(ValueError, match="levels must be"):
            queue_positions(_events(SCENARIO), levels="nope")


class TestQueueAgeGrid:
    # A at t0, B joins at t10, A leaves at t20 (bid level 100).
    GRID_EVENTS = [
        {"id": 1, "t": 0, "price": 100.0, "vol": 10.0, "action": "created"},
        {"id": 2, "t": 10, "price": 100.0, "vol": 5.0, "action": "created"},
        {"id": 1, "t": 20, "price": 100.0, "vol": 10.0, "action": "deleted"},
    ]

    def test_shape_and_per_rank_age(self) -> None:
        ages, times, mr = queue_age_grid(
            _events(self.GRID_EVENTS), side="bid", n_time=5
        )
        assert mr == 2  # deepest the bid queue gets is 2
        assert ages.shape == (2, 5)
        assert len(times) == 5
        # samples at t=0,5,10,15,20. At t=10 both rest: front(A)=10s, back(B)=0s.
        assert ages[0, 2] == 10.0
        assert ages[1, 2] == 0.0
        # front order is always older than the one behind it.
        assert ages[0, 3] > ages[1, 3]

    def test_front_is_row_zero_nan_when_short(self) -> None:
        ages, _, _ = queue_age_grid(_events(self.GRID_EVENTS), side="bid", n_time=5)
        # at t=0 only A rests -> rank 2 (row 1) is empty.
        assert np.isfinite(ages[0, 0])
        assert np.isnan(ages[1, 0])

    def test_side_filter(self) -> None:
        # All-ask events -> the bid grid is empty.
        ev = _events(
            [
                {
                    "id": 1,
                    "t": 0,
                    "price": 100.0,
                    "vol": 5.0,
                    "dir": "ask",
                    "action": "created",
                }
            ]
        )
        ages, _, mr = queue_age_grid(ev, side="bid", n_time=5)
        assert mr == 0 and ages.size == 0

    def test_empty_events(self) -> None:
        ages, times, mr = queue_age_grid(_events([]), side="bid", n_time=5)
        assert mr == 0 and ages.size == 0 and len(times) == 0

    def test_invalid_side_raises(self) -> None:
        with pytest.raises(ValueError, match="side must be"):
            queue_age_grid(_events(self.GRID_EVENTS), side="nope")
