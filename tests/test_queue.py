"""Tests for the FIFO queue-position engine (WS-4.1)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from ob_analytics.exceptions import ConfigError
from ob_analytics.queue import QUEUE_COLUMNS, queue_age_grid, queue_positions

TS = pd.Timestamp("2015-05-01 00:00:00")


def _events(rows: list[dict]) -> pd.DataFrame:
    """Build a classified events frame from compact (id, t, price, vol, dir, action).

    ``fill`` defaults to 0 and ``type`` to ``resting-limit``.
    """
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
                "fill": r.get("fill", 0),
                "type": r.get("type", "resting-limit"),
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
            "fill",
            "type",
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

    def test_empty_events_name_l3(self) -> None:
        # No events is what an L2 run has: an empty queue would read as "no
        # orders queued", so the function says it needs L3 instead.
        with pytest.raises(ConfigError, match=r"needs per-order \(L3\) events"):
            queue_positions(_events([]), levels="touch")

    def test_invalid_levels_raises(self) -> None:
        with pytest.raises(ValueError, match="levels must be"):
            queue_positions(_events(SCENARIO), levels="nope")


class TestQueueAgeGrid:
    # A at t0, B joins at t10, A leaves at t20 (bid level 100).
    GRID_EVENTS: ClassVar[list[dict]] = [
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

    def test_empty_events_name_l3(self) -> None:
        with pytest.raises(ConfigError, match=r"needs per-order \(L3\) events"):
            queue_age_grid(_events([]), side="bid", n_time=5)

    def test_invalid_side_raises(self) -> None:
        with pytest.raises(ValueError, match="side must be"):
            queue_age_grid(_events(self.GRID_EVENTS), side="nope")

    def test_the_composition_of_a_window_with_no_events_is_empty(self) -> None:
        # A zoom window can hold no events on an L3 run; that is an empty
        # grid, not the L2 refusal.
        from ob_analytics.visualization import prepare

        ev = _events(self.GRID_EVENTS)
        later = ev["timestamp"].max() + pd.Timedelta("1h")
        data = prepare.liquidity_at_touch(
            ev, start_time=later, end_time=later + pd.Timedelta("1s"), level="L3"
        )
        assert data["max_rank"] == 0 and data["ages"].size == 0


class TestLiveLevels:
    """The heap of live levels finds the same touch a scan of every level does.

    The touch used to be ``max``/``min`` over all live levels on every event,
    which made the call slow on a real day (#489); the heap must not change
    which level it reports.
    """

    @pytest.mark.parametrize("side", ["bid", "ask"])
    @pytest.mark.parametrize("prices", [range(5), range(400)])
    def test_best_matches_a_full_scan(self, side: str, prices: range) -> None:
        from ob_analytics.engine import Direction
        from ob_analytics.engine._queue import _LiveLevels

        direction = Direction[side.upper()]
        levels = _LiveLevels(direction)
        live: set[int] = set()
        rng = np.random.default_rng(7)
        # Enough churn to leave stale heap entries behind and rebuild the heap
        # several times, on a narrow and on a wide range of prices.
        for _ in range(20_000):
            price = int(rng.choice(prices))
            if price in live and rng.random() < 0.6:
                levels.discard(price)
                live.discard(price)
            else:
                levels.add(price)
                live.add(price)
            expected = None
            if live:
                expected = max(live) if direction == Direction.BID else min(live)
            assert levels.best() == expected


class TestArrivalQueue:
    """A level's rank and volume ahead match a walk of the queue (#520).

    ``emit`` used to walk each order's level to find these; the Fenwick trees
    must report the same numbers after every kind of change to the queue.
    """

    @staticmethod
    def _walk(queue: dict[int, float], oid: int) -> tuple[int, float, float]:
        rank, ahead = 1, 0
        for other_id, size in queue.items():
            if other_id == oid:
                return rank, ahead, size
            rank += 1
            ahead += size
        raise KeyError(oid)

    @pytest.mark.parametrize("tree_from", [2, 32])
    def test_adds_fills_cancels(
        self, tree_from: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ob_analytics.engine import _queue

        monkeypatch.setattr(_queue, "_TREE_FROM", tree_from)
        q = _queue._ArrivalQueue(exact=True)
        for oid, size in [(1, 5), (2, 7), (3, 11), (4, 13)]:
            q.append(oid, size)
        assert [q.position(oid) for oid in (1, 2, 3, 4)] == [
            (1, 0, 5),
            (2, 5, 7),
            (3, 12, 11),
            (4, 23, 13),
        ]
        q.resize(2, 3)  # a partial fill keeps the place
        assert q.position(2) == (2, 5, 3)
        assert q.position(4) == (4, 19, 13)
        q.remove(1)  # the front order leaves
        assert q.position(2) == (1, 0, 3)
        assert q.position(4) == (3, 14, 13)
        q.remove(3)  # a cancel in the middle
        assert q.position(4) == (2, 3, 13)
        q.append(1, 2)  # back again: to the back of the queue
        assert q.position(1) == (3, 16, 2)
        assert len(q) == 3

    def test_long_queue_matches_a_walk(self) -> None:
        from ob_analytics.engine import _queue

        q = _queue._ArrivalQueue(exact=True)
        reference: dict[int, float] = {}
        rng = np.random.default_rng(11)
        next_id = 0
        # Grow past the length that builds the trees, churn long enough to run
        # out of slots several times, then drain below the length that drops
        # them.
        step = 0
        while step < 25_000 or reference:
            step += 1
            draining = step >= 25_000
            grow = not draining and (len(reference) < 200 or rng.random() < 0.5)
            if grow:
                next_id += 1
                size = int(rng.integers(1, 1_000_000))
                q.append(next_id, size)
                reference[next_id] = size
            elif reference:
                oid = list(reference)[int(rng.integers(len(reference)))]
                if not draining and rng.random() < 0.4:
                    size = int(rng.integers(1, int(reference[oid]) + 1))
                    q.resize(oid, size)
                    reference[oid] = size
                else:
                    q.remove(oid)
                    del reference[oid]
            if reference and step % 7 == 0:
                for oid in list(reference)[:: max(1, len(reference) // 5)]:
                    assert q.position(oid) == self._walk(reference, oid)
            assert len(q) == len(reference)
            if step == 25_000:
                assert q._slot is not None, "the long queue should use the trees"
        assert not reference and len(q) == 0
        assert q._slot is None, "the drained queue should walk again"

    def test_float_sizes_are_walked(self) -> None:
        from ob_analytics.engine import _queue

        q = _queue._ArrivalQueue(exact=False)
        reference: dict[int, float] = {}
        for oid in range(100):
            q.append(oid, 0.1 * (oid + 1))
            reference[oid] = 0.1 * (oid + 1)
        for oid in range(0, 100, 3):
            q.remove(oid)
            del reference[oid]
        # Summed in queue order, so the front order has exactly nothing ahead.
        front = next(iter(reference))
        assert q.position(front)[:2] == (1, 0)
        for oid in reference:
            assert q.position(oid) == self._walk(reference, oid)

    @pytest.mark.parametrize("volume_dtype", ["int64", "float64"])
    def test_queue_positions_on_a_long_level(self, volume_dtype: str) -> None:
        """A 40-order level reports what a walk reports.

        Integer sizes are long enough to use the trees; float sizes are walked.
        """
        sizes = {
            oid: 0.1 * oid if volume_dtype == "float64" else oid for oid in range(1, 41)
        }
        rows = [
            {"id": oid, "t": oid, "price": 100, "vol": size, "action": "created"}
            for oid, size in sizes.items()
        ]
        # Cancel every third order, then touch the rest with no size change.
        t = len(rows)
        cancelled = set(range(1, 41, 3))
        for oid in sorted(cancelled):
            t += 1
            rows.append(
                {
                    "id": oid,
                    "t": t,
                    "price": 100,
                    "vol": sizes[oid],
                    "action": "deleted",
                }
            )
        resting = {oid: sizes[oid] for oid in sizes if oid not in cancelled}
        for oid, size in resting.items():
            t += 1
            rows.append(
                {"id": oid, "t": t, "price": 100, "vol": size, "action": "changed"}
            )
        ev = _events(rows).astype({"volume": volume_dtype})

        out = queue_positions(ev, levels="all")

        touched = out[out["action"] == "changed"].reset_index(drop=True)
        assert touched["id"].tolist() == list(resting)
        expected = [self._walk(resting, oid) for oid in resting]
        assert touched["rank"].tolist() == [rank for rank, _, _ in expected]
        assert touched["ahead_volume"].tolist() == [ahead for _, ahead, _ in expected]
        assert touched["ahead_volume"].iloc[0] == 0
