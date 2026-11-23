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
        grid = queue_age_grid(_events(self.GRID_EVENTS), side="bid", n_time=5)
        ages = grid.to_numpy()
        assert len(grid) == 2  # deepest the bid queue gets is 2
        assert grid.index.name == "rank" and grid.index.tolist() == [1, 2]
        assert ages.shape == (2, 5)
        assert len(grid.columns) == 5
        # samples at t=0,5,10,15,20. At t=10 both rest: front(A)=10s, back(B)=0s.
        assert ages[0, 2] == 10.0
        assert ages[1, 2] == 0.0
        # front order is always older than the one behind it.
        assert ages[0, 3] > ages[1, 3]

    def test_front_is_row_zero_nan_when_short(self) -> None:
        grid = queue_age_grid(_events(self.GRID_EVENTS), side="bid", n_time=5)
        ages = grid.to_numpy()
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
        grid = queue_age_grid(ev, side="bid", n_time=5)
        ages = grid.to_numpy()
        mr = len(grid)
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


def _reference_positions(events, *, touch_only: bool) -> dict[str, np.ndarray]:
    """Queue positions from a plain Python model of the FIFO rules.

    Kept as simple as it can be: a dictionary per level in arrival order,
    walked from the front, and the touch from ``max`` or ``min`` over the
    non-empty levels.  Ages are truncated to whole microseconds, with the
    whole seconds added after the division, as the golden tests pin them.
    The compiled engine must give the same arrays, bit for bit.
    """
    from ob_analytics.engine import Action

    visible = events.visible()
    # Prices and sizes are ints or floats, as the stream carries them.
    queues: dict[tuple[int, float], dict[int, float]] = {}
    level_of: dict[int, tuple[int, float]] = {}
    placed: dict[int, int] = {}
    out: dict[str, list] = {
        name: []
        for name in (
            "row",
            "price",
            "action",
            "rank",
            "queue_len",
            "ahead_volume",
            "remaining",
            "age_s",
        )
    }

    def emit(row: int, when: int, oid: int, level, action: int) -> None:
        direction, price = level
        if touch_only:
            live = [p for (d, p), q in queues.items() if d == direction and q]
            if price != (max(live) if direction == 0 else min(live)):
                return
        queue = queues[level]
        rank = 1
        ahead: float = 0
        for other, size in queue.items():
            if other == oid:
                break
            rank += 1
            ahead += size
        seconds, remainder = divmod(when - placed[oid], 1_000_000_000)
        age = seconds + (remainder // 1000) / 1e6
        for name, value in zip(
            out,
            (row, price, action, rank, len(queue), ahead, queue[oid], age),
            strict=True,
        ):
            out[name].append(value)

    def leave(oid: int, level) -> None:
        del level_of[oid]
        del queues[level][oid]

    def join(oid: int, level, volume) -> None:
        level_of[oid] = level
        queues.setdefault(level, {})[oid] = volume

    for row in visible.tolist():
        oid = int(events.order_id[row])
        when = int(events.timestamp[row])
        price = events.rests_at()[row].item()
        volume = events.volume[row].item()
        action = int(events.action[row])
        level = level_of.get(oid)
        if action == Action.CREATED:
            if level is not None:
                emit(row, when, oid, level, Action.DELETED)
                leave(oid, level)
            placed[oid] = when
            new_level = (int(events.direction[row]), price)
            join(oid, new_level, volume)
            emit(row, when, oid, new_level, Action.CREATED)
        elif level is None:
            continue
        elif action == Action.DELETED or volume <= 0:
            emit(row, when, oid, level, Action.DELETED)
            leave(oid, level)
        elif (level[0], price) != level:
            emit(row, when, oid, level, Action.DELETED)
            leave(oid, level)
            join(oid, (level[0], price), volume)
            emit(row, when, oid, (level[0], price), Action.CREATED)
        else:
            queues[level][oid] = volume
            emit(row, when, oid, level, Action.CHANGED)

    return {
        "row": np.array(out["row"], dtype=np.int64),
        "price": np.asarray(out["price"], dtype=events.rests_at().dtype),
        "action": np.array(out["action"], dtype=np.int8),
        "rank": np.array(out["rank"], dtype=np.int64),
        "queue_len": np.array(out["queue_len"], dtype=np.int64),
        "ahead_volume": np.asarray(out["ahead_volume"], dtype=events.volume.dtype),
        "remaining": np.asarray(out["remaining"], dtype=events.volume.dtype),
        "age_s": np.array(out["age_s"], dtype=np.float64),
    }


def _random_stream(seed: int, n: int):
    """A random event stream that reaches every rule of the queue engine.

    Orders are placed again while queued, moved, resized, resized to zero,
    deleted, and changed after they are gone.  Times tie and run backwards,
    so some ages are negative.  Some streams have float sizes, float prices,
    hidden and market orders, or resting prices that differ from the row's.
    """
    from ob_analytics.engine import OrderEvents

    rng = np.random.default_rng(seed)
    order_id = rng.integers(1, max(n // 4, 2) + 1, n).astype(np.int64)
    timestamp = rng.integers(0, 10**12, n).astype(np.int64)
    timestamp[: n * 4 // 5] = np.sort(timestamp[: n * 4 // 5])
    price = rng.integers(95, 106, n).astype(np.int64)
    if seed % 5 == 2:
        price = price / 7.0
    volume = rng.integers(0, 50, n).astype(np.int64)
    if seed % 3 == 1:
        volume = rng.random(n) * rng.choice([1e-8, 1.0, 3e5], n)
        volume[rng.random(n) < 0.1] = 0.0
    extra = {}
    if seed % 2:
        extra["is_hidden"] = rng.random(n) < 0.05
    if seed % 4 < 2:
        extra["is_market"] = rng.random(n) < 0.05
    if seed % 7 < 3:
        extra["resting_price"] = np.where(rng.random(n) < 0.8, price, price + 1)
    return OrderEvents(
        order_id=order_id,
        timestamp=timestamp,
        price=price,
        volume=volume,
        direction=rng.integers(0, 2, n).astype(np.int8),
        action=rng.choice([0, 0, 1, 1, 2], n).astype(np.int8),
        **extra,
    )


class TestCompiledReplay:
    """The compiled queue engine gives what a plain model of the rules gives."""

    @staticmethod
    def _walk(queue: dict[int, float], oid: int) -> tuple[int, float, float]:
        rank, ahead = 1, 0
        for other_id, size in queue.items():
            if other_id == oid:
                return rank, ahead, size
            rank += 1
            ahead += size
        raise KeyError(oid)

    @pytest.mark.parametrize("touch_only", [True, False])
    @pytest.mark.parametrize("seed", range(60))
    def test_matches_the_plain_model_bit_for_bit(
        self, seed: int, touch_only: bool
    ) -> None:
        from ob_analytics import engine

        events = _random_stream(seed, [0, 1, 7, 60, 400][seed % 5])
        got = engine.queue_positions(events, touch_only=touch_only)
        expected = _reference_positions(events, touch_only=touch_only)
        for name, want in expected.items():
            have = getattr(got, name)
            assert have.dtype == want.dtype, name
            if want.dtype.kind == "f":
                # Compare the bits, so -0.0 and NaN are checked too.
                have, want = have.view(np.int64), want.view(np.int64)
            np.testing.assert_array_equal(have, want, err_msg=name)

    @pytest.mark.parametrize(
        ("column", "value", "message"),
        [
            ("direction", 2, "BID or ASK"),
            ("timestamp", np.iinfo(np.int64).min, "timestamp"),
            ("resting_price", np.nan, "resting price"),
        ],
    )
    def test_rejects_rows_it_cannot_queue(
        self, column: str, value: float, message: str
    ) -> None:
        # The compiled loop does not check its array bounds, so a bad code has
        # to be stopped before it runs.
        import dataclasses

        from ob_analytics import engine

        events = _random_stream(2, 60)  # float prices, with resting prices
        bad = np.array(getattr(events, column))
        bad[events.visible()[0]] = value
        with pytest.raises(ValueError, match=message):
            engine.queue_positions(dataclasses.replace(events, **{column: bad}))

    def test_compiles_without_the_disk_cache_when_it_cannot_write_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # numba raises RuntimeError at decoration when no cache folder can be
        # written, as in a read-only install with a read-only home folder.
        from ob_analytics.engine import _queue_replay

        def njit(*args, **kwargs):
            if kwargs.get("cache"):
                raise RuntimeError("cannot cache function: no locator available")
            (func,) = args
            return func

        monkeypatch.setattr(_queue_replay, "njit", njit)
        assert _queue_replay._compiled(lambda x: x + 1)(1) == 2

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
