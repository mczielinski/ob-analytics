"""A price level must empty when the order resting on it goes away.

The level an order rests on is where its volume is added and where every later
reduction is taken off.  An order can change that level only one way: a
``changed`` row with no execution that carries a new price, which moves the
order (Databento's modify).  A row that reports an execution, or a delete,
never moves it, whatever price it carries.

Bitstamp reports a ``deleted`` whose price is not the price the order rested
at for a small share of orders.  ``price_level_volume`` adds an order's volume
at its ``created`` price, so it has to subtract it there too; subtracting at
the price the later row happens to carry leaves the volume on the created
level for the rest of the session, where every consumer reads it back as a
resting level that no order is on.

The per-order rebuild (``engine.book_state``) tracks orders by id and has
never had this problem, so these tests also pin the two rebuilds together.
"""

from typing import ClassVar

import numpy as np
import pandas as pd

from ob_analytics import engine
from ob_analytics._engine_frames import to_order_events
from ob_analytics.analytics import set_order_types
from ob_analytics.depth import price_level_volume

_BASE = pd.Timestamp("2026-01-05 10:00:00", tz="UTC")


def _events(rows: list[tuple]) -> pd.DataFrame:
    """Build a classified events frame.

    Each row is ``(event_id, id, t_seconds, price, volume, direction, action,
    fill)``.  Prices are integer ticks and volumes integer lots, per the schema.
    """
    ts = pd.Series(
        [_BASE + pd.Timedelta(seconds=r[2]) for r in rows], dtype="datetime64[ns, UTC]"
    )
    events = pd.DataFrame(
        {
            "event_id": np.array([r[0] for r in rows], dtype=np.int64),
            "id": np.array([r[1] for r in rows], dtype=np.int64),
            "timestamp": ts,
            "exchange_timestamp": ts.copy(),
            "price": np.array([r[3] for r in rows], dtype=np.int64),
            "volume": np.array([r[4] for r in rows], dtype=np.int64),
            "direction": pd.Categorical(
                [r[5] for r in rows], categories=["bid", "ask"], ordered=True
            ),
            "action": pd.Categorical(
                [r[6] for r in rows],
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "fill": np.array([r[7] for r in rows], dtype=np.int64),
        }
    )
    return set_order_types(
        events,
        pd.DataFrame(
            {
                "maker_event_id": pd.array([], dtype=object),
                "taker_event_id": pd.array([], dtype=object),
            }
        ),
    )


def _final_volume(depth: pd.DataFrame, price: int, direction: str) -> float:
    """Volume the level carries after the last row that touched it."""
    rows = depth[(depth["price"] == price) & (depth["direction"] == direction)]
    if rows.empty:
        return 0.0
    return float(rows.sort_values("timestamp", kind="stable")["volume"].iloc[-1])


def _level_at(depth: pd.DataFrame, price: int, direction: str, t: float) -> float:
    """Volume the level carries after every row up to *t* seconds."""
    at = _BASE + pd.Timedelta(seconds=t)
    rows = depth[
        (depth["price"] == price)
        & (depth["direction"] == direction)
        & (depth["timestamp"] <= at)
    ]
    if rows.empty:
        return 0.0
    return float(rows.sort_values("timestamp", kind="stable")["volume"].iloc[-1])


def _book_levels(events: pd.DataFrame, t: float) -> dict[tuple[str, int], float]:
    """Volume per ``(direction, price)`` in the per-order rebuild at *t*."""
    at = int((_BASE + pd.Timedelta(seconds=t)).value)
    book = engine.book_state(to_order_events(events, market=True), at=at)
    levels: dict[tuple[str, int], float] = {}
    for side in (book.bids, book.asks):
        for row in side.row:
            key = (str(events["direction"].iloc[row]), int(events["price"].iloc[row]))
            levels[key] = levels.get(key, 0.0) + float(events["volume"].iloc[row])
    return levels


def _depth_levels(depth: pd.DataFrame, t: float) -> dict[tuple[str, int], float]:
    """Volume per ``(direction, price)`` in the price-level rebuild at *t*."""
    at = _BASE + pd.Timedelta(seconds=t)
    last = (
        depth[depth["timestamp"] <= at]
        .sort_values("timestamp", kind="stable")
        .groupby(["direction", "price"], observed=True)["volume"]
        .last()
    )
    last = last[last > 0].reset_index()
    return {
        (str(d), int(p)): float(v)
        for d, p, v in zip(last["direction"], last["price"], last["volume"])
    }


def _assert_rebuilds_agree(rows: list[tuple]) -> None:
    """The two rebuilds report the same levels after every event."""
    events = _events(rows)
    depth = price_level_volume(events)
    for t in sorted({r[2] for r in rows}):
        assert _depth_levels(depth, t) == _book_levels(events, t), f"at t={t}"


class TestDeleteAtADifferentPrice:
    """The delete carries a price the order never rested at."""

    def test_created_level_empties(self):
        """The volume leaves the level it was added to, not the reported one."""
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "ask", "created", 0),
                    (2, 1, 1.0, 10_100, 500, "ask", "deleted", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_000, "ask") == 0.0

    def test_the_reported_level_is_untouched(self):
        """A level the order never rested on keeps the volume it really holds."""
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "ask", "created", 0),
                    (2, 2, 1.0, 10_100, 700, "ask", "created", 0),
                    (3, 1, 2.0, 10_100, 500, "ask", "deleted", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_000, "ask") == 0.0
        # Order 2 is still resting at 10_100; order 1's delete must not take
        # its volume, and the clip at zero must not hide the overdraw either.
        assert _final_volume(depth, 10_100, "ask") == 700.0

    def test_a_fill_reported_at_another_price(self):
        """An execution reported away from the resting price still reduces it."""
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "bid", "created", 0),
                    (2, 1, 1.0, 9_900, 300, "bid", "changed", 200),
                    (3, 1, 2.0, 9_900, 300, "bid", "deleted", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_000, "bid") == 0.0
        # An execution report does not move the order.
        assert _level_at(depth, 9_900, "bid", 1.0) == 0.0


class TestAnOrderThatMoves:
    """A ``changed`` row with no execution and a new price moves the order.

    Databento's modify can move an order to another price (issue #262).  The
    volume has to leave the old level and join the new one.
    """

    MOVED: ClassVar[list[tuple]] = [
        (1, 1, 0.0, 10_000, 500, "ask", "created", 0),
        (2, 1, 1.0, 10_100, 500, "ask", "changed", 0),
        (3, 1, 2.0, 10_100, 300, "ask", "changed", 200),
        (4, 1, 3.0, 10_100, 300, "ask", "deleted", 0),
    ]

    def test_the_old_level_empties_at_the_move(self):
        depth = price_level_volume(_events(self.MOVED))
        assert _level_at(depth, 10_000, "ask", 0.0) == 500.0
        assert _level_at(depth, 10_000, "ask", 1.0) == 0.0
        assert _final_volume(depth, 10_000, "ask") == 0.0

    def test_the_new_level_carries_the_order_until_it_leaves(self):
        depth = price_level_volume(_events(self.MOVED))
        assert _level_at(depth, 10_100, "ask", 0.0) == 0.0
        assert _level_at(depth, 10_100, "ask", 1.0) == 500.0
        assert _level_at(depth, 10_100, "ask", 2.0) == 300.0
        assert _final_volume(depth, 10_100, "ask") == 0.0

    def test_a_move_that_also_changes_the_size(self):
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "bid", "created", 0),
                    (2, 1, 1.0, 9_900, 400, "bid", "changed", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_000, "bid") == 0.0
        assert _final_volume(depth, 9_900, "bid") == 400.0

    def test_a_later_delete_at_the_original_price_leaves_the_new_level(self):
        # The delete reports a price the order no longer rests at; it is still
        # taken off where the order is.
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "ask", "created", 0),
                    (2, 2, 0.5, 10_000, 700, "ask", "created", 0),
                    (3, 1, 1.0, 10_100, 500, "ask", "changed", 0),
                    (4, 1, 2.0, 10_000, 500, "ask", "deleted", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_100, "ask") == 0.0
        assert _final_volume(depth, 10_000, "ask") == 700.0

    def test_a_move_back_and_forth(self):
        rows = [
            (1, 1, 0.0, 10_000, 500, "ask", "created", 0),
            (2, 1, 1.0, 10_100, 500, "ask", "changed", 0),
            (3, 1, 2.0, 10_000, 500, "ask", "changed", 0),
        ]
        depth = price_level_volume(_events(rows))
        assert _final_volume(depth, 10_000, "ask") == 500.0
        assert _final_volume(depth, 10_100, "ask") == 0.0

    def test_the_rebuilds_agree(self):
        _assert_rebuilds_agree(self.MOVED)


class TestARowAfterTheDelete:
    """A stray ``changed`` row after an order's delete does not bring it back."""

    def test_a_new_price_after_the_delete_is_not_a_move(self):
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "ask", "created", 0),
                    (2, 2, 0.5, 10_000, 700, "ask", "created", 0),
                    (3, 1, 1.0, 10_000, 500, "ask", "deleted", 0),
                    (4, 1, 2.0, 10_100, 500, "ask", "changed", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_000, "ask") == 700.0
        assert _final_volume(depth, 10_100, "ask") == 0.0

    def test_a_larger_size_after_the_delete_is_not_growth(self):
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "ask", "created", 0),
                    (2, 2, 0.5, 10_000, 700, "ask", "created", 0),
                    (3, 1, 1.0, 10_000, 500, "ask", "deleted", 0),
                    (4, 1, 2.0, 10_000, 900, "ask", "changed", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_000, "ask") == 700.0


class TestAnOrderThatGrows:
    """A ``changed`` row with no execution and a larger size adds the difference."""

    GROWN: ClassVar[list[tuple]] = [
        (1, 1, 0.0, 10_000, 500, "bid", "created", 0),
        (2, 1, 1.0, 10_000, 800, "bid", "changed", 0),
        (3, 1, 2.0, 10_000, 800, "bid", "deleted", 0),
    ]

    def test_the_added_size_is_counted(self):
        depth = price_level_volume(_events(self.GROWN))
        assert _level_at(depth, 10_000, "bid", 1.0) == 800.0

    def test_the_level_empties_when_it_leaves(self):
        depth = price_level_volume(_events(self.GROWN))
        assert _final_volume(depth, 10_000, "bid") == 0.0

    def test_growth_after_a_move_lands_on_the_new_level(self):
        rows = [
            (1, 1, 0.0, 10_000, 500, "bid", "created", 0),
            (2, 1, 1.0, 9_900, 500, "bid", "changed", 0),
            (3, 1, 2.0, 9_900, 900, "bid", "changed", 0),
        ]
        depth = price_level_volume(_events(rows))
        assert _final_volume(depth, 10_000, "bid") == 0.0
        assert _final_volume(depth, 9_900, "bid") == 900.0
        _assert_rebuilds_agree(rows)

    def test_the_rebuilds_agree(self):
        _assert_rebuilds_agree(self.GROWN)


class TestUnaffectedCases:
    """A venue that reports one price per order is unchanged."""

    def test_matching_prices_still_empty_the_level(self):
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "ask", "created", 0),
                    (2, 1, 1.0, 10_000, 500, "ask", "deleted", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_000, "ask") == 0.0

    def test_a_resting_order_keeps_its_volume(self):
        depth = price_level_volume(
            _events([(1, 1, 0.0, 10_000, 500, "ask", "created", 0)])
        )
        assert _final_volume(depth, 10_000, "ask") == 500.0


class TestAgainstTheBundledSample:
    """No level may hold volume that no order is resting on."""

    def test_no_phantom_levels_on_the_bitstamp_sample(self, bitstamp_sample_dir):
        from ob_analytics.bitstamp import BitstampSource
        from ob_analytics.pipeline import Pipeline

        result = Pipeline(source=BitstampSource()).run(
            str(bitstamp_sample_dir / "orders.csv.gz")
        )
        events, depth = result.events, result.depth

        # Read both books at the last real event, before the capture's
        # end-of-run synthetic deletes clear everything.
        end = events["timestamp"].max()
        levels = (
            depth[depth["timestamp"] < end]
            .sort_values("timestamp", kind="stable")
            .groupby(["direction", "price"], observed=True)["volume"]
            .last()
        )
        levels = levels[levels > 0]

        latest = (
            events[events["timestamp"] < end]
            .sort_values(["timestamp", "event_id"], kind="stable")
            .groupby("id")
            .tail(1)
        )
        resting = latest[
            (latest["action"] != "deleted")
            & (latest["volume"] > 0)
            & (~latest["type"].isin(["market", "market-limit"]))
        ]
        occupied = resting.groupby(["direction", "price"], observed=True)[
            "volume"
        ].sum()

        phantom = levels.index.difference(occupied.index)
        assert len(phantom) == 0, (
            f"{len(phantom)} price level(s) hold volume with no order resting "
            f"on them: {list(phantom)[:5]}"
        )
