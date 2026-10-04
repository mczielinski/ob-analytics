"""Coverage for the hand-written toy session in ``ob_analytics.datasets``.

The toy stream is the tutorial's teaching instrument and doubles as a
readable fixture, so these tests pin down everything the docs will claim
about it: loader-parity layout, canonical volume/fill semantics, the
exact order classification per actor, trade↔event consistency, the
scripted t=30 book, and that the pipeline stages and plot faces accept
frames this small.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ob_analytics.analytics import order_book, order_lifecycles, set_order_types
from ob_analytics.datasets import (
    ToyOrder,
    match_toy_orders,
    toy_events,
    toy_orders,
    toy_trades,
)
from ob_analytics.depth import depth_metrics, price_level_volume
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import Level
from ob_analytics.schemas import (
    validate_depth_df,
    validate_events_df,
    validate_trades_df,
)
from ob_analytics.visualization import plot, prepare

EXPECTED_CLASSES = {
    "Alice": "resting-limit",
    "Bob": "resting-limit",
    "Chen": "resting-limit",
    "Ivy": "resting-limit",
    "Erin": "resting-limit",
    "Gus": "resting-limit",
    "Frank": "market",
    "Iris": "market",
    "Sam": "market",
    "Hana": "market-limit",
    "Dana": "flashed-limit",
    "Eve": "flashed-limit",
}


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return toy_events()


@pytest.fixture(scope="module")
def trades() -> pd.DataFrame:
    return toy_trades()


@pytest.fixture(scope="module")
def classified(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    return set_order_types(events, trades)


class TestLayout:
    def test_loader_parity_columns_and_dtypes(self, events: pd.DataFrame) -> None:
        """Columns and dtypes match BitstampLoader output (plus ``actor``)."""
        assert list(events.columns) == [
            "original_number",
            "id",
            "timestamp",
            "exchange_timestamp",
            "price",
            "volume",
            "action",
            "direction",
            "event_id",
            "fill",
            "raw_event_type",
            "actor",
        ]
        assert events["id"].dtype == "int64"
        assert events["event_id"].dtype == "int64"
        assert str(events["timestamp"].dtype) == "datetime64[ns, UTC]"
        assert str(events["exchange_timestamp"].dtype) == "datetime64[ns, UTC]"
        assert events["price"].dtype == "int64"
        # Integer lots, the size counterpart of the integer-tick price above
        # (issue #226).
        assert events["volume"].dtype == "int64"
        assert events["fill"].dtype == "int64"
        assert events["action"].dtype == "category"
        assert list(events["action"].cat.categories) == [
            "created",
            "changed",
            "deleted",
        ]
        assert events["direction"].dtype == "category"
        assert list(events["direction"].cat.categories) == ["bid", "ask"]

    def test_shape_and_determinism(self, events: pd.DataFrame) -> None:
        assert len(events) == 24
        assert events["id"].nunique() == 12
        assert (events["event_id"] == range(1, 25)).all()
        pd.testing.assert_frame_equal(toy_events(), toy_events())

    def test_trades_layout(self, trades: pd.DataFrame) -> None:
        assert len(trades) == 5
        assert list(trades["direction"].cat.categories) == ["buy", "sell"]
        assert trades["timestamp"].is_monotonic_increasing


class TestCanonicalSemantics:
    def test_fill_equals_abs_volume_diff_per_order(self, events: pd.DataFrame) -> None:
        """The loader-invariant the whole pipeline assumes (§1.1)."""
        by_id = events.sort_values(["id", "event_id"])
        diffs = by_id.groupby("id")["volume"].diff().abs().fillna(0.0)
        assert (diffs.round(8) == by_id["fill"].round(8)).all()

    def test_remaining_volume_non_negative_and_monotone_under_fills(
        self, events: pd.DataFrame
    ) -> None:
        assert (events["volume"] >= 0).all()
        filled = events[events["fill"] > 0]
        by_id = events.sort_values(["id", "event_id"])
        for oid in filled["id"].unique():
            vols = by_id.loc[by_id["id"] == oid, "volume"].to_numpy()
            assert (vols[1:] <= vols[:-1]).all(), f"order {oid} volume increased"

    def test_full_fill_vs_cancel_deleted_row_shape(self, events: pd.DataFrame) -> None:
        """Full fills end deleted(volume=0, fill>0); cancels deleted(volume>0, fill=0)."""
        deleted = events[events["action"] == "deleted"]
        fills = deleted[deleted["fill"] > 0]
        cancels = deleted[deleted["fill"] == 0]
        assert (fills["volume"] == 0).all()
        assert (cancels["volume"] > 0).all()
        assert set(cancels["actor"]) == {"Dana", "Eve"}


class TestClassification:
    def test_every_actor_gets_expected_class(self, classified: pd.DataFrame) -> None:
        per_actor = (
            classified.groupby("actor", observed=True)["type"]
            .first()
            .astype(str)
            .to_dict()
        )
        assert per_actor == EXPECTED_CLASSES

    def test_no_unknown_or_pre_existing(self, classified: pd.DataFrame) -> None:
        assert (classified["type"] == "unknown").sum() == 0
        assert (classified["type"] == "pre-existing").sum() == 0

    def test_all_four_teaching_classes_present(self, classified: pd.DataFrame) -> None:
        present = set(classified["type"].astype(str))
        assert {"resting-limit", "market", "market-limit", "flashed-limit"} <= present


class TestContracts:
    def test_events_contract_after_classification(
        self, classified: pd.DataFrame
    ) -> None:
        validate_events_df(classified)

    def test_trades_contract(self, trades: pd.DataFrame) -> None:
        validate_trades_df(trades)

    def test_lifecycles_partition_all_orders(self, classified: pd.DataFrame) -> None:
        lc = order_lifecycles(classified)
        assert set(lc["id"]) == set(classified["id"])


class TestTradeEventConsistency:
    def test_maker_taker_events_carry_the_trade_volume(
        self, events: pd.DataFrame, trades: pd.DataFrame
    ) -> None:
        fill_by_eid = dict(zip(events["event_id"], events["fill"]))
        for row in trades.itertuples():
            assert fill_by_eid[row.maker_event_id] == row.volume
            assert fill_by_eid[row.taker_event_id] == row.volume

    def test_maker_taker_sides_oppose(
        self, events: pd.DataFrame, trades: pd.DataFrame
    ) -> None:
        side_by_eid = dict(zip(events["event_id"], events["direction"].astype(str)))
        for row in trades.itertuples():
            assert side_by_eid[row.maker_event_id] != side_by_eid[row.taker_event_id]

    def test_total_executed_volume_matches_taker_fill_tally(
        self, events: pd.DataFrame, trades: pd.DataFrame
    ) -> None:
        taker_eids = set(trades["taker_event_id"])
        taker_fills = events.loc[events["event_id"].isin(taker_eids), "fill"].sum()
        assert taker_fills == trades["volume"].sum() == 7.0


class TestBookAndDepth:
    def test_scripted_book_at_t30(self, classified: pd.DataFrame) -> None:
        tp = classified["timestamp"].iloc[0] + pd.Timedelta(seconds=30)
        snap = order_book(classified, tp=tp)
        bids, asks = snap["bids"], snap["asks"]
        assert bids["price"].max() == 99.0  # best bid
        assert asks["price"].min() == 101.0  # best ask — spread 2, mid 100
        assert bids.groupby("price")["volume"].sum().to_dict() == {98.0: 4.0, 99.0: 4.0}
        assert asks.groupby("price")["volume"].sum().to_dict() == {
            101.0: 2.0,
            102.0: 2.0,
        }
        # price-time priority: Alice (id 1) queued ahead of Ivy (id 4) at 99
        at_99 = bids[bids["price"] == 99.0]
        assert list(at_99["id"]) == [1, 4]

    def test_book_never_crossed_at_second_boundaries(
        self, classified: pd.DataFrame
    ) -> None:
        t0 = classified["timestamp"].iloc[0]
        for s in range(1, 61, 7):
            snap = order_book(classified, tp=t0 + pd.Timedelta(seconds=s))
            if len(snap["bids"]) and len(snap["asks"]):
                assert snap["bids"]["price"].max() < snap["asks"]["price"].min()

    def test_depth_pipeline_runs(self, classified: pd.DataFrame) -> None:
        depth = price_level_volume(classified)
        validate_depth_df(depth)
        summary = depth_metrics(depth)
        assert len(depth) > 0
        assert len(summary) > 0


class TestFacesRenderAtToyScale:
    """The tutorial renders these faces on the toy stream — they must accept N≈24."""

    def test_trade_tape_l2(self, trades: pd.DataFrame) -> None:
        fig = plot("trade_tape", level=Level.L2, **prepare.trades(trades))
        assert fig.axes
        plt.close(fig)

    def test_book_snapshot_l2_and_l3(self, classified: pd.DataFrame) -> None:
        tp = classified["timestamp"].iloc[0] + pd.Timedelta(seconds=30)
        snap = order_book(classified, tp=tp)
        for level, per_order in ((Level.L2, False), (Level.L3, True)):
            payload = prepare.book_snapshot(snap, per_order=per_order)
            fig = plot("book_snapshot", level=level, **payload)
            assert fig.axes
            plt.close(fig)

    def test_l3_payload_segments_the_99_queue(self, classified: pd.DataFrame) -> None:
        """Alice's and Ivy's orders at 99 stack as separate segments (0-2, 2-4)."""
        tp = classified["timestamp"].iloc[0] + pd.Timedelta(seconds=30)
        payload = prepare.book_snapshot(order_book(classified, tp=tp), per_order=True)
        at_99 = payload["bids"][payload["bids"]["price"] == 99.0]
        assert len(at_99) == 2
        segs = sorted(zip(at_99["seg_lo"], at_99["seg_hi"]))
        assert segs == [(0.0, 2.0), (2.0, 4.0)]


class TestMatchToyOrders:
    """The editable toy book (issue #121): the script matched by price-time priority."""

    @staticmethod
    def _sam_sweep_at_57(events: pd.DataFrame, trades: pd.DataFrame):
        """Move the second fill of Sam's sweep to t=57, as the toy session has it."""
        events, trades = events.copy(), trades.copy()
        later = events["timestamp"].iloc[0] + pd.Timedelta(seconds=57)
        events.loc[events["event_id"].isin([23, 24]), "timestamp"] = later
        events.loc[events["event_id"].isin([23, 24]), "exchange_timestamp"] = later
        trades.loc[trades.index[-1], "timestamp"] = later
        return events, trades

    def test_the_toy_script_rebuilds_the_toy_session(self) -> None:
        events, trades = match_toy_orders(toy_orders())
        # The one documented difference: a matching engine fills all of
        # Sam's sell at t=56.
        assert (
            events.loc[events["event_id"].isin([23, 24]), "timestamp"].dt.second == 56
        ).all()
        events, trades = self._sam_sweep_at_57(events, trades)
        pd.testing.assert_frame_equal(events, toy_events())
        pd.testing.assert_frame_equal(trades, toy_trades())

    def test_edited_session_is_canonical(self) -> None:
        orders = toy_orders()
        orders["Jo"] = ToyOrder(at=10, direction="bid", price=100, volume=2)
        del orders["Dana"]
        events, trades = match_toy_orders(orders)
        validate_events_df(set_order_types(events, trades))
        validate_trades_df(trades)
        # Every fill is a drop in the order's outstanding volume, and each
        # trade is two fills: the maker's and the taker's.
        fills = events[events["fill"] > 0].sort_values("event_id")
        by_id = events.sort_values(["id", "event_id"])
        drops = by_id.groupby("id")["volume"].diff().abs()
        assert (drops[fills.index] == fills["fill"]).all()
        assert trades["volume"].sum() * 2 == events["fill"].sum()

    def test_a_new_actor_gets_the_next_order_id(self) -> None:
        orders = toy_orders()
        orders["Jo"] = ToyOrder(at=10, direction="bid", price=97, volume=1)
        events, _ = match_toy_orders(orders)
        ids = events.groupby("actor")["id"].first()
        assert ids["Jo"] == 13
        assert ids["Alice"] == 1 and ids["Sam"] == 12

    def test_a_better_price_goes_ahead_of_an_earlier_order(self) -> None:
        orders = toy_orders()
        orders["Jo"] = ToyOrder(at=10, direction="bid", price=100, volume=1)
        _, trades = match_toy_orders(orders)
        sweep = trades[trades["taker_actor"] == "Sam"]
        assert list(sweep["maker_actor"]) == ["Jo", "Alice"]
        assert list(sweep["price"]) == [100, 99]

    def test_an_earlier_order_goes_first_at_one_price(self) -> None:
        orders = toy_orders()
        orders["Jo"] = ToyOrder(at=10, direction="bid", price=99, volume=1)
        _, trades = match_toy_orders(orders)
        sweep = trades[trades["taker_actor"] == "Sam"]
        assert list(sweep["maker_actor"]) == ["Alice", "Ivy"]

    def test_unfilled_remainder_rests_at_the_limit_price(self) -> None:
        orders = {
            "Ann": ToyOrder(at=0, direction="ask", price=101, volume=1),
            "Ben": ToyOrder(at=1, direction="bid", price=102, volume=3),
        }
        events, trades = match_toy_orders(orders)
        assert list(trades["price"]) == [101]
        ben = events[events["actor"] == "Ben"]
        assert list(ben["action"]) == ["created", "changed"]
        assert ben["volume"].iloc[-1] == 2 and ben["price"].iloc[-1] == 102

    def test_cancel_removes_what_still_rests(self) -> None:
        orders = toy_orders()
        orders["Alice"] = replace(orders["Alice"], cancel_at=30.0)
        events, trades = match_toy_orders(orders)
        alice = events[events["actor"] == "Alice"]
        assert list(alice["action"]) == ["created", "deleted"]
        assert alice["volume"].iloc[-1] == 2 and alice["fill"].iloc[-1] == 0
        assert "Alice" not in set(trades["maker_actor"])

    def test_cancel_after_a_full_fill_does_nothing(self) -> None:
        orders = toy_orders()
        orders["Alice"] = replace(orders["Alice"], cancel_at=58.0)
        events, _ = match_toy_orders(orders)
        alice = events[events["actor"] == "Alice"]
        assert list(alice["action"]) == ["created", "deleted"]
        assert alice["fill"].iloc[-1] == 2

    def test_arrivals_come_before_cancellations_at_one_time(self) -> None:
        # Ann is listed first, but Ben's arrival at t=10 still comes before
        # Ann's cancellation at t=10, so they trade.
        orders = {
            "Ann": ToyOrder(at=0, direction="ask", price=101, volume=1, cancel_at=10),
            "Ben": ToyOrder(at=10, direction="bid", price=101, volume=1),
        }
        events, trades = match_toy_orders(orders)
        assert list(trades["maker_actor"]) == ["Ann"]
        assert list(events["action"]) == ["created", "created", "deleted", "deleted"]

    def test_arrivals_at_one_time_follow_the_script_order(self) -> None:
        # Two bids arrive at once; the one listed first is first in the queue.
        orders = {
            "Cy": ToyOrder(at=1, direction="bid", price=99, volume=1),
            "Di": ToyOrder(at=1, direction="bid", price=99, volume=1),
            "Ed": ToyOrder(at=2, direction="ask", price=99, volume=1),
        }
        _, trades = match_toy_orders(orders)
        assert list(trades["maker_actor"]) == ["Cy"]

    def test_toy_orders_returns_a_fresh_copy(self) -> None:
        orders = toy_orders()
        del orders["Alice"]
        assert "Alice" in toy_orders()

    @pytest.mark.parametrize(
        ("order", "match"),
        [
            (ToyOrder(0, "buy", 100, 1), "direction"),  # ty: ignore[invalid-argument-type]
            (ToyOrder(0, "bid", 100.5, 1), "price"),  # ty: ignore[invalid-argument-type]
            (ToyOrder(0, "bid", 100, 0), "volume"),
            (ToyOrder(0, "bid", 100, True), "volume"),
            (ToyOrder(-1, "bid", 100, 1), "at"),
            (ToyOrder(float("nan"), "bid", 100, 1), "at"),
            (ToyOrder(5, "bid", 100, 1, cancel_at=5), "cancel_at"),
            (ToyOrder(5, "bid", 100, 1, cancel_at=float("nan")), "cancel_at"),
            (ToyOrder(5, "bid", 100, 1, cancel_at=float("inf")), "cancel_at"),
        ],
    )
    def test_bad_orders_are_refused(self, order: ToyOrder, match: str) -> None:
        with pytest.raises(ConfigError, match=match):
            match_toy_orders({"Jo": order})
