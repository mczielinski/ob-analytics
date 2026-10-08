"""Every rebuild puts an order at one price, and a book is uncrossed one way.

An order rests at the price its first row set, or the price of its latest
``changed`` row with no fill (a modify that moves it).  A row that reports a
fill never moves it, whatever price it carries: Bitstamp reports a taker's
execution at the price it traded at.  The per-order book, the depth table,
the windowed run's carry, and both queue rebuilds must agree on that price.
"""

from __future__ import annotations

import inspect
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from ob_analytics import analytics
from ob_analytics._windows import resting_orders
from ob_analytics.analytics import order_book
from ob_analytics.depth import price_level_volume
from ob_analytics.queue import queue_age_grid, queue_positions

_T0 = pd.Timestamp("2026-01-01", tz="UTC")


def _at(seconds: float) -> pd.Timestamp:
    return _T0 + pd.Timedelta(seconds=seconds)


def _events(rows: list[tuple]) -> pd.DataFrame:
    """A classified events frame.

    Each row is ``(event_id, id, t_seconds, price, volume, direction, action,
    fill, type)``.  Prices are ticks and sizes lots.
    """
    ts = pd.Series([_at(r[2]) for r in rows], dtype="datetime64[ns, UTC]")
    return pd.DataFrame(
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
            "type": [r[8] for r in rows],
        }
    )


# A bid created at 100, then a fill reported at 99 that leaves 3 lots resting.
FILL_ELSEWHERE = [
    (1, 7, 0.0, 100, 5, "bid", "created", 0, "resting-limit"),
    (2, 7, 1.0, 99, 3, "bid", "changed", 2, "resting-limit"),
]


class TestAFillReportedAtAnotherPrice:
    """The order stays at 100 in every rebuild."""

    def test_the_per_order_book(self):
        book = order_book(_events(FILL_ELSEWHERE), _at(2))
        assert book["bids"][["id", "price", "volume"]].to_dict("records") == [
            {"id": 7, "price": 100, "volume": 3}
        ]

    def test_the_depth_table(self):
        depth = price_level_volume(_events(FILL_ELSEWHERE))
        last = depth.groupby("price")["volume"].last().to_dict()
        assert last == {100: 3}

    def test_the_windowed_carry(self):
        carried = resting_orders(_events(FILL_ELSEWHERE))
        assert carried[["id", "price", "volume"]].to_dict("records") == [
            {"id": 7, "price": 100, "volume": 3}
        ]

    def test_the_queue_table(self):
        out = queue_positions(_events(FILL_ELSEWHERE), levels="all")
        assert out[["id", "price", "action", "remaining"]].to_dict("records") == [
            {"id": 7, "price": 100, "action": "created", "remaining": 5},
            {"id": 7, "price": 100, "action": "changed", "remaining": 3},
        ]

    def test_the_touch_queue_grid(self):
        # The order leaves the grid when its fill-reported row deletes it.
        rows = [
            *FILL_ELSEWHERE,
            (3, 8, 2.0, 98, 4, "bid", "created", 0, "resting-limit"),
            (4, 7, 3.0, 99, 0, "bid", "deleted", 3, "resting-limit"),
        ]
        ages, _, max_rank = queue_age_grid(_events(rows), side="bid", n_time=4)
        # Samples at 0, 1, 2 and 3 s: order 7 is the touch until it goes, then
        # order 8 (placed at 2 s) is.
        assert max_rank == 1
        assert ages[0].tolist() == [0.0, 1.0, 2.0, 1.0]


class TestAnOrderThatMoves:
    """A ``changed`` row with no fill and a new price moves the order."""

    MOVED: ClassVar[list[tuple]] = [
        (1, 1, 0.0, 100, 5, "bid", "created", 0, "resting-limit"),
        (2, 2, 1.0, 101, 4, "bid", "created", 0, "resting-limit"),
        (3, 1, 2.0, 101, 5, "bid", "changed", 0, "resting-limit"),
    ]

    def test_the_queue_moves_it_to_the_back_of_its_new_level(self):
        out = queue_positions(_events(self.MOVED), levels="all")
        moved = out.iloc[-1]
        assert (moved["id"], moved["price"], moved["rank"], moved["queue_len"]) == (
            1,
            101,
            2,
            2,
        )

    def test_the_touch_queue_grid_follows_it(self):
        ages, _, max_rank = queue_age_grid(_events(self.MOVED), side="bid", n_time=3)
        # At 2 s the touch (101) holds order 2 (age 1 s) then order 1 (age 2 s).
        assert max_rank == 2
        assert ages[:, -1].tolist() == [1.0, 2.0]

    def test_the_queue_reports_it_leaving_the_touch(self):
        # Bids 1 and 2 rest at 100; order 1 moves down to 98.  The touch table
        # shows order 1 leave the touch, as it shows a cancel.
        rows = [
            (1, 1, 0.0, 100, 5, "bid", "created", 0, "resting-limit"),
            (2, 2, 1.0, 100, 4, "bid", "created", 0, "resting-limit"),
            (3, 1, 2.0, 98, 5, "bid", "changed", 0, "resting-limit"),
        ]
        out = queue_positions(_events(rows), levels="touch")
        left = out.iloc[-1]
        assert (left["id"], left["price"], left["action"], left["rank"]) == (
            1,
            100,
            "deleted",
            1,
        )


class TestAnOrderPlacedAgain:
    """An id cancelled and placed again rests at its new price."""

    ROWS: ClassVar[list[tuple]] = [
        (1, 7, 0.0, 100, 5, "bid", "created", 0, "resting-limit"),
        (2, 7, 1.0, 100, 5, "bid", "deleted", 0, "resting-limit"),
        (3, 7, 2.0, 105, 4, "bid", "created", 0, "resting-limit"),
    ]

    def test_the_per_order_book(self):
        book = order_book(_events(self.ROWS), _at(3))
        assert book["bids"][["id", "price", "volume"]].to_dict("records") == [
            {"id": 7, "price": 105, "volume": 4}
        ]

    def test_the_queue_table(self):
        out = queue_positions(_events(self.ROWS), levels="all")
        assert out["price"].tolist() == [100, 100, 105]

    # The same id placed twice with no cancel between: it rests once, at its
    # latest price, as the per-order book shows it.
    TWICE: ClassVar[list[tuple]] = [
        (1, 7, 0.0, 100, 5, "bid", "created", 0, "resting-limit"),
        (2, 7, 1.0, 101, 4, "bid", "created", 0, "resting-limit"),
        (3, 8, 2.0, 99, 3, "bid", "created", 0, "resting-limit"),
        (4, 7, 3.0, 101, 4, "bid", "deleted", 0, "resting-limit"),
        (5, 8, 4.0, 99, 2, "bid", "changed", 1, "resting-limit"),
    ]

    def test_placed_twice_leaves_no_copy_in_the_queue_table(self):
        out = queue_positions(_events(self.TWICE), levels="touch")
        # Once order 7 is cancelled, order 8 at 99 is the touch, so its fill
        # at 4 s is a touch row.
        assert out[["id", "price", "rank"]].iloc[-1].tolist() == [8, 99, 1]

    def test_placed_twice_leaves_its_old_place_as_a_cancel_does(self):
        out = queue_positions(_events(self.TWICE), levels="all")
        first = out[out["id"] == 7].head(3)
        assert first[["price", "action"]].values.tolist() == [
            [100, "created"],
            [100, "deleted"],
            [101, "created"],
        ]

    def test_placed_twice_leaves_no_copy_in_the_grid(self):
        ages, _, max_rank = queue_age_grid(_events(self.TWICE), side="bid", n_time=5)
        # Samples at 0 to 4 s.  From 3 s order 7 is gone and order 8 (placed
        # at 2 s) is the touch.
        assert max_rank == 1
        assert ages[0].tolist() == [0.0, 0.0, 1.0, 1.0, 2.0]


class TestMarketOrdersInTheQueue:
    """A taker created at its limit never rests, so no queue holds it."""

    # Order 9 is a market buy created at 110, filled at 100 against order 1,
    # and deleted at the fill price.  Order 1 rests at 100 on the ask side;
    # order 2 is a bid at 99.
    TAKER: ClassVar[list[tuple]] = [
        (1, 1, 0.0, 100, 10, "ask", "created", 0, "resting-limit"),
        (2, 2, 0.5, 99, 6, "bid", "created", 0, "resting-limit"),
        (3, 9, 1.0, 110, 4, "bid", "created", 0, "market"),
        (4, 9, 1.0, 100, 0, "bid", "deleted", 4, "market"),
        (5, 1, 1.0, 100, 6, "ask", "changed", 4, "resting-limit"),
        (6, 2, 2.0, 99, 6, "bid", "deleted", 0, "resting-limit"),
    ]

    def test_the_grid_touch_is_the_resting_bid(self):
        ages, _, max_rank = queue_age_grid(_events(self.TAKER), side="bid", n_time=4)
        # Samples at 0.5, 1, 1.5 and 2 s: order 2 is the only bid on the
        # touch, until it is cancelled at 2 s.
        assert max_rank == 1
        np.testing.assert_array_equal(ages[0], [0.0, 0.5, 1.0, np.nan])

    def test_the_queue_table_leaves_it_out(self):
        out = queue_positions(_events(self.TAKER), levels="all")
        assert 9 not in out["id"].tolist()


@pytest.fixture(scope="module")
def sample_run_events(bitstamp_sample_dir):
    from ob_analytics.pipeline import Pipeline

    return Pipeline().run(bitstamp_sample_dir / "orders.csv.gz").events


@pytest.mark.parametrize("side", ["bid", "ask"])
def test_grid_touch_queue_matches_the_book_on_the_bundled_sample(
    sample_run_events, side
):
    """At every sample, the grid's touch holds the book's touch orders."""
    events = sample_run_events
    placed = events[events["action"] == "created"].groupby("id")["timestamp"].first()
    ages, times, _ = queue_age_grid(events, side=side, n_time=200)
    samples = pd.DatetimeIndex(times)
    if samples.tz is None:
        samples = samples.tz_localize("UTC")
    differ = 0
    for column, t in enumerate(samples):
        book = order_book(events, tp=t)
        frame = book["bids"] if side == "bid" else book["asks"]
        in_book: list[float] = []
        if not frame.empty:
            touch = frame["price"].max() if side == "bid" else frame["price"].min()
            at_touch = frame.loc[frame["price"] == touch, "id"]
            in_book = sorted((t - placed[at_touch]).dt.total_seconds())
        in_grid = sorted(a for a in ages[:, column] if not np.isnan(a))
        # The grid keeps ages to whole microseconds.
        same = len(in_grid) == len(in_book) and np.allclose(
            in_grid, in_book, rtol=0, atol=1e-6
        )
        differ += not same
    assert differ == 0, f"{differ} of {len(samples)} samples differ"


class TestOneUncross:
    """Two bids at 100 (id 2 placed first) and an ask at 99 between them."""

    TIE: ClassVar[list[tuple]] = [
        (1, 2, 1.0, 100, 1, "bid", "created", 0, "resting-limit"),
        (2, 3, 3.0, 99, 1, "ask", "created", 0, "resting-limit"),
        (3, 1, 5.0, 100, 1, "bid", "created", 0, "resting-limit"),
    ]

    def test_order_book_evicts_the_older_bid_first(self):
        book = order_book(_events(self.TIE), _at(10), uncross=True)
        assert book["bids"]["id"].tolist() == [1]
        assert book["asks"]["id"].tolist() == []

    def test_the_book_snapshot_shows_the_uncrossed_book(self):
        from ob_analytics.visualization._data import prepare_book_snapshot_data

        book = order_book(_events(self.TIE), _at(10), uncross=True)
        data = prepare_book_snapshot_data(book, per_order=True)
        assert len(data["bids"]) == 1
        assert data["asks"].empty

    def test_there_is_no_second_uncross_path(self):
        from ob_analytics.visualization._data import prepare_book_snapshot_data

        assert not hasattr(analytics, "uncross_book_sides")
        assert "uncross" not in inspect.signature(prepare_book_snapshot_data).parameters


@pytest.mark.parametrize("dtype", ["Int64", "int64[pyarrow]"])
def test_every_rebuild_keeps_the_price_dtype(dtype):
    events = _events(FILL_ELSEWHERE)
    events["price"] = events["price"].astype(dtype)
    assert str(order_book(events, _at(2))["bids"]["price"].dtype) == dtype
    assert str(price_level_volume(events)["price"].dtype) == dtype
    assert str(resting_orders(events)["price"].dtype) == dtype
    assert str(queue_positions(events, levels="all")["price"].dtype) == dtype


def test_order_book_has_no_unused_price_bounds():
    parameters = inspect.signature(order_book).parameters
    assert "min_bid" not in parameters
    assert "max_ask" not in parameters
