"""Tests for WS-6.0: feed classification, crossed-book handling, and the
per-run data-quality summary.

Three concerns, kept together because they share the matched-book /
diff-feed distinction:

* **Classification** — each format declares a :class:`FeedType`.
* **Uncrossing** — ``order_book(uncross=True)`` (and the frame-level
  :func:`uncross_book_sides`) evict crossed resting orders for display,
  mirroring the depth engine; the default stays faithful.
* **Data quality** — :func:`data_quality_summary` measures crossing,
  unmatched trades, duplicate ids, and pre-existing orders, with a
  ``_faithful_best_series`` that (unlike ``depth_summary``) does not
  pre-uncross.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ob_analytics import (
    BitstampFormat,
    DataQualitySummary,
    FeedType,
    LobsterFormat,
    data_quality_summary,
)
from ob_analytics.analytics import (
    _crossed_prefix_counts,
    _faithful_best_series,
    order_book,
    set_order_types,
    uncross_book_sides,
)
from ob_analytics.datasets import toy_events, toy_trades
from ob_analytics.depth import price_level_volume

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

_BASE = pd.Timestamp("2026-01-05 10:00:00")


def _canonical_events(rows: list[tuple]) -> pd.DataFrame:
    """Build a canonical (untyped) events frame.

    Each row is ``(event_id, id, t_seconds, price, volume, direction,
    action, fill)``.
    """
    ts = pd.Series([_BASE + pd.Timedelta(seconds=r[2]) for r in rows]).astype(
        "datetime64[ns]"
    )
    return pd.DataFrame(
        {
            "event_id": np.array([r[0] for r in rows], dtype=np.int64),
            "id": np.array([r[1] for r in rows], dtype=np.int64),
            "timestamp": ts,
            "exchange_timestamp": ts.copy(),
            "price": np.array([r[3] for r in rows], dtype=np.float64),
            "volume": np.array([r[4] for r in rows], dtype=np.float64),
            "direction": pd.Categorical(
                [r[5] for r in rows], categories=["bid", "ask"], ordered=True
            ),
            "action": pd.Categorical(
                [r[6] for r in rows],
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "fill": np.array([r[7] for r in rows], dtype=np.float64),
        }
    )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "maker_event_id": pd.array([], dtype=object),
            "taker_event_id": pd.array([], dtype=object),
        }
    )


def _classified(rows: list[tuple], trades: pd.DataFrame | None = None) -> pd.DataFrame:
    """Canonical events with the ``type`` column from ``set_order_types``."""
    trades = _empty_trades() if trades is None else trades
    return set_order_types(_canonical_events(rows), trades)


def crossed_events() -> pd.DataFrame:
    """A bid (100) resting above an ask (99), neither filling — the diff-feed
    pathology.  A third, non-crossing bid at t=60 gives the crossed state a
    measurable duration."""
    return _classified(
        [
            (1, 1, 0.0, 100.0, 2.0, "bid", "created", 0.0),
            (2, 2, 10.0, 99.0, 2.0, "ask", "created", 0.0),
            (3, 3, 60.0, 97.0, 1.0, "bid", "created", 0.0),
        ]
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_bitstamp_is_diff_feed(self):
        assert BitstampFormat().feed_type is FeedType.DIFF_FEED

    def test_lobster_is_matched_book(self):
        assert LobsterFormat().feed_type is FeedType.MATCHED_BOOK

    def test_feed_type_is_str_comparable(self):
        # The str mixin keeps CLI/JSON output and equality checks ergonomic.
        assert FeedType.DIFF_FEED == "diff_feed"
        assert FeedType.MATCHED_BOOK.value == "matched_book"

    def test_missing_feed_type_defaults_unknown(self):
        # Third-party formats predating the attribute read back as UNKNOWN,
        # so consumers never special-case format names.
        class _LegacyFormat:
            name = "legacy"

        assert getattr(_LegacyFormat(), "feed_type", FeedType.UNKNOWN) is (
            FeedType.UNKNOWN
        )


# ---------------------------------------------------------------------------
# order_book(uncross=...)
# ---------------------------------------------------------------------------


class TestUncrossOrderBook:
    def test_uncross_is_noop_on_matched_book(self):
        # The toy book is never crossed, so uncrossing must not change it.
        te = _classified_toy()
        faithful = order_book(te, uncross=False)
        display = order_book(te, uncross=True)
        assert faithful["bids"].equals(display["bids"])
        assert faithful["asks"].equals(display["asks"])

    def test_default_is_faithfully_crossed(self):
        book = order_book(crossed_events(), uncross=False)
        assert book["bids"]["price"].max() > book["asks"]["price"].min()

    def test_uncross_yields_uncrossed(self):
        book = order_book(crossed_events(), uncross=True)
        bids, asks = book["bids"], book["asks"]
        if not bids.empty and not asks.empty:
            assert bids["price"].max() < asks["price"].min()

    def test_uncross_evicts_the_stale_side(self):
        # The bid@100 (t=0) is older than the ask@99 (t=10); the fresher ask
        # wins, so the stale crossing bid is evicted while the ask survives.
        book = order_book(crossed_events(), uncross=True)
        assert 1 not in set(book["bids"]["id"])  # stale bid evicted
        assert 2 in set(book["asks"]["id"])  # fresh ask kept
        assert 3 in set(book["bids"]["id"])  # non-crossing bid kept

    def test_uncross_is_a_subset(self):
        faithful = order_book(crossed_events(), uncross=False)
        display = order_book(crossed_events(), uncross=True)
        for side in ("bids", "asks"):
            assert set(display[side]["id"]) <= set(faithful[side]["id"])

    def test_uncross_recomputes_liquidity(self):
        # After eviction, liquidity is the cumulative volume of the survivors.
        book = order_book(crossed_events(), uncross=True)
        for side in ("bids", "asks"):
            frame = book[side]
            if not frame.empty:
                np.testing.assert_allclose(
                    frame["liquidity"].to_numpy(),
                    frame["volume"].cumsum().to_numpy(),
                )

    def test_toy_touch_regression(self):
        # Concrete anchor that the faithful default is unchanged by the param.
        book = order_book(_classified_toy(), uncross=False)
        assert book["bids"]["price"].max() == 99.0
        assert book["asks"]["price"].min() == 102.0

    def test_uncross_book_sides_helper(self):
        # The public frame-level helper matches order_book(uncross=True) and
        # recomputes liquidity on the survivors.
        book = order_book(crossed_events(), uncross=False)
        bids, asks = uncross_book_sides(book["bids"], book["asks"])
        if not bids.empty and not asks.empty:
            assert bids["price"].max() < asks["price"].min()
        if not asks.empty:
            np.testing.assert_allclose(
                asks["liquidity"].to_numpy(), asks["volume"].cumsum().to_numpy()
            )


# ---------------------------------------------------------------------------
# _crossed_prefix_counts + _faithful_best_series (correctness contract)
# ---------------------------------------------------------------------------


class TestCrossedPrefixCounts:
    def test_no_cross(self):
        # best bid 99 < best ask 101 -> nothing evicted.
        nb, na = _crossed_prefix_counts(
            np.array([99.0]),
            np.array([_BASE.to_datetime64()]),
            np.array([101.0]),
            np.array([_BASE.to_datetime64()]),
        )
        assert (nb, na) == (0, 0)

    def test_evicts_older_side(self):
        older = _BASE.to_datetime64()
        newer = (_BASE + pd.Timedelta(seconds=5)).to_datetime64()
        # Fresh ask, stale bid -> evict the bid.
        nb, na = _crossed_prefix_counts(
            np.array([100.0]),
            np.array([older]),
            np.array([99.0]),
            np.array([newer]),
        )
        assert (nb, na) == (1, 0)
        # Fresh bid, stale ask -> evict the ask.
        nb, na = _crossed_prefix_counts(
            np.array([100.0]),
            np.array([newer]),
            np.array([99.0]),
            np.array([older]),
        )
        assert (nb, na) == (0, 1)


class TestFaithfulBestSeries:
    def _assert_matches_order_book(self, events: pd.DataFrame) -> None:
        depth = price_level_volume(events)
        series = _faithful_best_series(depth).sort_values("timestamp")
        for tp in sorted(events["timestamp"].unique()):
            book = order_book(events, tp=pd.Timestamp(tp), uncross=False)
            exp_bid = book["bids"]["price"].max() if not book["bids"].empty else np.nan
            exp_ask = book["asks"]["price"].min() if not book["asks"].empty else np.nan
            at_tp = series[series["timestamp"] <= tp]
            got_bid = at_tp["best_bid"].iloc[-1]
            got_ask = at_tp["best_ask"].iloc[-1]
            np.testing.assert_allclose(np.nan_to_num(got_bid), np.nan_to_num(exp_bid))
            np.testing.assert_allclose(np.nan_to_num(got_ask), np.nan_to_num(exp_ask))

    def test_matches_order_book_on_toy(self):
        self._assert_matches_order_book(_classified_toy())

    def test_matches_order_book_on_crossed(self):
        self._assert_matches_order_book(crossed_events())


# ---------------------------------------------------------------------------
# data_quality_summary
# ---------------------------------------------------------------------------


class TestDataQualitySummary:
    def test_toy_is_clean(self):
        te = _classified_toy()
        s = data_quality_summary(te, toy_trades(), feed_type=FeedType.MATCHED_BOOK)
        assert isinstance(s, DataQualitySummary)
        assert s.feed_type is FeedType.MATCHED_BOOK
        assert s.crossed_pct == 0.0
        assert s.crossed_episodes == 0
        assert s.unmatched_trades_pct == 0.0
        assert s.duplicate_event_ids == 0
        assert s.duplicate_created_ids == 0
        assert s.pre_existing_orders == 0
        assert s.n_events == 24
        assert s.n_orders == 12
        assert s.n_trades == 5

    def test_crossed_fixture_reports_high_crossing(self):
        ev = crossed_events()
        s = data_quality_summary(ev, _empty_trades(), feed_type=FeedType.DIFF_FEED)
        # Crossed from t=10 to t=60 out of a 60 s span -> ~83%.
        assert s.crossed_pct > 50.0
        assert s.crossed_episodes >= 1

    def test_unmatched_trades_counted(self):
        ev = crossed_events()
        trades = pd.DataFrame(
            {
                "maker_event_id": np.array([1, np.nan, 3], dtype=object),
                "taker_event_id": np.array([9, 9, np.nan], dtype=object),
            }
        )
        s = data_quality_summary(ev, trades)
        # 2 of 3 trades miss a maker or taker.
        assert s.unmatched_trades_pct == pytest.approx(200.0 / 3.0)

    def test_duplicate_event_ids_counted(self):
        # Two rows share event_id 2 (event_id must be globally unique).
        ev = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 2, 1.0, 101.0, 2.0, "ask", "created", 0.0),
                (2, 3, 2.0, 98.0, 1.0, "bid", "created", 0.0),
            ]
        )
        s = data_quality_summary(ev, _empty_trades())
        assert s.duplicate_event_ids == 1

    def test_duplicate_created_ids_counted(self):
        # Order id 1 is created twice.
        ev = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 1, 1.0, 99.0, 2.0, "bid", "created", 0.0),
                (3, 2, 2.0, 101.0, 2.0, "ask", "created", 0.0),
            ]
        )
        s = data_quality_summary(ev, _empty_trades())
        assert s.duplicate_created_ids == 1

    def test_pre_existing_orders_counted(self):
        # Order id 5 is only ever *changed* (no created row) -> pre-existing.
        ev = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 2, 1.0, 101.0, 2.0, "ask", "created", 0.0),
                (3, 5, 2.0, 98.0, 1.0, "bid", "changed", 0.0),
            ]
        )
        assert "pre-existing" in set(ev["type"].astype(str))
        s = data_quality_summary(ev, _empty_trades())
        assert s.pre_existing_orders == 1

    def test_depth_argument_matches_internal(self):
        ev = crossed_events()
        internal = data_quality_summary(ev, _empty_trades())
        explicit = data_quality_summary(
            ev, _empty_trades(), depth=price_level_volume(ev)
        )
        assert internal.to_dict() == explicit.to_dict()

    def test_to_dict_is_json_serialisable(self):
        s = data_quality_summary(crossed_events(), _empty_trades())
        payload = json.dumps(s.to_dict())
        assert json.loads(payload)["feed_type"] == "unknown"

    def test_render_reports_each_metric(self):
        s = data_quality_summary(
            crossed_events(), _empty_trades(), feed_type=FeedType.DIFF_FEED
        )
        text = s.render()
        assert "feed type" in text
        assert "crossed resting book" in text
        assert "diff feed" in text  # the interpretation note
        assert "pre-existing orders" in text


# ---------------------------------------------------------------------------
# prepare_book_snapshot_data(uncross=...)
# ---------------------------------------------------------------------------


class TestPrepareUncross:
    def test_prepare_uncrosses_book(self):
        from ob_analytics.visualization._data import prepare_book_snapshot_data

        book = order_book(crossed_events(), uncross=False)
        faithful = prepare_book_snapshot_data(book, uncross=False)
        display = prepare_book_snapshot_data(book, uncross=True)
        # Faithful stays crossed; the uncrossed view does not.
        assert faithful["bids"]["price"].max() > faithful["asks"]["price"].min()
        if not display["bids"].empty and not display["asks"].empty:
            assert display["bids"]["price"].max() < display["asks"]["price"].min()

    def test_prepare_uncross_ignores_timeless_book(self):
        from ob_analytics.visualization._data import prepare_book_snapshot_data

        # A synthetic ndarray book carries no timestamp; uncross must no-op
        # rather than raise.
        book = {
            "timestamp": _BASE.timestamp(),
            "bids": np.array([[100.0, 2.0, 2.0]]),
            "asks": np.array([[99.0, 2.0, 2.0]]),
        }
        out = prepare_book_snapshot_data(book, uncross=True)
        assert not out["bids"].empty


# ---------------------------------------------------------------------------
# Property tests — the uncross invariant on arbitrary resting books
# ---------------------------------------------------------------------------


def _resting_book_events(orders: list[tuple[int, float, int, str]]) -> pd.DataFrame:
    """Events frame of all-created resting-limit orders.

    Each order is ``(t_seconds, price, volume, direction)``.
    """
    rows = [
        (i + 1, i + 1, t, float(price), float(vol), direction, "created", 0.0)
        for i, (t, price, vol, direction) in enumerate(orders)
    ]
    ev = _canonical_events(rows)
    ev["type"] = pd.Categorical(
        ["resting-limit"] * len(rows),
        categories=[
            "unknown",
            "pre-existing",
            "flashed-limit",
            "resting-limit",
            "market-limit",
            "market",
        ],
        ordered=True,
    )
    return ev


_orders = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=30),  # t seconds
        st.integers(min_value=95, max_value=105),  # price
        st.integers(min_value=1, max_value=5),  # volume
        st.sampled_from(["bid", "ask"]),  # direction
    ),
    min_size=1,
    max_size=12,
)


class TestUncrossProperties:
    @settings(max_examples=200, deadline=None)
    @given(orders=_orders)
    def test_uncross_never_leaves_a_cross(self, orders):
        ev = _resting_book_events(orders)
        book = order_book(ev, uncross=True)
        bids, asks = book["bids"], book["asks"]
        if not bids.empty and not asks.empty:
            assert bids["price"].max() < asks["price"].min()

    @settings(max_examples=200, deadline=None)
    @given(orders=_orders)
    def test_uncross_is_a_subset_of_faithful(self, orders):
        ev = _resting_book_events(orders)
        faithful = order_book(ev, uncross=False)
        display = order_book(ev, uncross=True)
        for side in ("bids", "asks"):
            assert set(display[side]["id"]) <= set(faithful[side]["id"])

    @settings(max_examples=200, deadline=None)
    @given(orders=_orders)
    def test_uncross_is_idempotent_when_already_uncrossed(self, orders):
        ev = _resting_book_events(orders)
        faithful = order_book(ev, uncross=False)
        if faithful["bids"].empty or faithful["asks"].empty:
            return
        if faithful["bids"]["price"].max() < faithful["asks"]["price"].min():
            # An already-uncrossed book must be returned untouched.
            display = order_book(ev, uncross=True)
            assert faithful["bids"].equals(display["bids"])
            assert faithful["asks"].equals(display["asks"])


def _classified_toy() -> pd.DataFrame:
    return set_order_types(toy_events(), toy_trades())
