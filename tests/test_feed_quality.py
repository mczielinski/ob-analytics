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
from hypothesis import given, settings, strategies as st

from ob_analytics import (
    BitstampSource,
    DataQualitySummary,
    FeedType,
    LobsterSource,
    Severity,
    StaleOrder,
    data_quality_summary,
    detect_stale_orders,
)
from ob_analytics.analytics import (
    _faithful_best_series,
    order_book,
    set_order_types,
    uncross_book_sides,
)
from ob_analytics.datasets import toy_events, toy_trades
from ob_analytics.depth import price_level_volume
from ob_analytics.engine import crossed_prefix_counts

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
        assert BitstampSource().feed_type is FeedType.DIFF_FEED

    def test_lobster_is_matched_book(self):
        assert LobsterSource().feed_type is FeedType.MATCHED_BOOK

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
# crossed_prefix_counts + _faithful_best_series (correctness contract)
# ---------------------------------------------------------------------------


class TestCrossedPrefixCounts:
    def test_no_cross(self):
        # best bid 99 < best ask 101 -> nothing evicted.
        nb, na = crossed_prefix_counts(
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
        nb, na = crossed_prefix_counts(
            np.array([100.0]),
            np.array([older]),
            np.array([99.0]),
            np.array([newer]),
        )
        assert (nb, na) == (1, 0)
        # Fresh bid, stale ask -> evict the ask.
        nb, na = crossed_prefix_counts(
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
# The audit checks (issue #108)
# ---------------------------------------------------------------------------


class TestQualityChecks:
    """The pass/fail verdicts ``ob-analytics audit`` exits on."""

    def test_clean_run_passes_every_check(self):
        s = data_quality_summary(
            _classified_toy(), toy_trades(), feed_type=FeedType.MATCHED_BOOK
        )
        assert s.ok
        assert s.errors == ()
        assert s.warnings == ()
        assert "all passed" in s.render()

    def test_checks_are_ordered_errors_first(self):
        s = data_quality_summary(crossed_events(), _empty_trades())
        severities = [c.severity for c in s.checks]
        assert severities == sorted(
            severities, key=[Severity.ERROR, Severity.WARNING, Severity.INFO].index
        )

    def test_orphan_orders_counted(self):
        # Order 5 is changed and deleted with no created row.
        ev = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 5, 1.0, 98.0, 2.0, "bid", "changed", 0.0),
                (3, 5, 2.0, 98.0, 0.0, "bid", "deleted", 2.0),
            ]
        )
        s = data_quality_summary(ev, _empty_trades())
        assert s.orphan_orders == 1
        assert s.orphan_events == 2
        # Soft: an order resting before the capture began looks the same.
        assert s.ok
        assert [c.name for c in s.warnings] == ["orphan_orders"]

    def test_negative_volume_is_an_error(self):
        ev = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 2, 1.0, 101.0, -1.0, "ask", "created", 0.0),
            ]
        )
        s = data_quality_summary(ev, _empty_trades())
        assert s.negative_volume_rows == 1
        assert not s.ok
        assert "negative_volume" in {c.name for c in s.errors}

    def test_nonpositive_price_is_a_warning(self):
        ev = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 2, 1.0, 0.0, 2.0, "bid", "created", 0.0),
            ]
        )
        s = data_quality_summary(ev, _empty_trades())
        assert s.nonpositive_price_rows == 1
        assert s.ok  # reported, but not a failure
        assert "nonpositive_price" in {c.name for c in s.warnings}

    def test_venue_clock_after_receive_is_an_error(self):
        ev = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 2, 1.0, 101.0, 2.0, "ask", "created", 0.0),
            ]
        )
        # The venue stamped the second event a second after we received it.
        ev.loc[ev.index[1], "exchange_timestamp"] += pd.Timedelta(seconds=1)
        s = data_quality_summary(ev, _empty_trades())
        assert s.exchange_time_after_receive == 1
        assert not s.ok
        assert "exchange_time_after_receive" in {c.name for c in s.errors}

    def test_reordered_venue_clock_is_a_warning(self):
        ev = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 2, 1.0, 101.0, 2.0, "ask", "created", 0.0),
                (3, 3, 2.0, 98.0, 2.0, "bid", "created", 0.0),
            ]
        )
        # The middle event was stamped by the venue before the first one, so
        # the two reached the capture out of order.
        ev.loc[ev.index[1], "exchange_timestamp"] -= pd.Timedelta(seconds=10)
        s = data_quality_summary(ev, _empty_trades())
        assert s.exchange_time_reordered == 1
        assert s.exchange_time_after_receive == 0
        assert s.ok
        assert "exchange_time_reordered" in {c.name for c in s.warnings}

    def test_sequence_gap_is_an_error(self):
        ev = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 2, 1.0, 101.0, 2.0, "ask", "created", 0.0),
                (3, 3, 2.0, 98.0, 2.0, "bid", "created", 0.0),
            ]
        )
        ev["sequence"] = pd.array([1, 2, 7], dtype="Int64")  # 3-6 never arrived
        s = data_quality_summary(ev, _empty_trades())
        assert s.sequence_gaps == 4
        assert not s.ok
        assert "sequence_gaps" in {c.name for c in s.errors}

    @pytest.mark.parametrize(
        ("feed_type", "severity", "ok"),
        [
            (FeedType.MATCHED_BOOK, Severity.ERROR, False),
            (FeedType.DIFF_FEED, Severity.INFO, True),
            (FeedType.UNKNOWN, Severity.WARNING, True),
        ],
    )
    def test_crossing_severity_follows_the_feed_type(self, feed_type, severity, ok):
        """The same crossed book is a defect or a faithful replay by feed type."""
        s = data_quality_summary(crossed_events(), _empty_trades(), feed_type=feed_type)
        crossed = next(c for c in s.checks if c.name == "crossed_book")
        assert not crossed.passed
        assert crossed.severity is severity
        assert s.ok is ok

    def test_to_dict_carries_the_verdict(self):
        s = data_quality_summary(crossed_events(), _empty_trades())
        payload = json.loads(json.dumps(s.to_dict()))
        assert payload["ok"] is True
        assert {"name", "passed", "severity", "detail"} == set(payload["checks"][0])
        assert "orphan_orders" in {c["name"] for c in payload["checks"]}

    def test_render_lists_failed_checks_only(self):
        ev = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 2, 1.0, 101.0, -1.0, "ask", "created", 0.0),
            ]
        )
        text = data_quality_summary(ev, _empty_trades()).render()
        assert "ERROR   negative_volume" in text
        # INFO checks are context, not findings: they stay out of the verdict.
        assert "INFO" not in text


# ---------------------------------------------------------------------------
# Stale resting orders (issue #234)
# ---------------------------------------------------------------------------


def _trades(rows: list[tuple[float, float]]) -> pd.DataFrame:
    """Trades at ``(t_seconds, price)``, with the id columns the summary needs."""
    ts = pd.Series([_BASE + pd.Timedelta(seconds=t) for t, _ in rows]).astype(
        "datetime64[ns]"
    )
    return pd.DataFrame(
        {
            "timestamp": ts,
            "price": np.array([p for _, p in rows], dtype=np.float64),
            "maker_event_id": np.arange(len(rows), dtype=np.int64),
            "taker_event_id": np.arange(len(rows), dtype=np.int64) + 100,
        }
    )


def _book_with_stale_ask(delete_ask_at: float | None = None) -> pd.DataFrame:
    """A bid at 99, an ask at 101, and an ask at 103, all from t=0.

    A trade prints at 102 at t=10 (see :func:`_through_trade`), which the
    ask at 101 cannot survive.  By default the venue never reports that ask
    again; *delete_ask_at* reports its delete at that time instead.  A bid
    at t=100 marks the end of the capture.
    """
    rows = [
        (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
        (2, 2, 0.0, 101.0, 2.0, "ask", "created", 0.0),
        (3, 3, 0.0, 103.0, 2.0, "ask", "created", 0.0),
        (5, 4, 100.0, 98.0, 1.0, "bid", "created", 0.0),
    ]
    if delete_ask_at is not None:
        rows.append((4, 2, delete_ask_at, 101.0, 2.0, "ask", "deleted", 0.0))
    return _classified(sorted(rows, key=lambda r: r[2]))


def _through_trade() -> pd.DataFrame:
    return _trades([(10.0, 102.0)])


class TestStaleOrders:
    def test_finds_an_ask_a_trade_printed_through(self):
        (stale,) = detect_stale_orders(_book_with_stale_ask(), _through_trade())
        assert isinstance(stale, StaleOrder)
        assert stale.id == 2
        assert stale.direction == "ask"
        assert stale.price == 101.0
        assert stale.disproved_at == (_BASE + pd.Timedelta(seconds=10)).tz_localize(
            "UTC"
        )
        # From the trade at t=10 to the end of the capture at t=100, all of it
        # as the best ask.
        assert stale.stale_seconds == pytest.approx(90.0)
        assert stale.touch_seconds == pytest.approx(90.0)

    def test_finds_a_bid_a_trade_printed_through(self):
        events = _classified(
            [
                (1, 1, 0.0, 100.0, 2.0, "bid", "created", 0.0),
                (2, 2, 0.0, 102.0, 2.0, "ask", "created", 0.0),
                (3, 3, 60.0, 97.0, 1.0, "bid", "created", 0.0),
            ]
        )
        (stale,) = detect_stale_orders(events, _trades([(5.0, 98.0)]))
        assert (stale.id, stale.direction) == (1, "bid")
        assert stale.stale_seconds == pytest.approx(55.0)
        assert stale.touch_seconds == pytest.approx(55.0)

    def test_a_prompt_report_is_not_stale(self):
        # The venue reports the ask 200 ms after the trade: the normal lag of
        # a diff feed, well inside the one-second grace.
        events = _book_with_stale_ask(delete_ask_at=10.2)
        assert detect_stale_orders(events, _through_trade()) == ()

    def test_grace_sets_how_late_is_stale(self):
        events = _book_with_stale_ask(delete_ask_at=10.2)
        (stale,) = detect_stale_orders(
            events, _through_trade(), grace=pd.Timedelta(milliseconds=100)
        )
        assert stale.id == 2
        assert stale.stale_seconds == pytest.approx(0.2)

    def test_a_trade_at_the_same_instant_does_not_count(self):
        # A trade stamped in the same instant as the order's row cannot be
        # put before or after it, so it proves nothing.
        assert (
            detect_stale_orders(_book_with_stale_ask(), _trades([(0.0, 102.0)])) == ()
        )

    def test_a_trade_at_the_resting_price_does_not_count(self):
        # A print at the ask's own price is the ask trading, not a trade
        # through it.
        assert (
            detect_stale_orders(_book_with_stale_ask(), _trades([(10.0, 101.0)])) == ()
        )

    def test_the_worst_is_the_longest_at_the_touch(self):
        # Both asks sit below the trade at 104; only the one at 101 is the
        # best ask, so it comes first.
        stale = detect_stale_orders(_book_with_stale_ask(), _trades([(10.0, 104.0)]))
        assert [o.id for o in stale] == [2, 3]
        assert stale[1].touch_seconds == 0.0

    def test_tick_size_reports_quote_prices(self):
        (stale,) = detect_stale_orders(
            _book_with_stale_ask(), _through_trade(), tick_size=0.01
        )
        assert stale.price == pytest.approx(1.01)

    def test_toy_feed_has_none(self):
        assert detect_stale_orders(_classified_toy(), toy_trades()) == ()

    def test_does_not_change_the_book(self):
        events = _book_with_stale_ask()
        before = order_book(events)
        detect_stale_orders(events, _through_trade())
        after = order_book(events)
        assert 2 in set(after["asks"]["id"])
        assert before["asks"].equals(after["asks"])


class TestStaleOrdersInSummary:
    def test_summary_names_the_worst(self):
        s = data_quality_summary(
            _book_with_stale_ask(), _through_trade(), feed_type=FeedType.DIFF_FEED
        )
        assert [o.id for o in s.stale_orders] == [2]
        check = next(c for c in s.checks if c.name == "stale_orders")
        assert not check.passed
        assert check.severity is Severity.WARNING
        assert s.ok  # a warning: reported, but it does not fail the run
        text = s.render()
        assert "stale resting orders  : 1 (worst: ask 2 at 101" in text
        assert "for 1.5 min" in text

    def test_crossing_note_stops_saying_not_a_bug(self):
        s = data_quality_summary(
            _book_with_stale_ask(), _through_trade(), feed_type=FeedType.DIFF_FEED
        )
        crossed = next(c for c in s.checks if c.name == "crossed_book")
        assert "not a bug" not in crossed.detail
        assert "stale resting order" in crossed.detail

    def test_clean_diff_feed_keeps_the_note(self):
        s = data_quality_summary(
            crossed_events(), _empty_trades(), feed_type=FeedType.DIFF_FEED
        )
        assert s.stale_orders == ()
        assert "not a bug" in s.render()
        assert "stale resting orders  : 0" in s.render()
        assert next(c for c in s.checks if c.name == "stale_orders").passed

    def test_to_dict_carries_the_orders(self):
        s = data_quality_summary(
            _book_with_stale_ask(), _through_trade(), feed_type=FeedType.DIFF_FEED
        )
        (payload,) = json.loads(json.dumps(s.to_dict()))["stale_orders"]
        assert payload["id"] == 2
        assert payload["direction"] == "ask"
        assert payload["touch_seconds"] == pytest.approx(90.0)

    def test_names_the_opening_snapshot_ask_on_the_bitstamp_sample(
        self, bitstamp_sample_dir
    ):
        """The two orders the opening snapshot reported and the venue never
        mentioned again, and nothing else (#234)."""
        from ob_analytics.pipeline import Pipeline

        result = Pipeline(source=BitstampSource()).run(
            str(bitstamp_sample_dir / "orders.csv.gz")
        )
        s = data_quality_summary(
            result.events,
            result.trades,
            feed_type=FeedType.DIFF_FEED,
            depth=result.depth,
            tick_size=result.config.tick_size,
        )
        assert [o.id for o in s.stale_orders] == [
            2002347646152704,
            2002347642003458,
        ]
        worst = s.stale_orders[0]
        assert (worst.direction, worst.price) == ("ask", pytest.approx(78333.0))
        # It holds the ask touch for about 27 minutes after the first trade
        # that printed above it.
        assert worst.touch_seconds == pytest.approx(1645.7, abs=1.0)
        assert "ask 2002347646152704 at 78,333 held the ask touch for 27.4 min" in (
            s.render()
        )


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
