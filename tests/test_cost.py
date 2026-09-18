"""Tests for cost.py — effective/realized spread, impact, Amihud and Roll."""

import numpy as np
import pandas as pd
import pytest

from ob_analytics.cost import (
    COST_COLUMNS,
    CostSummary,
    amihud,
    cost_summary,
    roll_spread,
    transaction_costs,
)
from ob_analytics.exceptions import ConfigError, ObAnalyticsError

BASE = pd.Timestamp("2015-05-01 00:00:00", tz="UTC")


# ── Helpers ──────────────────────────────────────────────────────────


def _trades(prices, directions, volumes=None, offsets=None):
    """Build a minimal trades frame; *offsets* are seconds from BASE."""
    n = len(prices)
    if offsets is None:
        offsets = list(range(n))
    if volumes is None:
        volumes = [1.0] * n
    return pd.DataFrame(
        {
            "timestamp": [BASE + pd.Timedelta(seconds=s) for s in offsets],
            "price": list(prices),
            "volume": list(volumes),
            "direction": list(directions),
        }
    )


def _quotes(bids, asks, offsets=None):
    """Build a minimal quote frame in the depth_summary spelling."""
    if offsets is None:
        offsets = list(range(len(bids)))
    return pd.DataFrame(
        {
            "timestamp": [BASE + pd.Timedelta(seconds=s) for s in offsets],
            "best_bid_price": list(bids),
            "best_ask_price": list(asks),
        }
    )


# ── Effective / realized spread and price impact ─────────────────────


class TestTransactionCosts:
    def test_hand_worked_buy(self):
        """One buy, worked through by hand.

        Quotes: mid 100 at t=0, mid 102 at t=10.  A buy prints at 101 at
        t=5, so with a 5-second horizon the future mid is the 102 quote:

            effective = 2 * (+1) * (101 - 100) =  2
            realized  = 2 * (+1) * (101 - 102) = -2
            impact    = 2 * (+1) * (102 - 100) =  4

        and in basis points, against the mid of 100: 200, -200, 400.
        """
        quotes = _quotes([99, 101], [101, 103], offsets=[0, 10])
        trades = _trades([101], ["buy"], offsets=[5])

        costs = transaction_costs(trades, quotes, horizon="5s")

        assert list(costs.columns) == list(COST_COLUMNS)
        row = costs.iloc[0]
        assert row["mid_price"] == pytest.approx(100.0)
        assert row["future_mid_price"] == pytest.approx(102.0)
        assert row["effective_spread"] == pytest.approx(2.0)
        assert row["realized_spread"] == pytest.approx(-2.0)
        assert row["price_impact"] == pytest.approx(4.0)
        assert row["effective_spread_bps"] == pytest.approx(200.0)
        assert row["realized_spread_bps"] == pytest.approx(-200.0)
        assert row["price_impact_bps"] == pytest.approx(400.0)

    def test_hand_worked_sell(self):
        """The mirror image: a sell at 99 against the same mids.

        effective = 2 * (-1) * (99 - 100) = 2; the price then *falls* to a
        mid of 98, so the seller's information was right and the impact is
        positive: 2 * (-1) * (98 - 100) = 4.
        """
        quotes = _quotes([99, 97], [101, 99], offsets=[0, 10])
        trades = _trades([99], ["sell"], offsets=[5])

        row = transaction_costs(trades, quotes, horizon="5s").iloc[0]

        assert row["effective_spread"] == pytest.approx(2.0)
        assert row["realized_spread"] == pytest.approx(-2.0)
        assert row["price_impact"] == pytest.approx(4.0)

    def test_parts_add_to_the_whole(self):
        """realized + impact == effective, row by row."""
        rng = np.random.default_rng(0)
        n = 60
        quotes = _quotes(
            99 + rng.integers(0, 4, n),
            103 + rng.integers(0, 4, n),
            offsets=list(range(n)),
        )
        trades = _trades(
            101 + rng.integers(0, 4, 20),
            rng.choice(["buy", "sell"], 20),
            offsets=list(range(5, 45, 2)),
        )

        costs = transaction_costs(trades, quotes, horizon="5s")
        total = costs["realized_spread"] + costs["price_impact"]
        pd.testing.assert_series_equal(
            total, costs["effective_spread"], check_names=False
        )

    def test_mid_excludes_the_trades_own_quote(self):
        """A quote stamped with the trade is the book *after* it, so it is skipped.

        The mid at the trade's own instant is 110; the last quote strictly
        before it has a mid of 100.  Measuring against 110 would report a
        negative effective spread for a buy at 101.
        """
        quotes = _quotes([99, 109], [101, 111], offsets=[0, 5])
        trades = _trades([101], ["buy"], offsets=[5])

        row = transaction_costs(trades, quotes, horizon="1s").iloc[0]

        assert row["mid_price"] == pytest.approx(100.0)
        assert row["effective_spread"] == pytest.approx(2.0)

    def test_crossed_quote_is_skipped(self):
        """A book whose bid is above its ask has no mid, so the last good one is used."""
        quotes = _quotes([99, 120], [101, 100], offsets=[0, 4])
        trades = _trades([101], ["buy"], offsets=[5])

        row = transaction_costs(trades, quotes, horizon="1s").iloc[0]

        assert row["mid_price"] == pytest.approx(100.0)

    def test_no_quote_yet_leaves_the_row_unmeasured(self):
        """A trade before the first quote has nothing to measure against."""
        quotes = _quotes([99], [101], offsets=[10])
        trades = _trades([101], ["buy"], offsets=[5])

        row = transaction_costs(trades, quotes, horizon="1s").iloc[0]

        assert np.isnan(row["mid_price"])
        assert np.isnan(row["effective_spread"])

    def test_horizon_past_the_last_quote_is_not_measured(self):
        """The realized spread needs a real future mid, not the final quote reused."""
        quotes = _quotes([99, 99], [101, 101], offsets=[0, 10])
        trades = _trades([101, 101], ["buy", "buy"], offsets=[1, 9])

        costs = transaction_costs(trades, quotes, horizon="5s")

        # t=1 + 5s = 6s, inside the quotes; t=9 + 5s = 14s, past the end.
        assert costs["future_mid_price"].notna().iloc[0]
        assert np.isnan(costs["future_mid_price"].iloc[1])
        assert np.isnan(costs["realized_spread"].iloc[1])
        # The effective spread does not need the future, so it survives.
        assert costs["effective_spread"].iloc[1] == pytest.approx(2.0)

    def test_output_is_chronological(self):
        quotes = _quotes([99] * 12, [101] * 12)
        trades = _trades([101, 100, 102], ["buy"] * 3, offsets=[7, 3, 5])

        costs = transaction_costs(trades, quotes, horizon="1s")

        assert costs["timestamp"].is_monotonic_increasing
        assert list(costs["price"]) == [100, 102, 101]

    def test_horizon_is_recorded(self):
        quotes = _quotes([99] * 5, [101] * 5)
        trades = _trades([101], ["buy"], offsets=[1])

        assert transaction_costs(trades, quotes, horizon="2s").attrs["horizon"] == "2s"

    def test_direction_inferred_when_absent(self):
        """A feed with no aggressor side is classified against the quotes."""
        quotes = _quotes([99] * 12, [101] * 12)
        trades = _trades([101, 99, 101], ["buy"] * 3, offsets=[2, 4, 6]).drop(
            columns=["direction"]
        )

        costs = transaction_costs(trades, quotes, horizon="1s")

        # Lee-Ready against a mid of 100: above the mid is a buy, below a sell.
        assert list(costs["direction"]) == ["buy", "sell", "buy"]

    def test_native_direction_is_honored(self):
        """A native aggressor side is not second-guessed by a classifier."""
        quotes = _quotes([99] * 12, [101] * 12)
        # Priced like a buy, labelled a sell: the label wins.
        trades = _trades([101], ["sell"], offsets=[2])

        assert transaction_costs(trades, quotes, horizon="1s")["direction"].iloc[0] == (
            "sell"
        )

    def test_missing_column_raises(self):
        quotes = _quotes([99] * 5, [101] * 5)
        trades = _trades([101], ["buy"], offsets=[1]).drop(columns=["volume"])

        with pytest.raises(ConfigError, match="volume"):
            transaction_costs(trades, quotes)

    def test_empty_trades_raise(self):
        quotes = _quotes([99] * 5, [101] * 5)

        with pytest.raises(ObAnalyticsError):
            transaction_costs(_trades([], []), quotes)

    def test_bvc_sign_method_rejected(self):
        quotes = _quotes([99] * 5, [101] * 5)
        trades = _trades([101], ["buy"], offsets=[1])

        with pytest.raises(ConfigError, match="bvc"):
            transaction_costs(trades, quotes, sign_method="bvc")


# ── Session summary ──────────────────────────────────────────────────


class TestCostSummary:
    def test_weighted_by_trade_size(self):
        """A 9-unit trade counts nine times as much as a 1-unit trade.

        Effective spreads of 2 and 20 with volumes of 9 and 1 average to
        (2*9 + 20*1) / 10 = 3.8, not to the unweighted 11.
        """
        quotes = _quotes([99] * 20, [101] * 20)
        trades = _trades([101, 110], ["buy", "buy"], volumes=[9.0, 1.0], offsets=[2, 4])

        summary = cost_summary(transaction_costs(trades, quotes, horizon="1s"))

        assert isinstance(summary, CostSummary)
        assert summary.effective_spread == pytest.approx(3.8)
        assert summary.n_trades == 2
        assert summary.volume == pytest.approx(10.0)

    def test_realized_counted_separately_from_effective(self):
        """Trades past the horizon's reach drop out of the realized average only."""
        quotes = _quotes([99, 99], [101, 101], offsets=[0, 10])
        trades = _trades([101, 101], ["buy", "buy"], offsets=[1, 9])

        summary = cost_summary(transaction_costs(trades, quotes, horizon="5s"))

        assert summary.n_trades == 2
        assert summary.n_realized == 1
        assert summary.horizon == "5s"

    def test_nothing_measurable_is_nan_not_an_error(self):
        """A capture shorter than its own horizon still returns a summary."""
        quotes = _quotes([99, 99], [101, 101], offsets=[0, 2])
        trades = _trades([101], ["buy"], offsets=[1])

        summary = cost_summary(transaction_costs(trades, quotes, horizon="1h"))

        assert summary.n_realized == 0
        assert np.isnan(summary.realized_spread)
        assert summary.effective_spread == pytest.approx(2.0)

    def test_missing_column_raises(self):
        with pytest.raises(ConfigError, match="effective_spread"):
            cost_summary(pd.DataFrame({"timestamp": [BASE]}))


# ── Amihud ───────────────────────────────────────────────────────────


class TestAmihud:
    def test_hand_worked(self):
        """Four trades in one window, worked through by hand.

        Prices 100 -> 110 with sizes of 1 each: the return is
        |110/100 - 1| = 0.1, turnover is 100 + 102 + 104 + 110 = 416, so
        the illiquidity is 0.1 / 416.
        """
        trades = _trades([100, 102, 104, 110], ["buy"] * 4)

        result = amihud(trades)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["n_trades"] == 4
        assert row["abs_return"] == pytest.approx(0.1)
        assert row["turnover"] == pytest.approx(416.0)
        assert row["amihud"] == pytest.approx(0.1 / 416.0)

    def test_deeper_market_scores_lower(self):
        """The same price move on ten times the size is a tenth as illiquid."""
        thin = _trades([100, 110], ["buy"] * 2, volumes=[1.0, 1.0])
        deep = _trades([100, 110], ["buy"] * 2, volumes=[10.0, 10.0])

        assert amihud(deep).iloc[0]["amihud"] == pytest.approx(
            amihud(thin).iloc[0]["amihud"] / 10.0
        )

    def test_windowed(self):
        """One row per window that holds a trade, keyed by the window's start."""
        trades = _trades(
            [100, 101, 200, 202],
            ["buy"] * 4,
            offsets=[0, 30, 120, 150],
        )

        result = amihud(trades, window="1min")

        # 00:00 and 00:02 hold trades; 00:01 is empty and drops out.
        assert list(result["n_trades"]) == [2, 2]
        assert list(result["timestamp"]) == [BASE, BASE + pd.Timedelta(minutes=2)]

    def test_flat_price_is_perfectly_liquid(self):
        trades = _trades([100, 100, 100], ["buy"] * 3)

        assert amihud(trades).iloc[0]["amihud"] == pytest.approx(0.0)

    def test_empty_trades_raise(self):
        with pytest.raises(ObAnalyticsError):
            amihud(_trades([], []))


# ── Roll ─────────────────────────────────────────────────────────────


class TestRollSpread:
    @staticmethod
    def _bounce(n, half_spread, seed=0):
        """A flat mid of 100 with an iid ±*half_spread* bounce — Roll's own model."""
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], n)
        prices = 100 + half_spread * signs
        return _trades(prices, np.where(signs > 0, "buy", "sell"))

    def test_autocovariance_worked_by_hand(self):
        """The arithmetic, on five prices small enough to check on paper.

        Changes are 2, -1, 3, -1, averaging 0.75, so centred they are
        1.25, -1.75, 2.25, -1.75.  The lag-1 products sum to
        -2.1875 - 3.9375 - 3.9375 = -10.0625 over three terms, giving an
        autocovariance of -3.354166..., and 2*sqrt(3.354166...) = 3.66288.
        """
        trades = _trades([100, 102, 101, 104, 103], ["buy"] * 5)

        row = roll_spread(trades).iloc[0]

        assert row["autocovariance"] == pytest.approx(-10.0625 / 3)
        assert row["roll_spread"] == pytest.approx(2 * np.sqrt(10.0625 / 3))
        assert row["mean_price"] == pytest.approx(102.0)
        assert row["roll_spread_bps"] == pytest.approx(
            10_000 * 2 * np.sqrt(10.0625 / 3) / 102.0
        )

    def test_recovers_a_known_spread(self):
        """On Roll's own model, the estimate returns the spread that made it.

        A flat mid of 100 with an iid ±1 bounce is a spread of 2.  The
        estimator is noisy in small samples, so this takes a long run and
        allows a few per cent.
        """
        row = roll_spread(self._bounce(20_000, 1.0)).iloc[0]

        assert row["roll_spread"] == pytest.approx(2.0, rel=0.05)

    def test_wider_bounce_wider_spread(self):
        """Doubling the half-spread doubles the implied spread."""
        narrow = roll_spread(self._bounce(20_000, 1.0, seed=1)).iloc[0]
        wide = roll_spread(self._bounce(20_000, 2.0, seed=1)).iloc[0]

        assert wide["roll_spread"] == pytest.approx(2 * narrow["roll_spread"], rel=1e-6)

    def test_deterministic_alternation_overstates(self):
        """A perfectly alternating tape breaks the model and reads twice too wide.

        Roll assumes the aggressor side is independent from trade to trade.
        A tape that alternates exactly makes every price change ±2c instead
        of an average of ±c, so the estimate comes back at 2S rather than S.
        The estimator cannot tell; this is what its assumption is worth.
        """
        # Ends where it began, so the changes average to exactly zero.
        trades = _trades([99, 101] * 20 + [99], ["sell", "buy"] * 20 + ["sell"])

        row = roll_spread(trades).iloc[0]

        assert row["autocovariance"] == pytest.approx(-4.0)
        assert row["roll_spread"] == pytest.approx(4.0)

    def test_trending_tape_has_no_estimate(self):
        """A price that only rises has a positive autocovariance and no root."""
        trades = _trades(list(range(100, 140)), ["buy"] * 40)

        row = roll_spread(trades).iloc[0]

        assert row["autocovariance"] >= 0
        assert np.isnan(row["roll_spread"])
        assert np.isnan(row["roll_spread_bps"])

    def test_too_few_trades_to_estimate(self):
        """Three prints give one product, not a covariance."""
        trades = _trades([99, 101, 99], ["sell", "buy", "sell"])

        row = roll_spread(trades).iloc[0]

        assert row["n_trades"] == 3
        assert np.isnan(row["autocovariance"])
        assert np.isnan(row["roll_spread"])

    def test_windowed(self):
        """Each window is estimated on its own trades, not the whole run."""
        bounce = self._bounce(40, 1.0, seed=2)["price"].tolist()
        trades = _trades(
            bounce + list(range(200, 240)),
            ["buy"] * 80,
            offsets=list(range(40)) + list(range(60, 100)),
        )

        result = roll_spread(trades, window="1min")

        assert len(result) == 2
        assert result.iloc[0]["autocovariance"] < 0
        assert result.iloc[0]["roll_spread"] > 0
        # The second window only trends, so it has no estimate of its own --
        # which the whole-run figure would have hidden.
        assert np.isnan(result.iloc[1]["roll_spread"])

    def test_empty_trades_raise(self):
        with pytest.raises(ObAnalyticsError):
            roll_spread(_trades([], []))


# ── On the bundled sample ────────────────────────────────────────────


class TestBundledSample:
    """Every measure runs on the bundled Bitstamp capture (issue #110)."""

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        import ob_analytics as ob

        return ob.Pipeline().run(str(ob.sample_csv_path()))

    def test_costs_decompose(self, result):
        costs = transaction_costs(result.trades, result.depth_summary)

        assert len(costs) == len(result.trades)
        measured = costs.dropna(subset=["realized_spread"])
        assert not measured.empty
        np.testing.assert_allclose(
            measured["realized_spread"] + measured["price_impact"],
            measured["effective_spread"],
        )

    def test_summary_reports_what_it_could_measure(self, result):
        summary = cost_summary(
            transaction_costs(result.trades, result.depth_summary, horizon="5s")
        )

        assert summary.n_trades == len(result.trades)
        assert summary.n_realized <= summary.n_trades
        assert summary.horizon == "5s"
        # The median trade pays about the quoted spread; the mean is dragged
        # up by a tail, so only the sign and the order of magnitude are held.
        assert 0 < summary.effective_spread_bps < 100

    def test_amihud_and_roll_run(self, result):
        illiq = amihud(result.trades, window="5min")
        assert len(illiq) > 1
        assert (illiq["amihud"].dropna() >= 0).all()

        # This capture trends, so Roll has no real root -- the estimate is NaN
        # and the autocovariance beside it says why.
        roll = roll_spread(result.trades)
        assert roll.iloc[0]["autocovariance"] > 0
        assert np.isnan(roll.iloc[0]["roll_spread"])
