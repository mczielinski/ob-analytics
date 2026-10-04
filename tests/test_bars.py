"""Bars: the trade stream resampled into OHLCV rows (issue #148).

Covers the five built-in rules, the columns every bar carries, the registry
that makes a rule of your own usable by name, and the gallery face.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics.bars import (
    BAR_COLUMNS,
    BAR_RULES,
    bars,
    get_bar_rule,
    list_bar_rules,
    register_bar_rule,
)
from ob_analytics.exceptions import ConfigError, ObAnalyticsError

RULES = ("time", "tick", "volume", "dollar", "imbalance")


@pytest.fixture
def toy() -> pd.DataFrame:
    """Twelve trades on a 10-second grid, with a known buy/sell pattern."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="10s", tz="UTC"),
            "price": [100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103],
            "volume": [1, 2, 1, 3, 1, 2, 1, 1, 4, 1, 2, 1],
            "direction": pd.Categorical(
                [
                    "buy",
                    "buy",
                    "sell",
                    "sell",
                    "buy",
                    "sell",
                    "buy",
                    "buy",
                    "sell",
                    "buy",
                    "buy",
                    "sell",
                ],
                categories=["buy", "sell"],
            ),
        }
    )


@pytest.fixture
def _restore_registry():
    """Snapshot / restore the global bar-rule registry around the test."""
    before = dict(BAR_RULES._items)
    yield
    BAR_RULES._items.clear()
    BAR_RULES._items.update(before)


# ---------------------------------------------------------------------------
# Shape: the same columns and the same conservation laws, whichever rule cut
# ---------------------------------------------------------------------------


class TestShape:
    @pytest.mark.parametrize("rule", RULES)
    def test_columns_and_order(self, toy: pd.DataFrame, rule: str) -> None:
        out = bars(toy, rule)

        assert list(out.columns) == list(BAR_COLUMNS)
        assert out["bar"].tolist() == list(range(len(out)))
        assert out["timestamp_start"].is_monotonic_increasing
        assert out["timestamp_end"].is_monotonic_increasing
        # Bars partition the stream: one ends before the next begins.
        assert (
            out["timestamp_end"].iloc[:-1].to_numpy()
            < out["timestamp_start"].iloc[1:].to_numpy()
        ).all()

    @pytest.mark.parametrize("rule", RULES)
    def test_every_trade_lands_in_exactly_one_bar(
        self, toy: pd.DataFrame, rule: str
    ) -> None:
        out = bars(toy, rule)

        assert out["n_trades"].sum() == len(toy)
        assert out["volume"].sum() == toy["volume"].sum()
        assert out["turnover"].sum() == (toy["price"] * toy["volume"]).sum()

    @pytest.mark.parametrize("rule", RULES)
    def test_no_empty_bars(self, toy: pd.DataFrame, rule: str) -> None:
        assert (bars(toy, rule)["n_trades"] > 0).all()

    @pytest.mark.parametrize("rule", RULES)
    def test_ohlc_brackets_the_bar(self, toy: pd.DataFrame, rule: str) -> None:
        out = bars(toy, rule)

        assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
        assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()
        assert (out["vwap"] >= out["low"]).all()
        assert (out["vwap"] <= out["high"]).all()

    @pytest.mark.parametrize("rule", RULES)
    def test_signed_volume_splits_total_volume(
        self, toy: pd.DataFrame, rule: str
    ) -> None:
        out = bars(toy, rule)

        assert (out["buy_volume"] + out["sell_volume"] == out["volume"]).all()
        assert (out["signed_volume"] == out["buy_volume"] - out["sell_volume"]).all()

    @pytest.mark.parametrize("rule", RULES)
    def test_vwap_is_turnover_over_volume(self, toy: pd.DataFrame, rule: str) -> None:
        out = bars(toy, rule)

        np.testing.assert_allclose(out["vwap"], out["turnover"] / out["volume"])

    def test_integer_prices_stay_integers(self, toy: pd.DataFrame) -> None:
        # A pipeline result holds prices as whole ticks; bars must not float them.
        out = bars(toy, "tick", 4)

        for column in ("open", "high", "low", "close"):
            assert out[column].dtype == np.int64

    def test_trades_are_sorted_before_cutting(self, toy: pd.DataFrame) -> None:
        shuffled = toy.sample(frac=1.0, random_state=0)

        pd.testing.assert_frame_equal(bars(shuffled, "tick", 4), bars(toy, "tick", 4))


# ---------------------------------------------------------------------------
# What each rule promises
# ---------------------------------------------------------------------------


class TestClockRule:
    def test_bars_are_epoch_aligned_spans(self, toy: pd.DataFrame) -> None:
        out = bars(toy, "time", "30s")

        assert len(out) == 4
        assert out["n_trades"].tolist() == [3, 3, 3, 3]
        # Every bar sits inside one 30-second span of the clock.
        assert (
            out["timestamp_start"].dt.floor("30s")
            == out["timestamp_end"].dt.floor("30s")
        ).all()

    def test_ohlcv_of_the_first_bar(self, toy: pd.DataFrame) -> None:
        first = bars(toy, "time", "30s").iloc[0]

        assert (first["open"], first["high"], first["low"], first["close"]) == (
            100,
            102,
            100,
            102,
        )
        assert first["volume"] == 4
        assert first["buy_volume"] == 3
        assert first["sell_volume"] == 1
        assert first["vwap"] == pytest.approx(404 / 4)

    def test_a_span_with_no_trades_makes_no_bar(self) -> None:
        gappy = pd.DataFrame(
            {
                # Nothing trades in the middle minute.
                "timestamp": pd.to_datetime(
                    ["2026-01-01T00:00:00Z", "2026-01-01T00:02:00Z"]
                ),
                "price": [100, 101],
                "volume": [1, 1],
                "direction": ["buy", "buy"],
            }
        )

        assert len(bars(gappy, "time", "1min")) == 2

    def test_a_calendar_step_is_not_a_duration(self, toy: pd.DataFrame) -> None:
        with pytest.raises(ConfigError, match="fixed duration"):
            bars(toy, "time", "1ME")

    def test_the_threshold_is_recorded_as_a_duration(self, toy: pd.DataFrame) -> None:
        # However it was spelled, what cut the bars is reported as a duration.
        assert bars(toy, "time", "30s").attrs["bar_threshold"] == pd.Timedelta("30s")

    def test_duration_must_be_positive(self, toy: pd.DataFrame) -> None:
        with pytest.raises(ConfigError, match="positive duration"):
            bars(toy, "time", "0s")


class TestTickRule:
    def test_every_closed_bar_holds_the_count(self, toy: pd.DataFrame) -> None:
        out = bars(toy, "tick", 4)

        assert out["n_trades"].tolist() == [4, 4, 4]

    def test_leftover_trades_close_a_short_final_bar(self, toy: pd.DataFrame) -> None:
        out = bars(toy, "tick", 5)

        assert out["n_trades"].tolist() == [5, 5, 2]

    def test_count_must_be_at_least_one(self, toy: pd.DataFrame) -> None:
        with pytest.raises(ConfigError, match="count of 1 or more"):
            bars(toy, "tick", 0)


class TestAccumulationRules:
    def test_volume_bars_reach_the_threshold(self, toy: pd.DataFrame) -> None:
        out = bars(toy, "volume", 5)

        # Every bar but the leftover one closed on reaching the threshold.
        assert (out["volume"].iloc[:-1] >= 5).all()

    def test_dollar_bars_reach_the_threshold(self, toy: pd.DataFrame) -> None:
        out = bars(toy, "dollar", 500)

        assert (out["turnover"].iloc[:-1] >= 500).all()

    def test_imbalance_bars_reach_the_threshold_either_way(
        self, toy: pd.DataFrame
    ) -> None:
        out = bars(toy, "imbalance", 3)

        assert (out["signed_volume"].iloc[:-1].abs() >= 3).all()
        # Both directions close a bar, so the signs are not all one way.
        assert set(np.sign(out["signed_volume"].iloc[:-1])) == {1, -1}

    def test_one_huge_trade_closes_its_own_bar(self) -> None:
        lumpy = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01", periods=3, freq="1s", tz="UTC"
                ),
                "price": [100, 100, 100],
                "volume": [1, 1000, 1],
                "direction": ["buy", "buy", "buy"],
            }
        )

        out = bars(lumpy, "volume", 10)

        assert out["n_trades"].tolist() == [2, 1]
        assert out["volume"].tolist() == [1001, 1]

    @pytest.mark.parametrize("rule", ("volume", "dollar", "imbalance"))
    def test_threshold_must_be_positive(self, toy: pd.DataFrame, rule: str) -> None:
        with pytest.raises(ConfigError, match="positive threshold"):
            bars(toy, rule, 0)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaultThreshold:
    @pytest.mark.parametrize("rule", RULES)
    def test_lands_near_the_target(self, rule: str) -> None:
        rng = np.random.default_rng(0)
        n = 2_000
        trades = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01", periods=n, freq="1s", tz="UTC"
                ),
                "price": 100 + np.cumsum(rng.normal(0, 0.5, n)),
                "volume": rng.lognormal(0, 1, n),
                "direction": rng.choice(["buy", "sell"], n),
            }
        )

        out = bars(trades, rule, target_bars=40)

        # A rough aim, not a guarantee: lumpy sizes and cancelling flow both
        # pull the count around. Order of magnitude is what matters.
        assert 10 <= len(out) <= 160

    @pytest.mark.parametrize("rule", RULES)
    def test_rule_and_threshold_are_recorded(
        self, toy: pd.DataFrame, rule: str
    ) -> None:
        out = bars(toy, rule)

        assert out.attrs["bar_rule"] == rule
        assert out.attrs["bar_threshold"] is not None


# ---------------------------------------------------------------------------
# Trade signs
# ---------------------------------------------------------------------------


class TestTradeSigns:
    def test_native_direction_is_used_as_is(self, toy: pd.DataFrame) -> None:
        out = bars(toy, "tick", 12)

        assert out["buy_volume"].iloc[0] == 9
        assert out["sell_volume"].iloc[0] == 11

    def test_unlabelled_trades_fall_back_to_the_tick_rule(
        self, toy: pd.DataFrame
    ) -> None:
        out = bars(toy.drop(columns=["direction"]), "tick", 12)

        assert out["volume"].iloc[0] == toy["volume"].sum()
        assert out["buy_volume"].iloc[0] + out["sell_volume"].iloc[0] == 20

    def test_quotes_drive_lee_ready(self, toy: pd.DataFrame) -> None:
        # A mid far above every trade makes Lee-Ready call them all sells.
        quotes = pd.DataFrame(
            {
                "timestamp": [toy["timestamp"].iloc[0] - pd.Timedelta("1s")],
                "best_bid_price": [500.0],
                "best_ask_price": [500.0],
            }
        )

        out = bars(toy, "tick", 12, sign_method="lee_ready", quotes=quotes)

        assert out["buy_volume"].iloc[0] == 0
        assert out["sell_volume"].iloc[0] == 20

    def test_bulk_classification_is_not_a_per_trade_sign(
        self, toy: pd.DataFrame
    ) -> None:
        with pytest.raises(ConfigError, match="bvc"):
            bars(toy, "tick", 4, sign_method="bvc")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_rule_lists_the_registered_ones(self, toy: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="volume"):
            bars(toy, "fortnightly")

    def test_empty_trades(self) -> None:
        empty = pd.DataFrame(columns=["timestamp", "price", "volume"])

        with pytest.raises(ObAnalyticsError):
            bars(empty)

    def test_missing_columns(self, toy: pd.DataFrame) -> None:
        with pytest.raises(ConfigError, match="volume"):
            bars(toy.drop(columns=["volume"]))

    def test_a_single_trade_is_one_bar(self, toy: pd.DataFrame) -> None:
        out = bars(toy.head(1))

        assert len(out) == 1
        assert out["n_trades"].iloc[0] == 1
        assert out["open"].iloc[0] == out["close"].iloc[0] == 100


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_built_ins_are_registered(self) -> None:
        assert set(list_bar_rules()) == set(RULES)

    def test_lookup_is_case_insensitive(self) -> None:
        assert get_bar_rule("VOLUME") is get_bar_rule("volume")

    def test_a_rule_of_your_own_is_usable_by_name(
        self, toy: pd.DataFrame, _restore_registry
    ) -> None:
        class EveryOtherTrade:
            """A stand-in third-party rule, defined outside the core."""

            name = "every_other"

            def default_threshold(self, frame, target_bars):
                return 2

            def normalize(self, threshold):
                return int(threshold)

            def assign(self, frame, threshold):
                return np.arange(len(frame)) // int(threshold)

        register_bar_rule(EveryOtherTrade())

        out = bars(toy, "every_other")

        assert "every_other" in list_bar_rules()
        assert out["n_trades"].tolist() == [2] * 6

    def test_a_rule_returning_the_wrong_length_is_rejected(
        self, toy: pd.DataFrame, _restore_registry
    ) -> None:
        class Truncating:
            name = "truncating"

            def default_threshold(self, frame, target_bars):
                return 1

            def normalize(self, threshold):
                return threshold

            def assign(self, frame, threshold):
                return np.zeros(len(frame) - 1, dtype=np.int64)

        register_bar_rule(Truncating())

        with pytest.raises(ConfigError, match="bar indices"):
            bars(toy, "truncating")


# ---------------------------------------------------------------------------
# The gallery face
# ---------------------------------------------------------------------------


class TestFace:
    @pytest.mark.parametrize("backend", ("matplotlib", "plotly"))
    @pytest.mark.parametrize("rule", RULES)
    def test_renders_on_both_backends(
        self, toy: pd.DataFrame, rule: str, backend: str
    ) -> None:
        pytest.importorskip(backend)
        from ob_analytics.visualization import plot, prepare

        fig = plot("bars", backend=backend, **prepare.bars(bars(toy, rule)))

        assert fig is not None

    def test_panel_labels_itself_from_the_frame(self, toy: pd.DataFrame) -> None:
        from ob_analytics.visualization.gallery import bars_panel

        spec = bars_panel(bars(toy, "tick", 4))

        assert spec.plot_name == "bars"
        assert spec.title == "Bars (tick, 4)"

    def test_two_cuts_get_separate_file_stems(self, toy: pd.DataFrame) -> None:
        # Both draw the "bars" concept, but each needs its own image file.
        from ob_analytics.visualization.gallery import bars_panel

        clock = bars_panel(bars(toy, "time", "30s"))
        volume = bars_panel(bars(toy, "volume", 5))

        assert clock.plot_name == volume.plot_name == "bars"
        assert clock.name != volume.name

    def test_panel_takes_an_explicit_label(self, toy: pd.DataFrame) -> None:
        from ob_analytics.visualization.gallery import bars_panel

        assert bars_panel(bars(toy, "tick", 4), label="every 4th").title == (
            "Bars (every 4th)"
        )

    def test_clock_bars_get_a_time_axis(self, toy: pd.DataFrame) -> None:
        from ob_analytics.visualization import prepare

        data = prepare.bars(bars(toy, "time", "30s"))

        assert data["x_axis"] == "time"
        assert data["bar_width"] == pd.Timedelta("30s") * 0.8
        # Each candle sits over the middle of its own span.
        assert (
            data["x"]
            == pd.to_datetime(
                [
                    "2026-01-01T00:00:15Z",
                    "2026-01-01T00:00:45Z",
                    "2026-01-01T00:01:15Z",
                    "2026-01-01T00:01:45Z",
                ]
            )
        ).all()

    def test_clock_bars_never_overlap(self, toy: pd.DataFrame) -> None:
        from ob_analytics.visualization import prepare

        data = prepare.bars(bars(toy, "time", "30s"))
        gaps = data["x"].diff().dropna()

        assert (gaps >= data["bar_width"]).all()

    @pytest.mark.parametrize("rule", ("tick", "volume", "dollar", "imbalance"))
    def test_activity_bars_get_one_slot_each(
        self, toy: pd.DataFrame, rule: str
    ) -> None:
        # Activity bars cover wildly unequal spans, so a time axis would pile
        # them on top of each other; they are evenly spaced instead.
        from ob_analytics.visualization import prepare

        cut = bars(toy, rule)
        data = prepare.bars(cut)

        assert data["x_axis"] == "ordinal"
        assert data["x"].tolist() == list(range(len(cut)))
        assert data["bar_width"] < 1.0
        positions, labels = data["ticks"]
        assert len(positions) == len(labels) > 0
        assert all(labels)

    def test_a_frame_with_no_provenance_falls_back_to_slots(
        self, toy: pd.DataFrame
    ) -> None:
        from ob_analytics.visualization import prepare

        cut = bars(toy, "time", "30s")
        cut.attrs.clear()

        data = prepare.bars(cut)

        assert data["x_axis"] == "ordinal"
        assert data["label"] == ""
