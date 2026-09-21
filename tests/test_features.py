"""The feature table: one row per bar, one column per measurement (issue #149).

Covers the table's shape, the ten shipped features and their arithmetic, the
registry that makes a feature of your own usable by name, what happens when
the inputs do not carry what a feature reads, and — the point of the whole
thing — that no row reads data from after its own close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics.exceptions import ConfigError, ObAnalyticsError
from ob_analytics.features import (
    DEFAULT_FEATURES,
    FEATURES,
    INDEX_COLUMNS,
    KyleLambdaFeature,
    ReturnsFeature,
    VpinFeature,
    features,
    get_feature,
    list_features,
    readable_quotes,
    register_feature,
)

BOOK_FEATURES = ("spread", "mid_price", "micro_price", "imbalance", "depth")
#: The book features whose columns a single unreadable quote would corrupt.
READS_BOOK = ("spread", "mid_price", "micro_price", "imbalance")
TRADE_FEATURES = ("price", "returns", "flow", "vpin", "kyle_lambda")

#: A rule plus a threshold to cut with.  The threshold is always explicit: a
#: defaulted one is worked out from the whole trades frame, so the truncation
#: tests below would be comparing two different cuts.
CUTS = (
    ("time", "30s"),
    ("tick", 4),
    ("volume", 12.0),
    ("dollar", 1200.0),
    ("imbalance", 6.0),
)


@pytest.fixture
def toy_trades() -> pd.DataFrame:
    """Forty trades on a 5-second grid, with a repeating buy/sell pattern."""
    n = 40
    rng = np.random.default_rng(20260917)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="5s", tz="UTC"),
            "price": 100 + np.cumsum(rng.integers(-2, 3, size=n)),
            "volume": rng.integers(1, 6, size=n).astype(float),
            "direction": pd.Categorical(
                np.where(rng.random(n) > 0.45, "buy", "sell"),
                categories=["buy", "sell"],
            ),
        }
    )


@pytest.fixture
def toy_quotes(toy_trades: pd.DataFrame) -> pd.DataFrame:
    """A book snapshot between every pair of trades, with two depth bins."""
    n = 2 * len(toy_trades)
    rng = np.random.default_rng(626)
    mid = 100 + np.cumsum(rng.integers(-1, 2, size=n))
    half = rng.integers(1, 4, size=n)
    return pd.DataFrame(
        {
            # Offset half a step, so a quote falls between consecutive trades.
            "timestamp": pd.date_range(
                "2025-12-31 23:59:57.5", periods=n, freq="2500ms", tz="UTC"
            ),
            "best_bid_price": (mid - half).astype(float),
            "best_bid_vol": rng.integers(1, 20, size=n).astype(float),
            "best_ask_price": (mid + half).astype(float),
            "best_ask_vol": rng.integers(1, 20, size=n).astype(float),
            "bid_vol25bps": rng.integers(20, 60, size=n).astype(float),
            "bid_vol50bps": rng.integers(20, 60, size=n).astype(float),
            "ask_vol25bps": rng.integers(20, 60, size=n).astype(float),
            "ask_vol50bps": rng.integers(20, 60, size=n).astype(float),
        }
    )


@pytest.fixture
def _restore_registry():
    """Snapshot / restore the global feature registry around the test."""
    before = dict(FEATURES._items)
    yield
    FEATURES._items.clear()
    FEATURES._items.update(before)


class ConstantFeature:
    """A feature of one's own, writing one column of a known value."""

    name = "constant"
    columns = ("constant",)
    requires = frozenset()

    def compute(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {"constant": np.full(len(frame), 7.0)}


# ---------------------------------------------------------------------------
# No look-ahead: the reason the table exists in this shape
# ---------------------------------------------------------------------------


class TestNoLookAhead:
    """Truncating the inputs must not change the rows that survive.

    If any feature read data from after a row's close, removing that data
    would change the row.  Every bar rule cuts forward from the first trade,
    so a truncated run reproduces the earlier boundaries exactly and the two
    tables can be compared row for row.  The last row of the truncated table
    is dropped: its bar is the one the cut fell inside, and it is genuinely
    short of trades.
    """

    @staticmethod
    def _truncated(
        trades: pd.DataFrame, quotes: pd.DataFrame | None, share: float
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        cut = trades["timestamp"].iloc[int(len(trades) * share)]
        part_quotes = (
            None if quotes is None else quotes[quotes["timestamp"] <= cut].copy()
        )
        return trades[trades["timestamp"] <= cut].copy(), part_quotes

    @pytest.mark.parametrize(("rule", "threshold"), CUTS)
    def test_prefix_is_unchanged(self, toy_trades, toy_quotes, rule, threshold):
        full = features(toy_trades, toy_quotes, rule, threshold)
        part_trades, part_quotes = self._truncated(toy_trades, toy_quotes, 0.6)
        part = features(part_trades, part_quotes, rule, threshold).iloc[:-1]

        assert len(part) > 3, "the truncated run must be long enough to mean anything"
        pd.testing.assert_frame_equal(full.iloc[: len(part)], part)

    @pytest.mark.parametrize("name", DEFAULT_FEATURES)
    def test_each_feature_alone(self, toy_trades, toy_quotes, name):
        """Named one at a time, so a failure says which feature peeked."""
        full = features(toy_trades, toy_quotes, "tick", 4, include=[name])
        part_trades, part_quotes = self._truncated(toy_trades, toy_quotes, 0.6)
        part = features(part_trades, part_quotes, "tick", 4, include=[name]).iloc[:-1]

        pd.testing.assert_frame_equal(full.iloc[: len(part)], part)

    def test_later_quotes_do_not_reach_earlier_rows(self, toy_trades, toy_quotes):
        """Rewriting the book after a row's close leaves that row alone."""
        table = features(toy_trades, toy_quotes, "tick", 4)
        cut = table["timestamp"].iloc[5]

        tampered = toy_quotes.copy()
        later = tampered["timestamp"] > cut
        tampered.loc[later, "best_bid_price"] = 1.0
        tampered.loc[later, "best_ask_price"] = 999.0
        after = features(toy_trades, tampered, "tick", 4)

        pd.testing.assert_frame_equal(table.iloc[:6], after.iloc[:6])
        assert not table.iloc[6:]["spread"].equals(after.iloc[6:]["spread"])

    def test_a_feature_that_looks_forward_is_caught(
        self, toy_trades, _restore_registry
    ):
        """The truncation test has teeth: a peeking feature fails it."""

        class PeekingFeature:
            name = "peek"
            columns = ("peek",)
            requires = frozenset({"close"})

            def compute(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
                return {"peek": frame["close"].shift(-1).to_numpy(dtype=float)}

        register_feature(PeekingFeature())
        full = features(toy_trades, None, "tick", 4, include=["peek"])
        part_trades, _ = self._truncated(toy_trades, None, 0.6)
        part = features(part_trades, None, "tick", 4, include=["peek"]).iloc[:-1]

        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(full.iloc[: len(part)], part)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class TestShape:
    @pytest.mark.parametrize(("rule", "threshold"), CUTS)
    def test_columns_and_order(self, toy_trades, toy_quotes, rule, threshold):
        table = features(toy_trades, toy_quotes, rule, threshold)

        assert tuple(table.columns[:3]) == INDEX_COLUMNS
        for name in DEFAULT_FEATURES:
            for column in get_feature(name).columns:
                assert column in table.columns
        assert table.columns.is_unique

    def test_one_row_per_bar(self, toy_trades, toy_quotes):
        from ob_analytics.bars import bars

        table = features(toy_trades, toy_quotes, "volume", 12.0)
        bar_table = bars(toy_trades, "volume", 12.0)

        assert len(table) == len(bar_table)
        assert table["bar"].tolist() == bar_table["bar"].tolist()
        pd.testing.assert_series_equal(
            table["timestamp"], bar_table["timestamp_end"], check_names=False
        )

    def test_timestamps_are_ordered(self, toy_trades, toy_quotes):
        table = features(toy_trades, toy_quotes, "tick", 4)

        assert table["timestamp"].is_monotonic_increasing
        assert (table["timestamp_start"] <= table["timestamp"]).all()

    def test_target_bars_sets_the_default_cut(self, toy_trades, toy_quotes):
        table = features(toy_trades, toy_quotes, "tick", target_bars=8)

        assert len(table) == pytest.approx(8, abs=2)
        assert table.attrs["bar_rule"] == "tick"

    def test_attrs_record_the_cut_and_the_features(self, toy_trades, toy_quotes):
        table = features(toy_trades, toy_quotes, "volume", 12.0)

        assert table.attrs["bar_rule"] == "volume"
        assert table.attrs["bar_threshold"] == 12.0
        assert table.attrs["features"] == list(DEFAULT_FEATURES)
        assert table.attrs["features_skipped"] == []

    def test_empty_trades_raise(self, toy_quotes):
        with pytest.raises(ObAnalyticsError):
            features(toy_trades_empty(), toy_quotes)


def toy_trades_empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([], utc=True),
            "price": pd.Series(dtype=float),
            "volume": pd.Series(dtype=float),
        }
    )


# ---------------------------------------------------------------------------
# Selecting features, and what happens without quotes
# ---------------------------------------------------------------------------


class TestSelection:
    def test_no_quotes_skips_the_book_features(self, toy_trades):
        table = features(toy_trades)

        assert table.attrs["features"] == list(TRADE_FEATURES)
        assert sorted(table.attrs["features_skipped"]) == sorted(BOOK_FEATURES)
        assert "spread" not in table.columns
        assert "close" in table.columns

    def test_a_named_book_feature_without_quotes_raises(self, toy_trades):
        with pytest.raises(ConfigError, match="spread"):
            features(toy_trades, include=["spread"])

    def test_include_sets_the_column_order(self, toy_trades, toy_quotes):
        table = features(toy_trades, toy_quotes, include=["imbalance", "price"])

        assert list(table.columns) == [
            *INDEX_COLUMNS,
            "obi",
            "obi_depth",
            "open",
            "high",
            "low",
            "close",
            "vwap",
        ]

    def test_include_empty_gives_the_index_alone(self, toy_trades, toy_quotes):
        table = features(toy_trades, toy_quotes, include=[])

        assert list(table.columns) == list(INDEX_COLUMNS)
        assert table.attrs["features"] == []

    @pytest.mark.parametrize("include", [["returns", "returns"], ["spread", "Spread"]])
    def test_a_repeated_name_is_named_in_the_error(
        self, toy_trades, toy_quotes, include
    ):
        with pytest.raises(ConfigError, match="more than once"):
            features(toy_trades, toy_quotes, include=include)

    @pytest.mark.parametrize("include", ["spread", "returns"])
    def test_a_bare_string_is_refused_not_split(self, toy_trades, include):
        """A string is a sequence of strings; read as one it becomes letters."""
        with pytest.raises(ConfigError, match=rf"include=\[{include!r}\]"):
            features(toy_trades, include=include)

    def test_include_mistakes_are_reported_before_any_bars(self, toy_quotes):
        """No trades are read when the names alone are wrong."""
        empty = pd.DataFrame(columns=["timestamp", "price", "volume"])
        with pytest.raises(ConfigError, match="more than once"):
            features(empty, toy_quotes, include=["spread", "spread"])

    def test_unknown_feature_lists_the_registered_ones(self, toy_trades):
        with pytest.raises(KeyError, match="spread"):
            features(toy_trades, include=["not_a_feature"])

    def test_quotes_without_a_timestamp_raise(self, toy_trades, toy_quotes):
        with pytest.raises(ConfigError, match="timestamp"):
            features(toy_trades, toy_quotes.drop(columns=["timestamp"]))


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_shipped_features_are_registered(self):
        assert set(DEFAULT_FEATURES) <= set(list_features())

    def test_a_feature_of_your_own(self, toy_trades, toy_quotes, _restore_registry):
        register_feature(ConstantFeature())
        table = features(toy_trades, toy_quotes)

        assert (table["constant"] == 7.0).all()
        # Registered after the shipped ten, so its column comes last.
        assert table.columns[-1] == "constant"
        assert table.attrs["features"][-1] == "constant"

    def test_lookup_is_case_insensitive(self):
        assert get_feature("MICRO_PRICE").name == "micro_price"

    def test_a_feature_may_shadow_a_built_in(
        self, toy_trades, toy_quotes, _restore_registry
    ):
        class FlatSpread:
            name = "spread"
            columns = ("spread", "spread_bps")
            requires = frozenset()

            def compute(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
                return {
                    "spread": np.zeros(len(frame)),
                    "spread_bps": np.zeros(len(frame)),
                }

        register_feature(FlatSpread())
        table = features(toy_trades, toy_quotes)

        assert (table["spread"] == 0.0).all()

    def test_a_one_column_feature_names_its_column_after_itself(
        self, toy_trades, _restore_registry
    ):
        """So a second window registered beside the built-in has its own column."""
        register_feature(VpinFeature(name="vpin_5", window=5))
        table = features(toy_trades, None, "tick", 2, include=["vpin", "vpin_5"])

        assert list(table.columns[3:]) == ["vpin", "vpin_5"]
        assert not table["vpin"].equals(table["vpin_5"])

    def test_two_features_cannot_share_a_column(self, toy_trades, _restore_registry):
        """A renamed multi-column feature is refused, not silently merged."""
        register_feature(ReturnsFeature(name="returns_60", window=60))

        with pytest.raises(ConfigError, match="log_return"):
            features(toy_trades, None, "tick", 2, include=["returns", "returns_60"])

    def test_undeclared_column_is_reported(
        self, toy_trades, toy_quotes, _restore_registry
    ):
        class Mismatched:
            name = "mismatched"
            columns = ("declared",)
            requires = frozenset()

            def compute(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
                return {"returned": np.zeros(len(frame))}

        register_feature(Mismatched())
        with pytest.raises(ConfigError, match="declares columns"):
            features(toy_trades, toy_quotes, include=["mismatched"])

    def test_wrong_length_is_reported(self, toy_trades, toy_quotes, _restore_registry):
        class TooShort:
            name = "too_short"
            columns = ("too_short",)
            requires = frozenset()

            def compute(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
                return {"too_short": np.zeros(len(frame) - 1)}

        register_feature(TooShort())
        with pytest.raises(ConfigError, match="too_short"):
            features(toy_trades, toy_quotes, include=["too_short"])


# ---------------------------------------------------------------------------
# What each feature measures
# ---------------------------------------------------------------------------


class TestTradeFeatures:
    def test_price_columns_come_from_the_bars(self, toy_trades):
        from ob_analytics.bars import bars

        table = features(toy_trades, include=["price"])
        bar_table = bars(toy_trades, "time", None, target_bars=50)

        for column in ("open", "high", "low", "close", "vwap"):
            pd.testing.assert_series_equal(table[column], bar_table[column])

    def test_log_return_is_close_over_previous_close(self, toy_trades):
        table = features(toy_trades, None, "tick", 4, include=["price", "returns"])
        close = table["close"].to_numpy(dtype=float)

        assert np.isnan(table["log_return"].iloc[0])
        np.testing.assert_allclose(
            table["log_return"].to_numpy()[1:], np.log(close[1:] / close[:-1])
        )

    def test_realized_vol_is_the_trailing_standard_deviation(self, toy_trades):
        table = features(toy_trades, None, "tick", 2, include=["returns"])
        returns = table["log_return"]

        expected = returns.rolling(ReturnsFeature().window, min_periods=2).std(ddof=1)
        pd.testing.assert_series_equal(
            table["realized_vol"], expected, check_names=False
        )

    def test_trade_imbalance_is_the_signed_share(self, toy_trades):
        table = features(toy_trades, None, "tick", 4, include=["flow"])

        np.testing.assert_allclose(
            table["trade_imbalance"],
            table["signed_volume"] / table["volume"],
        )
        assert table["trade_imbalance"].between(-1.0, 1.0).all()

    def test_vpin_is_the_trailing_mean_absolute_imbalance(self, toy_trades):
        table = features(toy_trades, None, "volume", 12.0, include=["flow", "vpin"])

        imbalance = (table["signed_volume"] / table["volume"]).abs()
        expected = imbalance.rolling(VpinFeature().window, min_periods=1).mean()
        np.testing.assert_allclose(table["vpin"], expected)
        assert table["vpin"].between(0.0, 1.0).all()

    def test_kyle_lambda_is_the_trailing_regression_slope(self, toy_trades):
        table = features(
            toy_trades, None, "tick", 2, include=["price", "flow", "kyle_lambda"]
        )
        flow = table["signed_volume"].to_numpy(dtype=float)
        move = (table["close"] - table["open"]).to_numpy(dtype=float)
        window = KyleLambdaFeature().window

        row = len(table) - 1
        start = max(0, row - window + 1)
        x, y = flow[start : row + 1], move[start : row + 1]
        expected = np.polyfit(x, y, 1)[0]

        assert table["kyle_lambda"].iloc[row] == pytest.approx(expected)

    def test_trailing_windows_start_empty(self, toy_trades):
        table = features(
            toy_trades, None, "tick", 4, include=["returns", "kyle_lambda"]
        )

        assert np.isnan(table["log_return"].iloc[0])
        assert np.isnan(table["realized_vol"].iloc[0])
        assert np.isnan(table["kyle_lambda"].iloc[0])

    @pytest.mark.parametrize(
        ("make", "bad"),
        [
            (ReturnsFeature, 1),
            (ReturnsFeature, 0),
            (VpinFeature, 0),
            (KyleLambdaFeature, 1),
            (ReturnsFeature, 2.5),
            (VpinFeature, True),
        ],
    )
    def test_an_unusable_window_is_refused_when_built(self, make, bad):
        """Reported against the feature, not later from inside pandas."""
        with pytest.raises(ConfigError, match="window"):
            make(window=bad)

    def test_the_smallest_usable_windows_are_accepted(self):
        ReturnsFeature(window=2)
        VpinFeature(window=1)
        KyleLambdaFeature(window=2)

    @pytest.mark.parametrize("window", [np.int64(5), np.int32(5), np.uint8(5)])
    def test_a_numpy_integer_window_is_accepted(self, toy_trades, window):
        """A window worked out from data is often a numpy integer."""
        numpy_window = VpinFeature(name="vpin_np", window=window)
        python_window = VpinFeature(name="vpin_py", window=5)
        # The flow columns are what VPIN reads, so the table serves as its input.
        frame = features(toy_trades, None, "tick", 2, include=["flow"])

        np.testing.assert_allclose(
            np.asarray(numpy_window.compute(frame)["vpin_np"], dtype=float),
            np.asarray(python_window.compute(frame)["vpin_py"], dtype=float),
        )

    def test_a_reconfigured_window_is_honoured(self, toy_trades, _restore_registry):
        register_feature(ReturnsFeature(window=3))
        table = features(toy_trades, None, "tick", 2, include=["returns"])

        expected = table["log_return"].rolling(3, min_periods=2).std(ddof=1)
        pd.testing.assert_series_equal(
            table["realized_vol"], expected, check_names=False
        )


class TestBookFeatures:
    def test_the_book_is_the_one_prevailing_at_the_close(self, toy_trades, toy_quotes):
        table = features(toy_trades, toy_quotes, "tick", 4, include=["spread"])

        for row in range(len(table)):
            at = table["timestamp"].iloc[row]
            prevailing = toy_quotes[toy_quotes["timestamp"] <= at].iloc[-1]
            expected = prevailing["best_ask_price"] - prevailing["best_bid_price"]
            assert table["spread"].iloc[row] == pytest.approx(expected)

    def test_a_quote_at_the_close_counts(self, toy_trades, toy_quotes):
        """A quote stamped exactly at the close is part of the row's past."""
        table = features(toy_trades, toy_quotes, "tick", 4, include=["spread"])
        at = table["timestamp"].iloc[3]
        assert (toy_quotes["timestamp"] == at).any(), "the fixture must line up"

        moved = toy_quotes.copy()
        moved.loc[moved["timestamp"] == at, "best_bid_price"] -= 50.0
        after = features(toy_trades, moved, "tick", 4, include=["spread"])

        assert after["spread"].iloc[3] == table["spread"].iloc[3] + 50.0
        assert after["spread"].iloc[2] == table["spread"].iloc[2]

    def test_rows_before_the_first_quote_are_nan(self, toy_trades, toy_quotes):
        late = toy_quotes[toy_quotes["timestamp"] > toy_trades["timestamp"].iloc[6]]
        table = features(toy_trades, late, "tick", 2, include=["spread", "depth"])

        assert table[["spread", "bid_depth"]].iloc[0].isna().all()
        assert table[["spread", "bid_depth"]].iloc[-1].notna().all()

    def test_mid_and_micro_price(self, toy_trades, toy_quotes):
        from ob_analytics.depth import micro_price

        table = features(
            toy_trades, toy_quotes, "tick", 4, include=["mid_price", "micro_price"]
        )
        book = _book_at(table, toy_quotes)

        expected_mid = (book["best_bid_price"] + book["best_ask_price"]) / 2.0
        np.testing.assert_allclose(table["mid_price"], expected_mid)
        np.testing.assert_allclose(table["micro_price"], micro_price(book))
        np.testing.assert_allclose(
            table["micro_price_offset"], table["micro_price"] - table["mid_price"]
        )

    def test_micro_price_leans_toward_the_heavier_opposite_side(self):
        """A big bid and a thin ask put the micro-price above the mid."""
        trades = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01", periods=2, freq="1s", tz="UTC"
                ),
                "price": [100.0, 100.0],
                "volume": [1.0, 1.0],
                "direction": ["buy", "sell"],
            }
        )
        quotes = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01", periods=2, freq="1s", tz="UTC"
                ),
                "best_bid_price": [99.0, 99.0],
                "best_bid_vol": [100.0, 1.0],
                "best_ask_price": [101.0, 101.0],
                "best_ask_vol": [1.0, 100.0],
            }
        )
        table = features(trades, quotes, "tick", 1, include=["micro_price"])

        assert table["micro_price_offset"].iloc[0] > 0
        assert table["micro_price_offset"].iloc[1] < 0

    def test_spread_in_bps_is_the_spread_over_the_mid(self, toy_trades, toy_quotes):
        table = features(
            toy_trades, toy_quotes, "tick", 4, include=["spread", "mid_price"]
        )

        np.testing.assert_allclose(
            table["spread_bps"], table["spread"] / table["mid_price"] * 10_000.0
        )

    def test_imbalance_matches_book_imbalance(self, toy_trades, toy_quotes):
        from ob_analytics.depth import book_imbalance

        table = features(toy_trades, toy_quotes, "tick", 4, include=["imbalance"])
        book = _book_at(table, toy_quotes)

        np.testing.assert_allclose(table["obi"], book_imbalance(book, levels=1))
        # Two depth bins on each side, so the clamp brings five levels to three.
        np.testing.assert_allclose(table["obi_depth"], book_imbalance(book, levels=3))
        assert table["obi"].between(-1.0, 1.0).all()

    def test_depth_sums_every_bin(self, toy_trades, toy_quotes):
        table = features(toy_trades, toy_quotes, "tick", 4, include=["depth"])
        book = _book_at(table, toy_quotes)

        np.testing.assert_allclose(
            table["bid_depth"], book["bid_vol25bps"] + book["bid_vol50bps"]
        )
        np.testing.assert_allclose(
            table["ask_depth"], book["ask_vol25bps"] + book["ask_vol50bps"]
        )

    def test_depth_without_bins_reports_the_touch(self, toy_trades, toy_quotes):
        touch_only = toy_quotes[
            [
                "timestamp",
                "best_bid_price",
                "best_bid_vol",
                "best_ask_price",
                "best_ask_vol",
            ]
        ]
        table = features(toy_trades, touch_only, "tick", 4, include=["depth"])

        np.testing.assert_allclose(table["bid_depth"], table["best_bid_vol"])
        np.testing.assert_allclose(table["ask_depth"], table["best_ask_vol"])

    @staticmethod
    def _build(toy_trades, quotes):
        return features(toy_trades, quotes, "tick", 4, include=READS_BOOK)

    def _spoil_a_quote_a_row_reads(self, toy_trades, toy_quotes, bar, **values):
        """Spoil the quote *bar* actually reads, and also delete it.

        Picking the quote by position would usually land on one no bar reads,
        and then spoiling it changes nothing and the test passes however the
        join behaves.  The quote prevailing at a chosen row's close is one the
        table demonstrably depends on, which is what gives the comparison
        below its teeth.

        Returns the three tables: untouched, spoiled, and with that quote
        deleted.  Skipping a quote and never having it are the same thing, so
        the last two must match — and the first must differ from them, or the
        quote was not load-bearing after all.
        """
        plain = self._build(toy_trades, toy_quotes)
        at = plain["timestamp"].iloc[bar]
        reads = toy_quotes.index[toy_quotes["timestamp"] <= at][-1]

        spoiled = toy_quotes.copy()
        for column, value in values.items():
            spoiled.loc[reads, column] = value
        return (
            plain,
            self._build(toy_trades, spoiled),
            self._build(toy_trades, toy_quotes.drop(index=reads)),
        )

    def test_an_empty_book_side_is_not_read_as_a_price(self, toy_trades, toy_quotes):
        """The depth engine writes 0 for an empty side; it is a marker, not a price."""
        plain, spoiled, deleted = self._spoil_a_quote_a_row_reads(
            toy_trades, toy_quotes, bar=4, best_bid_price=0.0, best_bid_vol=0.0
        )

        pd.testing.assert_frame_equal(spoiled, deleted)
        assert not spoiled["spread"].equals(plain["spread"])
        # Read at face value that quote gives a spread the width of the
        # instrument, a mid at half the ask, and a micro-price of zero.
        assert (spoiled["spread"].dropna() < 20.0).all()
        assert (spoiled["micro_price"].dropna() > 0).all()

    def test_a_crossed_book_is_not_read_as_a_spread(self, toy_trades, toy_quotes):
        """A diff feed holds crossed orders; their midpoint is not a price."""
        plain, spoiled, deleted = self._spoil_a_quote_a_row_reads(
            toy_trades,
            toy_quotes,
            bar=4,
            best_bid_price=toy_quotes["best_ask_price"].max() + 5.0,
        )

        pd.testing.assert_frame_equal(spoiled, deleted)
        assert not spoiled["spread"].equals(plain["spread"])
        assert (spoiled["spread"].dropna() >= 0).all()

    def test_a_locked_book_is_kept(self, toy_trades, toy_quotes):
        """Bid equal to ask is a real state at a spread of zero, not a fault."""
        locked = toy_quotes.copy()
        at = locked["timestamp"].iloc[20]
        row = locked["timestamp"] == at
        locked.loc[row, "best_bid_price"] = locked.loc[row, "best_ask_price"]

        readable = readable_quotes(locked)

        assert (readable["timestamp"] == at).any()

    def test_no_readable_quote_gives_nan_not_zero(self, toy_trades, toy_quotes):
        """A row with nothing to read takes NaN, never a zero that reads as a price.

        The as-of join widens a whole-number price column to hold the NaN; a
        pipeline frame carries prices as integer ticks, so a fill of ``0``
        here would be indistinguishable from the empty-side marker the join
        was just taught to skip.
        """
        # Integer columns, the way a pipeline frame carries ticks and lots.
        integral = toy_quotes.astype(
            {column: "int64" for column in toy_quotes.columns if column != "timestamp"}
        )
        unreadable = integral.assign(best_bid_price=0, best_bid_vol=0)
        table = features(toy_trades, unreadable, "tick", 4, include=["spread", "depth"])

        assert table[["spread", "best_bid_vol", "bid_depth"]].isna().all().all()
        assert table["spread"].dtype == np.float64

        # And with some readable quotes, the rows before the first still do.
        late = integral[integral["timestamp"] > toy_trades["timestamp"].iloc[6]]
        partial = features(toy_trades, late, "tick", 2, include=["spread", "depth"])

        assert partial[["spread", "best_bid_vol"]].iloc[0].isna().all()
        assert partial[["spread", "best_bid_vol"]].iloc[-1].notna().all()

    def test_readable_quotes_leaves_an_untestable_frame_alone(self, toy_quotes):
        """A frame with no best bid/ask pair cannot be tested for either fault."""
        mid_only = toy_quotes[["timestamp"]].assign(mid=100.0)

        pd.testing.assert_frame_equal(readable_quotes(mid_only), mid_only)

    def test_a_quote_column_never_shadows_a_bar_column(self, toy_trades, toy_quotes):
        """A quotes frame carrying `close` must not displace the bar's own."""
        shadowing = toy_quotes.assign(close=-1.0)
        table = features(toy_trades, shadowing, "tick", 4, include=["price"])

        assert (table["close"] > 0).all()


def _book_at(table: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """The quote prevailing at each row's close, as its own frame."""
    return (
        pd.merge_asof(
            table[["timestamp"]],
            quotes.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        .drop(columns=["timestamp"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# The bundled sample: one call, a documented table
# ---------------------------------------------------------------------------


class TestBundledSample:
    def test_one_call_on_the_sample(self, sample_csv_path):
        from ob_analytics import Pipeline

        result = Pipeline().run(sample_csv_path)
        table = features(result.trades, result.depth_summary, "volume", 100)

        assert len(table) > 50
        assert table.attrs["features_skipped"] == []
        # Every shipped column is there, and the book was found for every row
        # (the depth summary starts before the first trade).
        for name in DEFAULT_FEATURES:
            for column in get_feature(name).columns:
                assert column in table.columns
        assert table[["spread", "mid_price", "obi"]].notna().all().all()
        assert (table["spread"] >= 0).all()
        assert table["vpin"].between(0.0, 1.0).all()
