"""Tests for ob_analytics._utils."""

import numpy as np
import pandas as pd
import pytest

from ob_analytics._utils import (
    decimal_places,
    lots_to_size,
    ticks_to_price_if_integer,
    validate_columns,
    validate_non_empty,
)
from ob_analytics.exceptions import ConfigError, ObAnalyticsError


class TestTicksToPriceIfInteger:
    def test_integer_ticks_are_converted(self):
        out = ticks_to_price_if_integer(
            pd.Series([25001, 25002], dtype="int64"), 0.01, decimals=2
        )
        assert list(out) == [250.01, 250.02]

    def test_float_prices_are_returned_unchanged(self):
        # Already a quote-currency price: converting again would rescale it.
        prices = pd.Series([250.01, 250.02])
        out = ticks_to_price_if_integer(prices, 0.01, decimals=2)
        assert out is prices

    def test_empty_integer_series_is_safe(self):
        out = ticks_to_price_if_integer(pd.Series([], dtype="int64"), 0.01)
        assert len(out) == 0


class TestDecimalPlaces:
    def test_whole_numbers_need_no_decimals(self):
        assert decimal_places([1.0, 2.0, 300.0]) == 0

    def test_satoshi_sizes_sit_on_eight_decimals(self):
        sizes = lots_to_size(np.array([1, 12345678, 250000000]), 1e-8, decimals=8)
        assert decimal_places(sizes) == 8

    def test_the_fewest_decimals_win(self):
        assert decimal_places([0.1, 0.02, 0.5]) == 2

    def test_float_arithmetic_is_on_no_grid(self):
        # 1000.1 - 1000.0 is 0.10000000000002274; it is not moved to 0.1.
        assert decimal_places([1000.1 - 1000.0, 0.25]) is None

    def test_a_large_size_stays_on_the_grid(self):
        # 138,800 BTC (the largest size in the bundled sample) is 1.388e13
        # satoshis, still inside the whole numbers a float holds exactly.
        sizes = lots_to_size(np.array([13880000000000, 3]), 1e-8, decimals=8)
        assert decimal_places(sizes) == 8

    def test_a_tiny_size_keeps_its_decimals(self):
        assert decimal_places([2.0, 3e-8]) == 8

    def test_no_decimal_grid_returns_none(self):
        assert decimal_places([1 / 3]) is None

    def test_a_value_that_is_not_finite_returns_none(self):
        assert decimal_places([0.1, np.nan]) is None
        assert decimal_places([0.1, np.inf]) is None

    def test_a_value_past_exact_float_integers_returns_none(self):
        assert decimal_places([2.0**60]) is None

    def test_a_sum_past_int64_returns_none(self):
        # Each value is exact, but together they do not fit in int64.
        assert decimal_places(np.full(1024, 2.0**52)) is None


class TestValidateColumns:
    def test_passes_when_columns_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_columns(df, {"a", "b"}, "test")

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ConfigError, match="missing required columns.*'c'"):
            validate_columns(df, {"a", "c"}, "test")

    def test_error_message_includes_context(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ConfigError, match="my_function"):
            validate_columns(df, {"y"}, "my_function")

    def test_empty_required_passes(self):
        df = pd.DataFrame({"a": [1]})
        validate_columns(df, set(), "test")


class TestValidateNonEmpty:
    def test_passes_when_non_empty(self):
        df = pd.DataFrame({"a": [1]})
        validate_non_empty(df, "test")

    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        with pytest.raises(ObAnalyticsError, match="empty DataFrame"):
            validate_non_empty(df, "test")
