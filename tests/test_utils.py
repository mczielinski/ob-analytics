"""Tests for ob_analytics._utils."""

import pandas as pd
import pytest

from ob_analytics._utils import (
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
