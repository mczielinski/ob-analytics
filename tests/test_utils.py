"""Tests for ob_analytics._utils."""

import pandas as pd
import pytest

from ob_analytics._utils import (
    validate_columns,
    validate_non_empty,
)
from ob_analytics.exceptions import ConfigError, ObAnalyticsError


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
