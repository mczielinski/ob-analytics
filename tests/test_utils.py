"""Tests for ob_analytics._utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics._utils import (
    interval_price_level_gaps,
    interval_sum_breaks,
    interval_vwap,
    norml,
    reverse_matrix,
    validate_columns,
    validate_non_empty,
    vector_diff,
    vwap,
)
from ob_analytics.exceptions import InsufficientDataError, InvalidDataError


class TestValidateColumns:
    def test_passes_when_columns_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_columns(df, {"a", "b"}, "test")

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(InvalidDataError, match="missing required columns.*'c'"):
            validate_columns(df, {"a", "c"}, "test")

    def test_error_message_includes_context(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(InvalidDataError, match="my_function"):
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
        with pytest.raises(InsufficientDataError, match="empty DataFrame"):
            validate_non_empty(df, "test")


class TestVectorDiff:
    def test_basic(self):
        result = vector_diff(np.array([10, 12, 15, 20]))
        np.testing.assert_array_equal(result, [10, 2, 3, 5])

    def test_single_element(self):
        result = vector_diff(np.array([42]))
        np.testing.assert_array_equal(result, [42])

    def test_zeros(self):
        result = vector_diff(np.array([0, 0, 0]))
        np.testing.assert_array_equal(result, [0, 0, 0])


class TestReverseMatrix:
    def test_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = reverse_matrix(df)
        pd.testing.assert_frame_equal(
            result, pd.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4]})
        )

    def test_numpy_array(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        result = reverse_matrix(arr)
        np.testing.assert_array_equal(result, [[5, 6], [3, 4], [1, 2]])

    def test_rejects_non_array(self):
        with pytest.raises(TypeError):
            reverse_matrix([1, 2, 3])


class TestNorml:
    def test_basic(self):
        result = norml(np.array([0.0, 5.0, 10.0]))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_custom_bounds(self):
        result = norml(np.array([5.0, 10.0, 15.0]), minv=0.0, maxv=20.0)
        np.testing.assert_allclose(result, [0.25, 0.5, 0.75])

    def test_constant_array(self):
        with np.errstate(invalid="ignore"):
            result = norml(np.array([5.0, 5.0]))
        assert all(np.isnan(result))


class TestIntervalSumBreaks:
    def test_basic(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        breaks = np.array([1, 3, 5])
        result = interval_sum_breaks(v, breaks)
        np.testing.assert_allclose(result, [3.0, 7.0, 11.0])

    def test_single_interval(self):
        v = np.array([10.0, 20.0, 30.0])
        breaks = np.array([2])
        result = interval_sum_breaks(v, breaks)
        np.testing.assert_allclose(result, [60.0])


class TestVwap:
    def test_basic(self):
        prices = np.array([100.0, 200.0])
        volumes = np.array([10.0, 10.0])
        assert vwap(prices, volumes) == 150.0

    def test_weighted(self):
        prices = np.array([100.0, 200.0])
        volumes = np.array([30.0, 10.0])
        assert vwap(prices, volumes) == 125.0


class TestIntervalVwap:
    def test_basic(self):
        price = np.array([100.0, 100.0, 200.0, 200.0])
        volume = np.array([10.0, 10.0, 10.0, 10.0])
        breaks = np.array([1, 3])
        result = interval_vwap(price, volume, breaks)
        np.testing.assert_allclose(result, [100.0, 200.0])


class TestIntervalPriceLevelGaps:
    def test_basic(self):
        volume = np.array([10.0, 0.0, 0.0, 5.0])
        breaks = np.array([1, 3])
        result = interval_price_level_gaps(volume, breaks)
        # cumsum of gaps [0,1,1,0] = [0,1,2,2]; intervals at [1,3] = [1,2]; diffs = [1,1]
        np.testing.assert_array_equal(result, [1, 1])
