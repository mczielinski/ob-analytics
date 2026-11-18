"""The one conversion every loader puts its prices and sizes through (#500).

A price between two ticks is refused, naming the column, an example and the
tick size; a size between two lots is rounded with a warning.  A value within
float error of a whole step is on the grid.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ob_analytics._utils import to_grid
from ob_analytics.exceptions import ConfigError


def _price(values, tick=0.01, **kwargs):
    return to_grid(values, tick, kind="price", column="px", where="here", **kwargs)


def _size(values, lot=1e-8, **kwargs):
    return to_grid(values, lot, kind="size", column="qty", where="here", **kwargs)


class TestPrices:
    def test_an_off_grid_price_is_refused_with_column_example_and_tick(self):
        with pytest.raises(ConfigError) as err:
            _price([1.00, 0.036, 2.00])
        message = str(err.value)
        assert message.startswith("here: 1 value(s) in the 'px' column")
        assert "0.036" in message
        assert "tick_size=0.01" in message

    def test_prices_on_the_grid_up_to_float_error_pass(self):
        # 0.036 / 0.001 is 35.99999999999999 in floats; 0.1 + 0.2 is
        # 0.30000000000000004.
        assert _price([0.036], tick=0.001).tolist() == [36]
        assert _price([0.1 + 0.2]).tolist() == [30]
        assert _price([97123.45, 1e6]).dtype == np.int64

    def test_rows_left_out_of_the_check_are_converted_unchecked(self):
        ticks = _price([1.00, -0.0001], checked=np.array([True, False]))
        assert ticks.tolist() == [100, 0]

    def test_a_price_that_is_not_a_number_is_refused(self):
        with pytest.raises(ConfigError, match="not a finite number"):
            _price([1.0, float("nan")])


class TestSizes:
    def test_an_off_grid_size_is_rounded_with_a_warning(self):
        with pytest.warns(UserWarning, match=r"'qty' column.*lot_size=0\.001"):
            lots = _size([0.0015, 0.002], lot=0.001)
        assert lots.tolist() == [2, 2]

    def test_sizes_on_the_grid_up_to_float_error_pass_quietly(self):
        # 1251.81488791 / 1e-8 is 125181488790.99998 in floats.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _size([1251.81488791]).tolist() == [125181488791]
            assert _size([300, 7], lot=1.0).tolist() == [300, 7]

    def test_a_size_too_large_for_int64_lots_is_refused(self):
        with pytest.raises(ConfigError, match="too large"):
            _size([1e12])
