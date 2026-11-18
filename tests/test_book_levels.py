"""What ``depth.book_levels`` promises (made public in #522).

It replays a run's depth table and gives the best few levels of each side
after every event: one row per event, best first, padded with zeros.
"""

from __future__ import annotations

import numpy as np
import pytest

from ob_analytics.depth import book_levels, price_level_volume
from ob_analytics.exceptions import ConfigError
from tests.test_price_level_stranding import _events


def _crossed_run():
    # Bids at 100 and 99, then an ask at 99 that crosses the bid at 100.
    events = _events(
        [
            (1, 1, 0, 100, 5, "bid", "created", 0),
            (2, 2, 1, 99, 3, "bid", "created", 0),
            (3, 3, 2, 99, 4, "ask", "created", 0),
        ]
    )
    return events, price_level_volume(events)


def test_one_row_per_event_best_first_padded_with_zeros():
    events, depth = _crossed_run()
    levels = book_levels(events, depth, 3, quoted=False)

    assert levels.bid_prices.shape == (3, 3)
    assert levels.bid_prices[0].tolist() == [100, 0, 0]
    assert levels.bid_volumes[0].tolist() == [5, 0, 0]
    assert levels.bid_prices[1].tolist() == [100, 99, 0]
    assert levels.ask_prices[1].tolist() == [0, 0, 0]
    assert levels.ask_prices[2].tolist() == [99, 0, 0]
    assert levels.ask_volumes[2].tolist() == [4, 0, 0]


def test_quoted_levels_leave_out_the_crossed_touch():
    events, depth = _crossed_run()
    every = book_levels(events, depth, 3, quoted=False)
    quoted = book_levels(events, depth, 3)

    # The bid at 100 is above the ask at 99, so the quoted book leaves it out.
    assert every.bid_prices[2, 0] == 100
    assert quoted.bid_prices[2, 0] != 100
    assert not np.array_equal(every.bid_prices, quoted.bid_prices)


def test_display_unit_prices_are_refused():
    events, depth = _crossed_run()
    with pytest.raises(ConfigError, match="integer ticks"):
        book_levels(events, depth.assign(price=depth["price"] * 0.25), 3)


def test_fractional_sizes_are_refused():
    events, depth = _crossed_run()
    with pytest.raises(ConfigError, match="whole lots"):
        book_levels(events, depth.assign(volume=depth["volume"] * 0.5), 3)


def test_whole_float_sizes_are_read_as_lots():
    events, depth = _crossed_run()
    levels = book_levels(events, depth.assign(volume=depth["volume"] * 1.0), 2)
    assert levels.bid_volumes[1].tolist() == [5, 3]


def test_a_missing_column_is_named():
    events, depth = _crossed_run()
    with pytest.raises(ConfigError, match="direction"):
        book_levels(events, depth.drop(columns="direction"), 3)


def test_num_levels_below_one_is_refused():
    events, depth = _crossed_run()
    with pytest.raises(ConfigError, match="num_levels"):
        book_levels(events, depth, 0)
