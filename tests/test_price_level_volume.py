"""What ``price_level_volume`` promises about the sizes it returns.

The depth table holds each level's size after every change to it.  On a
pipeline table the size is integer lots, whatever kind of change made it,
and a level never goes below zero unless the events remove more than they
added, which is a data problem the caller is told about.
"""

from __future__ import annotations

import warnings

import pytest

from ob_analytics.depth import price_level_volume
from tests.test_price_level_stranding import _events


class TestSizes:
    def test_a_partial_cancel_keeps_integer_sizes(self):
        # Created with 10 lots, then cut to 6 without an execution.
        events = _events(
            [
                (1, 1, 0, 100, 10, "bid", "created", 0),
                (2, 1, 1, 100, 6, "bid", "changed", 0),
            ]
        )
        depth = price_level_volume(events)

        assert list(depth["volume"]) == [10, 6]
        assert depth["volume"].dtype == "int64"


class TestALevelBelowZero:
    """A level the events take more from than they added is reported."""

    @staticmethod
    def _overdrawn():
        # Order 1 rests 5 at 100.  Its delete reports 5 removed *and* 5
        # filled, so the level loses 10.  Order 2 then adds 4.
        return _events(
            [
                (1, 1, 0, 100, 5, "bid", "created", 0),
                (2, 1, 1, 100, 5, "bid", "deleted", 5),
                (3, 2, 2, 100, 4, "bid", "created", 0),
            ]
        )

    def test_it_warns_naming_the_level_and_the_first_event(self):
        with pytest.warns(UserWarning, match="below zero") as caught:
            price_level_volume(self._overdrawn())

        message = str(caught[0].message)
        assert "bid level at price 100 " in message
        assert "event 2" in message

    def test_the_level_is_held_at_zero_and_a_later_order_starts_from_there(self):
        with pytest.warns(UserWarning, match="below zero"):
            depth = price_level_volume(self._overdrawn())

        # The delete writes one row for its cancel and one for its fill, which
        # would take the level to -5.  It reads 0, and the new order of 4 then
        # reads 4: the shortfall is not carried forward.
        assert list(depth["volume"]) == [5, 0, 0, 4]
        assert (depth["volume"] >= 0).all()

    def test_a_level_that_empties_does_not_warn(self):
        events = _events(
            [
                (1, 1, 0, 100, 5, "bid", "created", 0),
                (2, 1, 1, 100, 0, "bid", "deleted", 5),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            depth = price_level_volume(events)
        assert list(depth["volume"]) == [5, 0]


def test_float_rounding_on_no_grid_is_not_a_level_below_zero():
    # Float sizes on no decimal grid are summed as floats.  Cancelled in this
    # order, the four leave about -3e-17, which is rounding, not a loader
    # error.
    sizes = [
        0.236129492246605,
        0.07181083289788565,
        0.26294526924321115,
        0.017570410441558302,
    ]
    rows = [(i + 1, i + 1, i, 100, v, "bid", "created", 0) for i, v in enumerate(sizes)]
    rows += [
        (5 + k, i + 1, 4 + k, 100, sizes[i], "bid", "deleted", 0)
        for k, i in enumerate([1, 0, 2, 3])
    ]
    # The helper builds integer sizes, so the float sizes go in afterwards.
    events = _events([(*r[:4], 1, *r[5:]) for r in rows]).assign(
        volume=[r[4] for r in rows], fill=0.0
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        depth = price_level_volume(events)
    assert not [w for w in caught if "below zero" in str(w.message)]
    assert depth["volume"].iloc[-1] == 0.0


def _assert_no_level_below_zero(events) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        depth = price_level_volume(events)
    assert (depth["volume"] >= 0).all()


class TestBundledData:
    """The package's own data never takes a level below zero."""

    def test_the_toy_session(self):
        from ob_analytics.analytics import set_order_types
        from ob_analytics.datasets import toy_events, toy_trades

        _assert_no_level_below_zero(set_order_types(toy_events(), toy_trades()))

    def test_a_synthetic_session(self):
        from ob_analytics import Pipeline
        from ob_analytics.synth import (
            SynthConfig,
            SyntheticLoader,
            SyntheticTradeSource,
            generate_session,
        )

        session = generate_session(SynthConfig(seed=42, duration=90.0))
        result = Pipeline(
            loader=SyntheticLoader(session),
            trade_source=SyntheticTradeSource(session),
        ).run(None)
        _assert_no_level_below_zero(result.events)
