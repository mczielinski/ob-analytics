"""Tests for the demo runners' panel wiring (``ob_analytics._demos``).

A result stores prices as integer ticks; the gallery draws the quote currency.
A demo panel that reads price must therefore be handed display-unit trades, or
its y-axis shows raw ticks (about 100x the price at the default 0.01 tick).
"""

from __future__ import annotations

import pandas as pd
import pytest

from ob_analytics._demos import (
    _lobster_analytics_cards,
    run_bitstamp_demo,
)
from ob_analytics.pipeline import PipelineResult
from ob_analytics.visualization.gallery import display_result


@pytest.fixture
def halts(result: PipelineResult) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": [result.trades["timestamp"].iloc[0]]})


def _card(cards, key):
    return next(c for c in cards if c.name == key)


class TestLobsterAnalyticsCards:
    def test_fixture_stores_ticks(self, result: PipelineResult) -> None:
        # Guard: if prices were already floats the tests below would pass
        # whether or not the cards convert.
        assert pd.api.types.is_integer_dtype(result.trades["price"])

    def test_trading_halts_card_prices_are_display_units(
        self, result: PipelineResult, halts: pd.DataFrame
    ) -> None:
        card = _card(_lobster_analytics_cards(result, halts), "trading_halts")
        data = card.prepare(**card.prep_kwargs)

        expected = display_result(result).trades["price"].to_numpy()
        got = data["trades"]["price"].to_numpy()
        assert pd.api.types.is_float_dtype(data["trades"]["price"])
        assert got == pytest.approx(expected)
        assert got.max() < result.trades["price"].max()

    def test_no_halts_no_halts_card(self, result: PipelineResult) -> None:
        keys = {p.name for p in _lobster_analytics_cards(result, None)}
        assert "trading_halts" not in keys


def test_bitstamp_demo_without_trades_names_the_capture_command(tmp_path) -> None:
    (tmp_path / "orders.csv").write_text("id\n")
    with pytest.raises(FileNotFoundError, match="ob-analytics capture bitstamp"):
        run_bitstamp_demo(tmp_path, tmp_path / "out")
