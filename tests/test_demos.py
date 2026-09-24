"""Tests for the demo runners' panel wiring (``ob_analytics._demos``).

A result stores prices as integer ticks; the gallery draws the quote currency.
A demo panel that reads price must therefore be handed display-unit trades, or
its y-axis shows raw ticks (about 100x the price at the default 0.01 tick).
"""

from __future__ import annotations

import pandas as pd
import pytest

from ob_analytics._demos import _cost_panels, _lobster_analytics_panels
from ob_analytics.bitstamp import BitstampSource
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.visualization.gallery import display_result


@pytest.fixture(scope="module")
def result(tiny_bitstamp_orders_csv) -> PipelineResult:
    return Pipeline(source=BitstampSource()).run(str(tiny_bitstamp_orders_csv))


@pytest.fixture
def halts(result: PipelineResult) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": [result.trades["timestamp"].iloc[0]]})


def _panel(panels, key):
    return next(p for p in panels if p.name == key)


class TestLobsterAnalyticsPanels:
    def test_fixture_stores_ticks(self, result: PipelineResult) -> None:
        # Guard: if prices were already floats the tests below would pass
        # whether or not the panels convert.
        assert pd.api.types.is_integer_dtype(result.trades["price"])

    def test_trading_halts_panel_prices_are_display_units(
        self, result: PipelineResult, halts: pd.DataFrame
    ) -> None:
        panel = _panel(_lobster_analytics_panels(result, halts), "trading_halts")
        data = panel.prepare(**panel.prep_kwargs)

        expected = display_result(result).trades["price"].to_numpy()
        got = data["trades"]["price"].to_numpy()
        assert pd.api.types.is_float_dtype(data["trades"]["price"])
        assert got == pytest.approx(expected)
        assert got.max() < result.trades["price"].max()

    def test_ofi_horizon_matches_display_units(self, result: PipelineResult) -> None:
        # OFI is a volume ratio, so raw and display trades must agree.
        panel = _panel(_lobster_analytics_panels(result, None), "ofi_horizon")
        raw = panel.prepare(**panel.prep_kwargs)
        shown = panel.prepare(
            **{**panel.prep_kwargs, "trades": display_result(result).trades}
        )
        assert raw["ofi"] == pytest.approx(shown["ofi"], nan_ok=True)

    def test_no_halts_no_halts_panel(self, result: PipelineResult) -> None:
        keys = {p.name for p in _lobster_analytics_panels(result, None)}
        assert "trading_halts" not in keys


def test_cost_panel_bps_match_display_units(result: PipelineResult) -> None:
    # The cost panel is fed raw-tick frames on purpose: bps are ratios.
    from ob_analytics.cost import transaction_costs

    display = display_result(result)
    raw_costs = transaction_costs(result.trades, result.depth_summary)
    shown_costs = transaction_costs(display.trades, display.depth_summary)
    for col in ("effective_spread_bps", "realized_spread_bps", "price_impact_bps"):
        assert raw_costs[col].to_numpy() == pytest.approx(
            shown_costs[col].to_numpy(), nan_ok=True
        )
    assert isinstance(_cost_panels(result), list)
