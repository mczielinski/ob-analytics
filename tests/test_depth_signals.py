"""Tests for the depth-stage fair-value / pressure signals (issue #109).

Every expected value is hand-computed on a small scripted depth summary so the
formulas are pinned exactly, including the ``bid_sz + ask_sz == 0`` edge case
that must yield ``NaN`` rather than a divide-by-zero error.
"""

import numpy as np
import pandas as pd
import pytest

from ob_analytics.depth import book_imbalance, depth_signals, micro_price


def _summary() -> pd.DataFrame:
    """A three-row depth summary with two bps depth bins per side.

    Row 0 -- bid-heavy touch, asymmetric depth.
    Row 1 -- empty book (zero volume everywhere): the divide-by-zero edge case.
    Row 2 -- balanced touch and depth (micro-price == mid, OBI == 0).
    """
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:00:01", "2020-01-01 00:00:02"]
            ),
            "best_bid_price": [100.0, 100.0, 100.0],
            "best_bid_vol": [8.0, 0.0, 5.0],
            "best_ask_price": [101.0, 101.0, 102.0],
            "best_ask_vol": [2.0, 0.0, 5.0],
            "bid_vol25bps": [12.0, 0.0, 5.0],
            "bid_vol50bps": [4.0, 0.0, 0.0],
            "ask_vol25bps": [4.0, 0.0, 5.0],
            "ask_vol50bps": [6.0, 0.0, 0.0],
        }
    )


class TestMicroPrice:
    def test_size_weighted_mid_exact(self) -> None:
        # (100*2 + 101*8) / (2 + 8) = 1008 / 10 = 100.8
        mp = micro_price(_summary())
        assert mp.iloc[0] == pytest.approx(100.8)

    def test_balanced_touch_equals_mid(self) -> None:
        # Equal sizes -> the weighted mid collapses to the plain mid.
        # (100*5 + 102*5) / 10 = 101.0 = (100 + 102) / 2
        mp = micro_price(_summary())
        assert mp.iloc[2] == pytest.approx(101.0)

    def test_zero_size_is_nan_not_zero_division(self) -> None:
        mp = micro_price(_summary())
        assert np.isnan(mp.iloc[1])

    def test_result_is_indexed_like_input(self) -> None:
        df = _summary().set_index(np.array([10, 20, 30]))
        mp = micro_price(df)
        assert list(mp.index) == [10, 20, 30]

    def test_stoikov_adjustment_scalar_is_added(self) -> None:
        base = micro_price(_summary())
        adjusted = micro_price(_summary(), stoikov_adjustment=0.1)
        assert adjusted.iloc[0] == pytest.approx(base.iloc[0] + 0.1)
        assert adjusted.iloc[2] == pytest.approx(base.iloc[2] + 0.1)
        # The adjustment cannot rescue a zero-size row.
        assert np.isnan(adjusted.iloc[1])

    def test_stoikov_adjustment_series_is_added_elementwise(self) -> None:
        adjusted = micro_price(
            _summary(), stoikov_adjustment=pd.Series([0.1, 0.2, 0.3])
        )
        assert adjusted.iloc[0] == pytest.approx(100.9)
        assert adjusted.iloc[2] == pytest.approx(101.3)
        assert np.isnan(adjusted.iloc[1])


class TestBookImbalance:
    def test_touch_imbalance_exact(self) -> None:
        # (8 - 2) / (8 + 2) = 0.6
        obi = book_imbalance(_summary(), levels=1)
        assert obi.iloc[0] == pytest.approx(0.6)

    def test_touch_default_levels_is_one(self) -> None:
        assert book_imbalance(_summary()).equals(book_imbalance(_summary(), levels=1))

    def test_balanced_touch_is_zero(self) -> None:
        obi = book_imbalance(_summary(), levels=1)
        assert obi.iloc[2] == pytest.approx(0.0)

    def test_cumulative_over_one_bin(self) -> None:
        # levels=2 cumulates the first bin: (12 - 4) / (12 + 4) = 0.5
        obi = book_imbalance(_summary(), levels=2)
        assert obi.iloc[0] == pytest.approx(0.5)

    def test_cumulative_over_two_bins(self) -> None:
        # levels=3 cumulates both bins: bid = 12+4 = 16, ask = 4+6 = 10
        # (16 - 10) / (16 + 10) = 6 / 26
        obi = book_imbalance(_summary(), levels=3)
        assert obi.iloc[0] == pytest.approx(6.0 / 26.0)

    def test_zero_volume_is_nan(self) -> None:
        assert np.isnan(book_imbalance(_summary(), levels=1).iloc[1])
        assert np.isnan(book_imbalance(_summary(), levels=3).iloc[1])

    def test_levels_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="levels must be >= 1"):
            book_imbalance(_summary(), levels=0)

    def test_levels_beyond_available_bins_raises(self) -> None:
        # Only two bins per side, so levels=4 (needs 3 bins) is out of range.
        with pytest.raises(ValueError, match="depth bins"):
            book_imbalance(_summary(), levels=4)


class TestDepthSignals:
    def test_appends_expected_columns(self) -> None:
        out = depth_signals(_summary())
        for col in ("mid_price", "micro_price", "obi", "obi_depth"):
            assert col in out.columns

    def test_preserves_existing_columns(self) -> None:
        df = _summary()
        out = depth_signals(df)
        assert set(df.columns) <= set(out.columns)
        for col in df.columns:
            pd.testing.assert_series_equal(out[col], df[col])

    def test_does_not_mutate_input(self) -> None:
        df = _summary()
        before = list(df.columns)
        depth_signals(df)
        assert list(df.columns) == before

    def test_column_values_match_the_signal_functions(self) -> None:
        out = depth_signals(_summary())
        assert out["mid_price"].iloc[0] == pytest.approx(100.5)
        assert out["micro_price"].iloc[0] == pytest.approx(100.8)
        assert out["obi"].iloc[0] == pytest.approx(0.6)

    def test_depth_levels_clamped_to_available_bins(self) -> None:
        # Request more depth than exists (2 bins) -> clamp to levels=3, no error.
        out = depth_signals(_summary(), depth_levels=99)
        assert out["obi_depth"].iloc[0] == pytest.approx(6.0 / 26.0)

    def test_edge_row_is_nan(self) -> None:
        out = depth_signals(_summary())
        assert np.isnan(out["micro_price"].iloc[1])
        assert np.isnan(out["obi"].iloc[1])
        assert np.isnan(out["obi_depth"].iloc[1])


class TestBookSignalsFace:
    """The gallery face: prepare payload, matplotlib/plotly renderers, panel."""

    def test_prepare_payload_shape_and_clamp(self) -> None:
        from ob_analytics.visualization import _data

        payload = _data.prepare_book_signals_data(_summary(), levels=99)
        for key in ("timestamp", "mid", "microprice", "obi", "obi_depth", "levels"):
            assert key in payload
        # Only two bins, so levels clamps to 3.
        assert payload["levels"] == 3
        assert payload["microprice"][0] == pytest.approx(100.8)
        assert payload["obi"][0] == pytest.approx(0.6)

    def test_matplotlib_renderer_returns_figure(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        from ob_analytics.visualization import _data, plot

        payload = _data.prepare_book_signals_data(_summary())
        fig = plot("book_signals", backend="matplotlib", **payload)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plotly_renderer_returns_figure(self) -> None:
        pytest.importorskip("plotly")
        from ob_analytics.visualization import _data, plot

        payload = _data.prepare_book_signals_data(_summary())
        fig = plot("book_signals", backend="plotly", **payload)
        # A plotly figure exposes a data tuple of traces.
        assert len(fig.data) > 0

    def test_panel_helper_builds_spec(self) -> None:
        from ob_analytics.visualization.gallery import book_signals_panel

        spec = book_signals_panel(_summary(), levels=4)
        assert spec.plot_name == "book_signals"
        assert spec.prep_kwargs["levels"] == 4
