"""Tests for the plot concept table, ``visualization/_concepts.py``.

The table declares every plot of the package at ``(concept, level, backend)``;
the backends register their renderers from it, and ``plot``, ``plot_result``
and the gallery read it to know where a plot is drawn (audit #405).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ob_analytics.protocols import Level
from ob_analytics.visualization import (
    RENDERERS,
    FocusWindow,
    _concepts,
    busiest_window,
    focus_window,
    plot,
)
from ob_analytics.visualization._data import (
    _horizon_bands,
    padded_range,
    prepare_book_snapshot_data,
)
from ob_analytics.visualization._renderers import _BACKEND_MODULES, _load_backend


@pytest.fixture(scope="module", autouse=True)
def _load_builtin_backends():
    for backend in _concepts.BACKEND_PREFIXES:
        _load_backend(backend)


class TestTable:
    def test_the_builtin_backends_register_exactly_the_table(self):
        builtin = {
            key for key in RENDERERS.list() if key[2] in _concepts.BACKEND_PREFIXES
        }
        assert builtin == set(_concepts.PLOTS)

    def test_each_builtin_backend_is_a_known_module(self):
        assert set(_concepts.BACKEND_PREFIXES) <= set(_BACKEND_MODULES)

    def test_a_concept_is_declared_once(self):
        keys = [c.key for c in _concepts.CONCEPTS]
        assert len(keys) == len(set(keys))

    def test_a_concept_is_one_kind(self):
        for concept in _concepts.CONCEPTS:
            kinds = {level is None for level in concept.faces}
            assert len(kinds) == 1, concept.key

    def test_the_book_replay_is_declared_plotly_only(self):
        for level in (Level.L2, Level.L3):
            assert _concepts.backends("book_replay", level) == ("plotly",)

    def test_every_gallery_card_has_a_title_and_a_note(self):
        for concept in _concepts.CONCEPTS:
            if any(face.bind for face in concept.faces.values()):
                assert concept.title and concept.note, concept.key

    def test_a_plugin_renderer_is_placed_with_the_table(self):
        def fn(data, ax=None):
            return None

        RENDERERS.register(("trade_tape", Level.L2, "probe"), fn)
        try:
            assert "probe" in _concepts.backends("trade_tape", Level.L2)
            assert "matplotlib" in _concepts.backends("trade_tape", Level.L2)
        finally:
            RENDERERS._items.pop(("trade_tape", Level.L2, "probe"), None)


class TestMissingBackend:
    def test_plot_names_the_backends_that_draw_it(self):
        with pytest.raises(KeyError, match="drawn by: plotly"):
            plot("book_replay", Level.L2, backend="matplotlib")

    def test_plot_without_a_level_names_them_too(self):
        with pytest.raises(KeyError, match="drawn by: plotly"):
            plot("book_replay", backend="bokeh")

    def test_plot_names_the_levels_when_none_matches(self):
        with pytest.raises(KeyError, match="is drawn at L3"):
            plot("order_outcome", Level.L2, backend="matplotlib")

    def test_an_unknown_concept_says_so(self):
        with pytest.raises(KeyError, match="Unknown plot concept"):
            plot("no_such_plot", backend="matplotlib")

    def test_require_passes_a_drawn_plot(self):
        _concepts.require("trade_tape", Level.L3, "bokeh")


class TestBusiestWindow:
    @staticmethod
    def _frame(seconds):
        t0 = pd.Timestamp("2026-01-01", tz="UTC")
        return pd.DataFrame(
            {"timestamp": [t0 + pd.Timedelta(seconds=s) for s in seconds]}
        )

    def test_it_holds_the_most_rows(self):
        frame = self._frame([0, 100, 101, 102, 103, 300])
        start, end = busiest_window(frame, "10s")
        assert (start, end - start) == (frame["timestamp"][1], pd.Timedelta("10s"))

    def test_the_earlier_of_two_equal_windows_wins(self):
        frame = self._frame([0, 1, 100, 101, 500])
        start, _ = busiest_window(frame, "10s")
        assert start == frame["timestamp"][0]

    def test_it_does_not_run_past_the_last_row(self):
        frame = self._frame([0, 50, 98, 99, 100])
        start, end = busiest_window(frame, "10s")
        assert end == frame["timestamp"].iloc[-1]
        assert start == end - pd.Timedelta("10s")

    def test_unsorted_rows_give_the_same_window(self):
        frame = self._frame([0, 100, 101, 102, 103, 300])
        assert busiest_window(frame.iloc[::-1], "10s") == busiest_window(frame, "10s")

    def test_an_empty_frame_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            busiest_window(self._frame([]), "10s")

    def test_a_width_of_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            busiest_window(self._frame([0]), "0s")


class TestFocusWindow:
    def test_it_holds_a_price_band_only(self):
        assert [f.name for f in dataclasses.fields(FocusWindow)] == [
            "price_from",
            "price_to",
        ]

    def test_an_empty_tape_is_unclipped(self):
        assert focus_window(pd.DataFrame({"price": []})) == FocusWindow()


class TestSharedDrawingData:
    def test_padded_range(self):
        assert padded_range((10.0, 20.0)) == pytest.approx((9.6, 20.4))
        assert padded_range((5.0, 5.0)) == (4.0, 6.0)
        assert padded_range(None) is None

    def test_the_book_payload_carries_its_mid_and_curves(self):
        book = {
            "timestamp": 0,
            "bids": pd.DataFrame(
                {"id": [1, 2], "price": [99.0, 98.0], "volume": [1.0, 2.0]}
            ),
            "asks": pd.DataFrame(
                {"id": [3, 4], "price": [102.0, 101.0], "volume": [1.0, 1.0]}
            ),
        }
        data = prepare_book_snapshot_data(book, volume_scale=1.0)
        assert data["mid"] == pytest.approx(100.0)
        assert list(data["bid_curve"]["price"]) == [98.0, 99.0]
        assert list(data["ask_curve"]["price"]) == [101.0, 102.0]

    def test_horizon_bands_fill_as_far_as_the_value_reaches(self):
        bands = _horizon_bands(np.array([[0.5, -1.0, np.nan]]))
        assert bands["buy_bands"].shape == (1, 3, 3)
        # 0.5 fills the first band, half of the second (its 1/3 to 2/3 span
        # is reached up to 1/2), and none of the third.
        assert bands["buy_bands"][0, :, 0] == pytest.approx([1.0, 0.5, 0.0])
        assert bands["sell_bands"][0, :, 1] == pytest.approx([1.0, 1.0, 1.0])
        assert bands["buy_bands"][0, :, 2] == pytest.approx([0.0, 0.0, 0.0])
        assert bands["band_alphas"] == pytest.approx((0.25, 0.5, 0.75))
