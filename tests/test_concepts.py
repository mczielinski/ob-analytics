"""Tests for the plot concept table, ``visualization/_concepts.py``.

The table declares every plot of the package at ``(concept, level, backend)``;
each backend registers its renderers at the coordinates the table gives it
(a test below checks the two match), and ``plot``, ``plot_result`` and the
gallery read it to know where a plot is drawn (audit #405).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ob_analytics.exceptions import ConfigError, UnknownNameError
from ob_analytics.protocols import Level
from ob_analytics.visualization import (
    RENDERERS,
    FocusWindow,
    _concepts,
    busiest_window,
    focus_window,
    plot,
    prepare,
)
from ob_analytics.visualization._data import _horizon_bands, padded_range
from ob_analytics.visualization._renderers import (
    _BACKEND_MODULES,
    _load_backend,
    renders,
)


@pytest.fixture(scope="module", autouse=True)
def _load_builtin_backends():
    for backend in _concepts.BUILTIN_BACKENDS:
        _load_backend(backend)


class TestTable:
    def _builtin_registrations(self) -> set:
        return {key for key in RENDERERS.list() if key[2] in _concepts.BUILTIN_BACKENDS}

    def test_every_plot_in_the_table_has_a_renderer(self):
        # A renderer the table names and no backend registers fails here,
        # not when its backend is imported.
        missing = set(_concepts.PLOTS) - self._builtin_registrations()
        assert not missing, f"no renderer is registered at {sorted(missing, key=str)}"

    def test_no_builtin_backend_registers_a_plot_the_table_lacks(self):
        extra = self._builtin_registrations() - set(_concepts.PLOTS)
        assert not extra, f"the table does not declare {sorted(extra, key=str)}"

    def test_each_builtin_backend_is_a_known_module(self):
        assert set(_concepts.BUILTIN_BACKENDS) <= set(_BACKEND_MODULES)

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

    def test_a_plugin_renderer_is_placed_with_the_table(self, restore_registries):
        def fn(data, ax=None):
            return None

        RENDERERS.register(("trade_tape", Level.L2, "probe"), fn)
        assert "probe" in _concepts.backends("trade_tape", Level.L2)
        assert "matplotlib" in _concepts.backends("trade_tape", Level.L2)

    def test_renders_registers_the_function_at_each_coordinate(
        self, restore_registries
    ):
        def fn(data, ax=None):
            return None

        decorated = renders("probe_a", None, "probe")(
            renders("probe_b", None, "probe")(fn)
        )
        assert decorated is fn
        assert RENDERERS.get(("probe_a", None, "probe")) is fn
        assert RENDERERS.get(("probe_b", None, "probe")) is fn

    def test_the_shared_window_renderer_draws_both_measures(self):
        for backend in ("matplotlib", "plotly"):
            assert RENDERERS.get(("amihud", None, backend)) is RENDERERS.get(
                ("roll_spread", None, backend)
            )


class TestMissingBackend:
    def test_plot_names_the_backends_that_draw_it(self):
        with pytest.raises(UnknownNameError) as info:
            plot("book_replay", Level.L2, backend="matplotlib")
        assert str(info.value) == (
            "Unknown L2 'matplotlib' plot 'book_replay'. "
            "Backends that draw it: 'plotly'."
        )

    def test_plot_without_a_level_names_them_too(self):
        with pytest.raises(UnknownNameError, match="Backends that draw it: 'plotly'.$"):
            plot("book_replay", backend="bokeh")

    def test_plot_names_the_levels_when_none_matches(self):
        with pytest.raises(UnknownNameError, match="It is drawn at L3.$"):
            plot("order_outcome", Level.L2, backend="matplotlib")

    def test_an_unknown_concept_says_so(self):
        with pytest.raises(UnknownNameError, match="Unknown plot concept"):
            plot("no_such_plot", backend="matplotlib")

    def test_a_close_name_is_suggested(self):
        with pytest.raises(UnknownNameError, match="Did you mean 'book_replay'"):
            plot("book_replays", backend="plotly")

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
        data = prepare.book_snapshot(book, level="L2")
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


# ---------------------------------------------------------------------------
# What each face needs (#492)
# ---------------------------------------------------------------------------

_TABLES = ("events", "trades", "depth", "depth_summary")

_GALLERY_FACES = [
    pytest.param(concept.key, level, face, id=f"{concept.key}.{level}")
    for concept in _concepts.CONCEPTS
    for level, face in concept.faces.items()
    if face.bind is not None
]


@pytest.fixture(scope="module")
def l3_scene(result):
    """The gallery's scene of an L3 run, with LOBSTER's raw event type added.

    The tiny Bitstamp run has classified events and ``aggressiveness_bps``;
    the ``raw_event_type`` column gives the hidden-execution face its data,
    so the scene meets every face's needs.
    """
    from ob_analytics.visualization import gallery

    scenes: list[_concepts.Scene] = []
    original = gallery._concept_cards

    def capture(scene: _concepts.Scene) -> list:
        scenes.append(scene)
        return original(scene)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gallery, "_concept_cards", capture)
        gallery.build_gallery_model(result)
    (scene,) = scenes
    events = scene.events.assign(raw_event_type=1)
    events.loc[events.index[0], "raw_event_type"] = 5
    return dataclasses.replace(scene, events=events)


def _restricted(scene, needs):
    """*scene* with every table *needs* does not name emptied."""
    named = set(needs.tables) | set(needs.uses)
    return dataclasses.replace(
        scene,
        **{t: getattr(scene, t).iloc[0:0] for t in _TABLES if t not in named},
    )


class TestNeeds:
    def test_every_gallery_face_declares_needs(self):
        for concept in _concepts.CONCEPTS:
            for face in concept.faces.values():
                assert (face.needs is None) == (face.bind is None), concept.key

    def test_a_face_with_a_binder_and_no_needs_is_refused(self):
        with pytest.raises(ConfigError, match="needs"):
            _concepts.Face(("matplotlib",), bind=lambda s: _concepts.Binding(dict, {}))

    def test_an_unknown_table_is_refused(self):
        with pytest.raises(ConfigError, match="Unknown tables"):
            _concepts.Needs(("orders",))

    def test_event_columns_need_the_events(self):
        with pytest.raises(ConfigError, match="events table"):
            _concepts.Needs(("depth",), classified=True)

    def test_an_l2_face_needs_only_aggregated_data(self):
        for concept in _concepts.CONCEPTS:
            face = concept.faces.get(Level.L2)
            if face is not None and face.needs is not None:
                assert not face.needs.per_order, concept.key
                assert not face.needs.classified and not face.needs.columns

    def test_time_series_is_level_less(self):
        assert _concepts.placements("time_series") == (
            (None, "matplotlib"),
            (None, "plotly"),
        )

    @pytest.mark.parametrize(("key", "level", "face"), _GALLERY_FACES)
    def test_the_binder_reads_only_what_the_face_declares(
        self, l3_scene, key, level, face
    ):
        scene = _restricted(l3_scene, face.needs)
        assert face.needs.unmet(scene) is None
        binding = face.bind(scene)
        declared = set(face.needs.tables) | set(face.needs.uses)
        for name, value in binding.kwargs.items():
            for table in _TABLES:
                if value is getattr(scene, table) and table not in declared:
                    pytest.fail(f"{key}.{level} passes {table} as {name!r}")
        # The face prepares from the tables it declares alone.
        assert isinstance(binding.prepare(**binding.kwargs), dict)

    @pytest.mark.parametrize(("key", "level", "face"), _GALLERY_FACES)
    def test_each_declared_table_is_needed(self, l3_scene, key, level, face):
        for table in face.needs.tables:
            scene = dataclasses.replace(
                l3_scene, **{table: getattr(l3_scene, table).iloc[0:0]}
            )
            assert face.needs.unmet(scene), (key, level, table)

    def test_the_reasons_say_what_is_missing(self, l3_scene):
        needs = _concepts.Needs(("events",), classified=True, columns=("foo",))
        no_events = dataclasses.replace(l3_scene, events=l3_scene.events.iloc[0:0])
        assert needs.unmet(no_events) == (
            "it needs per-order events, and this run has none"
        )
        unclassified = dataclasses.replace(
            l3_scene, events=l3_scene.events.drop(columns="type")
        )
        reason = needs.unmet(unclassified)
        assert reason is not None and "order types" in reason
        assert needs.unmet(l3_scene) == (
            "it needs the events' foo, and this run's events lack it"
        )


class TestAvailableConcepts:
    def test_it_does_not_build_the_gallery_model(self, result, monkeypatch):
        from ob_analytics.visualization import available_concepts, gallery

        def fail(*args, **kwargs):
            raise AssertionError("available_concepts built the gallery model")

        monkeypatch.setattr(gallery, "build_gallery_model", fail)
        assert available_concepts(result)["queue_position"] == ["L3"]

    def test_it_agrees_with_the_gallery_model(self, result):
        from ob_analytics.visualization import available_concepts
        from ob_analytics.visualization.gallery import build_gallery_model

        listed = available_concepts(result)
        model = build_gallery_model(result)
        built = {
            c.key: sorted(lvl.value for lvl in c.variants)
            for c in model.concepts
            if c.variants
        }
        assert {k: v for k, v in listed.items() if v} == built
