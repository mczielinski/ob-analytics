"""Tests for ``ob_analytics.visualization.gallery``.

The gallery is driven by a :class:`GalleryModel` (leveled
:class:`PlotConcept` objects + level-less analytic :class:`PlotSpec` panels).
A *view* projects that model into cards; :func:`generate_gallery` renders and
saves them.  These tests build hand-made models + stub renderers so they don't
need a full pipeline (one test runs the tiny pipeline to check the real
inventory from :func:`build_gallery_model`).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest

from ob_analytics.visualization import RENDERERS, Level
from ob_analytics.visualization.gallery import (
    GalleryModel,
    PlotConcept,
    PlotSpec,
    _Card,
    _Panel,
    _project,
    _render_panel,
    _write_gallery_html,
    build_gallery_model,
    generate_gallery,
)


# ---------------------------------------------------------------------------
# Stub renderers
#
# The gallery dispatches through ``plot(concept, level, backend=...)``, which
# resolves a renderer from ``RENDERERS`` by the coordinate
# ``(concept, level, backend)``.  These stubs cover an L2 face, an L3 face (so
# ``"stub"`` can be made comparable), a deliberately-failing face, and a
# level-less analytic -- all without running a pipeline.  Matplotlib renderers
# take ``(data, ax)``; plotly renderers take ``(data,)``.
# ---------------------------------------------------------------------------


def _stub_mpl(data, ax=None):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def _stub_plotly(data):
    import plotly.graph_objects as go

    return go.Figure(data=go.Scatter(x=[0, 1], y=[0, 1]))


def _stub_fail_mpl(data, ax=None):
    raise RuntimeError("deliberate matplotlib failure")


def _stub_fail_plotly(data):
    raise RuntimeError("deliberate plotly failure")


_STUB_RENDERERS = {
    ("stub", Level.L2, "matplotlib"): _stub_mpl,
    ("stub", Level.L2, "plotly"): _stub_plotly,
    ("stub", Level.L3, "matplotlib"): _stub_mpl,
    ("stub", Level.L3, "plotly"): _stub_plotly,
    ("stubfail", Level.L2, "matplotlib"): _stub_fail_mpl,
    ("stubfail", Level.L2, "plotly"): _stub_fail_plotly,
    ("stubmetric", None, "matplotlib"): _stub_mpl,
    ("stubmetric", None, "plotly"): _stub_plotly,
}


@pytest.fixture(autouse=True)
def _register_stub_renderers():
    """Register stub renderers for gallery dispatch; clean up after."""
    for key, fn in _STUB_RENDERERS.items():
        RENDERERS.register(key, fn)
    yield
    for key in _STUB_RENDERERS:
        RENDERERS._items.pop(key, None)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _spec(name="stub", title="Stub", plot_name="stub", **kw) -> PlotSpec:
    return PlotSpec(name, title, plot_name, lambda: {}, kw)


def _l2_concept(key="stub", title="Stub") -> PlotConcept:
    return PlotConcept(key, title, {Level.L2: _spec(plot_name=key)})


def _comparable_concept(key="stub", title="Stub") -> PlotConcept:
    return PlotConcept(
        key,
        title,
        {Level.L2: _spec(plot_name=key), Level.L3: _spec(plot_name=key)},
    )


def _metric_spec() -> PlotSpec:
    return PlotSpec("stubmetric", "Stub Metric", "stubmetric", lambda: {}, {})


# ---------------------------------------------------------------------------
# Model semantics
# ---------------------------------------------------------------------------


class TestPlotConcept:
    def test_at_returns_variant(self) -> None:
        c = _l2_concept()
        assert c.at(Level.L2) is not None
        assert c.at(Level.L3) is None

    def test_comparable_requires_both_faces(self) -> None:
        assert not _l2_concept().comparable
        assert _comparable_concept().comparable

    def test_frozen(self) -> None:
        c = _l2_concept()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(c, "key", "other")


class TestGalleryModel:
    def test_analytics_defaults_empty(self) -> None:
        m = GalleryModel(concepts=[_l2_concept()])
        assert m.analytics == []


# ---------------------------------------------------------------------------
# View projection
# ---------------------------------------------------------------------------


class TestProject:
    def test_l2_renders_l2_face_across_backends(self) -> None:
        model = GalleryModel(concepts=[_l2_concept()])
        cards = _project(model, "l2", ["plotly", "matplotlib"])
        assert len(cards) == 1
        card = cards[0]
        assert card.title == "Stub -- L2"
        assert [p.backend for p in card.panels] == ["plotly", "matplotlib"]
        assert all(p.stem == "stub.L2" for p in card.panels)
        assert [p.role for p in card.panels] == ["primary", "secondary"]

    def test_l2_includes_analytics(self) -> None:
        model = GalleryModel(concepts=[_l2_concept()], analytics=[_metric_spec()])
        cards = _project(model, "l2", ["matplotlib"])
        titles = [c.title for c in cards]
        assert "Stub Metric" in titles
        metric_card = next(c for c in cards if c.title == "Stub Metric")
        assert metric_card.panels[0].level is None
        assert metric_card.panels[0].stem == "stubmetric"

    def test_l3_skips_l2_only_concepts_and_analytics(self) -> None:
        model = GalleryModel(concepts=[_l2_concept()], analytics=[_metric_spec()])
        assert _project(model, "l3", ["matplotlib"]) == []

    def test_l3_renders_l3_face_only(self) -> None:
        model = GalleryModel(
            concepts=[_comparable_concept()], analytics=[_metric_spec()]
        )
        cards = _project(model, "l3", ["matplotlib"])
        assert [c.title for c in cards] == ["Stub -- L3"]
        assert cards[0].panels[0].stem == "stub.L3"

    def test_both_renders_each_face_plus_analytics(self) -> None:
        model = GalleryModel(
            concepts=[_comparable_concept()], analytics=[_metric_spec()]
        )
        cards = _project(model, "both", ["matplotlib"])
        assert [c.title for c in cards] == ["Stub -- L2", "Stub -- L3", "Stub Metric"]

    def test_comparison_pairs_l2_l3_in_one_card(self) -> None:
        model = GalleryModel(
            concepts=[_comparable_concept(), _l2_concept("only", "OnlyL2")],
            analytics=[_metric_spec()],
        )
        cards = _project(model, "comparison", ["plotly", "matplotlib"])
        # Only the comparable concept appears; analytics + L2-only are excluded.
        assert len(cards) == 1
        card = cards[0]
        assert card.title == "Stub"
        assert [p.level for p in card.panels] == [Level.L2, Level.L3]
        assert [p.stem for p in card.panels] == ["stub.L2", "stub.L3"]
        assert all(p.role == "equal" for p in card.panels)
        # Single backend axis -> plotly when available.
        assert {p.backend for p in card.panels} == {"plotly"}

    def test_comparison_falls_back_to_matplotlib(self) -> None:
        model = GalleryModel(concepts=[_comparable_concept()])
        cards = _project(model, "comparison", ["matplotlib"])
        assert {p.backend for p in cards[0].panels} == {"matplotlib"}

    def test_unknown_view_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown view"):
            _project(GalleryModel(concepts=[]), "nope", ["matplotlib"])


# ---------------------------------------------------------------------------
# _render_panel
# ---------------------------------------------------------------------------


def _panel(backend="plotly", stem="stub.L2", rendered=True, role="primary") -> _Panel:
    return _Panel(
        concept="stub",
        level=Level.L2,
        prepare=lambda: {},
        prep_kwargs={},
        backend=backend,
        stem=stem,
        label=backend.capitalize(),
        panel_cls=f"{backend}-panel" if backend != "matplotlib" else "mpl-panel",
        role=role,
        rendered=rendered,
    )


class TestRenderPanel:
    def test_plotly_rendered_uses_iframe(self) -> None:
        html = _render_panel(_panel("plotly", "01.L2"), "Demo")
        assert "<iframe" in html
        assert 'src="plotly/01.L2.html"' in html
        assert "panel-primary" in html

    def test_matplotlib_rendered_uses_img(self) -> None:
        html = _render_panel(_panel("matplotlib", "01.L2", role="secondary"), "Demo")
        assert "<img" in html
        assert 'src="matplotlib/01.L2.png"' in html
        assert "mpl-panel" in html
        assert "panel-secondary" in html

    def test_not_rendered_shows_na(self) -> None:
        html = _render_panel(_panel("plotly", "x", rendered=False), "x")
        assert "Not available" in html
        assert "<iframe" not in html
        assert "<img" not in html


# ---------------------------------------------------------------------------
# _write_gallery_html
# ---------------------------------------------------------------------------


class TestWriteGalleryHtml:
    def test_renders_card_with_panels(self, tmp_path: Path) -> None:
        out = tmp_path / "g.html"
        card = _Card("Demo", [_panel("plotly"), _panel("matplotlib", role="secondary")])
        _write_gallery_html(out, [card], "Title")
        body = out.read_text().split("<body>", 1)[1]
        assert body.index("plotly-panel") < body.index("mpl-panel")
        assert "panel-primary" in body
        assert "panel-secondary" in body

    def test_empty_cards_show_placeholder(self, tmp_path: Path) -> None:
        out = tmp_path / "g.html"
        _write_gallery_html(out, [], "Title")
        assert "No plots for this view." in out.read_text()

    def test_failed_panel_renders_na(self, tmp_path: Path) -> None:
        out = tmp_path / "g.html"
        card = _Card("Demo", [_panel("plotly", rendered=False)])
        _write_gallery_html(out, [card], "Title")
        assert "Not available" in out.read_text()

    def test_html_escapes_card_titles(self, tmp_path: Path) -> None:
        out = tmp_path / "g.html"
        card = _Card("<script>X</script>", [_panel("matplotlib")])
        _write_gallery_html(out, [card], "Page")
        html = out.read_text()
        assert "<script>X</script>" not in html
        assert "&lt;script&gt;X&lt;/script&gt;" in html


# ---------------------------------------------------------------------------
# generate_gallery (end-to-end with a stub model)
# ---------------------------------------------------------------------------


class TestGenerateGallery:
    def _model(self) -> GalleryModel:
        return GalleryModel(concepts=[_l2_concept()])

    def test_writes_concept_level_file_stems(self, tmp_path: Path) -> None:
        path = generate_gallery(result=None, output_dir=tmp_path, model=self._model())
        assert path.exists()
        assert (tmp_path / "plotly" / "stub.L2.html").exists()
        assert (tmp_path / "matplotlib" / "stub.L2.png").exists()

    def test_default_backends_prefer_plotly(self, tmp_path: Path) -> None:
        path = generate_gallery(result=None, output_dir=tmp_path, model=self._model())
        body = path.read_text().split("<body>", 1)[1]
        assert body.index("plotly-panel") < body.index("mpl-panel")

    def test_explicit_backends_order(self, tmp_path: Path) -> None:
        path = generate_gallery(
            result=None,
            output_dir=tmp_path,
            model=self._model(),
            backends=["matplotlib", "plotly"],
        )
        body = path.read_text().split("<body>", 1)[1]
        assert body.index("mpl-panel") < body.index("plotly-panel")

    def test_view_comparison_single_backend_both_faces(self, tmp_path: Path) -> None:
        model = GalleryModel(concepts=[_comparable_concept()])
        path = generate_gallery(
            result=None, output_dir=tmp_path, model=model, view="comparison"
        )
        # Comparison collapses to plotly; both L2 and L3 land under plotly/.
        assert (tmp_path / "plotly" / "stub.L2.html").exists()
        assert (tmp_path / "plotly" / "stub.L3.html").exists()
        assert not (tmp_path / "matplotlib").exists()
        html = path.read_text()
        assert "L2 -- MBP" in html
        assert "L3 -- MBO" in html

    def test_view_l2_includes_analytics(self, tmp_path: Path) -> None:
        model = GalleryModel(concepts=[_l2_concept()], analytics=[_metric_spec()])
        path = generate_gallery(
            result=None,
            output_dir=tmp_path,
            model=model,
            view="l2",
            backends=["matplotlib"],
        )
        assert (tmp_path / "matplotlib" / "stubmetric.png").exists()
        assert "Stub Metric" in path.read_text()

    def test_renderer_failure_yields_na_card(self, tmp_path: Path) -> None:
        model = GalleryModel(concepts=[_l2_concept("stubfail", "Fails")])
        path = generate_gallery(
            result=None,
            output_dir=tmp_path,
            model=model,
            backends=["plotly", "matplotlib"],
        )
        html = path.read_text()
        assert "Fails -- L2" in html
        assert html.count("Not available") == 2  # one per backend

    def test_requires_result_or_model(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="requires either"):
            generate_gallery(result=None, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# build_gallery_model (real tiny pipeline)
# ---------------------------------------------------------------------------


class TestBuildGalleryModel:
    def test_inventory_levels(self, tiny_bitstamp_orders_csv) -> None:
        from ob_analytics.bitstamp import BitstampFormat
        from ob_analytics.pipeline import Pipeline

        result = Pipeline(format=BitstampFormat()).run(str(tiny_bitstamp_orders_csv))
        model = build_gallery_model(result)

        keys = {c.key for c in model.concepts}
        # The unconditional order-book concepts are always present.
        assert {
            "trade_tape",
            "depth_heatmap",
            "order_activity",
            "cancellations",
            "events_histogram",
        } <= keys
        # book_snapshot + depth_chart (aggregate vs per-order), cancellations
        # (volume map vs age x distance scatter) and order_activity (event map vs
        # lifecycle Gantt) ship both faces, so they are the comparable concepts;
        # every other concept is L2-only.
        comparable = {c.key for c in model.concepts if c.comparable}
        assert comparable == {
            "book_snapshot",
            "depth_chart",
            "cancellations",
            "order_activity",
        }
        for c in model.concepts:
            assert c.at(Level.L2) is not None
            if c.comparable:
                assert c.at(Level.L3) is not None
            else:
                assert c.at(Level.L3) is None
        # Analytics are appended by callers, not derived here.
        assert model.analytics == []
