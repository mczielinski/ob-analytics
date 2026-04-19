"""Tests for ``ob_analytics.visualization.gallery``.

These tests exercise the gallery generator directly with hand-built
``PlotSpec`` lists and stub ``plot_fn`` callables, so they don't depend
on running a full pipeline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest

from ob_analytics.visualization.gallery import (
    PlotSpec,
    _render_panel,
    _write_gallery_html,
    generate_gallery,
)


# ---------------------------------------------------------------------------
# Stub plot functions
# ---------------------------------------------------------------------------


def _stub_mpl(backend: str = "matplotlib"):
    """Return a tiny matplotlib or plotly figure depending on backend."""
    if backend == "matplotlib":
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig
    if backend == "plotly":
        import plotly.graph_objects as go

        return go.Figure(data=go.Scatter(x=[0, 1], y=[0, 1]))
    raise ValueError(f"unsupported backend: {backend}")


def _stub_failing(backend: str = "matplotlib"):
    raise RuntimeError(f"deliberate {backend} failure")


# ---------------------------------------------------------------------------
# _render_panel
# ---------------------------------------------------------------------------


class TestRenderPanel:
    def test_plotly_rendered_uses_iframe(self) -> None:
        html = _render_panel(
            "plotly", "01_demo", rendered=True, role="primary", escaped_title="Demo"
        )
        assert "<iframe" in html
        assert 'src="plotly/01_demo.html"' in html
        assert "plotly-panel" in html
        assert "panel-primary" in html

    def test_matplotlib_rendered_uses_img(self) -> None:
        html = _render_panel(
            "matplotlib",
            "01_demo",
            rendered=True,
            role="secondary",
            escaped_title="Demo",
        )
        assert "<img" in html
        assert 'src="matplotlib/01_demo.png"' in html
        assert "mpl-panel" in html
        assert "panel-secondary" in html

    def test_not_rendered_shows_na(self) -> None:
        html = _render_panel(
            "plotly", "x", rendered=False, role="primary", escaped_title="x"
        )
        assert "Not available" in html
        assert "<iframe" not in html
        assert "<img" not in html


# ---------------------------------------------------------------------------
# _write_gallery_html
# ---------------------------------------------------------------------------


class TestWriteGalleryHtml:
    def test_plotly_first_renders_as_primary(self, tmp_path: Path) -> None:
        out = tmp_path / "g.html"
        plots = [("01_demo", "Demo", {"plotly": True, "matplotlib": True})]
        _write_gallery_html(out, plots, "Title", ["plotly", "matplotlib"])
        body = out.read_text().split("<body>", 1)[1]
        plotly_pos = body.index("plotly-panel")
        mpl_pos = body.index("mpl-panel")
        assert plotly_pos < mpl_pos
        primary_idx = body.index("panel-primary")
        secondary_idx = body.index("panel-secondary")
        assert primary_idx < secondary_idx
        # The primary marker sits inside the plotly panel.
        assert body[plotly_pos:mpl_pos].count("panel-primary") == 1

    def test_matplotlib_only_has_no_secondary(self, tmp_path: Path) -> None:
        out = tmp_path / "g.html"
        plots = [("01_demo", "Demo", {"matplotlib": True})]
        _write_gallery_html(out, plots, "Title", ["matplotlib"])
        body = out.read_text().split("<body>", 1)[1]
        assert "panel-primary" in body
        assert "panel-secondary" not in body
        assert "plotly-panel" not in body

    def test_failed_backend_renders_na_panel(self, tmp_path: Path) -> None:
        out = tmp_path / "g.html"
        plots = [("01_demo", "Demo", {"plotly": False, "matplotlib": True})]
        _write_gallery_html(out, plots, "Title", ["plotly", "matplotlib"])
        body = out.read_text().split("<body>", 1)[1]
        plotly_chunk = body[body.index("plotly-panel") : body.index("mpl-panel")]
        assert "Not available" in plotly_chunk
        assert "iframe" not in plotly_chunk

    def test_html_escapes_plot_titles(self, tmp_path: Path) -> None:
        out = tmp_path / "g.html"
        plots = [("01", "<script>X</script>", {"matplotlib": True})]
        _write_gallery_html(out, plots, "Page", ["matplotlib"])
        html = out.read_text()
        assert "<script>X</script>" not in html
        assert "&lt;script&gt;X&lt;/script&gt;" in html


# ---------------------------------------------------------------------------
# generate_gallery
# ---------------------------------------------------------------------------


class TestGenerateGallery:
    def _specs(self) -> list[PlotSpec]:
        return [PlotSpec("01_demo", "Demo Plot", _stub_mpl, {})]

    def test_default_backends_prefer_plotly_when_available(
        self, tmp_path: Path
    ) -> None:
        # plotly is installed in the dev env; default ordering should be
        # ["plotly", "matplotlib"].
        path = generate_gallery(
            result=None,
            output_dir=tmp_path,
            specs=self._specs(),
        )
        assert path.exists()
        assert (tmp_path / "plotly" / "01_demo.html").exists()
        assert (tmp_path / "matplotlib" / "01_demo.png").exists()
        body = path.read_text().split("<body>", 1)[1]
        # Plotly comes first → primary class lands on the plotly panel.
        plotly_idx = body.index("plotly-panel")
        mpl_idx = body.index("mpl-panel")
        assert plotly_idx < mpl_idx

    def test_plotly_unavailable_falls_back_to_matplotlib(self, tmp_path: Path) -> None:
        # Simulate plotly missing via ImportError on `import plotly`.
        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def fake_import(name, *a, **kw):
            if name == "plotly":
                raise ImportError("simulated")
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=fake_import):
            path = generate_gallery(
                result=None,
                output_dir=tmp_path,
                specs=self._specs(),
            )
        assert path.exists()
        assert (tmp_path / "matplotlib" / "01_demo.png").exists()
        assert not (tmp_path / "plotly").exists()
        body = path.read_text().split("<body>", 1)[1]
        assert "plotly-panel" not in body

    def test_explicit_backends_preserved(self, tmp_path: Path) -> None:
        path = generate_gallery(
            result=None,
            output_dir=tmp_path,
            specs=self._specs(),
            backends=["matplotlib", "plotly"],  # mpl primary
        )
        body = path.read_text().split("<body>", 1)[1]
        # mpl listed first → mpl panel is primary.
        mpl_idx = body.index("mpl-panel")
        plotly_idx = body.index("plotly-panel")
        assert mpl_idx < plotly_idx

    def test_failure_in_one_backend_does_not_skip_others(self, tmp_path: Path) -> None:
        spec = PlotSpec("01_demo", "Demo", _stub_failing, {})
        path = generate_gallery(
            result=None,
            output_dir=tmp_path,
            specs=[spec],
            backends=["plotly", "matplotlib"],
        )
        assert path.exists()
        # Neither backend produced a file, but the HTML still mentions the card.
        html = path.read_text()
        assert "Demo" in html
        assert html.count("Not available") == 2  # one per backend

    def test_at_least_one_panel_per_card(self, tmp_path: Path) -> None:
        path = generate_gallery(
            result=None,
            output_dir=tmp_path,
            specs=self._specs(),
            backends=["matplotlib"],
        )
        html = path.read_text()
        # Single backend → exactly one panel per card.
        assert html.count('class="panel ') == 1


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
