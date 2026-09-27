"""The :class:`PlotTheme` value object shared by every rendering backend.

A theme is passed per call -- ``plot(..., theme=PlotTheme(...))`` -- and
applies only to that call; there is no global theme to set.  Each backend
turns the shared fields into its own terms:

* **matplotlib** -- ``style`` / ``context`` / ``font_scale`` go to
  :func:`seaborn.set_theme`, then ``rc`` on top.
* **plotly** -- a per-figure template: ``style`` picks the base template,
  ``context`` x ``font_scale`` sets the font size, then ``plotly_layout`` on
  top.
* **bokeh** -- ``style`` sets the background and grid, ``context`` x
  ``font_scale`` the text sizes, then ``bokeh_figure`` on top.

Every backend draws its colours from ``palette``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ob_analytics.visualization._palette import DEFAULT_PALETTE, Palette

#: Seaborn's context font multipliers.  The plotly and bokeh backends use the
#: same multipliers so ``context="talk"`` enlarges text in every backend.
CONTEXT_SCALES: dict[str, float] = {
    "paper": 0.8,
    "notebook": 1.0,
    "talk": 1.5,
    "poster": 2.0,
}

#: Seaborn styles, and whether each one draws on a dark (gray) background.
STYLES: dict[str, bool] = {
    "white": False,
    "whitegrid": False,
    "ticks": False,
    "dark": True,
    "darkgrid": True,
}


@dataclass(frozen=True)
class PlotTheme:
    """Configurable visual theme for ob-analytics plots, in every backend.

    Attributes
    ----------
    style : str
        Seaborn style name: ``"white"``, ``"whitegrid"``, ``"ticks"``,
        ``"dark"``, or ``"darkgrid"``.  In plotly and bokeh the two dark
        styles draw on seaborn's gray background and ``"ticks"`` draws
        outside tick marks.
    context : str
        Seaborn context name: ``"paper"``, ``"notebook"``, ``"talk"``, or
        ``"poster"``.  Scales text in every backend.
    font_scale : float
        Font scaling factor, applied on top of ``context`` in every backend.
    palette : Palette
        The colours every backend draws with.
    rc : dict[str, object]
        Matplotlib rc overrides applied on top of the seaborn theme
        (matplotlib only).
    plotly_layout : dict[str, object]
        Plotly layout properties applied on top of the theme's template
        (plotly only), e.g. ``{"font": {"family": "Georgia"}}``.
    bokeh_figure : dict[str, object]
        Keyword arguments passed to :func:`bokeh.plotting.figure` on top of
        the theme's defaults (bokeh only), e.g. ``{"width": 1200}``.
    """

    style: str = "white"
    context: str = "notebook"
    font_scale: float = 1.05
    palette: Palette = DEFAULT_PALETTE
    rc: dict[str, object] = field(
        # The reference style (Cleveland–McGill bundle): white background,
        # dotted light grid, no top/right spines, bold left-aligned titles.
        # Built on top of seaborn (style + context still apply).
        default_factory=lambda: {
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": DEFAULT_PALETTE.rule,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "lines.linewidth": 2.0,
        }
    )
    plotly_layout: dict[str, object] = field(
        # The same reference bundle as ``rc``: bold titles aligned with the
        # left edge of the plot area.
        default_factory=lambda: {
            "title": {
                "x": 0.0,
                "xanchor": "left",
                "xref": "paper",
                "font": {"weight": "bold"},
            },
        }
    )
    bokeh_figure: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ValueError(
                f"Unknown style {self.style!r}. Available: {sorted(STYLES)}"
            )
        if self.context not in CONTEXT_SCALES:
            raise ValueError(
                f"Unknown context {self.context!r}. Available: {sorted(CONTEXT_SCALES)}"
            )

    @property
    def dark(self) -> bool:
        """Whether the style draws on a dark (gray) background."""
        return STYLES[self.style]

    @property
    def text_scale(self) -> float:
        """Text size relative to the default theme (``context`` x ``font_scale``).

        The plotly and bokeh backends multiply their base font sizes by this,
        so the default theme keeps their base sizes exactly.
        """
        default = CONTEXT_SCALES["notebook"] * 1.05
        return CONTEXT_SCALES[self.context] * self.font_scale / default


#: Default theme applied when a renderer creates its own figure.  Pass a
#: ``theme=`` kwarg to :func:`~ob_analytics.visualization.plot` (or directly to
#: a renderer) to override it per call; there is no global mutable theme.
DEFAULT_THEME: PlotTheme = PlotTheme()
