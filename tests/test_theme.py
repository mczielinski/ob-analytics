"""PlotTheme applies across backends: one theme, the same look in each.

A theme passed to ``plot(..., theme=...)`` reaches the matplotlib, plotly,
and bokeh renderers, each of which draws its colours from ``theme.palette``
and scales its text by ``context`` x ``font_scale``.
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ob_analytics.visualization import (
    DEFAULT_PALETTE,
    DEFAULT_THEME,
    Level,
    Palette,
    PlotTheme,
    plot,
)
from ob_analytics.visualization._data import (
    prepare_book_snapshot_data,
    prepare_ofi_data,
    prepare_trades_data,
)

# Colours no default palette field uses, so finding them in a figure proves
# the theme's palette reached the renderer.
CUSTOM = Palette(bid="#123456", ask="#654321", buy="#0a0b0c", sell="#c0b0a0")


@pytest.fixture
def trades_data() -> dict:
    ts = pd.Timestamp("2015-05-01 01:00:00")
    trades = pd.DataFrame(
        {
            "timestamp": [ts + pd.Timedelta(seconds=i) for i in range(5)],
            "price": [236.50, 236.55, 236.45, 236.60, 236.50],
            "volume": [100, 200, 150, 300, 250],
            "direction": pd.Categorical(
                ["buy", "sell", "buy", "sell", "buy"], categories=["buy", "sell"]
            ),
        }
    )
    return prepare_trades_data(trades)


@pytest.fixture
def book_data() -> dict:
    book = {
        "timestamp": 1430445600,
        "bids": np.array([[236.50, 100, 100], [236.00, 200, 300]]),
        "asks": np.array([[237.00, 150, 150], [237.50, 250, 400]]),
    }
    return prepare_book_snapshot_data(book, per_order=False)


def _mpl_colours(fig) -> set[str]:
    """Every line / patch / collection colour in a matplotlib figure, as hex."""
    rgba: list = []
    for ax in fig.axes:
        rgba += [line.get_color() for line in ax.get_lines()]
        rgba += [p.get_facecolor() for p in ax.patches]
        for coll in ax.collections:
            rgba += list(coll.get_facecolor()) + list(coll.get_edgecolor())
    plt.close(fig)
    return {mcolors.to_hex(c).lower() for c in rgba}


def _colours(backend: str, fig) -> str | set[str]:
    """What to search for a colour in: a hex set (matplotlib) or JSON text."""
    if backend == "matplotlib":
        return _mpl_colours(fig)
    if backend == "plotly":
        return fig.to_json().lower()
    from bokeh.embed import json_item

    return json.dumps(json_item(fig)).lower()


def _skip_missing(backend: str) -> None:
    if backend == "plotly":
        pytest.importorskip("plotly")
    if backend == "bokeh":
        pytest.importorskip("bokeh")


# ---------------------------------------------------------------------------
# The value object
# ---------------------------------------------------------------------------


class TestPlotTheme:
    def test_default_palette(self) -> None:
        assert DEFAULT_THEME.palette == DEFAULT_PALETTE
        assert DEFAULT_PALETTE.bid == "#0072B2"
        assert DEFAULT_PALETTE.ask == "#D55E00"

    def test_default_text_scale_is_one(self) -> None:
        assert DEFAULT_THEME.text_scale == pytest.approx(1.0)

    def test_context_and_font_scale_multiply(self) -> None:
        theme = PlotTheme(context="talk", font_scale=1.05)
        assert theme.text_scale == pytest.approx(1.5)
        theme = PlotTheme(context="notebook", font_scale=2.1)
        assert theme.text_scale == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"style": "neon"}, "Unknown style"),
            ({"context": "huge"}, "Unknown context"),
        ],
    )
    def test_rejects_unknown_names(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            PlotTheme(**kwargs)

    def test_dark_styles(self) -> None:
        assert PlotTheme(style="darkgrid").dark
        assert PlotTheme(style="dark").dark
        assert not PlotTheme(style="whitegrid").dark


# ---------------------------------------------------------------------------
# The palette reaches every backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
class TestPaletteAcrossBackends:
    def test_trade_tape_uses_buy_and_sell(self, backend: str, trades_data) -> None:
        _skip_missing(backend)
        theme = PlotTheme(palette=CUSTOM)
        found = _colours(
            backend,
            plot("trade_tape", Level.L2, backend=backend, theme=theme, **trades_data),
        )
        assert CUSTOM.buy in found
        assert CUSTOM.sell in found

    def test_book_snapshot_uses_bid_and_ask(self, backend: str, book_data) -> None:
        _skip_missing(backend)
        theme = PlotTheme(palette=CUSTOM)
        fig = plot("book_snapshot", Level.L2, backend=backend, theme=theme, **book_data)
        found = _colours(backend, fig)
        assert CUSTOM.bid in found
        assert CUSTOM.ask in found

    def test_default_palette_without_a_theme(self, backend: str, book_data) -> None:
        _skip_missing(backend)
        found = _colours(
            backend, plot("book_snapshot", Level.L2, backend=backend, **book_data)
        )
        assert DEFAULT_PALETTE.bid.lower() in found
        assert DEFAULT_PALETTE.ask.lower() in found


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_ofi_bars_use_buy_and_sell(backend: str) -> None:
    _skip_missing(backend)
    ts = pd.date_range("2015-01-01", periods=4, freq="min")
    data = prepare_ofi_data(
        pd.DataFrame({"timestamp": ts, "ofi": [0.3, -0.5, 0.7, -0.4]})
    )
    fig = plot(
        "order_flow_imbalance", backend=backend, theme=PlotTheme(palette=CUSTOM), **data
    )
    found = _colours(backend, fig)
    assert CUSTOM.buy in found
    assert CUSTOM.sell in found


# ---------------------------------------------------------------------------
# Plotly: the theme becomes a per-figure template
# ---------------------------------------------------------------------------


class TestPlotlyTemplate:
    @pytest.fixture(autouse=True)
    def _plotly(self) -> None:
        pytest.importorskip("plotly")

    def test_default_font_size_and_left_title(self, trades_data) -> None:
        fig = plot("trade_tape", Level.L2, backend="plotly", **trades_data)
        layout = fig.layout.template.layout
        assert layout.font.size == pytest.approx(13)
        assert layout.title.x == 0
        assert layout.title.xanchor == "left"
        assert layout.title.font.weight == "bold"

    def test_context_scales_the_font(self, trades_data) -> None:
        theme = PlotTheme(context="talk")
        fig = plot("trade_tape", Level.L2, backend="plotly", theme=theme, **trades_data)
        assert fig.layout.template.layout.font.size == pytest.approx(13 * 1.5)

    def test_dark_style_uses_seaborn_background(self, trades_data) -> None:
        import plotly.io as pio

        fig = plot(
            "trade_tape",
            Level.L2,
            backend="plotly",
            theme=PlotTheme(style="darkgrid"),
            **trades_data,
        )
        seaborn_bg = pio.templates["seaborn"].layout.plot_bgcolor
        assert fig.layout.template.layout.plot_bgcolor == seaborn_bg

    def test_plotly_layout_goes_on_top(self, trades_data) -> None:
        theme = PlotTheme(plotly_layout={"font": {"family": "Georgia"}})
        fig = plot("trade_tape", Level.L2, backend="plotly", theme=theme, **trades_data)
        assert fig.layout.template.layout.font.family == "Georgia"
        # The theme's font size survives a layout override that omits it.
        assert fig.layout.template.layout.font.size == pytest.approx(13)

    def test_no_global_template_is_registered(self, trades_data) -> None:
        import plotly.io as pio

        before = (list(pio.templates), pio.templates.default)
        plot(
            "trade_tape",
            Level.L2,
            backend="plotly",
            theme=PlotTheme(style="darkgrid"),
            **trades_data,
        )
        assert (list(pio.templates), pio.templates.default) == before


# ---------------------------------------------------------------------------
# Bokeh: background, text size, figure overrides
# ---------------------------------------------------------------------------


class TestBokehTheme:
    @pytest.fixture(autouse=True)
    def _bokeh(self) -> None:
        pytest.importorskip("bokeh")

    def test_default_sizes(self, trades_data) -> None:
        fig = plot("trade_tape", Level.L2, backend="bokeh", **trades_data)
        assert fig.title.text_font_size == "13pt"
        assert fig.background_fill_color == "#ffffff"

    def test_context_scales_text(self, trades_data) -> None:
        fig = plot(
            "trade_tape",
            Level.L2,
            backend="bokeh",
            theme=PlotTheme(context="poster"),
            **trades_data,
        )
        assert fig.title.text_font_size == f"{13 * 2.0:g}pt"

    def test_dark_style_background(self, trades_data) -> None:
        fig = plot(
            "trade_tape",
            Level.L2,
            backend="bokeh",
            theme=PlotTheme(style="darkgrid"),
            **trades_data,
        )
        assert fig.background_fill_color == "#EAEAF2"

    def test_bokeh_figure_goes_on_top(self, trades_data) -> None:
        theme = PlotTheme(bokeh_figure={"width": 1234})
        fig = plot("trade_tape", Level.L2, backend="bokeh", theme=theme, **trades_data)
        assert fig.width == 1234


# ---------------------------------------------------------------------------
# plot_result threads the theme too
# ---------------------------------------------------------------------------


def test_plot_result_accepts_theme(tiny_bitstamp_orders_csv) -> None:
    pytest.importorskip("plotly")
    from ob_analytics.bitstamp import BitstampSource
    from ob_analytics.pipeline import Pipeline
    from ob_analytics.visualization import plot_result

    result = Pipeline(source=BitstampSource()).run(str(tiny_bitstamp_orders_csv))
    fig = plot_result(
        result, "trade_tape", backend="plotly", theme=PlotTheme(palette=CUSTOM)
    )
    found = fig.to_json().lower()
    assert CUSTOM.buy in found or CUSTOM.sell in found
