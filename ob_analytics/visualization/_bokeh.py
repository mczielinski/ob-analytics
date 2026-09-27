"""Bokeh interactive rendering backend for ob-analytics.

Each ``bokeh_*()`` function takes a prepared data dict (from
:mod:`~ob_analytics.visualization._data`) and returns a
:class:`bokeh.plotting.figure` with interactive zoom, pan, and hover --
suited to Bokeh / Panel server dashboards and streaming views, alongside the
static Matplotlib default and the Plotly interactive backend.

Covers the core concepts -- ``trade_tape``, ``depth_heatmap``,
``book_snapshot``, ``depth_chart`` -- at both L2 (aggregate) and L3
(per-order) resolution where the other backends do.

Install via ``pip install ob-analytics[bokeh]``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ob_analytics.exceptions import ConfigError
from ob_analytics.visualization._data import (
    biased_color_norm,
    book_bar_thickness,
    book_mid,
    check_book_payload_level,
    mpl_marker_area_to_plotly_size,
)
from ob_analytics.visualization._palette import Palette
from ob_analytics.visualization._theme import DEFAULT_THEME, PlotTheme


@lru_cache(maxsize=1)
def _import_bokeh() -> Any:
    """Lazy-import ``bokeh.plotting`` with a friendly error message.

    Cached so bokeh is imported once per process rather than on every
    render call.  ``lru_cache`` only stores successful returns, so when
    bokeh is missing the ``ConfigError`` is re-raised on each call exactly
    as before.
    """
    try:
        import bokeh.plotting as bpl

        return bpl
    except ImportError:
        raise ConfigError(
            "Bokeh is required for interactive visualizations. "
            "Install it with:  pip install ob-analytics[bokeh]"
        ) from None


# ---------------------------------------------------------------------------
# Shared layout helpers
# ---------------------------------------------------------------------------

_BASE_FIGURE_KWARGS: dict[str, Any] = {
    "width": 900,
    "height": 520,
    "tools": "pan,wheel_zoom,box_zoom,reset,save",
    "toolbar_location": "above",
}

# Plot-area background of the light and dark styles (the dark one is
# seaborn's gray, as in the matplotlib and plotly backends).
_LIGHT_BACKGROUND = "#ffffff"
_DARK_BACKGROUND = "#EAEAF2"

# Text sizes of the default theme (bokeh's own axis defaults); other themes
# scale them by ``PlotTheme.text_scale`` (context x font_scale).
_TITLE_FONT_PT = 13
_AXIS_LABEL_FONT_PX = 13
_TICK_LABEL_FONT_PX = 11


def _base_figure(bpl: Any, theme: PlotTheme, title: str = "", **kwargs: Any) -> Any:
    """Create a Bokeh figure styled by *theme*.

    ``theme.bokeh_figure`` goes on top of the theme's defaults; the face's own
    *kwargs* (axis types, ranges) go last.
    """
    opts: dict[str, Any] = {
        **_BASE_FIGURE_KWARGS,
        "background_fill_color": _DARK_BACKGROUND if theme.dark else _LIGHT_BACKGROUND,
        "title": title,
        "x_axis_type": "datetime",
    }
    opts.update(theme.bokeh_figure)
    opts.update(kwargs)
    fig = bpl.figure(**opts)
    scale = theme.text_scale
    fig.title.text_font_size = f"{_TITLE_FONT_PT * scale:g}pt"
    fig.title.align = "left"
    fig.axis.axis_label_text_font_size = f"{_AXIS_LABEL_FONT_PX * scale:g}px"
    fig.axis.major_label_text_font_size = f"{_TICK_LABEL_FONT_PX * scale:g}px"
    fig.grid.grid_line_dash = [2, 2]
    fig.grid.grid_line_alpha = 0.35
    if theme.dark:
        fig.grid.grid_line_color = "white"
        fig.grid.grid_line_alpha = 1.0
        fig.grid.grid_line_dash = []
    return fig


def _maybe_legend(fig: Any) -> None:
    """Place and enable click-to-hide on the legend, if any glyph declared one."""
    if fig.legend:
        fig.legend.location = "top_left"
        fig.legend.click_policy = "hide"


def _apply_padded_y_range(fig: Any, y_range: tuple[float, float] | None) -> None:
    """Set a 4%-padded y-range so spike prints are never cut at the axis edge."""
    if y_range is None:
        return
    lo, hi = y_range
    pad = (hi - lo) * 0.04 or 1.0
    fig.y_range.start = lo - pad
    fig.y_range.end = hi + pad


# ---------------------------------------------------------------------------
# Trade tape
# ---------------------------------------------------------------------------


def _bokeh_mid_line(fig: Any, mid_line: Any, pal: Palette) -> None:
    """Reference mid/microprice line, held constant until the next sample.

    "after" steps: the mid holds until the book changes; linear
    interpolation would paint a ramp between sparse samples.
    """
    if mid_line is None or mid_line.empty:
        return
    fig.step(
        x=mid_line["timestamp"],
        y=mid_line["mid"],
        mode="after",
        line_color=pal.reference_line,
        line_width=1,
        line_alpha=0.8,
    )


def _bokeh_lollipops(fig: Any, side: Any, color: str, label: str) -> None:
    """Stems (mid -> price) plus volume-sized markers for one tape side."""
    from bokeh.models import ColumnDataSource

    # One source shared by the stem + marker glyphs, so the (potentially
    # large, one-row-per-order) timestamp/price columns convert once.
    source = ColumnDataSource(
        {
            "timestamp": side["timestamp"],
            "mid": side["mid"],
            "price": side["price"],
            "size": mpl_marker_area_to_plotly_size(side["marker_area"].to_numpy()),
        }
    )
    fig.segment(
        x0="timestamp",
        y0="mid",
        x1="timestamp",
        y1="price",
        source=source,
        line_color=color,
        line_width=1,
        line_alpha=0.5,
    )
    fig.scatter(
        x="timestamp",
        y="price",
        source=source,
        size="size",
        fill_color=color,
        line_color=None,
        fill_alpha=0.9,
        legend_label=label,
    )


def bokeh_trades(data: dict, *, theme: PlotTheme = DEFAULT_THEME) -> Any:
    """Render the L2 signed-lollipop trade tape.

    Each trade is a stem from the mid line to its execution price, tipped by
    a volume-sized marker and coloured by aggressor side.  The price axis
    spans the full data extent (no quantile clip), so spike prints stay
    visible.
    """
    pal = theme.palette
    bpl = _import_bokeh()
    fig = _base_figure(
        bpl, theme, title="Trade Prices", x_axis_label="Time", y_axis_label="Price"
    )

    _bokeh_mid_line(fig, data.get("mid_line"), pal)
    for side, color, label in (
        (data["buys"], pal.buy, "buy (lifts ask)"),
        (data["sells"], pal.sell, "sell (hits bid)"),
    ):
        if side.empty:
            continue
        _bokeh_lollipops(fig, side, color, label)

    _apply_padded_y_range(fig, data.get("y_range"))
    _maybe_legend(fig)
    return fig


def bokeh_trade_tape_per_order(data: dict, *, theme: PlotTheme = DEFAULT_THEME) -> Any:
    """L3 (MBO) signed-lollipop trade tape with maker resting spans.

    Same signed lollipops as the L2 tape (stem mid -> price, marker sized by
    volume, coloured by aggressor side), plus the L3 differentiator: a faint
    span from each consumed maker order's creation to its fill.  Above the
    density threshold the lollipops are per-second VWAPs.
    """
    pal = theme.palette
    bpl = _import_bokeh()
    fig = _base_figure(
        bpl,
        theme,
        title="Trade tape with maker order lifecycles",
        x_axis_label="Time",
        y_axis_label="Execution Price",
    )

    dense = data.get("dense", False)
    span_alpha = 0.12 if dense else 0.35
    # Maker resting spans (horizontal), faint underneath the lollipops.
    for side, color in ((data["buys"], pal.buy), (data["sells"], pal.sell)):
        if side.empty:
            continue
        fig.segment(
            x0=side["created_ts"],
            y0=side["price"],
            x1=side["timestamp"],
            y1=side["price"],
            line_color=color,
            line_width=1.0,
            line_alpha=span_alpha,
        )

    _bokeh_mid_line(fig, data.get("mid_line"), pal)

    suffix = ", per-s VWAP" if dense else ""
    for side, color, label in (
        (data["lolli_buys"], pal.buy, f"buy (lifts ask){suffix}"),
        (data["lolli_sells"], pal.sell, f"sell (hits bid){suffix}"),
    ):
        if side.empty:
            continue
        _bokeh_lollipops(fig, side, color, label)

    _apply_padded_y_range(fig, data.get("y_range"))
    _maybe_legend(fig)
    return fig


# ---------------------------------------------------------------------------
# Depth heatmap
# ---------------------------------------------------------------------------

# Re-exported under the backend's own name: shared with the plotly backend's
# depth-heatmap color mapping (see _data.biased_color_norm's docstring).
_bokeh_color_field = biased_color_norm


def bokeh_price_levels(data: dict, *, theme: PlotTheme = DEFAULT_THEME) -> Any:
    """Render the price-level depth heatmap."""
    pal = theme.palette
    bpl = _import_bokeh()
    from bokeh.models import (
        ColorBar,
        ColumnDataSource,
        FixedTicker,
        HoverTool,
        LinearColorMapper,
    )
    from bokeh.palettes import Viridis256

    depth = data["depth"]
    spread = data["spread"]
    trades = data["trades"]
    show_mp = data["show_mp"]
    col_bias = data.get("col_bias", 1.0)

    fig = _base_figure(
        bpl,
        theme,
        title="Price Levels Over Time",
        x_axis_label="Time",
        y_axis_label="Limit Price",
    )

    if not depth.empty:
        vol = depth["volume"].fillna(0)
        color_t, tickvals, ticktext = _bokeh_color_field(
            depth["volume"].to_numpy(), col_bias
        )
        mapper = LinearColorMapper(palette=Viridis256, low=0.0, high=1.0)
        source = ColumnDataSource(
            {
                "x": depth["timestamp"],
                "y": depth["price"],
                "t": color_t,
                "volume": vol,
                "alpha": np.where(vol > 0, 0.8, 0.1),
            }
        )
        fig.scatter(
            x="x",
            y="y",
            source=source,
            size=4,
            fill_color={"field": "t", "transform": mapper},
            fill_alpha="alpha",
            line_color=None,
        )
        if tickvals:
            fig.add_layout(
                ColorBar(
                    color_mapper=mapper,
                    title="Volume",
                    ticker=FixedTicker(ticks=tickvals),
                    major_label_overrides=dict(zip(tickvals, ticktext, strict=True)),
                    width=8,
                ),
                "right",
            )
        fig.add_tools(
            HoverTool(
                tooltips=[
                    ("Time", "@x{%F %T}"),
                    ("Price", "@y{0.00}"),
                    ("Volume", "@volume{0.0000}"),
                ],
                formatters={"@x": "datetime"},
            )
        )

    if spread is not None and show_mp:
        if "best_bid_price" in spread and "best_ask_price" in spread:
            mp = (spread["best_bid_price"] + spread["best_ask_price"]) / 2
            fig.step(
                x=spread["timestamp"],
                y=mp,
                mode="after",
                line_color=pal.price_line,
                line_width=1.5,
                legend_label="Midprice",
            )
    elif spread is not None:
        if "best_ask_price" in spread:
            fig.step(
                x=spread["timestamp"],
                y=spread["best_ask_price"],
                mode="after",
                line_color=pal.ask,
                line_width=1.2,
                line_dash="dotted",
                legend_label="Best Ask",
            )
        if "best_bid_price" in spread:
            fig.step(
                x=spread["timestamp"],
                y=spread["best_bid_price"],
                mode="after",
                line_color=pal.bid,
                line_width=1.2,
                line_dash="dotted",
                legend_label="Best Bid",
            )

    if trades is not None and not trades.empty:
        buys = trades[trades["direction"] == "buy"]
        sells = trades[trades["direction"] == "sell"]
        if not sells.empty:
            fig.scatter(
                x=sells["timestamp"],
                y=sells["price"],
                marker="inverted_triangle",
                size=8,
                fill_color=pal.sell,
                line_color="white",
                legend_label="Sell Trades",
            )
        if not buys.empty:
            fig.scatter(
                x=buys["timestamp"],
                y=buys["price"],
                marker="triangle",
                size=8,
                fill_color=pal.buy,
                line_color="white",
                legend_label="Buy Trades",
            )

    y_range = data.get("y_range")
    if y_range is not None:
        fig.y_range.start, fig.y_range.end = y_range

    _maybe_legend(fig)
    return fig


# ---------------------------------------------------------------------------
# Book snapshot + depth chart
# ---------------------------------------------------------------------------


def _bokeh_book_bars(
    data: dict, *, per_order: bool, theme: PlotTheme = DEFAULT_THEME
) -> Any:
    """Horizontal book ladder: price on y, size on x, bids below / asks above.

    L2 draws one bar per price level; L3 segments each level into its
    individual orders with white separators, so equal-total levels with
    different composition read differently.
    """
    pal = theme.palette
    check_book_payload_level(data, per_order=per_order)
    bpl = _import_bokeh()
    from bokeh.models import Span

    bids = data["bids"]
    asks = data["asks"]
    fig = _base_figure(
        bpl,
        theme,
        title=data["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC"),
        x_axis_type="linear",
        x_axis_label="Size (per order)" if per_order else "Size",
        y_axis_label="Price",
    )

    thickness = book_bar_thickness(bids, asks) * 0.9
    # White per-order separators (dark ones vanished against the fill).
    line_color = "white" if per_order else None
    line_width = 1.0 if per_order else 0.0
    for side, color, label in ((bids, pal.bid, "Bid"), (asks, pal.ask, "Ask")):
        if side.empty:
            continue
        fig.hbar(
            y=side["price"],
            left=side["seg_lo"],
            right=side["seg_hi"],
            height=thickness,
            fill_color=color,
            line_color=line_color,
            line_width=line_width,
            legend_label=label,
        )

    mid = book_mid(bids, asks)
    if mid is not None:
        fig.add_layout(
            Span(
                location=mid,
                dimension="width",
                line_color=pal.rule,
                line_dash="dashed",
                line_width=1,
            )
        )

    if data["show_quantiles"]:
        for y_val in (*data["bid_quantiles"], *data["ask_quantiles"]):
            fig.add_layout(
                Span(
                    location=y_val,
                    dimension="width",
                    line_color=pal.reference_line,
                    line_dash="dotted",
                    line_width=0.8,
                )
            )

    fig.x_range.start = 0
    _maybe_legend(fig)
    return fig


def bokeh_book_snapshot_aggregate(
    data: dict, *, theme: PlotTheme = DEFAULT_THEME
) -> Any:
    """L2 (MBP) book snapshot: aggregate size per price level."""
    return _bokeh_book_bars(data, per_order=False, theme=theme)


def bokeh_book_snapshot_per_order(
    data: dict, *, theme: PlotTheme = DEFAULT_THEME
) -> Any:
    """L3 (MBO) book snapshot: each order a stacked segment within its level."""
    return _bokeh_book_bars(data, per_order=True, theme=theme)


def _bokeh_depth_curve(
    data: dict, *, per_order: bool, theme: PlotTheme = DEFAULT_THEME
) -> Any:
    """Cumulative-depth curve: stepped per level (L2) or per order (L3)."""
    pal = theme.palette
    check_book_payload_level(data, per_order=per_order)
    bpl = _import_bokeh()
    from bokeh.models import ColumnDataSource

    fig = _base_figure(
        bpl,
        theme,
        title=data["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC"),
        x_axis_type="linear",
        x_axis_label="Price",
        y_axis_label="Cumulative liquidity",
    )

    for side, color, label in (
        (data["bids"], pal.bid, "Bid"),
        (data["asks"], pal.ask, "Ask"),
    ):
        if side.empty:
            continue
        s = side.sort_values("price")
        # One source shared across the fill/step/scatter glyphs below, so the
        # price/liquidity columns are converted once per side, not per glyph.
        source = ColumnDataSource({"price": s["price"], "liquidity": s["liquidity"]})
        # "before" steps mirror the matplotlib ``where="pre"`` curve: the
        # level jumps to y_i before reaching x_i.
        fig.varea(
            x="price",
            y1=0,
            y2="liquidity",
            source=source,
            fill_color=color,
            fill_alpha=0.15,
        )
        fig.step(
            x="price",
            y="liquidity",
            source=source,
            mode="before",
            line_color=color,
            line_width=2,
            legend_label=label,
        )
        if per_order:
            fig.scatter(
                x="price",
                y="liquidity",
                source=source,
                size=6,
                fill_color=color,
                line_color=None,
            )

    fig.y_range.start = 0
    _maybe_legend(fig)
    return fig


def bokeh_depth_chart_aggregate(data: dict, *, theme: PlotTheme = DEFAULT_THEME) -> Any:
    """L2 (MBP) depth chart: cumulative liquidity stepped per price level."""
    return _bokeh_depth_curve(data, per_order=False, theme=theme)


def bokeh_depth_chart_per_order(data: dict, *, theme: PlotTheme = DEFAULT_THEME) -> Any:
    """L3 (MBO) depth chart: cumulative liquidity stepped per individual order."""
    return _bokeh_depth_curve(data, per_order=True, theme=theme)


# ---------------------------------------------------------------------------
# Renderer self-registration
#
# Imported here (not at module top) so RENDERERS -- defined in the package
# __init__ -- already exists when this (lazily imported) module is loaded.
from ob_analytics.visualization import RENDERERS, Level

# (concept, level, renderer); mirrors the matplotlib / plotly backends'
# coordinates for the core concepts this backend covers.
_L2 = Level.L2
_L3 = Level.L3
for _concept, _level, _fn in [
    ("trade_tape", _L2, bokeh_trades),
    ("trade_tape", _L3, bokeh_trade_tape_per_order),
    ("depth_heatmap", _L2, bokeh_price_levels),
    ("book_snapshot", _L2, bokeh_book_snapshot_aggregate),
    ("book_snapshot", _L3, bokeh_book_snapshot_per_order),
    ("depth_chart", _L2, bokeh_depth_chart_aggregate),
    ("depth_chart", _L3, bokeh_depth_chart_per_order),
]:
    RENDERERS.register((_concept, _level, "bokeh"), _fn)
