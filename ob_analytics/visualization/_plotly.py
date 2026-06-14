"""Plotly interactive rendering backend for ob-analytics.

Each ``plotly_*()`` function takes a prepared data dict (from
:mod:`~ob_analytics.visualization._data`) and returns a
:class:`plotly.graph_objects.Figure` with interactive zoom, pan, and hover.

Install via ``pip install ob-analytics[interactive]``.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from ob_analytics.exceptions import ConfigError
from ob_analytics.visualization._data import book_mid, mpl_marker_area_to_plotly_size
from ob_analytics.visualization._palette import (
    _ASK_COLOR,
    _BID_COLOR,
    _BUY_COLOR,
    _CANCELLED_COLOR,
    _FILLED_COLOR,
    _PARTIAL_COLOR,
    _SELL_COLOR,
)


@lru_cache(maxsize=1)
def _import_plotly() -> Any:
    """Lazy-import plotly with a friendly error message.

    Cached so plotly is imported once per process rather than on every
    render call.  ``lru_cache`` only stores successful returns, so when
    plotly is missing the ``ConfigError`` is re-raised on each call
    exactly as before.
    """
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        raise ConfigError(
            "Plotly is required for interactive visualizations. "
            "Install it with:  pip install ob-analytics[interactive]"
        ) from None


# ---------------------------------------------------------------------------
# Shared layout helpers
# ---------------------------------------------------------------------------

# Light theme, matching the matplotlib default (one polarity across the
# gallery; the reference bundle is light).
_BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified",
)


def _base_figure(go: Any, title: str = "", **kwargs: Any) -> Any:
    """Create a Plotly figure with the dark ob-analytics theme."""
    layout = {**_BASE_LAYOUT, "title": dict(text=title, x=0.5)}
    layout.update(kwargs)
    return go.Figure(layout=layout)


# ---------------------------------------------------------------------------
# Rendering functions
# ---------------------------------------------------------------------------


def plotly_time_series(data: dict) -> Any:
    """Render a time-series step plot."""
    go = _import_plotly()
    df = data["df"]
    fig = _base_figure(go, title=data["title"])
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=df["val"],
            mode="lines",
            line=dict(shape="hv", width=2, color="#5dade2"),
            name=data["y_label"],
        )
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text=data["y_label"])
    return fig


def plotly_trades(data: dict) -> Any:
    """Render the L2 signed-lollipop trade tape.

    Each trade is a stem from the mid line to its execution price, tipped by a
    volume-sized marker and coloured by aggressor side.  The price axis spans
    the full data extent (no quantile clip), so spike prints stay visible.
    """
    go = _import_plotly()
    fig = _base_figure(go, title="Trade Prices")

    mid_line = data.get("mid_line")
    if mid_line is not None and not mid_line.empty:
        fig.add_trace(
            go.Scattergl(
                x=mid_line["timestamp"],
                y=mid_line["mid"],
                mode="lines",
                line=dict(color="#888888", width=1),
                opacity=0.8,
                name="mid",
                hoverinfo="skip",
            )
        )
    for side, color, label in (
        (data["buys"], _BUY_COLOR, "buy (lifts ask)"),
        (data["sells"], _SELL_COLOR, "sell (hits bid)"),
    ):
        if side.empty:
            continue
        _plotly_lollipops(fig, go, side, color, label)

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Price")
    _apply_padded_y_range(fig, data.get("y_range"))
    return fig


def _format_volume(value: float) -> str:
    """Compact label for a colorbar volume tick."""
    if not math.isfinite(value):
        return ""
    a = abs(value)
    if a != 0 and (a < 1e-3 or a >= 1e6):
        return f"{value:.1e}"
    if a >= 1000:
        return f"{value:,.0f}"
    if a >= 1:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _biased_color_norm(
    volume: np.ndarray, col_bias: float, n_ticks: int = 5
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map volumes to [0, 1] color positions under a power/log bias.

    Mirrors the matplotlib backend (see ``mpl_price_levels``): ``col_bias``
    of ``1.0`` is linear, ``0 < col_bias < 1`` is a PowerNorm gamma that
    brightens low-volume levels, and ``col_bias <= 0`` selects log10. Returns
    the normalized color array plus ``marker.colorbar`` kwargs whose ticks sit
    in normalized space but are labeled in original volume units.
    """
    v = np.asarray(volume, dtype=float)
    if col_bias <= 0:
        finite = v[np.isfinite(v) & (v > 0)]
    else:
        finite = v[np.isfinite(v)]
    base_bar = dict(title="Volume", x=1.02, len=0.75)
    if finite.size == 0:
        return np.zeros_like(v), base_bar
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    if col_bias <= 0:
        lo, hi = math.log(vmin), math.log(vmax)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (np.log(np.clip(v, vmin, vmax)) - lo) / (hi - lo)

        def inv(tt: float) -> float:
            return math.exp(lo + tt * (hi - lo))
    else:
        gamma = col_bias
        base = (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin)
        t = base**gamma

        def inv(tt: float) -> float:
            return vmin + (tt ** (1.0 / gamma)) * (vmax - vmin)

    t = np.nan_to_num(t, nan=0.0)
    tickvals = [i / (n_ticks - 1) for i in range(n_ticks)]
    ticktext = [_format_volume(inv(tv)) for tv in tickvals]
    return t, {**base_bar, "tickvals": tickvals, "ticktext": ticktext}


def plotly_price_levels(data: dict) -> Any:
    """Render the price-level depth heatmap using Scattergl."""
    go = _import_plotly()
    depth = data["depth"]
    spread = data["spread"]
    trades = data["trades"]
    show_mp = data["show_mp"]
    col_bias = data.get("col_bias", 1.0)

    fig = _base_figure(go, title="Price Levels Over Time")

    if not depth.empty:
        vol = depth["volume"].fillna(0)
        # Mirror the matplotlib col_bias norm: feed normalized color positions
        # to the colorscale, keep colorbar ticks in real volume units, and let
        # true volume ride along as customdata so hover still reads true size.
        color_t, colorbar = _biased_color_norm(depth["volume"].to_numpy(), col_bias)

        fig.add_trace(
            go.Scattergl(
                x=depth["timestamp"],
                y=depth["price"],
                mode="markers",
                customdata=vol,
                marker=dict(
                    size=3,
                    color=color_t,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=1,
                    colorbar=colorbar,
                    opacity=np.where(vol > 0, 0.8, 0.1),
                ),
                hovertemplate=(
                    "Time: %{x}<br>Price: %{y:.2f}<br>"
                    "Volume: %{customdata:.4f}<extra></extra>"
                ),
                name="Depth",
            )
        )
        fig.update_layout(margin=dict(l=60, r=90, t=50, b=50))

    if spread is not None and show_mp:
        if "best_bid_price" in spread and "best_ask_price" in spread:
            mp = (spread["best_bid_price"] + spread["best_ask_price"]) / 2
            fig.add_trace(
                go.Scatter(
                    x=spread["timestamp"],
                    y=mp,
                    mode="lines",
                    line=dict(color="#222222", width=1.5, shape="hv"),
                    name="Midprice",
                )
            )
    elif spread is not None:
        if "best_ask_price" in spread:
            fig.add_trace(
                go.Scatter(
                    x=spread["timestamp"],
                    y=spread["best_ask_price"],
                    mode="lines",
                    line=dict(color=_ASK_COLOR, width=1.2, dash="dot"),
                    name="Best Ask",
                )
            )
        if "best_bid_price" in spread:
            fig.add_trace(
                go.Scatter(
                    x=spread["timestamp"],
                    y=spread["best_bid_price"],
                    mode="lines",
                    line=dict(color=_BID_COLOR, width=1.2, dash="dot"),
                    name="Best Bid",
                )
            )

    if trades is not None and not trades.empty:
        buys = trades[trades["direction"] == "buy"]
        sells = trades[trades["direction"] == "sell"]
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["price"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=8,
                        color=_SELL_COLOR,
                        line=dict(width=1, color="white"),
                    ),
                    name="Sell Trades",
                )
            )
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["timestamp"],
                    y=buys["price"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=8,
                        color=_BUY_COLOR,
                        line=dict(width=1, color="white"),
                    ),
                    name="Buy Trades",
                )
            )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Limit Price")
    y_range = data.get("y_range")
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


def plotly_event_map(data: dict) -> Any:
    """Render a limit-order event map."""
    go = _import_plotly()
    created = data["created"]
    deleted = data["deleted"]

    fig = _base_figure(go, title="Limit Order Event Map")

    col_map = {"bid": _BID_COLOR, "ask": _ASK_COLOR}

    if not created.empty:
        for direction in ["bid", "ask"]:
            subset = created[created["direction"] == direction]
            if subset.empty:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=subset["timestamp"],
                    y=subset["price"],
                    mode="markers",
                    marker=dict(
                        size=np.clip(subset["volume"] * 20, 3, 15),
                        color=col_map[direction],
                        opacity=0.6,
                    ),
                    name=f"Created ({direction})",
                    hovertemplate=(
                        "Time: %{x}<br>Price: %{y:.2f}<br>"
                        "Vol: %{customdata:.4f}<extra></extra>"
                    ),
                    customdata=subset["volume"],
                )
            )

    if not deleted.empty:
        for direction in ["bid", "ask"]:
            subset = deleted[deleted["direction"] == direction]
            if subset.empty:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=subset["timestamp"],
                    y=subset["price"],
                    mode="markers",
                    marker=dict(
                        size=np.clip(subset["volume"] * 20, 3, 15),
                        color=col_map[direction],
                        opacity=0.3,
                        symbol="x",
                    ),
                    name=f"Deleted ({direction})",
                    hovertemplate=(
                        "Time: %{x}<br>Price: %{y:.2f}<br>"
                        "Vol: %{customdata:.4f}<extra></extra>"
                    ),
                    customdata=subset["volume"],
                )
            )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Limit Price")
    return fig


def plotly_volume_map(data: dict) -> Any:
    """Render a volume map of flashed limit orders."""
    go = _import_plotly()
    events = data["events"]
    log_scale = data["log_scale"]
    col_map = {"bid": _BID_COLOR, "ask": _ASK_COLOR}

    fig = _base_figure(go, title="Volume Map of Flashed Limit Orders")

    for direction in ["bid", "ask"]:
        subset = events[events["direction"] == direction]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=subset["timestamp"],
                y=subset["volume"],
                mode="markers",
                marker=dict(size=4, color=col_map[direction], opacity=0.6),
                name=direction.capitalize(),
                hovertemplate="Time: %{x}<br>Volume: %{y:.4f}<extra></extra>",
            )
        )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Volume", type="log" if log_scale else "linear")
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    """``"#4477dd"`` -> ``"rgba(68,119,221,0.15)"``."""
    r, g, b = (int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _plotly_book_bars(data: dict, *, per_order: bool) -> Any:
    """Horizontal book ladder: price on y, size on x, bids below / asks above.

    L2 draws one bar per price level; L3 segments each level into its individual
    orders with white separators, so equal-total levels with different
    composition read differently.
    """
    go = _import_plotly()
    fig = _base_figure(go, title=data["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC"))

    # White per-order separators (dark ones vanished against the fill).
    line = dict(color="white", width=1.0) if per_order else dict(width=0)
    for side, color, label in (
        (data["bids"], _BID_COLOR, "Bid"),
        (data["asks"], _ASK_COLOR, "Ask"),
    ):
        if side.empty:
            continue
        fig.add_trace(
            go.Bar(
                y=side["price"],
                x=side["seg_hi"] - side["seg_lo"],
                base=side["seg_lo"],
                orientation="h",
                marker=dict(color=color, line=line),
                name=label,
                hovertemplate="Price: %{y:.2f}<br>Size: %{x:.4f}<extra></extra>",
            )
        )

    mid = book_mid(data["bids"], data["asks"])
    if mid is not None:
        fig.add_hline(y=mid, line_dash="dash", line_color="#444444", line_width=1)

    if data["show_quantiles"]:
        for y_val in (*data["bid_quantiles"], *data["ask_quantiles"]):
            fig.add_hline(y=y_val, line_dash="dot", line_color="#888888", line_width=1)

    fig.update_layout(barmode="overlay")
    fig.update_xaxes(
        title_text="Size (per order)" if per_order else "Size", rangemode="tozero"
    )
    fig.update_yaxes(title_text="Price")
    return fig


def _plotly_depth_curve(data: dict, *, per_order: bool) -> Any:
    """Cumulative-depth curve: stepped per level (L2) or per order (L3)."""
    go = _import_plotly()
    fig = _base_figure(go, title=data["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC"))

    mode = "lines+markers" if per_order else "lines"
    for side, color, label in (
        (data["bids"], _BID_COLOR, "Bid"),
        (data["asks"], _ASK_COLOR, "Ask"),
    ):
        if side.empty:
            continue
        s = side.sort_values("price")
        fig.add_trace(
            go.Scatter(
                x=s["price"],
                y=s["liquidity"],
                mode=mode,
                line=dict(shape="vh", color=color, width=2),
                marker=dict(size=6, color=color),
                name=label,
                fill="tozeroy",
                fillcolor=_rgba(color, 0.15),
            )
        )

    fig.update_xaxes(title_text="Price")
    fig.update_yaxes(title_text="Cumulative liquidity")
    return fig


def plotly_book_snapshot_aggregate(data: dict) -> Any:
    """L2 (MBP) book snapshot: aggregate size per price level."""
    return _plotly_book_bars(data, per_order=False)


def plotly_book_snapshot_per_order(data: dict) -> Any:
    """L3 (MBO) book snapshot: each order a stacked segment within its level."""
    return _plotly_book_bars(data, per_order=True)


def plotly_depth_chart_aggregate(data: dict) -> Any:
    """L2 (MBP) depth chart: cumulative liquidity stepped per price level."""
    return _plotly_depth_curve(data, per_order=False)


def plotly_depth_chart_per_order(data: dict) -> Any:
    """L3 (MBO) depth chart: cumulative liquidity stepped per individual order."""
    return _plotly_depth_curve(data, per_order=True)


# Log-decade ticks shared by both cancel panels (axes carry log10 values, so
# ticks sit at integer positions and are relabeled into human units).
_AGE_TICKVALS = list(range(-3, 3))
_AGE_TICKTEXT = ["1ms", "10ms", "100ms", "1s", "10s", "100s"]
_DIST_TICKVALS = list(range(-2, 3))
_DIST_TICKTEXT = ["0.01", "0.1", "1", "10", "100"]


def _log_density_grid(side: Any, xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    """Counts per (log10 age, log10 distance) bin, zeros masked to NaN.

    Binning server-side and shipping only the grid keeps the exported HTML
    small -- a raw ``Histogram2d``/``Scattergl`` would carry every cancelled
    order (megabytes); this carries one number per cell.
    """
    counts, _, _ = np.histogram2d(
        np.log10(side["age_s"]),
        np.log10(side["distance_from_touch"]),
        bins=[xedges, yedges],
    )
    return np.where(counts == 0, np.nan, counts).T  # (y, x) for Heatmap z


def plotly_cancellations_per_order(data: dict) -> Any:
    """L3 (MBO) cancellations: per-side log-log *density* of age x distance.

    A pre-binned ``go.Heatmap`` per side (small multiples, bid | ask) replaces
    the old raw ``Scattergl`` point cloud: it reveals the latent populations on
    log-log axes and keeps the exported HTML small (one grid of counts, not one
    marker per cancelled order).
    """
    go = _import_plotly()
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=("Cancelled bid", "Cancelled ask"),
    )
    fig.update_layout(**_BASE_LAYOUT)
    fig.update_layout(
        title=dict(text="Cancelled orders by age and distance from touch", x=0.5),
        coloraxis=dict(colorscale="Blues", colorbar=dict(title="Orders<br>per bin")),
        hovermode="closest",
    )

    sides = [(1, data["bids"]), (2, data["asks"])]
    populated = [s for _, s in sides if not s.empty]
    if populated:
        # Shared log-space bin edges so the two panels are directly comparable.
        all_x = np.log10(np.concatenate([s["age_s"].to_numpy() for s in populated]))
        all_y = np.log10(
            np.concatenate([s["distance_from_touch"].to_numpy() for s in populated])
        )
        xedges = np.linspace(all_x.min(), all_x.max(), 31)
        yedges = np.linspace(all_y.min(), all_y.max(), 31)
        xcent = (xedges[:-1] + xedges[1:]) / 2
        ycent = (yedges[:-1] + yedges[1:]) / 2

        for col, side in sides:
            if side.empty:
                continue
            fig.add_trace(
                go.Heatmap(
                    z=_log_density_grid(side, xedges, yedges),
                    x=xcent,
                    y=ycent,
                    coloraxis="coloraxis",
                    hovertemplate=(
                        "log10 age: %{x:.1f}<br>log10 dist: %{y:.1f}"
                        "<br>orders: %{z}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

    for col in (1, 2):
        fig.update_xaxes(
            title_text="Age at cancel",
            tickvals=_AGE_TICKVALS,
            ticktext=_AGE_TICKTEXT,
            row=1,
            col=col,
        )
    fig.update_yaxes(
        title_text="Distance from touch (bps)",
        tickvals=_DIST_TICKVALS,
        ticktext=_DIST_TICKTEXT,
        row=1,
        col=1,
    )
    return fig


def _segments_xy(start: Any, end: Any, y: Any) -> tuple[Any, Any]:
    """Interleave per-order spans into one ``None``-gapped line for Scattergl.

    Each order draws a horizontal segment ``(start, y) -> (end, y)``; a ``None``
    gap separates consecutive orders so the whole fate frame renders as a single
    WebGL trace (one trace per fate, like the L2 volume map this pairs with).
    """
    n = len(y)
    xs = np.empty(n * 3, dtype=object)
    xs[0::3] = start.to_numpy()
    xs[1::3] = end.to_numpy()
    xs[2::3] = None
    ys = np.empty(n * 3, dtype=object)
    ys[0::3] = y.to_numpy()
    ys[1::3] = y.to_numpy()
    ys[2::3] = None
    return xs, ys


def _vstem_xy(x: Any, y0: Any, y1: Any) -> tuple[Any, Any]:
    """Interleave vertical lollipop stems into one ``None``-gapped Scattergl line.

    Each trade draws a stem ``(x, y0) -> (x, y1)`` (mid -> price); a ``None`` gap
    separates consecutive stems so a whole side renders as a single WebGL trace.
    """
    n = len(x)
    xv = np.asarray(x)
    xs = np.empty(n * 3, dtype=object)
    xs[0::3] = xv
    xs[1::3] = xv
    xs[2::3] = None
    ys = np.empty(n * 3, dtype=object)
    ys[0::3] = np.asarray(y0)
    ys[1::3] = np.asarray(y1)
    ys[2::3] = None
    return xs, ys


def _plotly_lollipops(fig: Any, go: Any, side: Any, color: str, label: str) -> None:
    """Stems (mid -> price) plus volume-sized markers for one tape side."""
    xs, ys = _vstem_xy(side["timestamp"], side["mid"], side["price"])
    fig.add_trace(
        go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=1),
            opacity=0.5,
            name=label,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=side["timestamp"],
            y=side["price"],
            mode="markers",
            marker=dict(
                size=mpl_marker_area_to_plotly_size(side["marker_area"].to_numpy()),
                color=color,
                opacity=0.9,
                line=dict(width=0),
            ),
            name=label,
            hovertemplate="Time: %{x}<br>Price: %{y:.2f}<extra></extra>",
        )
    )


def _apply_padded_y_range(fig: Any, y_range: tuple[float, float] | None) -> None:
    """Set a 4%-padded y-range so spike prints are never cut at the axis edge."""
    if y_range is None:
        return
    lo, hi = y_range
    pad = (hi - lo) * 0.04 or 1.0
    fig.update_yaxes(range=[lo - pad, hi + pad])


def plotly_order_activity_per_order(data: dict) -> Any:
    """L3 (MBO) order activity: each order one lifecycle bar, coloured by fate.

    Uses ``Scattergl`` (WebGL) like the L2 ``plotly_event_map`` it pairs with: a
    per-order face draws one segment per limit order, so the line cloud is large
    and the SVG ``Scatter`` path does not scale.
    """
    go = _import_plotly()
    fig = _base_figure(go, title="Order lifecycles (place → outcome)")
    for side, color, label in (
        (data["filled"], _FILLED_COLOR, "filled"),
        (data["cancelled"], _CANCELLED_COLOR, "cancelled"),
        (data["resting"], _PARTIAL_COLOR, "still resting"),
    ):
        if side.empty:
            continue
        xs, ys = _segments_xy(side["start_ts"], side["end_ts"], side["price"])
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=1.5),
                opacity=0.6,
                name=label,
                hoverinfo="skip",
            )
        )
    shown_of = data.get("shown_of")
    if shown_of is not None:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.99,
            y=0.01,
            text=f"showing {shown_of[0]:,} of {shown_of[1]:,} orders",
            showarrow=False,
            font=dict(size=10, color="#555555"),
        )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Limit Price")
    y_range = data.get("y_range")
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


def plotly_liquidity_at_touch(data: dict) -> Any:
    """L2 (MBP) liquidity at the touch: best bid/ask resting size over time."""
    go = _import_plotly()
    fig = _base_figure(go, title="Liquidity at the touch")
    ts = data["timestamp"]
    for vol, color, label in (
        (data["bid_vol"], _BID_COLOR, "Best bid size"),
        (data["ask_vol"], _ASK_COLOR, "Best ask size"),
    ):
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=vol,
                mode="lines",
                line=dict(color=color, width=1.2, shape="hv"),
                name=label,
            )
        )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Size at touch")
    return fig


def plotly_order_outcome_per_order(data: dict) -> Any:
    """L3 (MBO) order outcome: each order as placement distance x size, by fate.

    Uses ``Scattergl`` (WebGL) because a per-order face draws one marker per limit
    order, so the point cloud is large and the SVG ``Scatter`` path does not scale.
    """
    go = _import_plotly()
    fig = _base_figure(go, title="Order outcome by placement distance and size")
    # Cancelled first (underneath) and faded; see the matplotlib backend.
    # A distance-binned fate variant is roadmap §3.8 (docs/plans/).
    for frame, color, label, pt_opacity in (
        (data["cancelled"], _CANCELLED_COLOR, "cancelled", 0.18),
        (data["partial"], _PARTIAL_COLOR, "partial", 0.6),
        (data["filled"], _FILLED_COLOR, "filled", 0.85),
    ):
        if frame.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=frame["distance_bps"],
                y=frame["placed"],
                mode="markers",
                marker=dict(
                    size=mpl_marker_area_to_plotly_size(
                        frame["marker_area"].to_numpy()
                    ),
                    color=color,
                    opacity=pt_opacity,
                    line=dict(width=0),
                ),
                name=label,
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="#888888", line_width=1)
    fig.update_xaxes(title_text="Placement distance from touch (bps)")
    fig.update_yaxes(title_text="Order size")
    return fig


def plotly_trade_tape_per_order(data: dict) -> Any:
    """L3 (MBO) signed-lollipop trade tape with maker resting spans.

    Same signed lollipops as the L2 tape (stem mid -> price, marker sized by
    volume, coloured by aggressor side), plus the L3 differentiator: a faint
    span from each consumed maker order's creation to its fill.  Trade prices
    are never clipped; above the density threshold the lollipops are per-second
    VWAPs (roadmap §3.4).  All clouds use ``Scattergl`` (WebGL) so they scale.
    """
    go = _import_plotly()
    fig = _base_figure(go, title="Trade tape with maker order lifecycles")

    dense = data.get("dense", False)
    span_opacity = 0.12 if dense else 0.35
    # Maker resting spans (horizontal), faint underneath the lollipops.
    for side, color in (
        (data["buys"], _BUY_COLOR),
        (data["sells"], _SELL_COLOR),
    ):
        if side.empty:
            continue
        xs, ys = _segments_xy(side["created_ts"], side["timestamp"], side["price"])
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=1.0),
                opacity=span_opacity,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    mid_line = data.get("mid_line")
    if mid_line is not None and not mid_line.empty:
        fig.add_trace(
            go.Scattergl(
                x=mid_line["timestamp"],
                y=mid_line["mid"],
                mode="lines",
                line=dict(color="#888888", width=1),
                opacity=0.8,
                name="mid",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    suffix = ", per-s VWAP" if dense else ""
    for side, color, label in (
        (data["lolli_buys"], _BUY_COLOR, f"buy (lifts ask){suffix}"),
        (data["lolli_sells"], _SELL_COLOR, f"sell (hits bid){suffix}"),
    ):
        if side.empty:
            continue
        _plotly_lollipops(fig, go, side, color, label)

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Execution Price")
    _apply_padded_y_range(fig, data.get("y_range"))
    return fig


def plotly_volume_percentiles(data: dict) -> Any:
    """Render volume-percentile stacked area chart."""
    go = _import_plotly()
    asks_cumsum = data["asks_cumsum"]
    bids_cumsum_neg = data["bids_cumsum_neg"]
    asks_cols = data["asks_cols"]
    bids_cols = data["bids_cols"]
    all_cols = data["all_cols"]
    colors_dict = data["colors_dict"]
    legend_names = data["legend_names"]
    max_ask = data["max_ask"]
    max_bid = data["max_bid"]
    volume_scale = data["volume_scale"]
    side_line = data["side_line"]

    fig = _base_figure(go, title="Volume Percentiles")

    # Convert matplotlib RGBA tuples to plotly rgb strings
    def _to_rgb(c: Any) -> str:
        if isinstance(c, tuple):
            return f"rgba({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)},{c[3]:.2f})"
        return str(c)

    label_map = dict(zip(all_cols, legend_names))

    # Asks (positive side) — draw from outermost to innermost for stacking
    for col in asks_cols:
        current = asks_cumsum[col].values
        fig.add_trace(
            go.Scatter(
                x=asks_cumsum.index,
                y=current,
                mode="lines",
                line=dict(width=0.5, color="black"),
                fill="tonexty" if col != asks_cols[0] else "tozeroy",
                fillcolor=_to_rgb(colors_dict[col]),
                name=label_map.get(col, col),
                showlegend=True,
                hovertemplate=f"{label_map.get(col, col)}: %{{y:.4f}}<extra></extra>",
            )
        )

    # Bids (negative side)
    for col in bids_cols:
        current = bids_cumsum_neg[col].values
        fig.add_trace(
            go.Scatter(
                x=bids_cumsum_neg.index,
                y=current,
                mode="lines",
                line=dict(width=0.5, color="black"),
                fill="tonexty" if col != bids_cols[0] else "tozeroy",
                fillcolor=_to_rgb(colors_dict[col]),
                name=label_map.get(col, col),
                showlegend=True,
                hovertemplate=f"{label_map.get(col, col)}: %{{y:.4f}}<extra></extra>",
            )
        )

    if side_line:
        fig.add_hline(y=0, line_color="#444444", line_width=0.5)

    y_range = volume_scale * max(max_ask, max_bid)
    fig.update_yaxes(range=[-y_range, y_range], title_text="Liquidity")
    fig.update_xaxes(title_text="Time")
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font_size=10,
        )
    )
    return fig


def plotly_events_histogram(data: dict) -> Any:
    """Render an events price/volume histogram."""
    go = _import_plotly()
    events = data["events"]
    val = data["val"]
    bw = data["bw"]

    fig = _base_figure(go, title=f"Events {val} distribution")

    for direction, color in [("bid", _BID_COLOR), ("ask", _ASK_COLOR)]:
        subset = events[events["direction"] == direction]
        if subset.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=subset[val],
                name=direction.capitalize(),
                marker_color=color,
                opacity=0.7,
                xbins=dict(size=bw) if bw is not None else None,
            )
        )

    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text=val.capitalize())
    fig.update_yaxes(title_text="Count")
    return fig


def plotly_vpin(data: dict) -> Any:
    """Render VPIN time series."""
    go = _import_plotly()
    vpin_df = data["vpin_df"]
    threshold = data["threshold"]

    fig = _base_figure(go, title="VPIN — Probability of Informed Trading")

    fig.add_trace(
        go.Bar(
            x=vpin_df["timestamp_end"],
            y=vpin_df["vpin"],
            marker_color="#5dade2",
            opacity=0.4,
            name="Per-bucket VPIN",
            hovertemplate="Time: %{x}<br>VPIN: %{y:.3f}<extra></extra>",
        )
    )

    if "vpin_avg" in vpin_df.columns:
        fig.add_trace(
            go.Scatter(
                x=vpin_df["timestamp_end"],
                y=vpin_df["vpin_avg"],
                mode="lines",
                line=dict(color="#e74c3c", width=2.5),
                name="VPIN (rolling avg)",
            )
        )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#f39c12",
        line_width=2,
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top left",
    )

    fig.update_yaxes(range=[0, 1.05], title_text="VPIN")
    fig.update_xaxes(title_text="Time")
    return fig


def plotly_order_flow_imbalance(data: dict) -> Any:
    """Render order flow imbalance bar chart."""
    go = _import_plotly()
    ofi_df = data["ofi_df"]
    trades = data["trades"]
    colors = data["colors"]

    fig = _base_figure(go, title="Order Flow Imbalance")

    fig.add_trace(
        go.Bar(
            x=ofi_df["timestamp"],
            y=ofi_df["ofi"],
            marker_color=colors,
            opacity=0.7,
            name="OFI",
            hovertemplate="Time: %{x}<br>OFI: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_color="#444444", line_width=0.5, opacity=0.6)
    fig.update_yaxes(range=[-1.05, 1.05], title_text="OFI")
    fig.update_xaxes(title_text="Time")

    if trades is not None and "price" in trades.columns:
        fig.add_trace(
            go.Scatter(
                x=trades["timestamp"],
                y=trades["price"],
                mode="lines",
                line=dict(color="#f1c40f", width=1.5),
                name="Price",
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title=dict(text="Price", font=dict(color="#f1c40f")),
                overlaying="y",
                side="right",
                tickfont=dict(color="#f1c40f"),
            ),
        )

    return fig


def plotly_kyle_lambda(data: dict) -> Any:
    """Render Kyle's Lambda regression scatter."""
    go = _import_plotly()
    reg_df = data["reg_df"]
    lambda_ = data["lambda_"]
    r_squared = data["r_squared"]
    t_stat = data["t_stat"]

    title = "Kyle's Lambda — Price Impact Regression"
    if not np.isnan(r_squared):
        title += f"<br><sub>R² = {r_squared:.3f}, t = {t_stat:.2f}</sub>"

    fig = _base_figure(go, title=title)

    fig.add_trace(
        go.Scatter(
            x=reg_df["signed_volume"],
            y=reg_df["delta_price"],
            mode="markers",
            marker=dict(
                size=7,
                color="#5dade2",
                opacity=0.6,
                line=dict(width=0.5, color="white"),
            ),
            name="Observations",
            hovertemplate=(
                "Signed Volume: %{x:.4f}<br>ΔPrice: %{y:.6f}<extra></extra>"
            ),
        )
    )

    if not np.isnan(lambda_):
        x_range = np.linspace(
            reg_df["signed_volume"].min(),
            reg_df["signed_volume"].max(),
            100,
        )
        intercept = (
            reg_df["delta_price"].mean() - lambda_ * reg_df["signed_volume"].mean()
        )
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=intercept + lambda_ * x_range,
                mode="lines",
                line=dict(color="#e74c3c", width=2.5),
                name=f"λ = {lambda_:.6f}",
            )
        )

    fig.add_hline(y=0, line_color="#444444", line_width=0.3, opacity=0.5)
    fig.add_vline(x=0, line_color="#444444", line_width=0.3, opacity=0.5)
    fig.update_xaxes(title_text="Signed Order Flow (net volume)")
    fig.update_yaxes(title_text="ΔPrice")
    return fig


# ---------------------------------------------------------------------------
# LOBSTER-enriched plots
# ---------------------------------------------------------------------------


def plotly_hidden_executions(data: dict) -> Any:
    """Render hidden execution volume overlaid on the trade price."""
    go = _import_plotly()
    trades = data["trades"]
    hidden = data["hidden"]
    has_hidden = data["has_hidden"]

    title = (
        "Hidden Order Executions"
        if has_hidden
        else "Hidden Order Executions (no hidden execution data)"
    )
    fig = _base_figure(go, title=title)

    if has_hidden and not hidden.empty:
        # Hue by aggressor side instead of a Reds-by-volume ramp: size already
        # encodes volume (bounded via ``normalized_marker_areas``), so colouring
        # by volume too washed typical prints out to near-white (roadmap §3.6).
        sizes = mpl_marker_area_to_plotly_size(data["marker_area"])
        direction = data.get("direction")
        col_map = {"bid": _BID_COLOR, "ask": _ASK_COLOR}
        if direction is not None:
            for d in ("bid", "ask"):
                mask = np.asarray(direction == d)
                if not mask.any():
                    continue
                subset = hidden[mask]
                fig.add_trace(
                    go.Scatter(
                        x=subset["timestamp"],
                        y=subset["price"],
                        mode="markers",
                        customdata=subset["volume"],
                        marker=dict(
                            size=sizes[mask],
                            color=col_map[d],
                            opacity=0.55,
                            line=dict(width=0.4, color="white"),
                        ),
                        name=f"Hidden ({d})",
                        hovertemplate=(
                            "Time: %{x}<br>Price: %{y:.2f}<br>"
                            "Volume: %{customdata}<extra></extra>"
                        ),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=hidden["timestamp"],
                    y=hidden["price"],
                    mode="markers",
                    customdata=hidden["volume"],
                    marker=dict(
                        size=sizes,
                        color="#7f8c8d",
                        opacity=0.55,
                        line=dict(width=0.4, color="white"),
                    ),
                    name="Hidden executions",
                    hovertemplate=(
                        "Time: %{x}<br>Price: %{y:.2f}<br>"
                        "Volume: %{customdata}<extra></extra>"
                    ),
                )
            )

    # Trade price drawn after the markers so it renders on top and stays legible.
    if not trades.empty:
        fig.add_trace(
            go.Scatter(
                x=trades["timestamp"],
                y=trades["price"],
                mode="lines",
                line=dict(color="#222222", width=1, shape="hv"),
                name="Trade price",
                opacity=0.9,
            )
        )

    if not has_hidden:
        fig.add_annotation(
            text="No hidden execution events (raw_event_type == 5)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#888"),
        )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Price")
    y_range = data.get("y_range")
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


def plotly_trading_halts(data: dict) -> Any:
    """Render trade price with shaded halt periods."""
    go = _import_plotly()
    trades = data["trades"]
    halt_periods = data["halt_periods"]
    has_halts = data["has_halts"]

    title = "Trading Halts" if has_halts else "Trading Halts (no halt data)"
    fig = _base_figure(go, title=title)

    if not trades.empty:
        fig.add_trace(
            go.Scatter(
                x=trades["timestamp"],
                y=trades["price"],
                mode="lines",
                line=dict(color="#5dade2", width=1, shape="hv"),
                name="Trade price",
                opacity=0.8,
            )
        )

    if has_halts and halt_periods:
        for i, (h_start, h_end) in enumerate(halt_periods):
            fig.add_vrect(
                x0=h_start,
                x1=h_end,
                fillcolor="#e74c3c",
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text="Halt" if i == 0 else None,
            )
    elif not has_halts:
        fig.add_annotation(
            text="No trading halt events (raw_event_type == 7)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#888"),
        )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Price")
    return fig


# ---------------------------------------------------------------------------
# Renderer self-registration
# ---------------------------------------------------------------------------
# Imported here (not at module top) so RENDERERS -- defined in the package
# __init__ -- already exists when this (lazily imported) module is loaded.
from ob_analytics.visualization import RENDERERS, Level  # noqa: E402

# (concept, level, renderer); mirrors the matplotlib backend's coordinates so
# every concept has both faces.  Analytics are level-less (``None``).
_L2 = Level.L2
_L3 = Level.L3
for _concept, _level, _fn in [
    ("time_series", _L2, plotly_time_series),
    ("trade_tape", _L2, plotly_trades),
    ("trade_tape", _L3, plotly_trade_tape_per_order),
    ("depth_heatmap", _L2, plotly_price_levels),
    ("order_activity", _L2, plotly_event_map),
    ("order_activity", _L3, plotly_order_activity_per_order),
    ("order_outcome", _L3, plotly_order_outcome_per_order),
    ("liquidity_at_touch", _L2, plotly_liquidity_at_touch),
    ("cancellations", _L2, plotly_volume_map),
    ("cancellations", _L3, plotly_cancellations_per_order),
    ("book_snapshot", _L2, plotly_book_snapshot_aggregate),
    ("book_snapshot", _L3, plotly_book_snapshot_per_order),
    ("depth_chart", _L2, plotly_depth_chart_aggregate),
    ("depth_chart", _L3, plotly_depth_chart_per_order),
    ("volume_percentiles", _L2, plotly_volume_percentiles),
    ("events_histogram", _L2, plotly_events_histogram),
    ("hidden_executions", _L2, plotly_hidden_executions),
    ("vpin", None, plotly_vpin),
    ("order_flow_imbalance", None, plotly_order_flow_imbalance),
    ("kyle_lambda", None, plotly_kyle_lambda),
    ("trading_halts", None, plotly_trading_halts),
]:
    RENDERERS.register((_concept, _level, "plotly"), _fn)
