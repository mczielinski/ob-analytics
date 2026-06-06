"""Plotly interactive rendering backend for ob-analytics.

Each ``plotly_*()`` function takes a prepared data dict (from
:mod:`~ob_analytics.visualization._data`) and returns a
:class:`plotly.graph_objects.Figure` with interactive zoom, pan, and hover.

Install via ``pip install ob-analytics[interactive]``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ob_analytics.exceptions import ConfigError
from ob_analytics.visualization._data import mpl_marker_area_to_plotly_size


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

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#2d2d2d",
    plot_bgcolor="#1e1e1e",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified",
)


def _base_figure(go: Any, title: str = "", **kwargs: Any) -> Any:
    """Create a Plotly figure with the dark ob-analytics theme."""
    layout = {**_DARK_LAYOUT, "title": dict(text=title, x=0.5)}
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
    """Render a trade-price step plot."""
    go = _import_plotly()
    filtered = data["filtered_trades"]
    fig = _base_figure(go, title="Trade Prices")
    fig.add_trace(
        go.Scatter(
            x=filtered["timestamp"],
            y=filtered["price"],
            mode="lines",
            line=dict(shape="hv", width=2, color="#5dade2"),
            name="Price",
            hovertemplate="Time: %{x}<br>Price: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Limit Price")
    return fig


def plotly_price_levels(data: dict) -> Any:
    """Render the price-level depth heatmap using Scattergl."""
    go = _import_plotly()
    depth = data["depth"]
    spread = data["spread"]
    trades = data["trades"]
    show_mp = data["show_mp"]

    fig = _base_figure(go, title="Price Levels Over Time")

    if not depth.empty:
        vol = depth["volume"].copy()
        vol = vol.fillna(0)
        vol_max = vol.max() if vol.max() > 0 else 1

        fig.add_trace(
            go.Scattergl(
                x=depth["timestamp"],
                y=depth["price"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=vol,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=vol_max,
                    colorbar=dict(title="Volume", x=1.02, len=0.75),
                    opacity=np.where(vol > 0, 0.8, 0.1),
                ),
                hovertemplate=(
                    "Time: %{x}<br>Price: %{y:.2f}<br>"
                    "Volume: %{marker.color:.4f}<extra></extra>"
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
                    line=dict(color="white", width=1.5, shape="hv"),
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
                    line=dict(color="#ff4444", width=1.2, dash="dot"),
                    name="Best Ask",
                )
            )
        if "best_bid_price" in spread:
            fig.add_trace(
                go.Scatter(
                    x=spread["timestamp"],
                    y=spread["best_bid_price"],
                    mode="lines",
                    line=dict(color="#44ff44", width=1.2, dash="dot"),
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
                        color="#ff4444",
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
                        color="#44ff44",
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
    data["events"]
    created = data["created"]
    deleted = data["deleted"]

    fig = _base_figure(go, title="Limit Order Event Map")

    col_map = {"bid": "#4444ff", "ask": "#ff4444"}

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
    col_map = {"bid": "#4444ff", "ask": "#ff4444"}

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


# Order-book snapshot palette, shared by the book_snapshot + depth_chart faces
# (kept identical to the matplotlib backend for cross-backend parity).
_BID_COLOR = "#4477dd"
_ASK_COLOR = "#dd4444"


def _rgba(hex_color: str, alpha: float) -> str:
    """``"#4477dd"`` -> ``"rgba(68,119,221,0.15)"``."""
    r, g, b = (int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _plotly_book_bars(data: dict, *, per_order: bool) -> Any:
    """Book-snapshot bars: one bar per level (L2) or stacked per order (L3)."""
    go = _import_plotly()
    fig = _base_figure(go, title=data["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC"))

    line = dict(color="#1e1e1e", width=0.6) if per_order else dict(width=0)
    for side, color, label in (
        (data["bids"], _BID_COLOR, "Bid"),
        (data["asks"], _ASK_COLOR, "Ask"),
    ):
        if side.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=side["price"],
                y=side["seg_hi"] - side["seg_lo"],
                base=side["seg_lo"],
                marker=dict(color=color, line=line),
                name=label,
                hovertemplate="Price: %{x:.2f}<br>Size: %{y:.4f}<extra></extra>",
            )
        )

    if data["show_quantiles"]:
        for x_val in data["bid_quantiles"]:
            fig.add_vline(x=x_val, line_dash="dash", line_color="#888888", line_width=1)
        for x_val in data["ask_quantiles"]:
            fig.add_vline(x=x_val, line_dash="dash", line_color="#888888", line_width=1)

    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Price")
    fig.update_yaxes(title_text="Size")
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


def plotly_cancellations_per_order(data: dict) -> Any:
    """L3 (MBO) cancellations: each cancelled order as an age x distance point.

    Uses ``Scattergl`` (WebGL) like the L2 ``plotly_volume_map`` it pairs with:
    a per-order face draws one marker per cancelled order, so the point cloud is
    large and the SVG ``Scatter`` path does not scale.
    """
    go = _import_plotly()
    fig = _base_figure(go, title="Cancelled orders by age and distance from touch")
    for side, color, label in (
        (data["bids"], _BID_COLOR, "Bid"),
        (data["asks"], _ASK_COLOR, "Ask"),
    ):
        if side.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=side["age_s"],
                y=side["distance_bps"],
                mode="markers",
                marker=dict(
                    size=mpl_marker_area_to_plotly_size(side["marker_area"].to_numpy()),
                    color=color,
                    opacity=0.4,
                ),
                name=label,
                hovertemplate=(
                    "Age: %{x:.2f}s<br>Distance: %{y:.2f} bps<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="#888888", line_width=1)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="Order age (s)")
    fig.update_yaxes(title_text="Placement distance from touch (bps)")
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
    np.zeros(len(asks_cumsum))
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
        fig.add_hline(y=0, line_color="white", line_width=0.5)

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

    for direction, color in [("bid", "#4444ff"), ("ask", "#ff4444")]:
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

    fig.add_hline(y=0, line_color="white", line_width=0.5, opacity=0.5)
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

    fig.add_hline(y=0, line_color="white", line_width=0.3, opacity=0.3)
    fig.add_vline(x=0, line_color="white", line_width=0.3, opacity=0.3)
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

    if not trades.empty:
        fig.add_trace(
            go.Scatter(
                x=trades["timestamp"],
                y=trades["price"],
                mode="lines",
                line=dict(color="#5dade2", width=1, shape="hv"),
                name="Trade price",
                opacity=0.7,
            )
        )

    if has_hidden and not hidden.empty:
        vol = hidden["volume"]
        vol_max = float(vol.max()) if vol.max() > 0 else 1.0
        marker_area = data["marker_area"]
        fig.add_trace(
            go.Scatter(
                x=hidden["timestamp"],
                y=hidden["price"],
                mode="markers",
                customdata=vol,
                marker=dict(
                    size=mpl_marker_area_to_plotly_size(marker_area),
                    color=vol,
                    colorscale="Reds",
                    cmin=0,
                    cmax=vol_max,
                    opacity=0.6,
                    line=dict(width=0.3, color="white"),
                ),
                name="Hidden executions",
                hovertemplate=(
                    "Time: %{x}<br>Price: %{y:.2f}<br>"
                    "Volume: %{customdata}<extra></extra>"
                ),
            )
        )
    elif not has_hidden:
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
    ("depth_heatmap", _L2, plotly_price_levels),
    ("order_activity", _L2, plotly_event_map),
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
