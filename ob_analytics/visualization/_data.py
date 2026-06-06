"""Backend-agnostic data preparation for visualization.

Each ``prepare_*()`` function extracts, filters, and transforms raw
DataFrames into plain dicts consumable by **any** rendering backend
(matplotlib, Plotly, Bokeh, …).  No rendering imports live here.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ob_analytics.depth import filter_depth


def _sanitize_spread(spread: pd.DataFrame) -> pd.DataFrame:
    """Drop spread rows with non-physical bid/ask (e.g. LOBSTER book warmup)."""
    if spread.empty:
        return spread
    mask = pd.Series(True, index=spread.index)
    if "best_bid_price" in spread.columns:
        mask &= spread["best_bid_price"] > 0
    if "best_ask_price" in spread.columns:
        mask &= spread["best_ask_price"] > 0
    if "best_bid_price" in spread.columns and "best_ask_price" in spread.columns:
        mask &= spread["best_ask_price"] >= spread["best_bid_price"]
    return spread[mask]


def price_y_range(*series: pd.Series | None) -> tuple[float, float] | None:
    """Shared (ymin, ymax) from one or more price columns."""
    parts: list[pd.Series] = []
    for s in series:
        if s is not None and not s.empty:
            parts.append(s.dropna())
    if not parts:
        return None
    prices = pd.concat(parts)
    return float(prices.min()), float(prices.max())


def volume_marker_areas(
    volume: pd.Series | np.ndarray, *, scale: float = 2.0
) -> np.ndarray:
    """Matplotlib scatter ``s`` values proportional to volume."""
    return np.asarray(volume, dtype=float) * scale


def mpl_marker_area_to_plotly_size(area: np.ndarray) -> np.ndarray:
    """Convert matplotlib scatter areas (pt²) to plotly marker diameters (px)."""
    return np.sqrt(np.maximum(area, 0.0)) * 0.8


# The volume-percentile chart colours a fixed 20-step rainbow gradient,
# shared verbatim by every rendering backend.  It was historically built
# at call time with ``matplotlib.colors.LinearSegmentedColormap`` from the
# ten anchor colours below; that pulled a rendering dependency into this
# backend-agnostic data layer.  The exact RGBA values are deterministic, so
# they are precomputed here once (round-trip-exact float literals) — keeping
# ``colors_dict`` byte-identical while removing the matplotlib import.
#
# To regenerate (e.g. to tweak the anchors), run:
#     from matplotlib.colors import LinearSegmentedColormap
#     anchors = ["#f92b20", "#fe701b", "#facd1f", "#d6fd1c", "#65fe1b",
#                "#1bfe42", "#1cfdb4", "#1fb9fa", "#1e71fb", "#261cfd"]
#     cmap = LinearSegmentedColormap.from_list("custom_cmap", anchors, N=20)
#     [cmap(i / 19) for i in range(20)]
_VOLUME_PERCENTILE_PALETTE: tuple[tuple[float, float, float, float], ...] = (
    (0.9764705882352941, 0.16862745098039217, 0.12549019607843137, 1.0),
    (0.9857585139318885, 0.29680082559339527, 0.11620227038183695, 1.0),
    (0.9950464396284829, 0.4249742002063983, 0.10691434468524252, 1.0),
    (0.9894736842105263, 0.5927760577915376, 0.11248710010319918, 1.0),
    (0.9820433436532507, 0.765531475748194, 0.11991744066047472, 1.0),
    (0.9283797729618163, 0.8732714138286894, 0.11723426212590299, 1.0),
    (0.8615067079463364, 0.9624355005159959, 0.11166150670794633, 1.0),
    (0.6992776057791538, 0.9933952528379774, 0.1085655314757482, 1.0),
    (0.48937048503611974, 0.9952528379772962, 0.10670794633642931, 1.0),
    (0.31971104231166153, 0.996078431372549, 0.1461300309597523, 1.0),
    (0.18224974200206398, 0.996078431372549, 0.21857585139318886, 1.0),
    (0.10670794633642931, 0.9952528379772962, 0.3529411764705883, 1.0),
    (0.1085655314757482, 0.9933952528379774, 0.5647058823529414, 1.0),
    (0.11166150670794633, 0.9500515995872034, 0.7492260061919503, 1.0),
    (0.117234262125903, 0.8237358101135188, 0.8792569659442727, 1.0),
    (0.12115583075335397, 0.6957688338493289, 0.9808049535603715, 1.0),
    (0.11929824561403508, 0.5620227038183693, 0.9826625386996903, 1.0),
    (0.11929824561403503, 0.42559339525283857, 0.9847265221878224, 1.0),
    (0.13415892672858618, 0.26769865841073276, 0.9884416924664603, 1.0),
    (0.14901960784313725, 0.10980392156862745, 0.9921568627450981, 1.0),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def infer_volume_scale(volumes: pd.Series | np.ndarray) -> float:
    """Pick a power-of-10 scale so the median volume lands in [0.1, 100].

    Useful for plot axes: integer-satoshi or share volumes (median ~1e8)
    benefit from being scaled down to single-digit numbers, while
    fractional FX volumes (median ~1e-4) benefit from being scaled up.

    Parameters
    ----------
    volumes : pandas.Series or numpy.ndarray
        Vector of positive volume samples.  Empty / non-positive medians
        fall back to a scale of ``1.0``.

    Returns
    -------
    float
        Multiplicative scale factor.  Multiplying *volumes* by the
        returned value yields a series whose median is in the
        ``[0.1, 100)`` range.
    """
    arr = np.asarray(volumes, dtype=float)
    if arr.size == 0:
        return 1.0
    median = float(np.nanmedian(arr))
    if not np.isfinite(median) or median <= 0:
        return 1.0
    exp = math.floor(math.log10(median))
    return 10.0 ** (-exp)


def _default_start_end(
    df: pd.DataFrame,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
    col: str = "timestamp",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fill *None* start/end with the min/max of *col*."""
    if start_time is None:
        start_time = df[col].min()
    if end_time is None:
        end_time = df[col].max()
    return start_time, end_time  # type: ignore[return-value]


def _price_axis_breaks(
    price_min: float,
    price_max: float,
) -> tuple[float, np.ndarray]:
    """Compute tick step and break array for a price axis."""
    price_range = price_max - price_min
    if price_range <= 0:
        return 1.0, np.array([price_min])
    price_by = 10 ** round(np.log10(price_range) - 1)
    y_breaks = np.arange(
        round(price_min / price_by) * price_by,
        round(price_max / price_by) * price_by,
        step=price_by,
    )
    return price_by, y_breaks


# ---------------------------------------------------------------------------
# Prepare functions — one per plot type
# ---------------------------------------------------------------------------


def prepare_time_series_data(
    timestamp: pd.Series,
    series: pd.Series,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    title: str = "time series",
    y_label: str = "series",
) -> dict[str, Any]:
    """Prepare data for a generic time-series step plot."""
    if len(timestamp) != len(series):
        raise ValueError("Length of timestamp and series must be the same.")

    df = pd.DataFrame({"ts": timestamp, "val": series})
    start_time, end_time = _default_start_end(df, start_time, end_time, col="ts")
    df = df[(df["ts"] >= start_time) & (df["ts"] <= end_time)]
    return {"df": df, "title": title, "y_label": y_label}


def prepare_trades_data(
    trades: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Prepare data for a trade-price step plot."""
    start_time, end_time = _default_start_end(trades, start_time, end_time)
    filtered = trades[
        (trades["timestamp"] >= start_time) & (trades["timestamp"] <= end_time)
    ]
    _, y_breaks = _price_axis_breaks(
        filtered["price"].min(),
        filtered["price"].max(),
    )
    return {"filtered_trades": filtered, "y_breaks": y_breaks}


def prepare_price_levels_data(
    depth: pd.DataFrame,
    spread: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    show_mp: bool = True,
    show_all_depth: bool = False,
    col_bias: float = 0.1,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    price_from: float | None = None,
    price_to: float | None = None,
    volume_from: float | None = None,
    volume_to: float | None = None,
    volume_scale: float | None = None,
    price_by: float | None = None,
) -> dict[str, Any]:
    """Prepare data for the price-level depth heatmap.

    When ``volume_scale`` is ``None`` (the default), an order-of-magnitude
    scale is auto-inferred from the input depth via
    :func:`infer_volume_scale`.
    """
    depth_local = depth.copy()
    if volume_scale is None:
        volume_scale = infer_volume_scale(depth_local["volume"])
    depth_local["volume"] = depth_local["volume"] * volume_scale

    if start_time is None:
        start_time = depth_local["timestamp"].iloc[0]
    if end_time is None:
        end_time = depth_local["timestamp"].iloc[-1]

    if spread is not None:
        spread = spread[
            (spread["timestamp"] >= start_time) & (spread["timestamp"] <= end_time)
        ]
        spread = _sanitize_spread(spread)
        if price_from is None:
            price_from = 0.995 * spread["best_bid_price"].min()
        if price_to is None:
            price_to = 1.005 * spread["best_ask_price"].max()

    if trades is not None:
        trades = trades[
            (trades["timestamp"] >= start_time) & (trades["timestamp"] <= end_time)
        ]
        if price_from is None:
            price_from = 0.995 * trades["price"].min()
        else:
            trades = trades[trades["price"] >= price_from]
        if price_to is None:
            price_to = 1.005 * trades["price"].max()
        else:
            trades = trades[trades["price"] <= price_to]

    if price_from is not None:
        depth_local = depth_local[depth_local["price"] >= price_from]
    if price_to is not None:
        depth_local = depth_local[depth_local["price"] <= price_to]
    if volume_from is not None:
        depth_local = depth_local[
            (depth_local["volume"] >= volume_from) | (depth_local["volume"] == 0)
        ]
    if volume_to is not None:
        depth_local = depth_local[depth_local["volume"] <= volume_to]

    depth_filtered = filter_depth(depth_local, start_time, end_time)

    if not show_all_depth:
        counts = depth_filtered.groupby("price", as_index=False)["timestamp"].agg(
            count="size",
            first_ts="min",
            last_ts="max",
        )
        unchanged = counts[
            (counts["count"] == 2)
            & (counts["first_ts"] == start_time)
            & (counts["last_ts"] == end_time)
        ]
        depth_filtered = depth_filtered[
            ~depth_filtered["price"].isin(unchanged["price"])
        ]

    depth_filtered.loc[depth_filtered["volume"] == 0, "volume"] = np.nan

    return {
        "depth": depth_filtered,
        "spread": spread,
        "trades": trades,
        "show_mp": show_mp,
        "col_bias": col_bias,
        "price_by": price_by,
        "y_range": price_y_range(depth_filtered["price"]),
    }


def prepare_event_map_data(
    events: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    price_from: float | None = None,
    price_to: float | None = None,
    volume_from: float | None = None,
    volume_to: float | None = None,
    volume_scale: float | None = None,
) -> dict[str, Any]:
    """Prepare data for a limit-order event map.

    ``volume_scale=None`` auto-infers a power-of-10 scale from the
    filtered events.
    """
    start_time, end_time = _default_start_end(events, start_time, end_time)

    events = events[
        (events["timestamp"] >= start_time)
        & (events["timestamp"] <= end_time)
        & ((events["type"] == "flashed-limit") | (events["type"] == "resting-limit"))
    ].copy()

    if volume_scale is None:
        volume_scale = infer_volume_scale(events["volume"])
    events["volume"] *= volume_scale

    if volume_from is not None:
        events = events[events["volume"] >= volume_from]
    if volume_to is not None:
        events = events[events["volume"] <= volume_to]

    if price_from is None:
        price_from = events["price"].quantile(0.01)
    if price_to is None:
        price_to = events["price"].quantile(0.99)

    events = events[(events["price"] >= price_from) & (events["price"] <= price_to)]

    created = events[events["action"] == "created"]
    deleted = events[events["action"] == "deleted"]

    price_by, _ = _price_axis_breaks(events["price"].min(), events["price"].max())

    return {
        "events": events,
        "created": created,
        "deleted": deleted,
        "price_by": price_by,
    }


def prepare_volume_map_data(
    events: pd.DataFrame,
    action: str = "deleted",
    event_type: list[str] | None = None,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    price_from: float | None = None,
    price_to: float | None = None,
    volume_from: float | None = None,
    volume_to: float | None = None,
    volume_scale: float | None = None,
    log_scale: bool = False,
) -> dict[str, Any]:
    """Prepare data for a volume map of flashed limit orders.

    ``volume_scale=None`` auto-infers a power-of-10 scale from the
    incoming events.
    """
    if event_type is None:
        event_type = ["flashed-limit"]
    if action not in ("deleted", "created"):
        raise ValueError(f"action must be 'deleted' or 'created', got {action!r}")

    start_time, end_time = _default_start_end(events, start_time, end_time)

    events = events.copy()
    if volume_scale is None:
        volume_scale = infer_volume_scale(events["volume"])
    events["volume"] *= volume_scale

    mask = (
        (events["action"] == action)
        & events["type"].isin(event_type)
        & (events["timestamp"] >= start_time)
        & (events["timestamp"] <= end_time)
    )
    events = events[mask]

    if price_from:
        events = events[events["price"] >= price_from]
    if price_to:
        events = events[events["price"] <= price_to]
    if volume_from is None:
        volume_from = events["volume"].quantile(0.0001)
    events = events[events["volume"] >= volume_from]
    if volume_to is None:
        volume_to = events["volume"].quantile(0.9999)
    events = events[events["volume"] <= volume_to]

    return {"events": events, "log_scale": log_scale}


def _as_book_side_frame(side: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Coerce one order-book side to a DataFrame with at least price + volume.

    :func:`ob_analytics.analytics.order_book` hands back per-order frames
    carrying ``id`` and ``bps``; the synthetic/legacy path hands back a
    3-column ``[price, volume, liquidity]`` ndarray.  Both normalise here.
    """
    if isinstance(side, np.ndarray):
        return pd.DataFrame(side, columns=["price", "volume", "liquidity"])
    return side.copy()


def _book_side(
    side: pd.DataFrame, *, ascending: bool, per_order: bool, scale: float
) -> pd.DataFrame:
    """Best-first side frame with stacking bounds + cumulative liquidity.

    *ascending* is the price sort that places the touch first (asks ascending,
    bids descending).  ``per_order=False`` sums every order at a price into one
    row (L2 / MBP); ``per_order=True`` keeps each order with within-level
    ``seg_lo``/``seg_hi`` stacking bounds (L3 / MBO).  Volume-like columns are
    multiplied by *scale*.
    """
    cols = ["price", "volume", "liquidity", "seg_lo", "seg_hi"]
    if side.empty:
        return pd.DataFrame(columns=cols)

    s = side.sort_values("price", ascending=ascending, kind="stable")
    if per_order:
        seg_hi = s.groupby("price", sort=False)["volume"].cumsum()
        out = pd.DataFrame(
            {
                "price": s["price"].to_numpy(),
                "volume": s["volume"].to_numpy(),
                "seg_lo": (seg_hi - s["volume"]).to_numpy(),
                "seg_hi": seg_hi.to_numpy(),
            }
        )
    else:
        agg = s.groupby("price", sort=False, as_index=False)["volume"].sum()
        vol = agg["volume"].to_numpy()
        out = pd.DataFrame(
            {
                "price": agg["price"].to_numpy(),
                "volume": vol,
                "seg_lo": np.zeros(len(vol)),
                "seg_hi": vol.copy(),
            }
        )
    out["liquidity"] = out["volume"].cumsum()
    out[["volume", "liquidity", "seg_lo", "seg_hi"]] *= scale
    return out


def _high_volume_prices(side: pd.DataFrame, q: float = 0.99) -> pd.Series:
    """Prices whose per-row volume is at or above the *q* quantile."""
    if side.empty:
        return pd.Series(dtype=float)
    return side.loc[side["volume"] >= side["volume"].quantile(q), "price"]


def prepare_book_snapshot_data(
    order_book: dict,
    per_order: bool = False,
    volume_scale: float | None = None,
    show_quantiles: bool = True,
) -> dict[str, Any]:
    """Prepare an order-book snapshot at one resolution.

    Feeds both the ``book_snapshot`` bars and the ``depth_chart`` curve.
    ``per_order=False`` collapses each price level to one row (L2 /
    Market-By-Price); ``per_order=True`` keeps every order as its own row with
    within-level stacking bounds (L3 / Market-By-Order).  ``volume_scale=None``
    auto-infers a power-of-10 scale from the combined bid/ask volumes.

    Returns ``bids``/``asks`` frames ordered best-first with columns ``price,
    volume, liquidity, seg_lo, seg_hi`` (volumes already scaled), plus
    high-volume ``*_quantiles`` price marks, ``timestamp``, ``volume_scale``,
    and the ``per_order`` flag.
    """
    bids = _as_book_side_frame(order_book["bids"])
    asks = _as_book_side_frame(order_book["asks"])

    if volume_scale is None:
        volume_scale = infer_volume_scale(
            np.concatenate([bids["volume"].to_numpy(), asks["volume"].to_numpy()])
        )

    bid_side = _book_side(
        bids, ascending=False, per_order=per_order, scale=volume_scale
    )
    ask_side = _book_side(asks, ascending=True, per_order=per_order, scale=volume_scale)

    bid_quantiles = pd.Series(dtype=float)
    ask_quantiles = pd.Series(dtype=float)
    if show_quantiles:
        bid_quantiles = _high_volume_prices(bid_side)
        ask_quantiles = _high_volume_prices(ask_side)

    timestamp = pd.to_datetime(order_book["timestamp"], unit="s", utc=True)

    return {
        "bids": bid_side,
        "asks": ask_side,
        "bid_quantiles": bid_quantiles,
        "ask_quantiles": ask_quantiles,
        "show_quantiles": show_quantiles,
        "per_order": per_order,
        "timestamp": timestamp,
        "volume_scale": volume_scale,
    }


def prepare_volume_percentiles_data(
    depth_summary: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    volume_scale: float | None = None,
    perc_line: bool = True,
    side_line: bool = True,
) -> dict[str, Any]:
    """Prepare data for volume-percentile stacked area chart.

    ``volume_scale=None`` auto-infers a power-of-10 scale from the
    aggregated bin volumes (after the time-window filter is applied).
    """
    if start_time is None:
        start_time = depth_summary["timestamp"].iloc[0]
    if end_time is None:
        end_time = depth_summary["timestamp"].iloc[-1]

    bid_names = [f"bid_vol{i}bps" for i in range(25, 501, 25)]
    ask_names = [f"ask_vol{i}bps" for i in range(25, 501, 25)]

    td = round((end_time - start_time).total_seconds())
    frequency = "mins" if td > 900 else "secs"
    delta = timedelta(seconds=60 if frequency == "mins" else 1)

    mask = (depth_summary["timestamp"] >= (start_time - delta)) & (
        depth_summary["timestamp"] <= end_time
    )
    ob = depth_summary.loc[mask, ["timestamp"] + bid_names + ask_names]
    ob = ob.drop_duplicates(subset="timestamp", keep="last")
    ob.set_index("timestamp", inplace=True)

    if frequency == "mins":
        intervals = pd.DatetimeIndex(ob.index).floor("min")
    else:
        intervals = pd.DatetimeIndex(ob.index).floor("s")

    aggregated = ob.groupby(intervals).mean()
    aggregated.index = aggregated.index + delta
    aggregated.reset_index(inplace=True)
    aggregated.rename(columns={"index": "timestamp"}, inplace=True)
    ob = aggregated

    bid_names_fmt = [f"bid_vol{int(i):03d}bps" for i in range(25, 501, 25)]
    ask_names_fmt = [f"ask_vol{int(i):03d}bps" for i in range(25, 501, 25)]
    ob.columns = pd.Index(["timestamp"] + bid_names_fmt + ask_names_fmt)

    max_ask = ob[ask_names_fmt].sum(axis=1).max()
    max_bid = ob[bid_names_fmt].sum(axis=1).max()

    if volume_scale is None:
        volume_scale = infer_volume_scale(
            ob[bid_names_fmt + ask_names_fmt].to_numpy().ravel()
        )

    melted_asks = ob.melt(
        id_vars="timestamp",
        value_vars=ask_names_fmt,
        var_name="percentile",
        value_name="liquidity",
    )
    melted_asks["percentile"] = pd.Categorical(
        melted_asks["percentile"],
        categories=ask_names_fmt[::-1],
        ordered=True,
    )
    melted_asks["liquidity"] *= volume_scale

    melted_bids = ob.melt(
        id_vars="timestamp",
        value_vars=bid_names_fmt,
        var_name="percentile",
        value_name="liquidity",
    )
    melted_bids["percentile"] = pd.Categorical(
        melted_bids["percentile"],
        categories=bid_names_fmt[::-1],
        ordered=True,
    )
    melted_bids["liquidity"] *= volume_scale

    # Two copies of the shared palette: one for the ask columns, one for
    # the bid columns (40 entries total, matching all_cols below).
    col_pal = list(_VOLUME_PERCENTILE_PALETTE) * 2

    legend_names = [f"+{int(i):03d}bps" for i in range(500, 49, -50)] + [
        f"-{int(i):03d}bps" for i in range(50, 501, 50)
    ]

    asks_pivot = melted_asks.pivot(
        index="timestamp",
        columns="percentile",
        values="liquidity",
    )
    bids_pivot = melted_bids.pivot(
        index="timestamp",
        columns="percentile",
        values="liquidity",
    )
    asks_pivot = asks_pivot[ask_names_fmt[::-1]]
    bids_pivot = bids_pivot[bid_names_fmt[::-1]]

    asks_cumsum = asks_pivot.cumsum(axis=1)
    bids_cumsum = bids_pivot.cumsum(axis=1)
    bids_cumsum_neg = -bids_cumsum

    asks_cols = asks_cumsum.columns.tolist()
    bids_cols = bids_cumsum.columns.tolist()
    all_cols = asks_cols + bids_cols
    colors_dict = dict(zip(all_cols, col_pal))

    return {
        "asks_cumsum": asks_cumsum,
        "bids_cumsum_neg": bids_cumsum_neg,
        "asks_cols": asks_cols,
        "bids_cols": bids_cols,
        "all_cols": all_cols,
        "colors_dict": colors_dict,
        "legend_names": legend_names,
        "max_ask": max_ask,
        "max_bid": max_bid,
        "volume_scale": volume_scale,
        "perc_line": perc_line,
        "side_line": side_line,
    }


def prepare_events_histogram_data(
    events: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    val: str = "volume",
    bw: float | None = None,
) -> dict[str, Any]:
    """Prepare data for an events price/volume histogram."""
    if val not in ("volume", "price"):
        raise ValueError(f"val must be 'volume' or 'price', got {val!r}")
    start_time, end_time = _default_start_end(events, start_time, end_time)
    filtered = events[
        (events["timestamp"] >= start_time) & (events["timestamp"] <= end_time)
    ]
    return {"events": filtered, "val": val, "bw": bw}


def prepare_vpin_data(
    vpin_df: pd.DataFrame,
    threshold: float = 0.7,
) -> dict[str, Any]:
    """Prepare data for a VPIN time-series chart."""
    if len(vpin_df) > 1:
        bar_width = 0.6 * (
            (vpin_df["timestamp_end"].iloc[-1] - vpin_df["timestamp_end"].iloc[0])
            / max(len(vpin_df) - 1, 1)
        )
    else:
        bar_width = 0.001
    return {"vpin_df": vpin_df, "threshold": threshold, "bar_width": bar_width}


def prepare_ofi_data(
    ofi_df: pd.DataFrame,
    trades: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Prepare data for an order-flow-imbalance bar chart.

    Bar colours are a backend-agnostic green/red split on the sign of the
    OFI.  Bar *width* depends on the renderer's x-axis units (matplotlib
    uses date numbers), so it is computed by each backend rather than here.
    """
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in ofi_df["ofi"]]
    return {
        "ofi_df": ofi_df,
        "trades": trades,
        "colors": colors,
    }


def prepare_kyle_lambda_data(kyle_result: Any) -> dict[str, Any]:
    """Prepare data for Kyle's Lambda regression scatter."""
    reg_df = kyle_result.regression_df
    lambda_ = kyle_result.lambda_
    r_squared = kyle_result.r_squared
    t_stat = kyle_result.t_stat
    return {
        "reg_df": reg_df,
        "lambda_": lambda_,
        "r_squared": r_squared,
        "t_stat": t_stat,
    }


# ---------------------------------------------------------------------------
# LOBSTER-enriched plots (gracefully degrade for non-LOBSTER data)
# ---------------------------------------------------------------------------


def prepare_hidden_executions_data(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Prepare data for a hidden-executions overlay chart.

    Shows hidden execution volume (``raw_event_type == 5``) scattered
    over the trade price time series.  Degrades to an empty plot with a
    message when no hidden execution data is present.
    """
    has_hidden = (
        "raw_event_type" in events.columns
        and events["raw_event_type"].notna().any()
        and (events["raw_event_type"] == 5).any()
    )

    if has_hidden:
        hidden = events[events["raw_event_type"] == 5].copy()
    else:
        hidden = pd.DataFrame(columns=events.columns)

    start_time, end_time = _default_start_end(trades, start_time, end_time)
    trades_f = trades[
        (trades["timestamp"] >= start_time) & (trades["timestamp"] <= end_time)
    ]
    if not hidden.empty:
        hidden = hidden[
            (hidden["timestamp"] >= start_time) & (hidden["timestamp"] <= end_time)
        ]

    price_series: list[pd.Series] = []
    if not trades_f.empty:
        price_series.append(trades_f["price"])
    if not hidden.empty:
        price_series.append(hidden["price"])

    return {
        "trades": trades_f,
        "hidden": hidden,
        "has_hidden": has_hidden and not hidden.empty,
        "y_range": price_y_range(*price_series),
        "marker_area": volume_marker_areas(hidden["volume"])
        if not hidden.empty
        else np.array([]),
    }


def prepare_trading_halts_data(
    trades: pd.DataFrame,
    halts: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Prepare data for a trading-halts overlay chart.

    Draws vertical shaded bands on the trade price chart for halt
    periods.  Accepts either a halts DataFrame directly or extracts
    halt events (``raw_event_type == 7``) from the events DataFrame.
    """
    if halts is None and events is not None:
        if (
            "raw_event_type" in events.columns
            and events["raw_event_type"].notna().any()
            and (events["raw_event_type"] == 7).any()
        ):
            halts = events[events["raw_event_type"] == 7].copy()

    has_halts = halts is not None and not halts.empty

    start_time, end_time = _default_start_end(trades, start_time, end_time)
    trades_f = trades[
        (trades["timestamp"] >= start_time) & (trades["timestamp"] <= end_time)
    ]

    halt_periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if has_halts:
        assert halts is not None
        halts_f = halts[
            (halts["timestamp"] >= start_time) & (halts["timestamp"] <= end_time)
        ].sort_values("timestamp")
        # Group consecutive halt events into periods.
        # Halts come in pairs: start (direction == -1 or price == -1)
        # and end (direction == 1 or next non-halt event).
        # Simple heuristic: treat each halt timestamp as a point and
        # create thin bands around them.
        for _, row in halts_f.iterrows():
            ts = row["timestamp"]
            half = pd.Timedelta(seconds=30)
            halt_periods.append((ts - half, ts + half))

    return {
        "trades": trades_f,
        "halt_periods": halt_periods,
        "has_halts": has_halts,
    }
