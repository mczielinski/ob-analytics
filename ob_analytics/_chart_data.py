"""Backend-agnostic data preparation for visualization.

Each ``prepare_*()`` function extracts, filters, and transforms raw
DataFrames into plain dicts consumable by **any** rendering backend
(matplotlib, Plotly, Bokeh, …).  No rendering imports live here.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ob_analytics._utils import reverse_matrix
from ob_analytics.depth import filter_depth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    price_min: float, price_max: float,
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
        filtered["price"].min(), filtered["price"].max(),
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
    volume_scale: float = 1,
    price_by: float | None = None,
) -> dict[str, Any]:
    """Prepare data for the price-level depth heatmap."""
    depth_local = depth.copy()
    depth_local["volume"] = depth_local["volume"] * volume_scale

    if start_time is None:
        start_time = depth_local["timestamp"].iloc[0]
    if end_time is None:
        end_time = depth_local["timestamp"].iloc[-1]

    if spread is not None:
        spread = spread[
            (spread["timestamp"] >= start_time) & (spread["timestamp"] <= end_time)
        ]
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
            count="size", first_ts="min", last_ts="max",
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
    }


def prepare_event_map_data(
    events: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    price_from: float | None = None,
    price_to: float | None = None,
    volume_from: float | None = None,
    volume_to: float | None = None,
    volume_scale: float = 1,
) -> dict[str, Any]:
    """Prepare data for a limit-order event map."""
    start_time, end_time = _default_start_end(events, start_time, end_time)

    events = events[
        (events["timestamp"] >= start_time)
        & (events["timestamp"] <= end_time)
        & ((events["type"] == "flashed-limit") | (events["type"] == "resting-limit"))
    ].copy()

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
    volume_scale: float = 1,
    log_scale: bool = False,
) -> dict[str, Any]:
    """Prepare data for a volume map of flashed limit orders."""
    if event_type is None:
        event_type = ["flashed-limit"]
    if action not in ("deleted", "created"):
        raise ValueError(f"action must be 'deleted' or 'created', got {action!r}")

    start_time, end_time = _default_start_end(events, start_time, end_time)

    events = events.copy()
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


def prepare_current_depth_data(
    order_book: dict,
    volume_scale: float = 1,
    show_quantiles: bool = True,
    show_volume: bool = True,
) -> dict[str, Any]:
    """Prepare data for order book depth snapshot."""
    bids = reverse_matrix(order_book["bids"])
    asks = reverse_matrix(order_book["asks"])

    # reverse_matrix may return ndarray; ensure we have DataFrames
    if isinstance(bids, np.ndarray):
        bids = pd.DataFrame(bids, columns=["price", "volume", "liquidity"])
    if isinstance(asks, np.ndarray):
        asks = pd.DataFrame(asks, columns=["price", "volume", "liquidity"])

    x = np.concatenate([
        bids["price"].values,
        [bids["price"].values[-1]],
        [asks["price"].values[0]],
        asks["price"].values,
    ])
    y1 = (
        np.concatenate([bids["liquidity"].values, [0], [0], asks["liquidity"].values])
        * volume_scale
    )
    y2 = (
        np.concatenate([bids["volume"].values, [0], [0], asks["volume"].values])
        * volume_scale
    )
    side = ["bid"] * (len(bids) + 1) + ["ask"] * (len(asks) + 1)

    depth_df = pd.DataFrame({"price": x, "liquidity": y1, "volume": y2, "side": side})

    bid_quantiles = pd.Series(dtype=float)
    ask_quantiles = pd.Series(dtype=float)
    if show_quantiles:
        bq = bids["volume"].quantile(0.99)
        bid_quantiles = bids.loc[bids["volume"] >= bq, "price"]
        aq = asks["volume"].quantile(0.99)
        ask_quantiles = asks.loc[asks["volume"] >= aq, "price"]

    timestamp = pd.to_datetime(order_book["timestamp"], unit="s", utc=True)

    return {
        "depth_df": depth_df,
        "bids": bids,
        "asks": asks,
        "bid_quantiles": bid_quantiles,
        "ask_quantiles": ask_quantiles,
        "show_volume": show_volume,
        "show_quantiles": show_quantiles,
        "timestamp": timestamp,
    }


def prepare_volume_percentiles_data(
    depth_summary: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    volume_scale: float = 1,
    perc_line: bool = True,
    side_line: bool = True,
) -> dict[str, Any]:
    """Prepare data for volume-percentile stacked area chart."""
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

    melted_asks = ob.melt(
        id_vars="timestamp", value_vars=ask_names_fmt,
        var_name="percentile", value_name="liquidity",
    )
    melted_asks["percentile"] = pd.Categorical(
        melted_asks["percentile"], categories=ask_names_fmt[::-1], ordered=True,
    )
    melted_asks["liquidity"] *= volume_scale

    melted_bids = ob.melt(
        id_vars="timestamp", value_vars=bid_names_fmt,
        var_name="percentile", value_name="liquidity",
    )
    melted_bids["percentile"] = pd.Categorical(
        melted_bids["percentile"], categories=bid_names_fmt[::-1], ordered=True,
    )
    melted_bids["liquidity"] *= volume_scale

    from matplotlib.colors import LinearSegmentedColormap

    colors_list = [
        "#f92b20", "#fe701b", "#facd1f", "#d6fd1c", "#65fe1b",
        "#1bfe42", "#1cfdb4", "#1fb9fa", "#1e71fb", "#261cfd",
    ]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors_list, N=20)
    col_pal = [cmap(i / 19) for i in range(20)]
    col_pal *= 2

    legend_names = [f"+{int(i):03d}bps" for i in range(500, 49, -50)] + [
        f"-{int(i):03d}bps" for i in range(50, 501, 50)
    ]

    asks_pivot = melted_asks.pivot(
        index="timestamp", columns="percentile", values="liquidity",
    )
    bids_pivot = melted_bids.pivot(
        index="timestamp", columns="percentile", values="liquidity",
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
    """Prepare data for an order-flow-imbalance bar chart."""
    import matplotlib.dates as mdates

    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in ofi_df["ofi"]]
    if len(ofi_df) > 1:
        median_gap = ofi_df["timestamp"].diff().median()
        bar_width = (
            mdates.date2num(ofi_df["timestamp"].iloc[0] + median_gap * 0.8)
            - mdates.date2num(ofi_df["timestamp"].iloc[0])
        )
    else:
        bar_width = 0.001
    return {
        "ofi_df": ofi_df,
        "trades": trades,
        "colors": colors,
        "bar_width": bar_width,
    }


def prepare_kyle_lambda_data(kyle_result: object) -> dict[str, Any]:
    """Prepare data for Kyle's Lambda regression scatter."""
    reg_df = kyle_result.regression_df  # type: ignore[union-attr]
    lambda_ = kyle_result.lambda_  # type: ignore[union-attr]
    r_squared = kyle_result.r_squared  # type: ignore[union-attr]
    t_stat = kyle_result.t_stat  # type: ignore[union-attr]
    return {
        "reg_df": reg_df,
        "lambda_": lambda_,
        "r_squared": r_squared,
        "t_stat": t_stat,
    }
