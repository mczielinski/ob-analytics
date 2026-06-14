"""Backend-agnostic data preparation for visualization.

Each ``prepare_*()`` function extracts, filters, and transforms raw
DataFrames into plain dicts consumable by **any** rendering backend
(matplotlib, Plotly, Bokeh, …).  No rendering imports live here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ob_analytics.depth import filter_depth


@dataclass(frozen=True)
class FocusWindow:
    """A shared display window for gallery faces.

    One clipping decision, made once per gallery build, instead of every
    face inventing its own (raw-price percentiles here, ±kσ there): faces
    that receive the same bounds stay comparable on shared axes.  ``None``
    on either axis means no clipping there.
    """

    start_time: pd.Timestamp | None = None
    end_time: pd.Timestamp | None = None
    price_from: float | None = None
    price_to: float | None = None


def focus_window(
    trades: pd.DataFrame,
    *,
    k_sigma: float = 3.0,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> FocusWindow:
    """Mid-anchored :class:`FocusWindow`: trades-median ± *k_sigma*·σ.

    Trades happen at the touch, so their median anchors the window where
    the action is; the σ band keeps far-from-touch flashed orders from
    stretching the price axis.  Degenerate tapes (empty, or zero/NaN price
    dispersion) yield an unclipped price axis.
    """
    if trades.empty:
        return FocusWindow(start_time, end_time, None, None)
    mid = float(trades["price"].median())
    std = float(trades["price"].std())
    if not math.isfinite(std) or std <= 0:
        return FocusWindow(start_time, end_time, None, None)
    return FocusWindow(
        start_time, end_time, max(0.0, mid - k_sigma * std), mid + k_sigma * std
    )


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


def normalized_marker_areas(
    volume: pd.Series | np.ndarray,
    *,
    lo: float = 10.0,
    hi: float = 120.0,
    ref_quantile: float = 0.95,
) -> np.ndarray:
    """Bounded, outlier-robust matplotlib scatter ``s`` values.

    Where :func:`volume_marker_areas` is raw-proportional and unbounded -- fine
    for sparse overlays but degenerate when one whale order is 10,000x the
    median -- this maps volume *linearly* into ``[lo, hi]`` against its
    *ref_quantile*: anything at or above that reference saturates at *hi*, and
    everything below scales down toward *lo* (volume ``0`` lands exactly on
    *lo*, so no point is invisible).  Suited to the dense L3 per-order scatters
    that draw one marker per order across a heavy-tailed volume range.
    """
    arr = np.asarray(volume, dtype=float)
    if arr.size == 0:
        return np.empty(0, dtype=float)
    ref = float(np.nanquantile(arr, ref_quantile))
    if not np.isfinite(ref) or ref <= 0:
        return np.full(arr.shape, lo, dtype=float)
    frac = np.clip(arr / ref, 0.0, 1.0)
    return lo + frac * (hi - lo)


def lollipop_marker_areas(
    volume: pd.Series | np.ndarray,
    *,
    lo: float = 14.0,
    hi: float = 90.0,
    ref_quantile: float = 0.95,
) -> np.ndarray:
    """Rank-3 (cube-root) matplotlib scatter ``s`` values for the trade tape.

    The trade tape's whole point is that one whale print should *pop* without
    drowning the small ones.  :func:`normalized_marker_areas` is linear, so a
    median print and a 10x print look nearly identical at the bottom of the
    ``[lo, hi]`` band; raw area (rank-5) goes the other way and visually
    compresses everything large.  This sits between them: the marker *diameter*
    scales as the cube root of volume (a "rank-3" encoding in the bundle's
    taxonomy), i.e. area as ``volume**(2/3)``, mapped into ``[lo, hi]`` against
    *ref_quantile* so the tail saturates rather than blowing up the axis.
    """
    arr = np.asarray(volume, dtype=float)
    if arr.size == 0:
        return np.empty(0, dtype=float)
    ref = float(np.nanquantile(arr, ref_quantile))
    if not np.isfinite(ref) or ref <= 0:
        return np.full(arr.shape, lo, dtype=float)
    frac = np.clip(np.cbrt(np.clip(arr, 0.0, None) / ref), 0.0, 1.0)
    return lo + frac * (hi - lo)


def mpl_marker_area_to_plotly_size(area: np.ndarray) -> np.ndarray:
    """Convert matplotlib scatter areas (pt²) to plotly marker diameters (px)."""
    return np.sqrt(np.maximum(area, 0.0)) * 0.8


def _volume_percentile_palette(
    n: int,
) -> tuple[tuple[float, float, float, float], ...]:
    """Volume-percentile gradient, shared by every rendering backend.

    Percentile rank is ordered data, so the *n* steps walk monotonically in
    luminance (pale → saturated blue) — hue carries no order.  Ramp direction
    and per-side hue split are revisited in roadmap §3.7 (docs/plans/).
    """
    span = max(n - 1, 1)
    return tuple(
        (
            0.88 + (0.03 - 0.88) * (i / span),
            0.92 + (0.19 - 0.92) * (i / span),
            0.98 + (0.42 - 0.98) * (i / span),
            1.0,
        )
        for i in range(n)
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


def _trade_tape_mid(
    trades: pd.DataFrame,
    spread: pd.DataFrame | None,
) -> pd.DataFrame:
    """Reference mid line for the trade tape, one row per trade timestamp.

    Lollipop stems hang off a mid/microprice line.  When a *spread* frame is
    available (``best_bid_price`` / ``best_ask_price``), the anchor is the true
    ``(bid + ask) / 2`` sampled as-of each trade.  Otherwise the tape is
    self-contained and the anchor is a rolling-median proxy of the trade price
    itself -- a smooth spine that still lets buys sit above and sells below
    without pretending to be the book mid.
    """
    out = trades[["timestamp"]].copy()
    if spread is not None and not spread.empty:
        sp = spread[["timestamp", "best_bid_price", "best_ask_price"]].dropna()
        if not sp.empty:
            sp = sp.sort_values("timestamp", kind="stable")
            sp["mid"] = (sp["best_bid_price"] + sp["best_ask_price"]) / 2.0
            merged = pd.merge_asof(
                out.sort_values("timestamp", kind="stable"),
                sp[["timestamp", "mid"]],
                on="timestamp",
                direction="nearest",
            )
            out["mid"] = merged.set_index(out.sort_values("timestamp").index)["mid"]
            if out["mid"].notna().any():
                return out
    # Fallback: rolling median of trade price (centered, robust to the spike).
    n = len(out)
    window = max(3, min(21, n // 5 or 1))
    out["mid"] = (
        trades["price"]
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    return out


def prepare_trades_data(
    trades: pd.DataFrame,
    spread: pd.DataFrame | None = None,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Prepare data for the L2 signed-lollipop trade tape.

    Each trade is one lollipop: a stem from the mid/microprice line up to the
    execution price (buys lift the ask, so they sit above; sells hit the bid,
    below), tipped by a marker whose size encodes the trade volume on a
    rank-3 (cube-root) scale.  The price axis is padded to the *data extent*
    within the time window -- trade prices are **never** quantile-clipped, so a
    rare spike print stays fully visible (roadmap §3.4).

    Pass *spread* (e.g. :func:`ob_analytics.depth.get_spread`) to anchor stems
    on the true book mid; without it, a rolling-median price proxy is used.
    """
    start_time, end_time = _default_start_end(trades, start_time, end_time)
    filtered = trades[
        (trades["timestamp"] >= start_time) & (trades["timestamp"] <= end_time)
    ]
    _, y_breaks = _price_axis_breaks(
        filtered["price"].min() if not filtered.empty else 0.0,
        filtered["price"].max() if not filtered.empty else 1.0,
    )

    mid_line = _trade_tape_mid(filtered, spread)
    tape = filtered[["timestamp", "price", "volume", "direction"]].copy()
    tape["mid"] = mid_line["mid"].to_numpy()
    tape["marker_area"] = lollipop_marker_areas(tape["volume"])

    # No clipping: the y-range is the full price extent within the window, so
    # the prints a tape exists to show (spikes) are never cut mid-marker.
    y_range = price_y_range(filtered["price"]) if not filtered.empty else None

    return {
        "filtered_trades": filtered,
        "y_breaks": y_breaks,
        "buys": tape[tape["direction"] == "buy"],
        "sells": tape[tape["direction"] == "sell"],
        "mid_line": mid_line.sort_values("timestamp", kind="stable"),
        "y_range": y_range,
    }


def prepare_price_levels_data(
    depth: pd.DataFrame,
    spread: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    show_mp: bool = True,
    show_all_depth: bool = False,
    col_bias: float = 1.0,
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

    ``col_bias`` is the gamma of a power-law color normalization
    (``volume ** col_bias`` mapped onto the colormap). ``1.0`` (the
    default) is linear: high-volume walls stand out against a dark
    field. Values in ``(0, 1)`` progressively brighten low-volume
    levels, revealing near-touch structure in heavy-tailed books
    (``0.1`` matches the R package's palette bias). Values ``<= 0``
    select a log10 scale.

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


def prepare_order_activity_l3_data(
    events: pd.DataFrame,
    volume_scale: float | None = None,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    price_from: float | None = None,
    price_to: float | None = None,
    max_spans: int = 2000,
    marker_threshold: int = 300,
) -> dict[str, Any]:
    """Per-order lifecycle Gantt: each order one span place -> outcome, by fate.

    The L3 (MBO) counterpart to the ``order_activity`` event map.  Where the L2
    map scatters *decoupled* created/deleted events, this links each order's
    events by ``id`` into a single horizontal span -- from creation to removal --
    placed at the order's price and split by terminal **outcome**:

    - **filled** (``filled``/``partial``): the order executed; the span ends at
      the fill-exhaustion time.
    - **cancelled**: placed and pulled without executing; ends at the delete.
    - **resting**: still on the book at window end; the span extends to
      *end_time* to show it had not yet terminated.

    The price axis is clipped to the 1st--99th percentile (like the event map)
    unless *price_from* / *price_to* are given.  Line **width encodes size** and
    the terminal **marker** (x filled, o cancelled) is drawn only when few
    enough spans survive (``marker_threshold``).

    Dense books (LOBSTER emits 10^4--10^5 lifelines) are degraded by density:
    above *max_spans* total spans, every fill/partial/resting span is kept and
    only the cancellation flood is sampled, with a ``shown_of`` ratio the
    renderer annotates.

    Spans come from :func:`~ob_analytics.analytics.order_lifecycles`, so an
    order ends when it is deleted **or fully executed** (LOBSTER fills never
    emit a delete); only genuinely still-resting orders extend to the window
    end.  Lifecycles overlapping the window are clipped to it.
    """
    from ob_analytics.analytics import order_lifecycles

    start_time, end_time = _default_start_end(events, start_time, end_time)

    life = order_lifecycles(events)
    life = life[life["type"].isin(["flashed-limit", "resting-limit"])]

    # Lifecycles overlapping the window, clipped to it for display.
    overlaps = (life["placed_ts"] <= end_time) & (
        life["end_ts"].isna() | (life["end_ts"] >= start_time)
    )
    spans = life[overlaps].copy()
    placed = spans["placed_ts"]
    spans["start_ts"] = placed.where(placed >= start_time, start_time)
    ended = spans["end_ts"].fillna(end_time)
    spans["end_ts"] = ended.where(ended <= end_time, end_time)
    spans = spans.rename(columns={"placed_vol": "volume"})

    if volume_scale is None:
        volume_scale = infer_volume_scale(spans["volume"]) if not spans.empty else 1.0
    spans["volume"] = spans["volume"] * volume_scale

    if not spans.empty:
        if price_from is None:
            price_from = spans["price"].quantile(0.01)
        if price_to is None:
            price_to = spans["price"].quantile(0.99)
        spans = spans[(spans["price"] >= price_from) & (spans["price"] <= price_to)]

    # Three terminal fates; filled/partial collapse to "filled".  Colour and the
    # end-marker encode fate; line width encodes size.
    spans = spans.assign(
        fate=spans["outcome"].map(
            {
                "filled": "filled",
                "partial": "filled",
                "cancelled": "cancelled",
                "resting": "resting",
            }
        )
    )

    # Degrade by density: keep every fill (the rare, telling outcome) and
    # sample the cancelled/resting flood down to the remaining budget so all
    # three fates stay represented; report the ratio for an annotation.
    n_total = len(spans)
    shown_of = None
    if n_total > max_spans:
        keep = spans[spans["fate"] == "filled"]
        flood = spans[spans["fate"] != "filled"]
        budget = max(0, max_spans - len(keep))
        if len(flood) > budget:
            flood = flood.sample(n=budget, random_state=0)
        spans = pd.concat([keep, flood]).sort_index()
        shown_of = (len(spans), n_total)

    if spans.empty:
        price_by, y_range = 1.0, None
    else:
        price_by, _ = _price_axis_breaks(spans["price"].min(), spans["price"].max())
        y_range = (float(spans["price"].min()), float(spans["price"].max()))

    # Line width encodes size: lw = 0.5 + 4*(size / max size).
    vmax = spans["volume"].max() if not spans.empty else 0.0
    if vmax > 0:
        spans = spans.assign(linewidth=0.5 + 4.0 * (spans["volume"] / vmax))
    else:
        spans = spans.assign(linewidth=1.0)

    return {
        "filled": spans[spans["fate"] == "filled"],
        "cancelled": spans[spans["fate"] == "cancelled"],
        "resting": spans[spans["fate"] == "resting"],
        "volume_scale": volume_scale,
        "price_by": price_by,
        "y_range": y_range,
        "shown_of": shown_of,
        "show_markers": n_total <= marker_threshold,
    }


def prepare_liquidity_at_touch_data(
    depth_summary: pd.DataFrame,
    volume_scale: float | None = None,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """L2 (MBP) liquidity at the touch: best bid/ask resting size over time.

    Two aggregate time series read straight from the depth summary -- the volume
    resting at the best bid (``best_bid_vol``) and at the best ask
    (``best_ask_vol``).  Size at a price level carries no order identity, so this
    is an L2 quantity by construction; the per-order L3 counterpart (queue
    composition at the touch) needs FIFO reconstruction and is deferred.
    """
    start_time, end_time = _default_start_end(depth_summary, start_time, end_time)
    win = depth_summary[
        (depth_summary["timestamp"] >= start_time)
        & (depth_summary["timestamp"] <= end_time)
    ]
    bid_vol = win["best_bid_vol"]
    ask_vol = win["best_ask_vol"]
    if volume_scale is None:
        combined = pd.concat([bid_vol, ask_vol])
        volume_scale = infer_volume_scale(combined) if not combined.empty else 1.0
    return {
        "timestamp": win["timestamp"].reset_index(drop=True),
        "bid_vol": (bid_vol * volume_scale).reset_index(drop=True),
        "ask_vol": (ask_vol * volume_scale).reset_index(drop=True),
        "volume_scale": volume_scale,
    }


def prepare_order_outcome_l3_data(
    events: pd.DataFrame,
    volume_scale: float | None = None,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    bps_quantiles: tuple[float, float] = (0.05, 0.95),
) -> dict[str, Any]:
    """L3 (MBO) order outcome: each limit order as outcome vs placement.

    For every order created in the window, one point at its placement --
    ``x = aggressiveness_bps`` (signed distance from the prevailing best at
    creation; ``>0`` improved the touch) against ``y = size`` -- coloured by a
    competing-risks *outcome* from
    :func:`~ob_analytics.analytics.order_lifecycles`:

    - **filled**: the whole order executed.
    - **partial**: some volume executed, the remainder was removed.
    - **cancelled**: removed without any execution.

    Orders still resting at window end are censored and dropped.  Outcomes
    derive from the canonical ``fill`` column (the order's own executions),
    so they reflect the *visible* book on every format.  Placement distance
    is heavy-tailed, so it is quantile-clipped to *bps_quantiles*.
    """
    from ob_analytics.analytics import order_lifecycles

    start_time, end_time = _default_start_end(events, start_time, end_time)

    life = order_lifecycles(events)
    created = life[
        (life["placed_ts"] >= start_time) & (life["placed_ts"] <= end_time)
    ].rename(
        columns={
            "aggressiveness_bps": "distance_bps",
            "placed_vol": "placed",
            "placed_ts": "timestamp",
        }
    )
    created = created[created["outcome"] != "resting"].dropna(subset=["distance_bps"])

    if volume_scale is None:
        volume_scale = (
            infer_volume_scale(created["placed"]) if not created.empty else 1.0
        )
    created = created.copy()
    created["placed"] = created["placed"] * volume_scale

    if not created.empty:
        lo_q, hi_q = bps_quantiles
        bps_lo = created["distance_bps"].quantile(lo_q)
        bps_hi = created["distance_bps"].quantile(hi_q)
        created = created[
            (created["distance_bps"] >= bps_lo) & (created["distance_bps"] <= bps_hi)
        ]

    created["marker_area"] = normalized_marker_areas(created["placed"])
    return {
        "filled": created[created["outcome"] == "filled"],
        "partial": created[created["outcome"] == "partial"],
        "cancelled": created[created["outcome"] == "cancelled"],
        "volume_scale": volume_scale,
    }


def _per_second_vwap(tr: pd.DataFrame) -> pd.DataFrame:
    """Collapse a tape to one volume-weighted lollipop per (second, side).

    Dense tapes (LOBSTER: tens of thousands of prints) saturate into an
    unreadable smear of overlapping markers.  Bucketing to one-second VWAPs
    keeps the shape of the tape -- when and on which side size traded -- while
    cutting the marker count by one to two orders of magnitude.  ``mid`` and
    ``marker_area`` are recomputed downstream against the aggregated rows.
    """
    if tr.empty:
        return tr.iloc[0:0][["timestamp", "price", "volume", "direction"]]
    g = tr.assign(_sec=tr["timestamp"].dt.floor("1s"))
    g = g.assign(_pv=g["price"] * g["volume"])
    agg = (
        g.groupby(["_sec", "direction"], observed=True)
        .agg(volume=("volume", "sum"), pv=("_pv", "sum"))
        .reset_index()
    )
    agg = agg[agg["volume"] > 0]
    agg["price"] = agg["pv"] / agg["volume"]
    agg = agg.rename(columns={"_sec": "timestamp"}).drop(columns="pv")
    return agg[["timestamp", "price", "volume", "direction"]]


def prepare_trade_tape_l3_data(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    spread: pd.DataFrame | None = None,
    volume_scale: float | None = None,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    price_from: float | None = None,
    price_to: float | None = None,
    density_threshold: int = 1500,
) -> dict[str, Any]:
    """L3 (MBO) signed-lollipop trade tape with maker resting spans.

    The L2 tape shows each execution as a signed lollipop (stem from the mid to
    the price, marker sized by volume, coloured by aggressor side).  The L3 face
    keeps those lollipops and adds, for every execution, the life of the *maker*
    order it consumed: a thin horizontal span from that order's creation to the
    fill, drawn at the maker's price -- the L3 differentiator, revealing how long
    the resting liquidity behind each trade had waited.

    Trade prices are **never** quantile-clipped (roadmap §3.4): the y-axis is
    padded to the data extent within the time window, so spike prints stay fully
    visible.  Pass explicit *price_from*/*price_to* only to crop deliberately.

    When the window holds more than *density_threshold* executions the lollipops
    are aggregated to one-second VWAPs (``dense=True``); the per-trade maker
    spans are still returned so the renderer can draw them thin underneath.
    """
    start_time, end_time = _default_start_end(trades, start_time, end_time)
    tr = trades[
        (trades["timestamp"] >= start_time) & (trades["timestamp"] <= end_time)
    ][["timestamp", "price", "volume", "direction", "maker_event_id"]]

    # Map each trade's maker event -> the maker order id -> its creation time.
    event_to_id = events[["event_id", "id"]]
    created_ts = (
        events[events["action"] == "created"]
        .groupby("id")["timestamp"]
        .min()
        .rename("created_ts")
    )
    tr = tr.merge(
        event_to_id, left_on="maker_event_id", right_on="event_id", how="inner"
    ).merge(created_ts, on="id", how="inner")

    # Optional deliberate price crop only; default keeps every print on-axis.
    if not tr.empty:
        if price_from is not None:
            tr = tr[tr["price"] >= price_from]
        if price_to is not None:
            tr = tr[tr["price"] <= price_to]

    if volume_scale is None:
        volume_scale = infer_volume_scale(tr["volume"]) if not tr.empty else 1.0
    tr = tr.copy()
    tr["volume"] = tr["volume"] * volume_scale

    mid_line = _trade_tape_mid(tr, spread)
    tr["mid"] = mid_line["mid"].to_numpy()
    tr["marker_area"] = lollipop_marker_areas(tr["volume"])

    dense = len(tr) > density_threshold
    if dense:
        agg = _per_second_vwap(tr)
        agg_mid = _trade_tape_mid(agg, spread)
        agg["mid"] = agg_mid["mid"].to_numpy()
        agg["marker_area"] = lollipop_marker_areas(agg["volume"])
        lolli_buys = agg[agg["direction"] == "buy"]
        lolli_sells = agg[agg["direction"] == "sell"]
    else:
        lolli_buys = tr[tr["direction"] == "buy"]
        lolli_sells = tr[tr["direction"] == "sell"]

    # y-range spans every print (and its mid), never a quantile clip.
    y_range = price_y_range(tr["price"], mid_line["mid"]) if not tr.empty else None

    return {
        "buys": tr[tr["direction"] == "buy"],
        "sells": tr[tr["direction"] == "sell"],
        "lolli_buys": lolli_buys,
        "lolli_sells": lolli_sells,
        "mid_line": mid_line.sort_values("timestamp", kind="stable"),
        "dense": dense,
        "volume_scale": volume_scale,
        "y_range": y_range,
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


def prepare_cancellations_l3_data(
    events: pd.DataFrame,
    volume_scale: float | None = None,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    age_floor_s: float = 1e-3,
    distance_floor_bps: float = 0.05,
    distance_cap_bps: float = 1_000.0,
) -> dict[str, Any]:
    """Per-order cancellations: each cancelled order as one age x distance point.

    The L3 (MBO) counterpart to the ``cancellations`` volume map.  Where the L2
    map aggregates cancelled *volume* over price and time, this keeps each
    cancelled order as a distinct point: ``age_s`` (seconds it rested between
    creation and deletion) against ``distance_from_touch`` (absolute bps from
    the prevailing best at creation), split by side.

    Cancellations are deleted *flashed-limit* orders, matching the L2 face.
    Both axes are heavy-tailed and span several orders of magnitude -- the
    latent populations (fleeting sub-100ms fishing at the touch, patient
    human-scale orders, deep orders pulled later) only separate on **log-log**
    axes, so the face is a per-side density (hexbin), not a linear scatter.
    Instead of clipping the tails (which also clips the structure) the floors
    *age_floor_s* / *distance_floor_bps* keep instant cancels and at-touch
    placements on-scale (log cannot show zero).  *distance_cap_bps* drops only
    the degenerate aggressiveness values (orders priced against a near-empty
    opposite best, which blow up to absurd bps) -- not real resting depth.
    """
    start_time, end_time = _default_start_end(events, start_time, end_time)
    deleted = events[
        (events["action"] == "deleted")
        & (events["type"] == "flashed-limit")
        & (events["timestamp"] >= start_time)
        & (events["timestamp"] <= end_time)
    ][["id", "timestamp", "volume", "direction"]]
    created = events[events["action"] == "created"][
        ["id", "timestamp", "aggressiveness_bps"]
    ].rename(columns={"timestamp": "created_ts", "aggressiveness_bps": "distance_bps"})

    cancels = deleted.merge(created, on="id", how="inner").dropna(
        subset=["distance_bps"]
    )
    cancels["age_s"] = (cancels["timestamp"] - cancels["created_ts"]).dt.total_seconds()
    cancels = cancels[cancels["age_s"] >= 0].copy()

    if volume_scale is None:
        volume_scale = (
            infer_volume_scale(cancels["volume"]) if not cancels.empty else 1.0
        )
    cancels["volume"] = cancels["volume"] * volume_scale

    # Floor onto the log scale instead of quantile-clipping the tails: instant
    # cancels land in a thin "age floor" band, at-touch placements in a
    # "distance floor" band, and the deep/patient structure is preserved.
    cancels["age_s"] = cancels["age_s"].clip(lower=age_floor_s)
    cancels["distance_from_touch"] = (
        cancels["distance_bps"].abs().clip(lower=distance_floor_bps)
    )
    cancels = cancels[cancels["distance_from_touch"] <= distance_cap_bps]

    bids = cancels[cancels["direction"] == "bid"]
    asks = cancels[cancels["direction"] == "ask"]
    return {"bids": bids, "asks": asks, "volume_scale": volume_scale}


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
        # Biggest order first within each level: the largest order anchors at
        # the axis (seg_lo=0), so a whale level reads as one long segment and a
        # crowd of small orders reads as a dashed run of short ones.
        s = s.sort_values(
            ["price", "volume"], ascending=[ascending, False], kind="stable"
        )
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


def _high_volume_prices(
    side: pd.DataFrame, q: float = 0.99, max_n: int = 3
) -> pd.Series:
    """Up to *max_n* prices whose *level* volume is at or above the *q* quantile.

    Volume is aggregated per price first, so a per-order (L3) side does not emit
    one guide line per order row -- only the genuinely heavy levels, capped at
    *max_n* so the snapshot never drowns under full-height guides.
    """
    if side.empty:
        return pd.Series(dtype=float)
    agg = side.groupby("price", as_index=False)["volume"].sum()
    hot = agg[agg["volume"] >= agg["volume"].quantile(q)]
    return hot.sort_values("volume", ascending=False)["price"].head(max_n)


def _window_levels(side: pd.DataFrame, top_n: int | None) -> pd.DataFrame:
    """Keep the rows of a best-first side at its *top_n* nearest price levels.

    Counts distinct price *levels*, not order rows, so an L3 side keeps every
    order resting at the kept levels.  Windowing to the touch is what keeps the
    ladder's bars tall enough that per-order separators stay legible.
    """
    if top_n is None or side.empty:
        return side
    keep = side["price"].drop_duplicates().head(top_n)
    return side[side["price"].isin(keep)]


def book_mid(bids: pd.DataFrame, asks: pd.DataFrame) -> float | None:
    """Midprice between the best bid and best ask, or ``None`` if one-sided.

    Both sides arrive best-first; the touch is the highest bid and the lowest
    ask.  Used by both backends to draw the ladder's mid reference line.
    """
    if bids.empty or asks.empty:
        return None
    return (float(bids["price"].max()) + float(asks["price"].min())) / 2


def prepare_book_snapshot_data(
    order_book: dict,
    per_order: bool = False,
    volume_scale: float | None = None,
    show_quantiles: bool = False,
    top_n: int | None = 40,
) -> dict[str, Any]:
    """Prepare an order-book snapshot at one resolution.

    Feeds both the ``book_snapshot`` ladder and the ``depth_chart`` curve.
    ``per_order=False`` collapses each price level to one row (L2 /
    Market-By-Price); ``per_order=True`` keeps every order as its own row with
    within-level stacking bounds (L3 / Market-By-Order).  ``volume_scale=None``
    auto-infers a power-of-10 scale from the combined bid/ask volumes.

    ``top_n`` windows each side to its *N* nearest price levels (``None`` =
    whole book); the default keeps the touch region where bars stay tall enough
    that per-order separators read.  ``show_quantiles`` overlays up to three
    heavy-level guide lines per side and is off by default.

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

    bid_side = _window_levels(
        _book_side(bids, ascending=False, per_order=per_order, scale=volume_scale),
        top_n,
    )
    ask_side = _window_levels(
        _book_side(asks, ascending=True, per_order=per_order, scale=volume_scale),
        top_n,
    )

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

    The BPS bins are discovered from the ``(bid|ask)_vol<N>bps`` columns of
    *depth_summary*, so any ``depth_bps`` / ``depth_bins`` configuration works
    (the previous hardcoded 25–500 bps range raised ``KeyError`` for anything
    else).  ``volume_scale=None`` auto-infers a power-of-10 scale from the
    aggregated bin volumes (after the time-window filter is applied).
    """
    if start_time is None:
        start_time = depth_summary["timestamp"].iloc[0]
    if end_time is None:
        end_time = depth_summary["timestamp"].iloc[-1]

    bps_levels = sorted(
        int(m.group(1))
        for c in depth_summary.columns
        if (m := re.fullmatch(r"bid_vol(\d+)bps", str(c)))
    )
    if not bps_levels:
        raise ValueError(
            "prepare_volume_percentiles_data: no 'bid_vol<N>bps' columns in "
            f"depth_summary. Columns: {list(depth_summary.columns)}"
        )
    bid_names = [f"bid_vol{b}bps" for b in bps_levels]
    ask_names = [f"ask_vol{b}bps" for b in bps_levels]

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

    pad = max(3, len(str(bps_levels[-1])))
    bid_names_fmt = [f"bid_vol{b:0{pad}d}bps" for b in bps_levels]
    ask_names_fmt = [f"ask_vol{b:0{pad}d}bps" for b in bps_levels]
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
    # the bid columns (2N entries total, matching all_cols below).
    col_pal = list(_volume_percentile_palette(len(bps_levels))) * 2

    # Every other bin, outermost first per side (matches the legacy fixed
    # labels at the default 25x20 configuration).
    desc = bps_levels[::-1]
    legend_names = [f"+{b:0{pad}d}bps" for b in desc[::2]] + [
        f"-{b:0{pad}d}bps" for b in desc[::2][::-1]
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
    price_from: float | None = None,
    price_to: float | None = None,
) -> dict[str, Any]:
    """Prepare data for an events price/volume histogram.

    ``price_from``/``price_to`` clip the events to a price window before
    binning.  Without it the price face collapses to a single 1px spike: even
    the 1st-99th percentile of a heavy-tailed book still spans the far-from-
    touch flashed orders, so the caller passes a mid-anchored focus window (the
    same one the depth heatmap uses) to keep the near-touch distribution legible.
    """
    if val not in ("volume", "price"):
        raise ValueError(f"val must be 'volume' or 'price', got {val!r}")
    start_time, end_time = _default_start_end(events, start_time, end_time)
    filtered = events[
        (events["timestamp"] >= start_time) & (events["timestamp"] <= end_time)
    ]
    if price_from is not None:
        filtered = filtered[filtered["price"] >= price_from]
    if price_to is not None:
        filtered = filtered[filtered["price"] <= price_to]
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
