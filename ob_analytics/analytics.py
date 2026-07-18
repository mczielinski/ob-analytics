"""Format-agnostic post-processing analytics.

Contains functions that operate on the outputs of any pipeline run,
regardless of the originating data format (Bitstamp, LOBSTER, etc.):
aggressiveness, trade impacts, order type classification, and
point-in-time order book reconstruction.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ob_analytics._utils import validate_columns, validate_non_empty
from ob_analytics.depth import price_level_volume
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import FeedType


def _event_diff_bps(
    events: pd.DataFrame, depth_summary: pd.DataFrame, direction: int
) -> pd.DataFrame:
    """Per-event aggressiveness in BPS vs the contemporaneous best price.

    *direction* is ``1`` for bids, ``-1`` for asks. Helper for
    :func:`order_aggressiveness`.
    """
    side = "bid" if direction == 1 else "ask"
    orders = events[
        (events["direction"] == side)
        & (events["action"] != "changed")
        & events["type"].isin(["flashed-limit", "resting-limit"])
    ].sort_values(by="timestamp", kind="stable")

    missing = ~orders["timestamp"].isin(depth_summary["timestamp"])
    if missing.any():
        logger.debug(
            "order_aggressiveness: {}/{} {} order timestamps not in "
            "depth_summary (merge_asof will handle gracefully)",
            missing.sum(),
            len(orders),
            side,
        )

    best_price_col = f"best_{side}_price"

    depth_summary_sorted = depth_summary.sort_values("event_id")
    orders = orders.sort_values("event_id")

    merged = pd.merge_asof(
        orders,
        depth_summary_sorted[["event_id", best_price_col]],
        on="event_id",
        direction="backward",
        allow_exact_matches=False,
    )

    merged = merged.dropna(subset=[best_price_col]).copy()
    best = merged[best_price_col]

    diff_price = direction * (merged["price"] - best)
    diff_bps = 10000 * diff_price / best
    return pd.DataFrame({"event_id": merged["event_id"], "diff_bps": diff_bps})


def order_aggressiveness(
    events: pd.DataFrame, depth_summary: pd.DataFrame
) -> pd.DataFrame:
    """Calculate order aggressiveness with respect to the best bid or ask in BPS.

    Parameters
    ----------
    events : pandas.DataFrame
        The events DataFrame (must contain ``direction``, ``action``, ``type``,
        ``timestamp``, ``event_id``, ``price`` columns).
    depth_summary : pandas.DataFrame
        The order book summary statistics DataFrame (must contain ``timestamp``
        and ``event_id`` columns).

    Returns
    -------
    pandas.DataFrame
        The events DataFrame with an added ``aggressiveness_bps`` column.
    """
    validate_columns(
        events,
        {"direction", "action", "type", "timestamp", "event_id", "price"},
        "order_aggressiveness(events)",
    )
    validate_columns(
        depth_summary,
        {"timestamp"},
        "order_aggressiveness(depth_summary)",
    )

    bid_diff = _event_diff_bps(events, depth_summary, 1)
    ask_diff = _event_diff_bps(events, depth_summary, -1)
    # Work on a copy: the caller's frame must not grow columns as a side
    # effect (the merges below already produce new frames).
    events = events.copy()
    events["aggressiveness_bps"] = np.nan

    if not bid_diff.empty:
        events = pd.merge(events, bid_diff, on="event_id", how="left")
        events["aggressiveness_bps"] = events["aggressiveness_bps"].fillna(
            events["diff_bps"]
        )
        events.drop(columns=["diff_bps"], inplace=True)

    if not ask_diff.empty:
        events = pd.merge(events, ask_diff, on="event_id", how="left")
        events["aggressiveness_bps"] = events["aggressiveness_bps"].fillna(
            events["diff_bps"]
        )
        events.drop(columns=["diff_bps"], inplace=True)

    return events


def trade_impacts(trades: pd.DataFrame) -> pd.DataFrame:
    """Generate a DataFrame containing order book impact summaries.

    Aggregates trade records by taker order ID to summarise how each
    aggressive order swept through the book (price range, number of fills,
    total volume, VWAP, duration).

    Parameters
    ----------
    trades : pandas.DataFrame
        The trades DataFrame (must contain ``taker``, ``price``, ``volume``,
        ``timestamp``, ``direction`` columns).

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarising market order impacts with columns:
        ``id``, ``min_price``, ``max_price``, ``vwap``, ``hits``, ``vol``,
        ``start_time``, ``end_time``, ``dir``.
    """
    validate_columns(
        trades,
        {"taker", "price", "volume", "timestamp", "direction"},
        "trade_impacts",
    )
    validate_non_empty(trades, "trade_impacts")

    trades_pv = trades.assign(_pv=trades["price"] * trades["volume"])
    impacts = (
        trades_pv.groupby("taker")
        .agg(
            id=("taker", "last"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            hits=("taker", "size"),
            vol=("volume", "sum"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            dir=("direction", "last"),
            pv_sum=("_pv", "sum"),
        )
        .reset_index(drop=True)
    )
    impacts["vwap"] = impacts["pv_sum"] / impacts["vol"]
    cols = [
        "id",
        "min_price",
        "max_price",
        "vwap",
        "hits",
        "vol",
        "start_time",
        "end_time",
        "dir",
    ]
    return impacts[cols]


# ---------------------------------------------------------------------------
# Order type classification
# ---------------------------------------------------------------------------


def set_order_types(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Determine limit order types.

    Classifies each order as one of: *market*, *resting-limit*,
    *flashed-limit*, or *market-limit*, based on how the order interacts
    with the book over its lifetime.

    Parameters
    ----------
    events : pandas.DataFrame
        The limit order events DataFrame.
    trades : pandas.DataFrame
        The executions DataFrame.

    Returns
    -------
    pandas.DataFrame
        The events DataFrame with an updated 'type' column indicating order types.
    """
    validate_columns(
        events,
        {"id", "price", "action", "event_id", "direction"},
        "set_order_types(events)",
    )
    validate_columns(
        trades,
        {"maker_event_id", "taker_event_id"},
        "set_order_types(trades)",
    )
    validate_non_empty(events, "set_order_types")

    # Work on a copy: the caller's frame must not gain the 'type' column as
    # a side effect.
    events = events.copy()
    events["type"] = pd.Categorical(
        np.repeat("unknown", len(events)),
        categories=[
            "unknown",
            "pre-existing",
            "flashed-limit",
            "resting-limit",
            "market-limit",
            "market",
        ],
        ordered=True,
    )

    created = events[events["action"] == "created"].sort_values(by="id", kind="stable")
    deleted = events[events["action"] == "deleted"].sort_values(by="id", kind="stable")
    changed = events[events["action"] == "changed"]

    created_deleted_ids = created[
        created["id"].isin(deleted["id"]) & ~created["id"].isin(changed["id"])
    ]["id"]

    cd_created = created[created["id"].isin(created_deleted_ids)][["id", "volume"]]
    cd_deleted = deleted[deleted["id"].isin(created_deleted_ids)][["id", "volume"]]
    cd_merged = cd_created.merge(cd_deleted, on="id", suffixes=("_created", "_deleted"))
    flashed_ids = set(
        cd_merged.loc[cd_merged["volume_created"] == cd_merged["volume_deleted"], "id"]
    )
    forever_ids = set(
        created[
            ~created["id"].isin(changed["id"]) & ~created["id"].isin(deleted["id"])
        ]["id"]
    )

    maker_event_ids_set = set(trades["maker_event_id"].dropna())
    taker_event_ids_set = set(trades["taker_event_id"].dropna())

    maker_ids = set(events[events["event_id"].isin(maker_event_ids_set)]["id"])
    taker_ids = set(events[events["event_id"].isin(taker_event_ids_set)]["id"])

    pure_maker_ids = maker_ids - taker_ids
    ml_ids = taker_ids & maker_ids
    mo_ids = taker_ids - maker_ids

    events.loc[events["id"].isin(flashed_ids), "type"] = "flashed-limit"
    events.loc[events["id"].isin(forever_ids | pure_maker_ids), "type"] = (
        "resting-limit"
    )
    events.loc[events["id"].isin(ml_ids), "type"] = "market-limit"
    events.loc[events["id"].isin(mo_ids), "type"] = "market"

    # Orders first seen mid-stream (no created row): the pre-existing opening
    # book and hidden executions.  An explicit class instead of "unknown" —
    # they are structurally unclassifiable, not classification failures.
    # Trade-derived labels above take precedence.
    pre_ids = set(events["id"]) - set(created["id"])
    events.loc[events["id"].isin(pre_ids) & (events["type"] == "unknown"), "type"] = (
        "pre-existing"
    )

    unidentified = (events["type"] == "unknown").sum()
    if unidentified > 0:
        logger.warning("Could not identify {} orders", unidentified)

    return events


# ---------------------------------------------------------------------------
# Order lifecycles
# ---------------------------------------------------------------------------


def order_lifecycles(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse events into one row per order: placement → outcome.

    The canonical lifecycle table (one derivation, shared by the L3 faces
    and the order-book reconstruction).  Relies on the schemas.py volume
    contract: ``volume`` is the outstanding size after each event and
    ``fill`` the executed delta, so an order is *terminated* when a
    ``deleted`` row arrives **or its outstanding size reaches zero** —
    the latter is how fully-executed LOBSTER orders end, which never emit
    a ``deleted`` event.

    Parameters
    ----------
    events : pandas.DataFrame
        Events satisfying the schemas.py contract.  Orders without a
        ``created`` row (pre-existing book, hidden executions) are
        excluded — their placement is unknown.

    Returns
    -------
    pandas.DataFrame
        One row per order id:

        * ``id``, ``direction``, ``price`` — placement identity.
        * ``type`` — classifier label (when the column is present).
        * ``placed_ts``, ``placed_vol`` — from the ``created`` row.
        * ``filled_vol`` — total executed quantity (Σ ``fill``).
        * ``end_ts`` — termination time (``NaT`` while still resting;
          callers clip to their window end for display).
        * ``outcome`` — ``filled`` / ``partial`` / ``cancelled`` /
          ``resting``.  Flashed orders are the ``cancelled`` subset whose
          ``type`` is ``flashed-limit``.
        * ``aggressiveness_bps`` — placement distance (when present).
    """
    validate_columns(
        events,
        {"id", "timestamp", "price", "volume", "direction", "action", "fill"},
        "order_lifecycles",
    )
    validate_non_empty(events, "order_lifecycles")

    created = events[events["action"] == "created"]
    agg: dict[str, tuple[str, str]] = {
        "placed_ts": ("timestamp", "first"),
        "placed_vol": ("volume", "first"),
        "price": ("price", "first"),
        "direction": ("direction", "first"),
    }
    if "type" in events.columns:
        agg["type"] = ("type", "first")
    if "aggressiveness_bps" in events.columns:
        agg["aggressiveness_bps"] = ("aggressiveness_bps", "first")
    life = created.groupby("id", sort=False).agg(**agg)  # type: ignore[call-overload]

    life["filled_vol"] = (
        events.groupby("id", sort=False)["fill"].sum().reindex(life.index).fillna(0.0)
    )

    # Termination: explicit delete, or outstanding size exhausted (the
    # created row itself is excluded so zero-size placements don't
    # self-terminate).
    deleted_ts = (
        events.loc[events["action"] == "deleted"]
        .groupby("id", sort=False)["timestamp"]
        .min()
    )
    non_created = events[events["action"] != "created"]
    exhausted_ts = (
        non_created.loc[non_created["volume"] <= 0]
        .groupby("id", sort=False)["timestamp"]
        .min()
    )
    end = pd.concat([deleted_ts.rename("a"), exhausted_ts.rename("b")], axis=1).min(
        axis=1
    )
    life["end_ts"] = end.reindex(life.index)

    terminated = life["end_ts"].notna()
    placed = life["placed_vol"]
    filled = life["filled_vol"]
    # Bitstamp volumes are 8-dp floats; fills summed per order can drift by
    # float epsilon, so "fully executed" allows a vanishing tolerance.
    full = filled >= placed - 1e-9
    outcome = pd.Series("resting", index=life.index)
    outcome[terminated & full & (placed > 0)] = "filled"
    outcome[terminated & ~full & (filled > 0)] = "partial"
    outcome[terminated & (filled <= 0)] = "cancelled"
    life["outcome"] = outcome

    return life.reset_index()


# ---------------------------------------------------------------------------
# Order book reconstruction
# ---------------------------------------------------------------------------


def _active_bids(active_orders: pd.DataFrame) -> pd.DataFrame:
    """Bid side of *active_orders*: best-first, with bps + cumulative liquidity."""
    bids = active_orders[
        (active_orders["direction"] == "bid") & (active_orders["type"] != "market")
    ]
    bids = bids.sort_values(by=["price", "id"], ascending=[False, True], kind="stable")
    first_price = bids.iloc[0]["price"] if not bids.empty else np.nan
    bids["bps"] = (
        ((first_price - bids["price"]) / first_price) * 10000
        if not bids.empty
        else np.nan
    )
    bids["liquidity"] = bids["volume"].cumsum()
    return bids


def _active_asks(active_orders: pd.DataFrame) -> pd.DataFrame:
    """Ask side of *active_orders*: best-first, with bps + cumulative liquidity."""
    asks = active_orders[
        (active_orders["direction"] == "ask") & (active_orders["type"] != "market")
    ]
    asks = asks.sort_values(by=["price", "id"], ascending=[True, True], kind="stable")
    first_price = asks.iloc[0]["price"] if not asks.empty else np.nan
    asks["bps"] = (
        ((asks["price"] - first_price) / first_price) * 10000
        if not asks.empty
        else np.nan
    )
    asks["liquidity"] = asks["volume"].cumsum()
    return asks


def _crossed_prefix_counts(
    bid_prices: np.ndarray,
    bid_ts: np.ndarray,
    ask_prices: np.ndarray,
    ask_ts: np.ndarray,
) -> tuple[int, int]:
    """How many best-end bids / asks to evict to uncross two book sides.

    *bid_prices* descend from the best bid, *ask_prices* ascend from the best
    ask, each paired with its order ``timestamp``.  Walks the touch: while the
    top bid is priced at or above the top ask (crossed, or locked when equal),
    evict the older of the two touching orders — the static-snapshot analogue
    of :class:`~ob_analytics.depth.DepthMetricsEngine` trusting the fresher
    quote.  The evicted orders are exactly the contiguous best-end prefixes, so
    the two returned counts describe the eviction completely.
    """
    bi = 0
    ai = 0
    n_bid = bid_prices.size
    n_ask = ask_prices.size
    while bi < n_bid and ai < n_ask and bid_prices[bi] >= ask_prices[ai]:
        if bid_ts[bi] <= ask_ts[ai]:
            bi += 1
        else:
            ai += 1
    return bi, ai


def _uncross_active_orders(active_orders: pd.DataFrame) -> pd.DataFrame:
    """Drop crossed resting orders so a reconstructed snapshot is uncrossed.

    Static-snapshot mirror of the depth engine's crossed-level eviction (see
    :meth:`~ob_analytics.depth.DepthMetricsEngine.update_side`): at the crossed
    or locked touch, keep the fresher quote and evict the older opposing order,
    repeating until ``best_bid < best_ask``.  Recency uses ``timestamp`` (the
    receive clock the depth engine also processes in).  Market-type rows never
    rest on the book, so they are excluded from the crossing test and always
    retained.  The evicted rows are removed from *active_orders* with the index
    and every column preserved, so the caller's ``_active_bids`` /
    ``_active_asks`` recompute ``bps`` and ``liquidity`` against the surviving
    touch.
    """
    resting = active_orders[active_orders["type"] != "market"]
    bids = resting[resting["direction"] == "bid"].sort_values(
        by=["price", "timestamp"], ascending=[False, True], kind="stable"
    )
    asks = resting[resting["direction"] == "ask"].sort_values(
        by=["price", "timestamp"], ascending=[True, True], kind="stable"
    )
    n_bid, n_ask = _crossed_prefix_counts(
        bids["price"].to_numpy(),
        bids["timestamp"].to_numpy(),
        asks["price"].to_numpy(),
        asks["timestamp"].to_numpy(),
    )
    if n_bid == 0 and n_ask == 0:
        return active_orders
    evicted = bids.index[:n_bid].union(asks.index[:n_ask])
    return active_orders.drop(index=evicted)


def uncross_book_sides(
    bids: pd.DataFrame, asks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evict crossed levels from two reconstructed book sides *for display*.

    The frame-level counterpart of ``order_book(..., uncross=True)`` for
    callers that already hold per-order book sides — e.g. the ``book_snapshot``
    / ``depth_chart`` visualization prepares.  Both frames are returned
    best-first (bids by descending price, asks by ascending price) with the
    crossed best-end orders removed so ``best_bid < best_ask``; ``liquidity``
    is recomputed when present and every other column is preserved.

    Parameters
    ----------
    bids, asks : pandas.DataFrame
        Per-order book sides carrying at least ``price`` and ``timestamp``
        (as returned in :func:`order_book`'s ``"bids"`` / ``"asks"`` frames).

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        The uncrossed ``(bids, asks)`` sides, best-first.
    """
    if not bids.empty:
        bids = bids.sort_values("price", ascending=False, kind="stable")
    if not asks.empty:
        asks = asks.sort_values("price", ascending=True, kind="stable")
    if bids.empty or asks.empty:
        return bids, asks

    n_bid, n_ask = _crossed_prefix_counts(
        bids["price"].to_numpy(),
        bids["timestamp"].to_numpy(),
        asks["price"].to_numpy(),
        asks["timestamp"].to_numpy(),
    )
    bids = bids.iloc[n_bid:]
    asks = asks.iloc[n_ask:]
    if "liquidity" in bids.columns:
        bids = bids.assign(liquidity=bids["volume"].cumsum())
    if "liquidity" in asks.columns:
        asks = asks.assign(liquidity=asks["volume"].cumsum())
    return bids, asks


def order_book(
    events: pd.DataFrame,
    tp: datetime | None = None,
    max_levels: int | None = None,
    bps_range: int = 0,
    min_bid: float = 0,
    max_ask: float = np.inf,
    uncross: bool = False,
) -> dict[str, datetime | pd.Timestamp | pd.DataFrame]:
    """Reconstruct the order book at a specific point in time.

    Parameters
    ----------
    events : pandas.DataFrame
        DataFrame containing order events.
    tp : datetime.datetime or pandas.Timestamp, optional
        The point in time at which to evaluate the order book.
        If None, uses the latest event timestamp in the data.
    max_levels : int, optional
        The maximum number of price levels to include for bids and asks.
    bps_range : int, optional
        Basis points range to filter the bids and asks. Default is 0.
    min_bid : float, optional
        Minimum bid price. Default is 0.
    max_ask : float, optional
        Maximum ask price. Default is infinity.
    uncross : bool, optional
        When ``True``, evict crossed resting orders so the snapshot satisfies
        ``best_bid < best_ask`` — a *display* convenience mirroring the depth
        engine's crossed-level eviction (see :func:`_uncross_active_orders`).
        The default is ``False``: the reconstruction stays **faithful** to the
        feed, so a diff feed's genuinely crossed resting orders (see
        :class:`~ob_analytics.protocols.FeedType`) are replayed as-is rather
        than silently uncrossed. Has no effect on a matched-book feed, which is
        never crossed.

    Returns
    -------
    dict[str, datetime.datetime or pandas.DataFrame]
        A dictionary containing:
        - 'timestamp': The evaluation timestamp.
        - 'asks': DataFrame of active ask orders.
        - 'bids': DataFrame of active bid orders.
    """
    validate_columns(
        events,
        {
            "action",
            "timestamp",
            "id",
            "direction",
            "type",
            "price",
            "volume",
            "exchange_timestamp",
        },
        "order_book",
    )
    validate_non_empty(events, "order_book")

    if tp is None:
        tp = events["timestamp"].max()

    pct_range = bps_range * 0.0001

    # Active orders at *tp* under the canonical schema: an order rests on the
    # book iff it was submitted (has a ``created`` row), and its latest event
    # at or before *tp* is neither a delete nor left it exhausted.  The
    # outstanding-size check is what removes fully-executed LOBSTER orders,
    # which never emit a ``deleted`` event and previously lingered as
    # phantoms (89% of "active" orders on the AAPL sample, crossing the
    # book).  Rows are chronological within each id for every loader, so the
    # per-id tail is the latest state.
    win = events[events["timestamp"] <= tp]
    last_state = win.groupby("id", sort=False).tail(1)
    created_ids = win.loc[win["action"] == "created", "id"].unique()
    active_orders = last_state[
        last_state["id"].isin(created_ids)
        & (last_state["action"] != "deleted")
        & (last_state["volume"] > 0)
    ]

    if active_orders["id"].duplicated().any():
        raise ConfigError(
            "Duplicate order IDs found in active orders. "
            "This indicates a data integrity issue."
        )

    # Opt-in display uncrossing: evict crossed resting orders before deriving
    # the per-side frames, so bps/liquidity below anchor on the surviving
    # touch.  The default leaves the faithful (possibly crossed) book intact.
    if uncross:
        active_orders = _uncross_active_orders(active_orders)

    asks = _active_asks(active_orders)
    asks = asks[
        ["id", "timestamp", "exchange_timestamp", "price", "volume", "liquidity", "bps"]
    ]
    asks = asks.iloc[::-1].reset_index(drop=True)

    bids = _active_bids(active_orders)
    bids = bids[
        ["id", "timestamp", "exchange_timestamp", "price", "volume", "liquidity", "bps"]
    ]

    if pct_range > 0:
        if not asks.empty:
            max_ask_price = asks.iloc[-1]["price"] * (1 + pct_range)
            asks = asks[asks["price"] <= max_ask_price]
        if not bids.empty:
            min_bid_price = bids.iloc[0]["price"] * (1 - pct_range)
            bids = bids[bids["price"] >= min_bid_price]

    if max_levels is not None:
        asks = asks.tail(max_levels).reset_index(drop=True)
        bids = bids.head(max_levels).reset_index(drop=True)

    return {"timestamp": tp, "asks": asks, "bids": bids}


# ---------------------------------------------------------------------------
# Per-run data-quality summary
# ---------------------------------------------------------------------------


def _faithful_best_series(depth: pd.DataFrame) -> pd.DataFrame:
    """Faithful best bid / best ask over time from a price-level-volume frame.

    Unlike ``depth_summary`` — which the depth engine already *uncrosses* —
    this reads the raw resting-book depth from
    :func:`~ob_analytics.depth.price_level_volume` and reports the touch
    without eviction, so a diff feed's crossed intervals (``best_bid >
    best_ask``) survive to be measured.  One chronological pass; each side
    keeps a lazy-deletion heap, so the running best is amortised ``O(log L)``.

    Parameters
    ----------
    depth : pandas.DataFrame
        Columns ``timestamp``, ``price``, ``volume``, ``direction`` — the
        cumulative resting volume per price level (``0`` = level empty), as
        emitted by :func:`~ob_analytics.depth.price_level_volume`.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp``, ``best_bid``, ``best_ask`` (``NaN`` where the
        side is empty), one row per input row in timestamp order.
    """
    validate_columns(
        depth, {"timestamp", "price", "volume", "direction"}, "_faithful_best_series"
    )
    ordered = depth.sort_values("timestamp", kind="stable")
    ts = ordered["timestamp"].to_numpy()
    prices = ordered["price"].to_numpy(dtype=np.float64)
    vols = ordered["volume"].to_numpy(dtype=np.float64)
    is_bid = ordered["direction"].to_numpy() == "bid"

    n = len(ordered)
    best_bid = np.full(n, np.nan)
    best_ask = np.full(n, np.nan)

    bid_vol: dict[float, float] = {}
    ask_vol: dict[float, float] = {}
    bid_heap: list[float] = []  # max-heap by price, stored negated
    ask_heap: list[float] = []  # min-heap by price

    for i in range(n):
        p = float(prices[i])
        v = float(vols[i])
        if is_bid[i]:
            bid_vol[p] = v
            if v > 0:
                heapq.heappush(bid_heap, -p)
            while bid_heap and bid_vol.get(-bid_heap[0], 0.0) <= 0:
                heapq.heappop(bid_heap)
        else:
            ask_vol[p] = v
            if v > 0:
                heapq.heappush(ask_heap, p)
            while ask_heap and ask_vol.get(ask_heap[0], 0.0) <= 0:
                heapq.heappop(ask_heap)
        if bid_heap:
            best_bid[i] = -bid_heap[0]
        if ask_heap:
            best_ask[i] = ask_heap[0]

    return pd.DataFrame({"timestamp": ts, "best_bid": best_bid, "best_ask": best_ask})


def _crossed_time_fraction(best: pd.DataFrame) -> tuple[float, int]:
    """Time-weighted crossed fraction (0–1) and crossed-episode count.

    A state is *crossed* when both sides exist and ``best_bid > best_ask``.
    The book holds each row's state over the interval to the next row, so the
    fraction is duration-weighted (matching the "book was crossed for ~90 s"
    reading) rather than row-weighted.  Episodes count contiguous crossed
    intervals (rising edges).
    """
    ts = best["timestamp"].to_numpy()
    bid = best["best_bid"].to_numpy()
    ask = best["best_ask"].to_numpy()
    crossed = (~np.isnan(bid)) & (~np.isnan(ask)) & (bid > ask)
    if crossed.size == 0:
        return 0.0, 0

    episodes = int(crossed[0]) + int(np.sum(crossed[1:] & ~crossed[:-1]))
    if crossed.size < 2:
        return (1.0 if crossed[0] else 0.0), episodes

    dt = np.diff(ts) / np.timedelta64(1, "ns")  # interval lengths in ns
    total = float(dt.sum())
    if total <= 0:  # all events share a timestamp -> fall back to row share
        return float(crossed.mean()), episodes
    crossed_time = float(dt[crossed[:-1]].sum())
    return crossed_time / total, episodes


@dataclass(frozen=True)
class DataQualitySummary:
    """Per-run data-quality metrics for a reconstructed session.

    Built by :func:`data_quality_summary`.  All percentages are 0–100 floats.

    Attributes
    ----------
    feed_type : FeedType
        The source's declared crossing invariant (see
        :class:`~ob_analytics.protocols.FeedType`); sets expectations for
        ``crossed_pct``.
    n_events, n_orders, n_trades : int
        Row / distinct-order / trade counts.
    crossed_pct : float
        Percentage of session *time* the faithful book is crossed
        (``best_bid > best_ask``).  Expected ``~0`` for a matched book; a
        genuine, faithfully-replayed property of a diff feed.
    crossed_episodes : int
        Number of distinct crossed intervals.
    unmatched_trades_pct : float
        Percentage of trades missing a resolved ``maker_event_id`` or
        ``taker_event_id`` (could not be tied to a resting order).
    duplicate_event_ids : int
        Count of ``event_id`` values occurring more than once (``event_id``
        should be globally unique — any non-zero value is suspect).
    duplicate_created_ids : int
        Count of order ids with more than one ``created`` event.
    pre_existing_orders : int
        Distinct orders resting before the capture window (classifier label
        ``pre-existing`` — structurally unclassifiable, not failures).
    """

    feed_type: FeedType
    n_events: int
    n_orders: int
    n_trades: int
    crossed_pct: float
    crossed_episodes: int
    unmatched_trades_pct: float
    duplicate_event_ids: int
    duplicate_created_ids: int
    pre_existing_orders: int

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a plain, JSON-serialisable dict."""
        return {
            "feed_type": str(self.feed_type.value),
            "n_events": self.n_events,
            "n_orders": self.n_orders,
            "n_trades": self.n_trades,
            "crossed_pct": self.crossed_pct,
            "crossed_episodes": self.crossed_episodes,
            "unmatched_trades_pct": self.unmatched_trades_pct,
            "duplicate_event_ids": self.duplicate_event_ids,
            "duplicate_created_ids": self.duplicate_created_ids,
            "pre_existing_orders": self.pre_existing_orders,
        }

    def _crossed_note(self) -> str:
        """One-line reading of ``crossed_pct`` given the feed type."""
        if self.feed_type == FeedType.MATCHED_BOOK:
            return (
                "as expected for a matched book"
                if self.crossed_pct <= 0.05
                else "UNEXPECTED for a matched book — check reconstruction/data"
            )
        if self.feed_type == FeedType.DIFF_FEED:
            return "expected for a diff feed — faithful replay, not a bug"
        return "feed type undeclared"

    def render(self) -> str:
        """Return a fixed-width, human-readable report block."""
        lines = [
            "Data quality summary",
            f"  feed type             : {self.feed_type.value}",
            f"  events / orders       : {self.n_events:,} / {self.n_orders:,}",
            f"  trades                : {self.n_trades:,}",
            f"  crossed resting book  : {self.crossed_pct:.2f}% of session "
            f"({self.crossed_episodes} episode(s)) [{self._crossed_note()}]",
            f"  unmatched trades      : {self.unmatched_trades_pct:.2f}%",
            f"  duplicate event ids   : {self.duplicate_event_ids}",
            f"  duplicate created ids : {self.duplicate_created_ids}",
            f"  pre-existing orders   : {self.pre_existing_orders}",
        ]
        return "\n".join(lines)


def data_quality_summary(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    feed_type: FeedType = FeedType.UNKNOWN,
    depth: pd.DataFrame | None = None,
) -> DataQualitySummary:
    """Summarise the data quality of one reconstructed session.

    Surfaces the health signals that matter before trusting a feed — most
    importantly how crossed the resting book is, which distinguishes a matched
    book from a diff feed (see :class:`~ob_analytics.protocols.FeedType`).

    Parameters
    ----------
    events : pandas.DataFrame
        Classified events (must carry the canonical columns **and** the
        ``type`` column from :func:`set_order_types`).
    trades : pandas.DataFrame
        The trades frame, with ``maker_event_id`` / ``taker_event_id``.
    feed_type : FeedType, optional
        The source's declared feed type, recorded on the summary and used to
        interpret ``crossed_pct``.  Read it off the format:
        ``getattr(fmt, "feed_type", FeedType.UNKNOWN)``.
    depth : pandas.DataFrame, optional
        A faithful price-level-volume frame (e.g. ``PipelineResult.depth``).
        When ``None`` it is computed from *events* via
        :func:`~ob_analytics.depth.price_level_volume`.  **Do not** pass
        ``depth_summary`` — that is already uncrossed and would report ~0%.

    Returns
    -------
    DataQualitySummary
    """
    validate_columns(
        events,
        {"event_id", "id", "action", "direction", "price", "type"},
        "data_quality_summary(events)",
    )
    validate_non_empty(events, "data_quality_summary")
    validate_columns(
        trades,
        {"maker_event_id", "taker_event_id"},
        "data_quality_summary(trades)",
    )

    if depth is None:
        depth = price_level_volume(events)

    if depth.empty:
        crossed_pct, crossed_episodes = 0.0, 0
    else:
        crossed_frac, crossed_episodes = _crossed_time_fraction(
            _faithful_best_series(depth)
        )
        crossed_pct = 100.0 * crossed_frac

    n_trades = len(trades)
    if n_trades:
        unmatched = (
            trades["maker_event_id"].isna() | trades["taker_event_id"].isna()
        ).sum()
        unmatched_pct = 100.0 * float(unmatched) / n_trades
    else:
        unmatched_pct = 0.0

    event_id_counts = events["event_id"].value_counts()
    duplicate_event_ids = int((event_id_counts > 1).sum())

    created_ids = events.loc[events["action"] == "created", "id"]
    created_counts = created_ids.value_counts()
    duplicate_created_ids = int((created_counts > 1).sum())

    pre_existing_orders = int(
        events.loc[events["type"] == "pre-existing", "id"].nunique()
    )

    return DataQualitySummary(
        feed_type=feed_type,
        n_events=len(events),
        n_orders=int(events["id"].nunique()),
        n_trades=n_trades,
        crossed_pct=crossed_pct,
        crossed_episodes=crossed_episodes,
        unmatched_trades_pct=unmatched_pct,
        duplicate_event_ids=duplicate_event_ids,
        duplicate_created_ids=duplicate_created_ids,
        pre_existing_orders=pre_existing_orders,
    )
