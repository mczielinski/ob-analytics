"""Order book depth computation and metrics.

Contains :class:`DepthMetricsEngine` for computing limit order book depth
metrics, along with :func:`price_level_volume`, :func:`filter_depth`,
:func:`depth_metrics` (backward-compatible wrapper), and :func:`get_spread`.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from ob_analytics._utils import (
    validate_columns,
    validate_non_empty,
)
from ob_analytics.config import PipelineConfig


@lru_cache(maxsize=256)
def _cached_breaks(range_len: int, bins: int) -> np.ndarray:
    """Compute bin boundaries for a given price range length."""
    breaks = ((np.arange(1, bins + 1) * range_len + bins - 1) // bins) - 1
    breaks[-1] = breaks[-1] - 1
    return breaks


def _interval_sums_sorted(
    idxs: np.ndarray,
    vols: np.ndarray,
    range_len: int,
    breaks: np.ndarray,
) -> np.ndarray:
    """Bin per-level volumes whose offsets *idxs* are sorted ascending.

    Contract: byte-identical to ``np.diff`` over ``np.cumsum(dense)[breaks]``,
    where ``dense`` is the length-``range_len`` array with ``dense[idxs] =
    vols`` and zeros everywhere else.  Because volumes are non-negative and
    ``x + 0.0 == x`` for finite ``x``, accumulating the active levels in
    ascending-offset order reproduces ``cumsum(dense)`` at every index
    bit-for-bit; the prefix sums are sampled at ``breaks`` and differenced.

    *idxs* must already be clipped to ``[0, range_len)``.
    """
    bins = len(breaks)
    if idxs.size == 0:
        return np.zeros(bins, dtype=np.float64)

    prefix = np.cumsum(vols)

    # prefix[j] equals the dense cumulative sum at the j-th active offset.
    # Negative break indices address from the end, matching the numpy fancy
    # indexing ``cs[breaks]`` of the dense reference.
    breaks_norm = np.where(breaks < 0, range_len + breaks, breaks)
    pos = np.searchsorted(idxs, breaks_norm, side="right") - 1
    intervals = np.where(pos >= 0, prefix[np.clip(pos, 0, prefix.size - 1)], 0.0)

    return np.concatenate(([intervals[0]], np.diff(intervals)))


def _interval_sums_sparse(
    levels: dict[int, float],
    best: int,
    side: int,
    range_len: int,
    breaks: np.ndarray,
) -> np.ndarray:
    """Sum active book volume into the bins delimited by *breaks*.

    Dict adapter over :func:`_interval_sums_sorted`, kept for the dense
    cumsum oracle test; the engine hot path holds price-sorted arrays and
    calls the core directly.
    """
    idx_list: list[int] = []
    vol_list: list[float] = []
    for p, v in levels.items():
        idx = (p - best) if side == 1 else (best - p)
        if 0 <= idx < range_len:
            idx_list.append(idx)
            vol_list.append(v)

    order = np.argsort(idx_list, kind="stable")
    idxs = np.asarray(idx_list)[order]
    vols = np.asarray(vol_list, dtype=np.float64)[order]
    return _interval_sums_sorted(idxs, vols, range_len, breaks)


class DepthMetricsEngine:
    """Incrementally compute order book depth metrics.

    Replaces the monolithic :func:`depth_metrics` function with a
    stateful, testable class.  :meth:`compute` processes a whole depth
    frame; internally each event is applied via :meth:`update_side`,
    which writes one metrics row into a pre-allocated numpy buffer.

    Each book side is held as a pair of parallel numpy arrays sorted
    ascending by integer price: best lookup is O(1) (asks at index 0, bids
    at index -1), membership is O(log L) via ``searchsorted``, crossed-level
    eviction is a contiguous slice, and BPS-bin sums vectorize over the
    in-window slice instead of iterating every active level in Python
    (levels average ~1.8k per side on the bundled sample, making that
    iteration the pipeline's former hot loop).

    Output is written into a pre-allocated numpy matrix and converted to a
    DataFrame once at the end; bin boundaries are ``@lru_cache``-d.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:
        self._config = config or PipelineConfig()

        self._ask_prices: np.ndarray = np.empty(0, dtype=np.int64)
        self._ask_vols: np.ndarray = np.empty(0, dtype=np.float64)
        self._bid_prices: np.ndarray = np.empty(0, dtype=np.int64)
        self._bid_vols: np.ndarray = np.empty(0, dtype=np.float64)

        self._bps = self._config.depth_bps
        self._bins = self._config.depth_bins
        self._row_len = 2 * (2 + self._bins)

    # ── Diagnostic views (state lives in the sorted arrays) ──────────

    @property
    def _ask_levels(self) -> dict[int, float]:
        """Active ask levels as ``{price: volume}`` (diagnostic view)."""
        return dict(zip(self._ask_prices.tolist(), self._ask_vols.tolist()))

    @property
    def _bid_levels(self) -> dict[int, float]:
        """Active bid levels as ``{price: volume}`` (diagnostic view)."""
        return dict(zip(self._bid_prices.tolist(), self._bid_vols.tolist()))

    @property
    def _best_ask(self) -> int | None:
        return int(self._ask_prices[0]) if self._ask_prices.size else None

    @property
    def _best_ask_vol(self) -> float:
        return float(self._ask_vols[0]) if self._ask_vols.size else 0.0

    @property
    def _best_bid(self) -> int | None:
        return int(self._bid_prices[-1]) if self._bid_prices.size else None

    @property
    def _best_bid_vol(self) -> float:
        return float(self._bid_vols[-1]) if self._bid_vols.size else 0.0

    def compute(self, depth: pd.DataFrame) -> pd.DataFrame:
        """Process an entire depth DataFrame and return metrics.

        This is the main entry point, equivalent to the legacy
        :func:`depth_metrics` function.

        Parameters
        ----------
        depth : pandas.DataFrame
            Price-level volume data with columns ``timestamp``,
            ``price``, ``volume``, ``direction``.

        Returns
        -------
        pandas.DataFrame
            Depth summary with ``timestamp``, ``best_bid_price``,
            ``best_bid_vol``, ``best_ask_price``, ``best_ask_vol``,
            and volume-in-BPS-bin columns (e.g. ``bid_vol25bps``).
        """
        validate_columns(
            depth,
            {"timestamp", "price", "volume", "direction"},
            "DepthMetricsEngine.compute",
        )
        validate_non_empty(depth, "DepthMetricsEngine.compute")

        multiplier = self._config.price_multiplier
        ordered = depth.sort_values(by="timestamp", kind="stable")

        prices_int = (multiplier * ordered["price"]).round().astype(int).values
        volumes = ordered["volume"].values
        sides = np.where(ordered["direction"].values == "bid", 0, 1)

        n = len(ordered)
        result = np.zeros((n, self._row_len), dtype=np.float64)

        for i in range(n):
            if i > 0:
                result[i] = result[i - 1]
            self.update_side(int(prices_int[i]), volumes[i], int(sides[i]), result[i])

        col_names = self._column_names()
        metrics = pd.DataFrame(result, columns=col_names)

        if "event_id" in ordered.columns:
            timestamps = ordered.reset_index(drop=True)[["timestamp", "event_id"]]
        else:
            timestamps = ordered.reset_index(drop=True)["timestamp"]
        res = pd.concat([timestamps, metrics], axis=1)

        price_cols = ["best_bid_price", "best_ask_price"]
        res[price_cols] = round(
            res[price_cols] / multiplier, self._config.price_decimals
        )

        return res

    def update_side(
        self, price: int, volume: float, side: int, out: np.ndarray
    ) -> None:
        """Process one depth event and write a metrics row into *out*.

        Parameters
        ----------
        price : int
            Price in integer units (e.g. cents).
        volume : float
            Volume at this price level (0 means deletion).
        side : int
            0 = bid, 1 = ask.
        out : np.ndarray
            Pre-allocated 1-D array of length ``row_len`` to fill.
        """
        if side == 1:
            prices, vols = self._ask_prices, self._ask_vols
        else:
            prices, vols = self._bid_prices, self._bid_vols

        evicted = False
        if volume > 0:
            i = int(np.searchsorted(prices, price))
            if i < prices.size and prices[i] == price:
                vols[i] = volume
            else:
                prices = np.insert(prices, i, price)
                vols = np.insert(vols, i, volume)
            # A resting bid and ask coexist only when bid_price < ask_price.
            # Trust the fresh quote: evict any *stale* opposing levels it
            # strictly crosses (e.g. an orphaned best whose delete event is
            # missing from the feed).  Equal-price touches are left intact,
            # so genuine locked books are still tolerated.  Crossed levels
            # are contiguous at the opposing array's best end, so eviction
            # is a single slice.
            if side == 1:
                opp_p, opp_v = self._bid_prices, self._bid_vols
                if opp_p.size and opp_p[-1] > price:
                    # new ask -> bids strictly above it are crossed
                    k = int(np.searchsorted(opp_p, price, side="right"))
                    self._bid_prices = opp_p[:k].copy()
                    self._bid_vols = opp_v[:k].copy()
                    evicted = True
            else:
                opp_p, opp_v = self._ask_prices, self._ask_vols
                if opp_p.size and opp_p[0] < price:
                    # new bid -> asks strictly below it are crossed
                    k = int(np.searchsorted(opp_p, price, side="left"))
                    self._ask_prices = opp_p[k:].copy()
                    self._ask_vols = opp_v[k:].copy()
                    evicted = True
        else:
            i = int(np.searchsorted(prices, price))
            if i < prices.size and prices[i] == price:
                prices = np.delete(prices, i)
                vols = np.delete(vols, i)

        if side == 1:
            self._ask_prices, self._ask_vols = prices, vols
        else:
            self._bid_prices, self._bid_vols = prices, vols

        if evicted:
            # Eviction mutated the opposing book; emit its metrics too,
            # otherwise compute() carries the stale opposing columns over.
            self._write_side_metrics(1 - side, out)
        self._write_side_metrics(side, out)

    def _write_side_metrics(self, side: int, out: np.ndarray) -> None:
        if side == 1:
            offset = 2 + self._bins
            prices, vols = self._ask_prices, self._ask_vols
        else:
            offset = 0
            prices, vols = self._bid_prices, self._bid_vols

        if not prices.size:
            out[offset] = 0
            out[offset + 1] = 0
            out[offset + 2 : offset + 2 + self._bins] = 0
            return

        best = int(prices[0]) if side == 1 else int(prices[-1])
        out[offset] = best
        out[offset + 1] = vols[0] if side == 1 else vols[-1]

        # Window covered by the BPS bins, mirroring the legacy code:
        #   ask: arange(best_ask, end_value + 1) inclusive ascending.
        #   bid: arange(best_bid, end_value - 1, -1) inclusive descending.
        # The in-window levels are a contiguous slice of the sorted arrays;
        # offsets are handed to the binning core in ascending order (bids
        # reversed), reproducing the dense cumsum accumulation order.
        if side == 1:
            end_value = round((1 + self._bps * self._bins * 0.0001) * best) + 1
            range_len = end_value - best + 1
            k = int(np.searchsorted(prices, best + range_len, side="left"))
            idxs = prices[:k] - best
            win_vols = vols[:k]
        else:
            end_value = round((1 - self._bps * self._bins * 0.0001) * best)
            range_len = best - end_value + 1
            j = int(np.searchsorted(prices, best - range_len, side="right"))
            idxs = (best - prices[j:])[::-1]
            win_vols = vols[j:][::-1]

        breaks = _cached_breaks(range_len, self._bins)
        out[offset + 2 : offset + 2 + self._bins] = _interval_sums_sorted(
            idxs, win_vols, range_len, breaks
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _column_names(self) -> list[str]:
        bps, bins = self._bps, self._bins

        def pct_names(name: str) -> list[str]:
            return [f"{name}{i}bps" for i in range(bps, bps * bins + 1, bps)]

        return (
            ["best_bid_price", "best_bid_vol"]
            + pct_names("bid_vol")
            + ["best_ask_price", "best_ask_vol"]
            + pct_names("ask_vol")
        )


# ── Standalone functions ──────────────────────────────────────────────


def price_level_volume(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate the cumulative volume for each price level over time.

    Parameters
    ----------
    events : pandas.DataFrame
        A pandas DataFrame containing limit order events.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with the cumulative volume for each price level.
    """
    validate_columns(
        events,
        {
            "event_id",
            "id",
            "timestamp",
            "exchange_timestamp",
            "price",
            "volume",
            "direction",
            "action",
            "fill",
            "type",
        },
        "price_level_volume",
    )
    validate_non_empty(events, "price_level_volume")

    def directional_price_level_volume(dir_events: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "event_id",
            "id",
            "timestamp",
            "exchange_timestamp",
            "price",
            "volume",
            "direction",
            "action",
        ]

        added_volume = dir_events[
            (dir_events["action"] == "created") & (dir_events["type"] != "market")
        ][cols]

        cancelled_volume = dir_events[
            (dir_events["action"] == "deleted")
            & (dir_events["volume"] > 0)
            & (dir_events["type"] != "market")
        ][cols]
        cancelled_volume["volume"] = -cancelled_volume["volume"]
        cancelled_volume = cancelled_volume[
            cancelled_volume["id"].isin(added_volume["id"])
        ]

        filled_volume = dir_events[
            (dir_events["fill"] > 0) & (dir_events["type"] != "market")
        ][
            [
                "event_id",
                "id",
                "timestamp",
                "exchange_timestamp",
                "price",
                "fill",
                "direction",
                "action",
            ]
        ]
        filled_volume["fill"] = -filled_volume["fill"]
        filled_volume = filled_volume[filled_volume["id"].isin(added_volume["id"])]
        filled_volume.columns = pd.Index(cols)

        # Cancel-reductions: changed rows that shrink the order's outstanding
        # size without an execution (LOBSTER partial cancels).  Bitstamp has
        # none by construction — `fill` covers every Bitstamp volume drop —
        # so this frame is empty there.  The drop is read off the canonical
        # outstanding-size column (schemas.py).
        outstanding_drop = (
            dir_events.groupby("id")["volume"].shift() - dir_events["volume"]
        )
        reduced_volume = dir_events[
            (dir_events["action"] == "changed")
            & (dir_events["fill"] == 0)
            & (outstanding_drop > 0)
            & (dir_events["type"] != "market")
        ][cols].copy()
        if not reduced_volume.empty:
            reduced_volume["volume"] = -outstanding_drop[reduced_volume.index]
            reduced_volume = reduced_volume[
                reduced_volume["id"].isin(added_volume["id"])
            ]

        volume_deltas = pd.concat(
            [added_volume, cancelled_volume, filled_volume, reduced_volume]
        )
        volume_deltas = volume_deltas.sort_values(
            by=["price", "timestamp"], kind="stable"
        )

        volume_deltas["volume"] = volume_deltas.groupby("price")["volume"].cumsum()
        volume_deltas["volume"] = volume_deltas["volume"].clip(lower=0)

        return volume_deltas[["event_id", "timestamp", "price", "volume", "direction"]]

    bids = events[events["direction"] == "bid"]
    depth_bid = directional_price_level_volume(bids)
    asks = events[events["direction"] == "ask"]
    depth_ask = directional_price_level_volume(asks)
    depth_data = pd.concat([depth_bid, depth_ask])
    return depth_data.sort_values(by="timestamp", kind="stable")


def filter_depth(
    d: pd.DataFrame, from_timestamp: pd.Timestamp, to_timestamp: pd.Timestamp
) -> pd.DataFrame:
    """Filter depth data within a specified time range.

    Parameters
    ----------
    d : pandas.DataFrame
        DataFrame containing depth data.
    from_timestamp : pandas.Timestamp
        Start of the time range.
    to_timestamp : pandas.Timestamp
        End of the time range.

    Returns
    -------
    pandas.DataFrame
        Filtered depth data within the specified time range.
    """
    validate_columns(d, {"timestamp", "price", "volume"}, "filter_depth")

    pre = d[d["timestamp"] <= from_timestamp]
    pre = pre.sort_values(by=["price", "timestamp"], kind="stable")

    pre = pre.drop_duplicates(subset="price", keep="last")
    pre = pre[pre["volume"] > 0].copy()

    if not pre.empty:
        pre.loc[:, "timestamp"] = pre["timestamp"].where(
            pre["timestamp"] >= from_timestamp, from_timestamp
        )

    mid = d[(d["timestamp"] > from_timestamp) & (d["timestamp"] < to_timestamp)]
    range_combined = pd.concat([pre, mid])

    open_ends = range_combined.drop_duplicates(subset="price", keep="last")
    open_ends = open_ends[open_ends["volume"] > 0].copy()
    open_ends["timestamp"] = to_timestamp
    open_ends["volume"] = 0

    range_combined = pd.concat([range_combined, open_ends])
    range_combined = range_combined.sort_values(
        by=["price", "timestamp"], kind="stable"
    )

    return range_combined


def depth_metrics(depth: pd.DataFrame, bps: int = 25, bins: int = 20) -> pd.DataFrame:
    """Compute limit order book depth metrics.

    This is a convenience wrapper around :class:`DepthMetricsEngine`.

    Parameters
    ----------
    depth : pandas.DataFrame
        DataFrame containing depth data.
    bps : int, optional
        Basis points increment for volume bins. Default is 25.
    bins : int, optional
        Number of bins to use for volume aggregation. Default is 20.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing depth metrics over time.
    """
    config = PipelineConfig(depth_bps=bps, depth_bins=bins)
    return DepthMetricsEngine(config).compute(depth)


def get_spread(depth_summary: pd.DataFrame) -> pd.DataFrame:
    """Extract the bid/ask spread from the depth summary.

    Parameters
    ----------
    depth_summary : pandas.DataFrame
        A pandas DataFrame containing depth summary statistics.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with the bid/ask spread data.
    """
    validate_columns(
        depth_summary,
        {
            "timestamp",
            "best_bid_price",
            "best_bid_vol",
            "best_ask_price",
            "best_ask_vol",
        },
        "get_spread",
    )

    spread = depth_summary[
        [
            "timestamp",
            "best_bid_price",
            "best_bid_vol",
            "best_ask_price",
            "best_ask_vol",
        ]
    ]
    changes = (
        spread[
            ["best_bid_price", "best_bid_vol", "best_ask_price", "best_ask_vol"]
        ].diff()
        != 0
    ).any(axis=1)
    return spread[changes]
