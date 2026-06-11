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


def _interval_sums_sparse(
    levels: dict[int, float],
    best: int,
    side: int,
    range_len: int,
    breaks: np.ndarray,
) -> np.ndarray:
    """Sum active book volume into the bins delimited by *breaks*.

    Contract: byte-identical to ``np.diff`` over ``np.cumsum(dense)[breaks]``,
    where ``dense`` is the length-``range_len`` array holding each active
    level's volume at ``idx = price - best`` (asks) / ``best - price`` (bids)
    and zeros everywhere else.  Because volumes are non-negative and
    ``x + 0.0 == x`` for finite ``x``, accumulating only the active levels in
    ascending-``idx`` order reproduces ``cumsum(dense)`` at every index
    bit-for-bit; the prefix sums are sampled at ``breaks`` and differenced.

    Iterating the whole levels dict per call is the pipeline's hot loop
    (levels average ~1.8k entries per side on the bundled sample); the
    vectorized rework is roadmap §2.1 (docs/plans/).
    """
    idxs: list[int] = []
    vols: list[float] = []
    for p, v in levels.items():
        idx = (p - best) if side == 1 else (best - p)
        if 0 <= idx < range_len:
            idxs.append(idx)
            vols.append(v)

    bins = len(breaks)
    if not idxs:
        return np.zeros(bins, dtype=np.float64)

    order = np.argsort(idxs, kind="stable")
    idxs_sorted = np.asarray(idxs)[order]
    prefix = np.cumsum(np.asarray(vols, dtype=np.float64)[order])

    # cs[j] over the dense array equals the prefix sum of active vols with
    # idx <= j.  Negative break indices address from the end, matching the
    # numpy fancy indexing ``cs[breaks]`` they replace.
    breaks_norm = np.where(breaks < 0, range_len + breaks, breaks)
    pos = np.searchsorted(idxs_sorted, breaks_norm, side="right") - 1
    intervals = np.where(pos >= 0, prefix[np.clip(pos, 0, len(prefix) - 1)], 0.0)

    return np.concatenate(([intervals[0]], np.diff(intervals)))


class DepthMetricsEngine:
    """Incrementally compute order book depth metrics.

    Replaces the monolithic :func:`depth_metrics` function with a
    stateful, testable class.  :meth:`compute` processes a whole depth
    frame; internally each event is applied via :meth:`update_side`,
    which writes one metrics row into a pre-allocated numpy buffer.

    Fixes over the legacy R implementation:

    * **Dynamic price support** -- uses ``dict[int, float]`` instead of
      ``np.zeros(1_000_000)``, so any price range and precision works.
    * **Correct best-price initialisation** -- ``min()`` for asks and
      ``max()`` for bids (the R code had them inverted).
    * **Correct best-price tracking** -- ``min()``/``max()`` over active
      levels only (hundreds of entries) instead of scanning a 1M array.
    * **Numpy output** -- pre-allocates a numpy matrix and converts to
      DataFrame only at the end (5-10x faster than ``DataFrame.iloc``
      in the hot loop).
    * **Breaks caching** -- ``@lru_cache`` avoids recomputing bin
      boundaries every iteration.
    * **Correct ``best_bid_vol``** -- when a new higher bid arrives,
      ``best_bid_vol`` is set to the new volume.

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

        self._ask_levels: dict[int, float] = {}
        self._bid_levels: dict[int, float] = {}
        self._best_ask: int | None = None
        self._best_ask_vol: float = 0.0
        self._best_bid: int | None = None
        self._best_bid_vol: float = 0.0

        self._bps = self._config.depth_bps
        self._bins = self._config.depth_bins
        self._row_len = 2 * (2 + self._bins)

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
            levels = self._ask_levels
            opposing = self._bid_levels
        else:
            levels = self._bid_levels
            opposing = self._ask_levels

        evicted = False
        if volume > 0:
            levels[price] = volume
            # A resting bid and ask coexist only when bid_price < ask_price.
            # Trust the fresh quote: evict any *stale* opposing levels it
            # strictly crosses (e.g. an orphaned best whose delete event is
            # missing from the feed).  Equal-price touches are left intact,
            # so genuine locked books are still tolerated.  The cheap best-vs-
            # price comparison keeps the common, non-crossing event O(1); the
            # full scan only runs when an actual cross is present.
            if side == 1 and self._best_bid is not None and self._best_bid > price:
                # new ask -> bids strictly above it are crossed
                for op in [op for op in opposing if op > price]:
                    del opposing[op]
                evicted = True
            elif side == 0 and self._best_ask is not None and self._best_ask < price:
                # new bid -> asks strictly below it are crossed
                for op in [op for op in opposing if op < price]:
                    del opposing[op]
                evicted = True
        elif price in levels:
            del levels[price]

        self._refresh_best(side, price, volume)
        if evicted:
            # Eviction mutated the opposing book; refresh and emit its metrics
            # too, otherwise compute() carries the stale opposing columns over.
            self._recompute_best(1 - side)
            self._write_side_metrics(1 - side, out)
        self._write_side_metrics(side, out)

    def _refresh_best(self, side: int, price: int, volume: float) -> None:
        if side == 1:
            levels = self._ask_levels
            current_best = self._best_ask
        else:
            levels = self._bid_levels
            current_best = self._best_bid

        if not levels:
            if side == 1:
                self._best_ask = None
                self._best_ask_vol = 0.0
            else:
                self._best_bid = None
                self._best_bid_vol = 0.0
            return

        new_best: int | None = current_best
        new_vol: float | None = None

        if volume > 0:
            better = (
                ((price < current_best) if side == 1 else (price > current_best))
                if current_best is not None
                else True
            )
            if better:
                new_best = price
                new_vol = volume
            elif price == current_best:
                new_vol = volume
        elif current_best is not None and price == current_best:
            new_best = min(levels) if side == 1 else max(levels)
            new_vol = levels[new_best]

        if new_vol is None:
            return

        if side == 1:
            self._best_ask = new_best
            self._best_ask_vol = new_vol
        else:
            self._best_bid = new_best
            self._best_bid_vol = new_vol

    def _recompute_best(self, side: int) -> None:
        """Recompute a side's best price/volume from its active levels.

        Used after a bulk eviction of crossed opposing levels, where the
        single-level :meth:`_refresh_best` fast-path does not apply.
        """
        if side == 1:
            if self._ask_levels:
                self._best_ask = min(self._ask_levels)
                self._best_ask_vol = self._ask_levels[self._best_ask]
            else:
                self._best_ask = None
                self._best_ask_vol = 0.0
        else:
            if self._bid_levels:
                self._best_bid = max(self._bid_levels)
                self._best_bid_vol = self._bid_levels[self._best_bid]
            else:
                self._best_bid = None
                self._best_bid_vol = 0.0

    def _write_side_metrics(self, side: int, out: np.ndarray) -> None:
        if side == 1:
            offset = 2 + self._bins
            best = self._best_ask
            best_vol = self._best_ask_vol
            levels = self._ask_levels
        else:
            offset = 0
            best = self._best_bid
            best_vol = self._best_bid_vol
            levels = self._bid_levels

        if best is None:
            out[offset] = 0
            out[offset + 1] = 0
            out[offset + 2 : offset + 2 + self._bins] = 0
            return

        out[offset] = best
        out[offset + 1] = best_vol

        # Window covered by the BPS bins, mirroring the legacy code:
        #   ask: arange(best_ask, end_value + 1) inclusive ascending.
        #   bid: arange(best_bid, end_value - 1, -1) inclusive descending.
        if side == 1:
            end_value = round((1 + self._bps * self._bins * 0.0001) * best) + 1
            range_len = end_value - best + 1
        else:
            end_value = round((1 - self._bps * self._bins * 0.0001) * best)
            range_len = best - end_value + 1

        # Sum the active price levels (~1.8k per side on the bundled sample)
        # straight into the BPS bins, without materialising the full,
        # mostly-zero integer-price window (potentially O(price * bps * bins)).
        breaks = _cached_breaks(range_len, self._bins)
        out[offset + 2 : offset + 2 + self._bins] = _interval_sums_sparse(
            levels, best, side, range_len, breaks
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
            (
                (dir_events["action"] == "created")
                | ((dir_events["action"] == "changed") & (dir_events["fill"] == 0))
            )
            & (dir_events["type"] != "market")
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

        volume_deltas = pd.concat([added_volume, cancelled_volume, filled_volume])
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
