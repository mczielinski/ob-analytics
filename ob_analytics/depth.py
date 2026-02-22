"""Order book depth computation and metrics.

Contains :class:`DepthMetricsEngine` for computing limit order book depth
metrics, along with :func:`price_level_volume`, :func:`filter_depth`,
:func:`depth_metrics` (backward-compatible wrapper), and :func:`get_spread`.
"""


from functools import lru_cache

import numpy as np
import pandas as pd

from ob_analytics._utils import interval_sum_breaks, validate_columns, validate_non_empty
from ob_analytics.config import PipelineConfig


@lru_cache(maxsize=256)
def _cached_breaks(range_len: int, bins: int) -> np.ndarray:
    """Compute bin boundaries for a given price range length."""
    breaks = ((np.arange(1, bins + 1) * range_len + bins - 1) // bins) - 1
    breaks[-1] = breaks[-1] - 1
    return breaks


class DepthMetricsEngine:
    """Incrementally compute order book depth metrics.

    Replaces the monolithic :func:`depth_metrics` function with a
    stateful, testable class.  Each call to :meth:`update` processes one
    depth event and returns a metrics row as a numpy array.

    Fixes over the legacy implementation:

    * **Dynamic price support** -- uses ``dict[int, int]`` instead of
      ``np.zeros(1_000_000)``, so any price range and precision works.
    * **Correct best-price tracking** -- ``min()``/``max()`` over active
      levels only (hundreds of entries) instead of scanning a 1M array.
    * **Numpy output** -- pre-allocates a numpy matrix and converts to
      DataFrame only at the end (5-10x faster than ``DataFrame.iloc``
      in the hot loop).
    * **Breaks caching** -- ``@lru_cache`` avoids recomputing bin
      boundaries every iteration.
    * **Correct ``best_bid_vol``** by default -- when a new higher bid
      arrives, ``best_bid_vol`` is set to the new volume.  Set
      ``compat_mode=True`` to replicate the R bug where it is left stale.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.
    compat_mode : bool
        If *True*, replicate the R package's ``best.bid.vol`` bug for
        parity testing.  Default *True* to avoid breaking existing
        comparisons.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        compat_mode: bool = True,
    ) -> None:
        self._config = config or PipelineConfig()
        self._compat_mode = compat_mode

        self._ask_levels: dict[int, int] = {}
        self._bid_levels: dict[int, int] = {}
        self._best_ask: int | None = None
        self._best_ask_vol: int = 0
        self._best_bid: int | None = None
        self._best_bid_vol: int = 0

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
            depth, {"timestamp", "price", "volume", "direction"}, "DepthMetricsEngine.compute"
        )
        validate_non_empty(depth, "DepthMetricsEngine.compute")

        multiplier = self._config.price_multiplier
        ordered = depth.sort_values(by="timestamp", kind="stable")

        prices_int = (multiplier * ordered["price"]).round().astype(int).values
        volumes = ordered["volume"].values
        sides = np.where(ordered["direction"].values == "bid", 0, 1)

        self._initialise_best(np.asarray(prices_int), np.asarray(sides))

        n = len(ordered)
        result = np.zeros((n, self._row_len), dtype=np.float64)

        for i in range(n):
            if i > 0:
                result[i] = result[i - 1]
            self.update(int(prices_int[i]), volumes[i], int(sides[i]), result[i])

        col_names = self._column_names()
        metrics = pd.DataFrame(result, columns=col_names)

        if "event_id" in ordered.columns:
            timestamps = ordered.reset_index(drop=True)[["timestamp", "event_id"]]
        else:
            timestamps = ordered.reset_index(drop=True)["timestamp"]
        res = pd.concat([timestamps, metrics], axis=1)

        price_cols = ["best_bid_price", "best_ask_price"]
        res[price_cols] = round(res[price_cols] / multiplier, self._config.price_decimals)

        return res

    def update(
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
            self._update_ask(price, volume, out)
        else:
            self._update_bid(price, volume, out)

    # ── Internal: ask side ────────────────────────────────────────────

    def _update_ask(self, price: int, volume: float, out: np.ndarray) -> None:
        if self._best_bid is not None and price <= self._best_bid:
            return

        int_vol = int(volume)
        if int_vol > 0:
            self._ask_levels[price] = int_vol
        elif price in self._ask_levels:
            del self._ask_levels[price]

        self._refresh_best_ask(price, int_vol)
        self._write_ask_metrics(out)

    def _refresh_best_ask(self, price: int, volume: int) -> None:
        if not self._ask_levels:
            self._best_ask = None
            self._best_ask_vol = 0
            return
        if volume > 0:
            if self._best_ask is None or price < self._best_ask:
                self._best_ask = price
                self._best_ask_vol = volume
            elif price == self._best_ask:
                self._best_ask_vol = volume
        elif self._best_ask is not None and price == self._best_ask:
            self._best_ask = min(self._ask_levels)
            self._best_ask_vol = self._ask_levels[self._best_ask]

    def _write_ask_metrics(self, out: np.ndarray) -> None:
        offset = 2 + self._bins
        if self._best_ask is None:
            out[offset] = 0
            out[offset + 1] = 0
            out[offset + 2: offset + 2 + self._bins] = 0
            return

        out[offset] = self._best_ask
        out[offset + 1] = self._best_ask_vol

        end_value = round((1 + self._bps * self._bins * 0.0001) * self._best_ask) + 1
        price_range = np.arange(self._best_ask, end_value + 1, 1, dtype=int)
        vol_array = np.array(
            [self._ask_levels.get(p, 0) for p in price_range], dtype=np.float64
        )
        breaks = self._compute_breaks(len(price_range))
        out[offset + 2: offset + 2 + self._bins] = interval_sum_breaks(vol_array, breaks)

    # ── Internal: bid side ────────────────────────────────────────────

    def _update_bid(self, price: int, volume: float, out: np.ndarray) -> None:
        if self._best_ask is not None and price >= self._best_ask:
            return

        int_vol = int(volume)
        if int_vol > 0:
            self._bid_levels[price] = int_vol
        elif price in self._bid_levels:
            del self._bid_levels[price]

        self._refresh_best_bid(price, int_vol)
        self._write_bid_metrics(out)

    def _refresh_best_bid(self, price: int, volume: int) -> None:
        if not self._bid_levels:
            self._best_bid = None
            self._best_bid_vol = 0
            return
        if volume > 0:
            if self._best_bid is None or price > self._best_bid:
                self._best_bid = price
                if not self._compat_mode:
                    self._best_bid_vol = volume
            elif price == self._best_bid:
                self._best_bid_vol = volume
        elif self._best_bid is not None and price == self._best_bid:
            self._best_bid = max(self._bid_levels)
            self._best_bid_vol = self._bid_levels[self._best_bid]

    def _write_bid_metrics(self, out: np.ndarray) -> None:
        if self._best_bid is None:
            out[0] = 0
            out[1] = 0
            out[2: 2 + self._bins] = 0
            return

        out[0] = self._best_bid
        out[1] = self._best_bid_vol

        end_value = round((1 - self._bps * self._bins * 0.0001) * self._best_bid)
        price_range = np.arange(self._best_bid, end_value - 1, -1, dtype=int)
        vol_array = np.array(
            [self._bid_levels.get(p, 0) for p in price_range], dtype=np.float64
        )
        breaks = self._compute_breaks(len(price_range))
        out[2: 2 + self._bins] = interval_sum_breaks(vol_array, breaks)

    # ── Helpers ───────────────────────────────────────────────────────

    def _initialise_best(self, prices: np.ndarray, sides: np.ndarray) -> None:
        """Set initial best bid/ask to match the legacy R behaviour."""
        ask_prices = prices[sides == 1]
        bid_prices = prices[sides == 0]
        if len(ask_prices) > 0:
            self._best_ask = int(ask_prices.max())
        if len(bid_prices) > 0:
            self._best_bid = int(bid_prices.min())

    def _compute_breaks(self, range_len: int) -> np.ndarray:
        return _cached_breaks(range_len, self._bins)

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
            "event_id", "id", "timestamp", "exchange_timestamp",
            "price", "volume", "direction", "action", "fill", "type",
        },
        "price_level_volume",
    )
    validate_non_empty(events, "price_level_volume")

    def directional_price_level_volume(dir_events: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "event_id", "id", "timestamp", "exchange_timestamp",
            "price", "volume", "direction", "action",
        ]

        added_volume = dir_events[
            (
                (dir_events["action"] == "created")
                | ((dir_events["action"] == "changed") & (dir_events["fill"] == 0))
            )
            & (dir_events["type"] != "pacman")
            & (dir_events["type"] != "market")
        ][cols]

        cancelled_volume = dir_events[
            (dir_events["action"] == "deleted")
            & (dir_events["volume"] > 0)
            & (dir_events["type"] != "pacman")
            & (dir_events["type"] != "market")
        ][cols]
        cancelled_volume["volume"] = -cancelled_volume["volume"]
        cancelled_volume = cancelled_volume[
            cancelled_volume["id"].isin(added_volume["id"])
        ]

        filled_volume = dir_events[
            (dir_events["fill"] > 0)
            & (dir_events["type"] != "pacman")
            & (dir_events["type"] != "market")
        ][
            [
                "event_id", "id", "timestamp", "exchange_timestamp",
                "price", "fill", "direction", "action",
            ]
        ]
        filled_volume["fill"] = -filled_volume["fill"]
        filled_volume = filled_volume[filled_volume["id"].isin(added_volume["id"])]
        filled_volume.columns = pd.Index(cols)

        volume_deltas = pd.concat([added_volume, cancelled_volume, filled_volume])
        volume_deltas = volume_deltas.sort_values(by=["price", "timestamp"], kind="stable")

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
    pre = pre[pre["volume"] > 0]

    if not pre.empty:
        pre["timestamp"] = pre["timestamp"].clip(lower=from_timestamp)  # type: ignore

    mid = d[(d["timestamp"] > from_timestamp) & (d["timestamp"] < to_timestamp)]
    range_combined = pd.concat([pre, mid])

    open_ends = range_combined.drop_duplicates(subset="price", keep="last")
    open_ends = open_ends[open_ends["volume"] > 0].copy()
    open_ends["timestamp"] = to_timestamp
    open_ends["volume"] = 0

    range_combined = pd.concat([range_combined, open_ends])
    range_combined = range_combined.sort_values(by=["price", "timestamp"], kind="stable")

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
    return DepthMetricsEngine(config, compat_mode=True).compute(depth)


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
        {"timestamp", "best_bid_price", "best_bid_vol", "best_ask_price", "best_ask_vol"},
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
