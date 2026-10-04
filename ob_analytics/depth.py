"""Order book depth computation and metrics.

Contains :class:`PriceLevelBook`, the book at L2 kept one depth row at a
time; :class:`DepthMetricsEngine` for computing limit order book depth
metrics on it, along with :func:`price_level_volume`, :func:`filter_depth`,
:func:`depth_metrics` (backward-compatible wrapper), :func:`get_spread`, and
:func:`price_level_snapshots`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ob_analytics import _engine_frames
from ob_analytics._utils import (
    validate_columns,
    validate_non_empty,
)
from ob_analytics.config import PipelineConfig
from ob_analytics.schemas import time_order_keys

if TYPE_CHECKING:
    from ob_analytics.analytics import OrderBookSnapshot


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


class PriceLevelBook:
    """The book at L2: the size resting at each price level, kept one row at a time.

    Each side is a pair of parallel numpy arrays sorted ascending by price, so
    the best ask is the first entry of its side and the best bid the last.
    Lookups are a ``searchsorted``, and evicting crossed levels is one slice.

    :meth:`update` applies one row of the depth table: a positive volume sets
    the size at that price, and zero removes the price level.  A positive
    update also evicts every opposing price level it strictly crosses.  The
    fresh quote is trusted over the older opposing level, whose delete is most
    likely missing from the feed.  Equal prices are left alone, so a locked
    book is kept.

    :class:`DepthMetricsEngine` builds the depth summary on this class, so a
    book kept here holds the depth summary's touch after every row.

    Parameters
    ----------
    price_dtype : numpy dtype, optional
        The dtype prices are held in: ``int64`` ticks (the default), or
        ``float64`` for a depth table already converted to the quote currency.

    Attributes
    ----------
    bid_prices, bid_volumes : numpy.ndarray
        The bid side, ascending by price.
    ask_prices, ask_volumes : numpy.ndarray
        The ask side, ascending by price.
    """

    __slots__ = ("ask_prices", "ask_volumes", "bid_prices", "bid_volumes")

    def __init__(self, price_dtype: type | np.dtype = np.int64) -> None:
        self.ask_prices: np.ndarray = np.empty(0, dtype=price_dtype)
        self.ask_volumes: np.ndarray = np.empty(0, dtype=np.float64)
        self.bid_prices: np.ndarray = np.empty(0, dtype=price_dtype)
        self.bid_volumes: np.ndarray = np.empty(0, dtype=np.float64)

    def update(self, price: float, volume: float, side: int) -> bool:
        """Apply one depth row and return whether it evicted opposing levels.

        Parameters
        ----------
        price : int or float
            The price level, in the book's price units.
        volume : float
            The size now resting at *price*; ``0`` removes the level.
        side : int
            ``0`` for a bid, ``1`` for an ask (the codes of
            :class:`ob_analytics.engine.Direction`).

        Returns
        -------
        bool
            ``True`` when the update evicted crossed levels from the other
            side, so that side changed too.
        """
        if side == 1:
            prices, vols = self.ask_prices, self.ask_volumes
        else:
            prices, vols = self.bid_prices, self.bid_volumes

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
                opp_p, opp_v = self.bid_prices, self.bid_volumes
                if opp_p.size and opp_p[-1] > price:
                    # new ask -> bids strictly above it are crossed
                    k = int(np.searchsorted(opp_p, price, side="right"))
                    self.bid_prices = opp_p[:k].copy()
                    self.bid_volumes = opp_v[:k].copy()
                    evicted = True
            else:
                opp_p, opp_v = self.ask_prices, self.ask_volumes
                if opp_p.size and opp_p[0] < price:
                    # new bid -> asks strictly below it are crossed
                    k = int(np.searchsorted(opp_p, price, side="left"))
                    self.ask_prices = opp_p[k:].copy()
                    self.ask_volumes = opp_v[k:].copy()
                    evicted = True
        else:
            i = int(np.searchsorted(prices, price))
            if i < prices.size and prices[i] == price:
                prices = np.delete(prices, i)
                vols = np.delete(vols, i)

        if side == 1:
            self.ask_prices, self.ask_volumes = prices, vols
        else:
            self.bid_prices, self.bid_volumes = prices, vols
        return evicted


class DepthMetricsEngine:
    """Incrementally compute order book depth metrics.

    Replaces the monolithic :func:`depth_metrics` function with a
    stateful, testable class.  :meth:`compute` processes a whole depth
    frame; internally each event is applied via :meth:`update_side`,
    which writes one metrics row into a pre-allocated numpy buffer.

    The book itself is a :class:`PriceLevelBook`, which holds each side as
    parallel numpy arrays sorted ascending by integer price.  BPS-bin sums
    vectorize over the in-window slice of those arrays instead of iterating
    every active level in Python (levels average ~1.8k per side on the
    bundled sample, making that iteration the pipeline's former hot loop).

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

        self._book = PriceLevelBook()

        self._bps = self._config.depth_bps
        self._bins = self._config.depth_bins
        self._row_len = 2 * (2 + self._bins)

    # ── Diagnostic views (state lives in the PriceLevelBook) ─────────

    @property
    def _ask_levels(self) -> dict[int, float]:
        """Active ask levels as ``{price: volume}`` (diagnostic view)."""
        book = self._book
        return dict(zip(book.ask_prices.tolist(), book.ask_volumes.tolist()))

    @property
    def _bid_levels(self) -> dict[int, float]:
        """Active bid levels as ``{price: volume}`` (diagnostic view)."""
        book = self._book
        return dict(zip(book.bid_prices.tolist(), book.bid_volumes.tolist()))

    @property
    def _best_ask(self) -> int | None:
        prices = self._book.ask_prices
        return int(prices[0]) if prices.size else None

    @property
    def _best_ask_vol(self) -> float:
        vols = self._book.ask_volumes
        return float(vols[0]) if vols.size else 0.0

    @property
    def _best_bid(self) -> int | None:
        prices = self._book.bid_prices
        return int(prices[-1]) if prices.size else None

    @property
    def _best_bid_vol(self) -> float:
        vols = self._book.bid_volumes
        return float(vols[-1]) if vols.size else 0.0

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

        # Replay in the canonical event order (``schemas.time_order_keys``:
        # timestamp, then sequence / event_id / ingest_seq where present), the
        # order the per-order reconstructions use.  Each output row is then the
        # book after the event it names, which is what a reader looking for
        # "the book just before this event" relies on.  Sorting on the
        # timestamp alone replayed one instant's rows bids first, then asks,
        # by price.  The crossed-level eviction in ``PriceLevelBook.update``
        # depends on replay order, so the two orders can end an instant on
        # different books.  A frame with no tie-break column (a hand-built
        # frame, or an L2 frame loaded without ``track_sequence``) keeps its
        # own order within an instant, as before.
        ordered = depth.sort_values(by=time_order_keys(depth), kind="stable")

        # Price is already an integer tick count (issue #155), so the engine
        # bins and compares levels on exact integers — no multiply-and-round.
        # A float column (a legacy or hand-built frame) is read as tick-valued
        # and rounded to the nearest integer tick.
        price_values = ordered["price"].to_numpy()
        if np.issubdtype(price_values.dtype, np.floating):
            prices_int = np.rint(price_values).astype(np.int64)
        else:
            prices_int = price_values.astype(np.int64)
        volumes = ordered["volume"].values
        sides = np.where(ordered["direction"].values == "bid", 0, 1)

        n = len(ordered)
        result = np.zeros((n, self._row_len), dtype=np.float64)

        # The first row starts from the book the engine already holds, so a
        # second call carries on from where the first left off.  Each row
        # writes only the side it changes, and a fresh engine's empty book
        # writes zeros, so a single call is unaffected.
        self._write_side_metrics(0, result[0])
        self._write_side_metrics(1, result[0])
        for i in range(n):
            if i > 0:
                result[i] = result[i - 1]
            self.update_side(int(prices_int[i]), volumes[i], int(sides[i]), result[i])

        col_names = self._column_names()

        # Which columns are exact counts rather than measurements: the two best
        # prices are always tick counts, and every volume column is a lot count
        # when the input carried lot counts.  A frame still holding float sizes
        # in the base asset -- one built by hand, or read from a file written
        # before sizes were integers -- is left alone, because rounding it onto
        # a lot grid would truncate a sub-lot size such as 0.5 to zero and lose
        # the level rather than report it.
        exact = {"best_bid_price", "best_ask_price"}
        if np.issubdtype(np.asarray(volumes).dtype, np.integer):
            exact |= {name for name in col_names if "vol" in name}

        # Each column is converted as it is lifted out of the row buffer, which
        # is one float64 matrix because it also holds the scale-free bps
        # columns.  Slicing the assembled frame and casting that instead would
        # hold a float copy and an int copy of every volume column at the same
        # time -- about 100 MiB on a 300k-event session, and the reason the
        # scale benchmark guards this.
        columns: dict[str, np.ndarray] = {
            name: (
                result[:, index].astype(np.int64)
                if name in exact
                else result[:, index].copy()
            )
            for index, name in enumerate(col_names)
        }
        del result
        metrics = pd.DataFrame(columns, copy=False)

        if "event_id" in ordered.columns:
            timestamps = ordered.reset_index(drop=True)[["timestamp", "event_id"]]
        else:
            timestamps = ordered.reset_index(drop=True)["timestamp"]
        return pd.concat([timestamps, metrics], axis=1)

    def update_side(
        self, price: int, volume: float, side: int, out: np.ndarray
    ) -> None:
        """Process one depth event and write a metrics row into *out*.

        Parameters
        ----------
        price : int
            Price as an integer tick count.
        volume : float
            Volume at this price level (0 means deletion).
        side : int
            0 = bid, 1 = ask.
        out : np.ndarray
            Pre-allocated 1-D array of length ``row_len`` to fill.
        """
        evicted = self._book.update(price, volume, side)
        if evicted:
            # Eviction mutated the opposing book; emit its metrics too,
            # otherwise compute() carries the stale opposing columns over.
            self._write_side_metrics(1 - side, out)
        self._write_side_metrics(side, out)

    def _write_side_metrics(self, side: int, out: np.ndarray) -> None:
        if side == 1:
            offset = 2 + self._bins
            prices, vols = self._book.ask_prices, self._book.ask_volumes
        else:
            offset = 0
            prices, vols = self._book.bid_prices, self._book.bid_volumes

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
        A pandas DataFrame with the cumulative volume for each price level:
        one row per change to a level, holding the level's volume after the
        event named by ``event_id``.  Rows are in the canonical event order
        (:func:`~ob_analytics.schemas.time_order_keys`), and the ``sequence``
        and ``ingest_seq`` columns are carried over from *events* when it has
        them, so the depth rows sort the same way the events do.
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

    # The tie-break columns of the event order other than ``event_id``.  They
    # travel with each row so the depth rows can be put in the same order as
    # the events.
    order_keys = [
        k for k in time_order_keys(events) if k not in ("timestamp", "event_id")
    ]

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
            *order_keys,
        ]

        added_volume = dir_events[
            (dir_events["action"] == "created") & (dir_events["type"] != "market")
        ][cols]

        # The level each order's volume actually sits on.  It is set by the
        # order's first row -- its `created` row, whenever the feed delivers
        # rows in order -- and moved only by a `changed` row that reports no
        # execution; every other row is subtracted at the level the order sits
        # on rather than at whatever price the row itself carries, so an
        # order's `+v` and `-v` always cancel on one level and a level can only
        # empty to exactly zero.
        #
        # The two can differ.  Bitstamp reports a `deleted` whose price is not
        # the price the order rested at for 1.3% of orders, and reports an
        # execution at the price it traded at, which need not be the order's
        # own.  Subtracting at the reported price strands the volume on the
        # resting level for the rest of the session, where it is read back as a
        # resting level that no order is on.  A `changed` row with no execution
        # and a new price is different: the order really has moved (Databento's
        # modify), so its volume leaves the old level and joins the new one.
        # The per-order rebuild (`engine.book_state`) tracks orders by id and
        # reads each one at the price of its latest row; this keeps the
        # price-level rebuild consistent with it.
        order_key = dir_events["id"]
        sets_level = ~order_key.duplicated() | (
            (dir_events["action"] == "changed") & (dir_events["fill"] == 0)
        )
        resting_price = (
            dir_events["price"]
            .where(sets_level)
            .groupby(order_key)
            .ffill()
            .astype(dir_events["price"].dtype)
        )
        previous_price = resting_price.groupby(order_key).shift()
        previous_volume = dir_events.groupby(order_key)["volume"].shift()
        # A move or a growth can only happen to an order that is still on the
        # book: one that was submitted here and whose previous row did not
        # delete it.  A stray row after the delete must not bring it back.
        amendable = (
            (dir_events["action"] == "changed")
            & (dir_events["fill"] == 0)
            & previous_price.notna()
            & (dir_events.groupby(order_key)["action"].shift() != "deleted")
            & (dir_events["type"] != "market")
            & dir_events["id"].isin(added_volume["id"])
        )
        moved = amendable & (resting_price != previous_price)

        cancelled_volume = dir_events[
            (dir_events["action"] == "deleted")
            & (dir_events["volume"] > 0)
            & (dir_events["type"] != "market")
        ][cols].copy()
        cancelled_volume["price"] = resting_price[cancelled_volume.index]
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
                *order_keys,
            ]
        ]
        filled_volume = filled_volume.copy()
        filled_volume["price"] = resting_price[filled_volume.index]
        filled_volume["fill"] = -filled_volume["fill"]
        filled_volume = filled_volume[filled_volume["id"].isin(added_volume["id"])]
        filled_volume.columns = pd.Index(cols)

        # Cancel-reductions: changed rows that shrink the order's outstanding
        # size without an execution (LOBSTER partial cancels).  Bitstamp has
        # none by construction — `fill` covers every Bitstamp volume drop —
        # so this frame is empty there.  The drop is read off the canonical
        # outstanding-size column (schemas.py).  A move is left out: its size
        # change is part of the move below.
        outstanding_drop = previous_volume - dir_events["volume"]
        resized = (
            (dir_events["action"] == "changed")
            & (dir_events["fill"] == 0)
            & ~moved
            & (dir_events["type"] != "market")
        )
        reduced_volume = dir_events[resized & (outstanding_drop > 0)][cols].copy()
        if not reduced_volume.empty:
            reduced_volume["price"] = resting_price[reduced_volume.index]
            reduced_volume["volume"] = -outstanding_drop[reduced_volume.index]
            reduced_volume = reduced_volume[
                reduced_volume["id"].isin(added_volume["id"])
            ]

        # Growth and moves are reported only by a venue that amends orders in
        # place.  Neither LOBSTER nor Bitstamp does, so these frames are
        # empty there and are left out of the concatenation altogether, which
        # keeps the output for those feeds exactly as it was.
        volume_dtype = dir_events["volume"].dtype
        grown_volume = dir_events[amendable & ~moved & (outstanding_drop < 0)][
            cols
        ].copy()
        grown_volume["price"] = resting_price[grown_volume.index]
        grown_volume["volume"] = (-outstanding_drop[grown_volume.index]).astype(
            volume_dtype
        )

        # A move takes the order's previous outstanding size off the level it
        # left and puts its new size on the level it joined.
        left_volume = dir_events[moved & (previous_volume > 0)][cols].copy()
        left_volume["price"] = previous_price[left_volume.index].astype(
            resting_price.dtype
        )
        left_volume["volume"] = (-previous_volume[left_volume.index]).astype(
            volume_dtype
        )
        joined_volume = dir_events[moved & (dir_events["volume"] > 0)][cols]

        amended = [
            frame
            for frame in (grown_volume, left_volume, joined_volume)
            if not frame.empty
        ]
        volume_deltas = pd.concat(
            [added_volume, cancelled_volume, filled_volume, reduced_volume, *amended]
        )
        # Each level's running total is taken in event order, so every row
        # holds the level's volume after its own event.  Sorting on timestamp
        # alone would leave the changes one instant makes to a level in the
        # order the frames were joined above, and a replay in event order
        # would then end the instant on the wrong volume.
        volume_deltas = volume_deltas.sort_values(
            by=["price", *time_order_keys(volume_deltas)], kind="stable"
        )

        volume_deltas["volume"] = volume_deltas.groupby("price")["volume"].cumsum()
        volume_deltas["volume"] = volume_deltas["volume"].clip(lower=0)

        return volume_deltas[
            ["event_id", "timestamp", "price", "volume", "direction", *order_keys]
        ]

    bids = events[events["direction"] == "bid"]
    depth_bid = directional_price_level_volume(bids)
    asks = events[events["direction"] == "ask"]
    depth_ask = directional_price_level_volume(asks)
    depth_data = pd.concat([depth_bid, depth_ask])
    return depth_data.sort_values(by=time_order_keys(depth_data), kind="stable")


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


def price_level_snapshots(
    depth: pd.DataFrame,
    times: Iterable[datetime | pd.Timestamp],
    max_levels: int | None = None,
) -> list[OrderBookSnapshot]:
    """Return the book at L2 at each of *times*, replayed from the depth table.

    One pass over *depth* drives a :class:`PriceLevelBook`, the book
    :class:`DepthMetricsEngine` builds the depth summary on, and copies it out
    at each instant.  So the touch of each snapshot equals the depth summary's
    last row at or before that instant, and a stale level that a fresher
    opposing quote crossed is evicted here as it is there.  This is the L2
    counterpart of :func:`ob_analytics.analytics.order_book`, which rebuilds
    the per-order book at one instant from the events.

    Parameters
    ----------
    depth : pandas.DataFrame
        The depth table: ``timestamp``, ``price``, ``volume`` and
        ``direction``.  Prices are integer ticks, or floats in the quote
        currency when the table was converted for display; both are kept as
        given.
    times : iterable of datetime.datetime or pandas.Timestamp
        The instants to take a snapshot at, in any order.  A row counts at an
        instant when its timestamp is at or before it.
    max_levels : int, optional
        Keep only this many price levels per side, nearest the touch first.
        ``None`` keeps every level.

    Returns
    -------
    list of OrderBookSnapshot
        One per entry of *times*, in the same order.  Each has the shape
        :func:`~ob_analytics.analytics.order_book` returns, with one row per
        price level: ``bids`` best first and ``asks`` best last, each with
        ``price``, ``volume`` and ``liquidity`` (cumulative volume from the
        touch).

    Raises
    ------
    TypeError
        If an instant and the depth timestamps disagree about being
        time-zone aware.
    """
    validate_columns(
        depth, {"timestamp", "price", "volume", "direction"}, "price_level_snapshots"
    )
    instants = [_engine_frames.instant_ns(t, like=depth["timestamp"]) for t in times]

    # The same playback order as DepthMetricsEngine.compute: the canonical
    # event order, so the book at the end of an instant is the one the depth
    # summary ends that instant on.
    ordered = depth.sort_values(by=time_order_keys(depth), kind="stable")
    stamps = _engine_frames.nanoseconds(ordered["timestamp"])
    price_values = ordered["price"].to_numpy()
    volume_values = ordered["volume"].to_numpy()
    volume_dtype = volume_values.dtype
    is_float = np.issubdtype(price_values.dtype, np.floating)
    book = PriceLevelBook(price_dtype=np.float64 if is_float else np.int64)
    prices = price_values.tolist()
    volumes = volume_values.tolist()
    sides = (ordered["direction"].to_numpy() != "bid").astype(int).tolist()

    def side_frame(side_prices: np.ndarray, side_volumes: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "price": side_prices,
                "volume": side_volumes.astype(volume_dtype),
                "liquidity": np.cumsum(side_volumes).astype(volume_dtype),
            }
        )

    snapshots: list[OrderBookSnapshot | None] = [None] * len(instants)
    row = 0
    for k in np.argsort(instants, kind="stable"):
        stop = int(np.searchsorted(stamps, instants[k], side="right"))
        for i in range(row, stop):
            book.update(prices[i], volumes[i], sides[i])
        row = max(row, stop)
        # Both sides best first; a slice of None keeps every level.
        bids = side_frame(
            book.bid_prices[::-1][:max_levels], book.bid_volumes[::-1][:max_levels]
        )
        asks = side_frame(book.ask_prices[:max_levels], book.ask_volumes[:max_levels])
        snapshots[k] = {
            "timestamp": _engine_frames.timestamps(
                np.array([instants[k]]), like=depth["timestamp"]
            )[0],
            "bids": bids,
            # order_book's convention: asks best last.
            "asks": asks.iloc[::-1].reset_index(drop=True),
        }
    return [s for s in snapshots if s is not None]


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


# ── Fair-value / pressure signals ─────────────────────────────────────
#
# Two standard microstructure signals derived from the depth summary the
# engine already produces: the size-weighted mid (micro-price) and the
# order-book imbalance (OBI).  Both read the best bid/ask price and volume,
# and OBI can also cumulate the per-bps depth-bin volume columns.


def bin_volume_columns(depth_summary: pd.DataFrame, side: str) -> list[str]:
    """Return the per-bps depth-bin volume columns for *side*, touch outward.

    These are the ``{side}_vol{N}bps`` aggregates written by
    :class:`DepthMetricsEngine` -- each the resting volume in one bps ring out
    from the touch.  They are discovered from the frame and ordered by their
    bps distance rather than assumed, so a summary built with a non-default
    ``bps``/``bins`` configuration is handled correctly.

    Parameters
    ----------
    depth_summary : pandas.DataFrame
        Depth summary produced by :func:`depth_metrics`.
    side : str
        ``"bid"`` or ``"ask"``.

    Returns
    -------
    list of str
        Matching column names, ordered from the touch outward.
    """
    pattern = re.compile(rf"^{side}_vol(\d+)bps$")
    matched: list[tuple[int, str]] = []
    for column in depth_summary.columns:
        found = pattern.match(column)
        if found is not None:
            matched.append((int(found.group(1)), column))
    return [column for _, column in sorted(matched)]


def micro_price(
    depth_summary: pd.DataFrame,
    *,
    stoikov_adjustment: pd.Series | np.ndarray | float | None = None,
) -> pd.Series:
    """Size-weighted mid price (the micro-price) at the touch.

    The micro-price weights each best quote by the size resting on the
    *opposite* side::

        micro = (best_bid_price * best_ask_vol + best_ask_price * best_bid_vol)
                / (best_bid_vol + best_ask_vol)

    so it leans toward the side carrying the heavier opposite book -- the
    direction price is more likely to move.  It is equivalent to the plain mid
    plus a spread-scaled touch imbalance::

        micro = mid + (spread / 2) * (best_bid_vol - best_ask_vol)
                                     / (best_bid_vol + best_ask_vol)

    Stoikov (2018), "The micro-price: A high frequency estimator of future
    prices", refines this by replacing the linear imbalance term with a
    correction ``g(imbalance, spread)`` fitted from data.  Fitting that
    correction is outside this function: pass a fitted per-row correction as
    *stoikov_adjustment* to add it to the weighted mid and obtain the refined
    estimator.  The default (``None``) returns the plain weighted mid, which is
    the zeroth-order Stoikov estimator.

    Parameters
    ----------
    depth_summary : pandas.DataFrame
        Depth summary with ``best_bid_price``, ``best_bid_vol``,
        ``best_ask_price`` and ``best_ask_vol`` columns.
    stoikov_adjustment : pandas.Series, numpy.ndarray, float or None, optional
        Optional additive Stoikov correction (see above).  Default ``None``.

    Returns
    -------
    pandas.Series
        The micro-price per row, indexed like *depth_summary*.  Rows where
        ``best_bid_vol + best_ask_vol == 0`` are ``NaN`` rather than a
        divide-by-zero error.
    """
    validate_columns(
        depth_summary,
        {"best_bid_price", "best_bid_vol", "best_ask_price", "best_ask_vol"},
        "micro_price",
    )

    bid_price = depth_summary["best_bid_price"].to_numpy(dtype=float)
    ask_price = depth_summary["best_ask_price"].to_numpy(dtype=float)
    bid_vol = depth_summary["best_bid_vol"].to_numpy(dtype=float)
    ask_vol = depth_summary["best_ask_vol"].to_numpy(dtype=float)

    denominator = bid_vol + ask_vol
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted = np.where(
            denominator > 0,
            (bid_price * ask_vol + ask_price * bid_vol) / denominator,
            np.nan,
        )

    if stoikov_adjustment is not None:
        weighted = weighted + np.asarray(stoikov_adjustment, dtype=float)

    return pd.Series(weighted, index=depth_summary.index, name="micro_price")


def book_imbalance(depth_summary: pd.DataFrame, levels: int = 1) -> pd.Series:
    """Order-book imbalance (OBI) from resting volume.

    OBI is the signed share of resting volume on the bid side::

        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)

    ranging from ``-1`` (all volume on the ask) to ``+1`` (all on the bid);
    ``0`` is a balanced book.

    Parameters
    ----------
    depth_summary : pandas.DataFrame
        Depth summary produced by :func:`depth_metrics`.
    levels : int, optional
        Depth over which to measure the imbalance.  ``1`` (default) uses the
        touch only -- ``best_bid_vol`` and ``best_ask_vol``.  ``levels > 1``
        cumulates resting volume over the first ``levels - 1`` bps depth bins
        (the ``{side}_vol{N}bps`` columns, which already include the touch), so
        the measured window grows monotonically with *levels*.

    Returns
    -------
    pandas.Series
        OBI per row, indexed like *depth_summary*.  Rows whose total volume is
        zero are ``NaN`` rather than a divide-by-zero error.

    Raises
    ------
    ValueError
        If *levels* is below ``1`` or exceeds the available depth bins.
    """
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    if levels == 1:
        validate_columns(
            depth_summary, {"best_bid_vol", "best_ask_vol"}, "book_imbalance"
        )
        bid_vol = depth_summary["best_bid_vol"].to_numpy(dtype=float)
        ask_vol = depth_summary["best_ask_vol"].to_numpy(dtype=float)
    else:
        bid_cols = bin_volume_columns(depth_summary, "bid")
        ask_cols = bin_volume_columns(depth_summary, "ask")
        take = levels - 1
        available = min(len(bid_cols), len(ask_cols))
        if take > available:
            raise ValueError(
                f"levels={levels} needs {take} depth bins, but the summary has "
                f"{available}"
            )
        bid_vol = depth_summary[bid_cols[:take]].to_numpy(dtype=float).sum(axis=1)
        ask_vol = depth_summary[ask_cols[:take]].to_numpy(dtype=float).sum(axis=1)

    denominator = bid_vol + ask_vol
    with np.errstate(invalid="ignore", divide="ignore"):
        imbalance = np.where(denominator > 0, (bid_vol - ask_vol) / denominator, np.nan)

    return pd.Series(imbalance, index=depth_summary.index, name="book_imbalance")


def depth_signals(
    depth_summary: pd.DataFrame,
    *,
    depth_levels: int = 5,
    stoikov_adjustment: pd.Series | np.ndarray | float | None = None,
) -> pd.DataFrame:
    """Append fair-value and pressure signal columns to a depth summary.

    Returns a *copy* of *depth_summary* with four columns added; existing
    columns are left untouched, so current consumers keep working.

    ``mid_price``
        Plain mid ``(best_bid_price + best_ask_price) / 2``.
    ``micro_price``
        Size-weighted mid from :func:`micro_price`.
    ``obi``
        Touch order-book imbalance -- :func:`book_imbalance` with ``levels=1``.
    ``obi_depth``
        Cumulative-depth order-book imbalance over *depth_levels* --
        :func:`book_imbalance` with ``levels=depth_levels``.

    Parameters
    ----------
    depth_summary : pandas.DataFrame
        Depth summary produced by :func:`depth_metrics`.
    depth_levels : int, optional
        Depth for the cumulative ``obi_depth`` column.  Default ``5`` (the
        touch plus the first four bps depth bins).  Clamped to the number of
        depth bins actually present, so a summary with fewer bins does not
        raise.
    stoikov_adjustment : pandas.Series, numpy.ndarray, float or None, optional
        Forwarded to :func:`micro_price`.

    Returns
    -------
    pandas.DataFrame
        A copy of *depth_summary* with the signal columns appended.
    """
    validate_columns(
        depth_summary,
        {"best_bid_price", "best_bid_vol", "best_ask_price", "best_ask_vol"},
        "depth_signals",
    )

    available = min(
        len(bin_volume_columns(depth_summary, "bid")),
        len(bin_volume_columns(depth_summary, "ask")),
    )
    effective_levels = max(1, min(depth_levels, available + 1))

    result = depth_summary.copy()
    result["mid_price"] = (
        depth_summary["best_bid_price"].to_numpy(dtype=float)
        + depth_summary["best_ask_price"].to_numpy(dtype=float)
    ) / 2.0
    result["micro_price"] = micro_price(
        depth_summary, stoikov_adjustment=stoikov_adjustment
    )
    result["obi"] = book_imbalance(depth_summary, levels=1)
    result["obi_depth"] = book_imbalance(depth_summary, levels=effective_levels)
    return result
