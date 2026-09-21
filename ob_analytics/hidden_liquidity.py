"""Hidden liquidity: iceberg orders and trades against hidden orders.

Some of the size a venue will trade is not in the visible book.  Two kinds
leave a footprint in an L3 stream, and each has a function here:

* :func:`detect_icebergs` finds **iceberg orders**.  An iceberg shows a small
  displayed peak and keeps the rest in reserve.  When the peak is filled, the
  venue shows the next slice as a new order, at the same price and on the same
  side, a moment later.  The detector looks for that refill: a resting order
  filled to zero, followed within ``max_delay`` by a new order at its price
  level.  Slices linked this way are chained into one suspected iceberg.
* :func:`hidden_trades` finds **trades against hidden orders**: a trade that
  printed strictly inside the visible spread standing before its maker's fill.
  No visible order rested at that price, so the maker was an order the book
  did not show.

Both are inferences from the public stream.  A trader who re-quotes by hand
within ``max_delay`` looks like an iceberg, and a hidden order resting at the
touch is not inside the spread, so it is not flagged.

On LOBSTER data the two can be checked against the venue's own labels: event
type 5 is an execution against a hidden order.  On one day of AAPL (2012-06-21)
85% of the type-5 executions are inside the visible spread, and 12% follow a
visible peak filled at the same price and instant, which is an iceberg's
reserve.  The how-to guide on hidden liquidity gives the measured recall and
precision.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ob_analytics._utils import validate_columns
from ob_analytics.engine import HIDDEN_ORDER_ID
from ob_analytics.exceptions import ConfigError

__all__ = [
    "ICEBERG_MAX_DELAY",
    "IcebergDetection",
    "detect_icebergs",
    "hidden_trades",
]

#: Default longest wait between a peak being filled and the next slice
#: appearing.  On Nasdaq (LOBSTER AAPL, 2012-06-21) refills cluster between
#: 0.2 and 0.3 ms, and a new order at the same price follows only about 1% of
#: cancellations within 0.4 ms, so one millisecond keeps nearly every refill
#: while letting few unrelated orders in.  A venue with a slower refill, or
#: timestamps taken on receipt far from the venue, needs a longer wait.
ICEBERG_MAX_DELAY = pd.Timedelta("1ms")

_ICEBERG_COLUMNS = [
    "iceberg",
    "direction",
    "price",
    "start",
    "end",
    "slices",
    "refills",
    "same_size_refills",
    "peak",
    "executed",
    "median_delay_s",
    "confidence",
]

_SLICE_COLUMNS = ["iceberg", "slice", "id", "timestamp", "volume", "fill", "delay_s"]

_CONFIDENCE = pd.CategoricalDtype(["low", "medium", "high"], ordered=True)


@dataclass(frozen=True)
class IcebergDetection:
    """Result of :func:`detect_icebergs`.

    Attributes
    ----------
    icebergs : pandas.DataFrame
        One row per suspected iceberg, in order of ``start``:

        - ``iceberg`` — number of the iceberg, from 1.
        - ``direction``, ``price`` — the side and price level it rested at.
        - ``start`` — when the first slice was placed (or first seen).
        - ``end`` — the last event of the last slice.
        - ``slices`` — visible orders in the chain, the first one included.
        - ``refills`` — ``slices - 1``.
        - ``same_size_refills`` — refills whose displayed size equals the
          displayed size of the slice they replaced.
        - ``peak`` — displayed size of the first slice, in lots; ``<NA>`` when
          the first slice was already resting when the stream began.
        - ``executed`` — size filled across all slices, in lots.
        - ``median_delay_s`` — median wait from a peak filled to its refill.
        - ``confidence`` — ``high`` when there are at least two refills and
          every one matches the displayed size, ``medium`` when at least one
          does, ``low`` otherwise.  An ordered categorical.

    slices : pandas.DataFrame
        One row per visible order in a chain: ``iceberg``, ``slice`` (from 1),
        order ``id``, ``timestamp`` placed (or first seen), displayed
        ``volume``, total ``fill``, and ``delay_s`` from the previous slice
        being filled to this one appearing (``NaN`` for the first slice).
    """

    icebergs: pd.DataFrame
    slices: pd.DataFrame


def detect_icebergs(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    max_delay: str | pd.Timedelta = ICEBERG_MAX_DELAY,
) -> IcebergDetection:
    """Find suspected iceberg orders from the refills they leave behind.

    A slice is *filled out* when its last fill as a maker leaves it with
    nothing outstanding.  A refill is the first new order placed at the same
    side and price at or after that fill, within *max_delay*, and, at the
    same instant, after it in event order.  When several slices at one price level are filled out
    together, a new order is matched to the one whose displayed size it
    equals, and otherwise to the one filled out first.  Each order joins at
    most one refill as the new slice and at most one as the old.

    Parameters
    ----------
    events : pandas.DataFrame
        L3 events with ``event_id``, ``id``, ``timestamp``, ``price``,
        ``volume``, ``action``, ``direction`` and ``fill``.
    trades : pandas.DataFrame
        Trades with ``maker_event_id``.  Only a maker's fill can empty a
        displayed peak, so an aggressor filled to zero is never a slice.
    max_delay : str or pandas.Timedelta
        Longest wait from a peak being filled to its refill.  See
        :data:`ICEBERG_MAX_DELAY` for the default and why.

    Returns
    -------
    IcebergDetection
        The suspected icebergs and the slices each one is made of.  Both
        frames are empty, with their columns, when none is found.

    Raises
    ------
    ConfigError
        If a required column is missing, or *max_delay* is negative.
    """
    validate_columns(
        events,
        {
            "event_id",
            "id",
            "timestamp",
            "price",
            "volume",
            "action",
            "direction",
            "fill",
        },
        "detect_icebergs",
    )
    validate_columns(trades, {"maker_event_id"}, "detect_icebergs")
    delay = pd.Timedelta(max_delay)
    if delay < pd.Timedelta(0):
        raise ConfigError("detect_icebergs: max_delay must not be negative.")

    visible = events[events["id"] != HIDDEN_ORDER_ID]
    # Time order first: not every loader numbers events chronologically
    # (Bitstamp numbers them by order id), so event_id only breaks ties.
    visible = visible.sort_values(["timestamp", "event_id"], kind="stable")
    filled_out = _filled_out(visible, trades)
    created = visible.loc[
        visible["action"] == "created",
        ["event_id", "id", "timestamp", "price", "volume", "direction"],
    ]
    links = _refill_links(filled_out, created, delay)
    if not links:
        return _empty_detection()
    return _build_detection(visible, links)


def _filled_out(visible: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Each order's last maker fill, where that fill left nothing outstanding."""
    maker_ids = pd.to_numeric(trades["maker_event_id"], errors="coerce").dropna()
    makers = visible[visible["event_id"].isin(maker_ids.astype("int64"))]
    last = makers.drop_duplicates("id", keep="last")
    emptied = (last["action"] == "deleted") | (last["volume"] == 0)
    return last.loc[emptied, ["event_id", "id", "timestamp", "price", "direction"]]


def _refill_links(
    filled_out: pd.DataFrame, created: pd.DataFrame, delay: pd.Timedelta
) -> list[tuple[int, int, pd.Timedelta]]:
    """Return ``(old_id, new_id, delay)`` for every refill found.

    A backward as-of join first keeps only the new orders placed within
    *delay* of some filled-out slice at their price level, so the matching
    loop walks a small fraction of the ``created`` rows.
    """
    if filled_out.empty or created.empty:
        return []
    keys = ["direction", "price"]
    near = pd.merge_asof(
        created.sort_values("timestamp", kind="stable"),
        filled_out[[*keys, "timestamp"]]
        .rename(columns={"timestamp": "filled_at"})
        .assign(timestamp=lambda f: f["filled_at"])
        .sort_values("timestamp", kind="stable"),
        on="timestamp",
        by=keys,
        direction="backward",
        tolerance=delay,
    )
    candidates = created[
        created["event_id"].isin(near.loc[near["filled_at"].notna(), "event_id"])
    ]
    if candidates.empty:
        return []

    # One timeline per price level: filled-out slices and new orders, in time
    # order with event_id breaking ties, so a new order is only ever matched
    # to a slice filled before it.
    timeline = pd.concat(
        [
            filled_out.assign(kind=0, size=-1),
            candidates.drop(columns="volume").assign(kind=1, size=candidates["volume"]),
        ]
    )
    # Sizes for matching a new slice to the peak it replaces.
    peaks = created.drop_duplicates("id").set_index("id")["volume"]

    links: list[tuple[int, int, pd.Timedelta]] = []
    for _, level in timeline.sort_values(
        ["timestamp", "event_id"], kind="stable"
    ).groupby(keys, observed=True, sort=False):
        pending: deque[tuple[pd.Timestamp, int, float]] = deque()
        for ts, oid, kind, size in zip(
            level["timestamp"], level["id"], level["kind"], level["size"]
        ):
            if kind == 0:
                pending.append((ts, int(oid), float(peaks.get(oid, np.nan))))
                continue
            while pending and ts - pending[0][0] > delay:
                pending.popleft()
            if not pending:
                continue
            match = next((p for p in pending if p[2] == size), pending[0])
            pending.remove(match)
            links.append((match[1], int(oid), ts - match[0]))
    return links


def _build_detection(
    visible: pd.DataFrame, links: list[tuple[int, int, pd.Timedelta]]
) -> IcebergDetection:
    """Chain refill links into icebergs and summarise each one."""
    successor = {old: (new, wait) for old, new, wait in links}
    has_parent = {new for _, new, _ in links}
    heads = [old for old, _, _ in links if old not in has_parent]

    per_order = visible.groupby("id", sort=False).agg(
        first_ts=("timestamp", "first"),
        last_ts=("timestamp", "last"),
        fill=("fill", "sum"),
        direction=("direction", "first"),
        price=("price", "first"),
    )
    created = visible[visible["action"] == "created"].drop_duplicates("id")
    displayed = created.set_index("id")["volume"]

    rows = []
    for head in heads:
        chain, waits = [head], [np.nan]
        while chain[-1] in successor:
            new, wait = successor[chain[-1]]
            chain.append(new)
            waits.append(wait.total_seconds())
        rows.append((chain, waits))
    first_seen = per_order["first_ts"].to_dict()
    rows.sort(key=lambda r: first_seen[r[0][0]])

    slice_rows = []
    for number, (chain, waits) in enumerate(rows, start=1):
        for position, (oid, wait) in enumerate(zip(chain, waits), start=1):
            slice_rows.append(
                {
                    "iceberg": number,
                    "slice": position,
                    "id": oid,
                    "timestamp": per_order.at[oid, "first_ts"],
                    "volume": displayed.get(oid, pd.NA),
                    "fill": per_order.at[oid, "fill"],
                    "delay_s": wait,
                }
            )
    slices = pd.DataFrame(slice_rows, columns=_SLICE_COLUMNS)
    slices["volume"] = slices["volume"].astype("Int64")
    slices["timestamp"] = slices["timestamp"].astype(visible["timestamp"].dtype)

    by_iceberg = slices.groupby("iceberg", sort=True)
    same_size = by_iceberg["volume"].apply(
        lambda v: int((v.iloc[1:].to_numpy() == v.iloc[:-1].to_numpy()).sum())
    )
    heads_ordered = [chain[0] for chain, _ in rows]
    tails = [chain[-1] for chain, _ in rows]
    icebergs = pd.DataFrame(
        {
            "iceberg": np.arange(1, len(rows) + 1),
            "direction": per_order.loc[heads_ordered, "direction"].to_numpy(),
            "price": per_order.loc[heads_ordered, "price"].to_numpy(),
            "start": by_iceberg["timestamp"].first().to_numpy(),
            "end": per_order.loc[tails, "last_ts"].to_numpy(),
            "slices": by_iceberg.size().to_numpy(),
            "refills": by_iceberg.size().to_numpy() - 1,
            "same_size_refills": same_size.to_numpy(),
            # The first slice's own size, not ``first()``: a first slice with
            # no ``created`` row has no known peak, and ``first()`` skips NA.
            "peak": slices.loc[slices["slice"] == 1, "volume"].to_numpy(),
            "executed": by_iceberg["fill"].sum().to_numpy(),
            "median_delay_s": by_iceberg["delay_s"].median().to_numpy(),
        }
    )
    icebergs["direction"] = icebergs["direction"].astype(visible["direction"].dtype)
    icebergs["start"] = icebergs["start"].astype(visible["timestamp"].dtype)
    icebergs["end"] = icebergs["end"].astype(visible["timestamp"].dtype)
    icebergs["peak"] = icebergs["peak"].astype("Int64")
    icebergs["confidence"] = pd.Categorical(
        np.select(
            [
                (icebergs["refills"] >= 2)
                & (icebergs["same_size_refills"] == icebergs["refills"]),
                icebergs["same_size_refills"] >= 1,
            ],
            ["high", "medium"],
            default="low",
        ),
        dtype=_CONFIDENCE,
    )
    return IcebergDetection(icebergs=icebergs[_ICEBERG_COLUMNS], slices=slices)


def _empty_detection() -> IcebergDetection:
    icebergs = pd.DataFrame(columns=_ICEBERG_COLUMNS)
    icebergs["confidence"] = icebergs["confidence"].astype(_CONFIDENCE)
    return IcebergDetection(
        icebergs=icebergs, slices=pd.DataFrame(columns=_SLICE_COLUMNS)
    )


def hidden_trades(
    events: pd.DataFrame, trades: pd.DataFrame, depth_summary: pd.DataFrame
) -> pd.DataFrame:
    """Return the trades that printed strictly inside the visible spread.

    Each trade is compared with the spread standing just before its maker's
    fill: the depth summary is read by an as-of join at the maker event's
    timestamp, with that instant itself excluded.  The maker event is used
    rather than the trade's own timestamp because some feeds report the fill
    on the order stream before the trade print arrives, and by the print the
    maker has already left the book.  Excluding the instant stops a sweep that
    empties a price level from making its own later prints look inside the
    spread.  A trade with no maker event falls back to its own timestamp.

    A trade is kept when both sides of the book were present and not crossed,
    and ``best_bid_price < price < best_ask_price``.  The depth summary holds
    visible orders only, so such a trade executed against an order the book
    did not show.  A hidden order resting at or behind the touch is missed: its
    trades print at a visible price.

    Parameters
    ----------
    events : pandas.DataFrame
        L3 events with ``event_id`` and ``timestamp``.
    trades : pandas.DataFrame
        Trades with ``timestamp``, ``price`` and ``maker_event_id``.
    depth_summary : pandas.DataFrame
        The run's depth summary, with ``timestamp``, ``best_bid_price`` and
        ``best_ask_price``.

    Returns
    -------
    pandas.DataFrame
        The matching rows of *trades*, index kept, with the standing
        ``best_bid_price`` and ``best_ask_price`` added.

    Raises
    ------
    ConfigError
        If a required column is missing.
    """
    validate_columns(events, {"event_id", "timestamp"}, "hidden_trades")
    validate_columns(trades, {"timestamp", "price", "maker_event_id"}, "hidden_trades")
    validate_columns(
        depth_summary,
        {"timestamp", "best_bid_price", "best_ask_price"},
        "hidden_trades",
    )
    unique_events = events.drop_duplicates("event_id")
    maker = pd.to_numeric(trades["maker_event_id"], errors="coerce").astype("float64")
    pos = pd.Index(unique_events["event_id"]).get_indexer(pd.Index(maker))
    at = trades["timestamp"].copy()
    known = pos >= 0
    at.iloc[np.flatnonzero(known)] = unique_events["timestamp"].to_numpy()[pos[known]]

    touch = depth_summary[["timestamp", "best_bid_price", "best_ask_price"]]
    standing = (
        pd.merge_asof(
            pd.DataFrame(
                # ``.array`` keeps the tz-aware dtype; ``to_numpy()`` would give
                # an object array, which ``merge_asof`` rejects when empty.
                {"timestamp": at.array, "row": np.arange(len(trades))}
            ).sort_values("timestamp", kind="stable"),
            touch.sort_values("timestamp", kind="stable"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=False,
        )
        .sort_values("row")
        .reset_index(drop=True)
    )
    # An instant with no book before it has no spread: NaN compares False.
    bid = standing["best_bid_price"].to_numpy(dtype="float64")
    ask = standing["best_ask_price"].to_numpy(dtype="float64")
    price = trades["price"].to_numpy(dtype="float64")
    inside = (bid > 0) & (ask > 0) & (bid < ask) & (bid < price) & (price < ask)
    out = trades.iloc[np.flatnonzero(inside)].copy()
    out["best_bid_price"] = bid[inside].astype("int64")
    out["best_ask_price"] = ask[inside].astype("int64")
    return out
