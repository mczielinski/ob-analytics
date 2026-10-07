"""Trade-sign classification for feeds without native maker/taker labels.

ob-analytics gets true trade signs for free on L3 crypto: Bitstamp ships
``buy_order_id`` / ``sell_order_id``, so :func:`~ob_analytics.analytics.set_order_types`
resolves maker vs taker and the trades frame carries a real ``direction``
(the taker's aggressor side).  **L2 / aggregated feeds don't label the
aggressor side** — and many CCXT sources don't either — so signed-flow
analytics (:func:`~ob_analytics.flow_toxicity.compute_vpin`,
:func:`~ob_analytics.flow_toxicity.order_flow_imbalance`) have nothing to
work with.

This module adds the three standard trade-sign classifiers so signed-flow
analytics work on every feed:

* **Tick rule** (:func:`tick_rule`) — sign from the last price change.
* **Lee–Ready** (:func:`lee_ready`) — quote-midpoint test with a tick-rule
  fallback for at-the-mid trades.  Needs quotes, e.g. the
  ``best_bid_price`` / ``best_ask_price`` columns of a pipeline
  ``depth_summary``, and reads the mid of the quote each trade arrived into
  (:func:`mid_before`).
* **BVC** (:func:`bulk_volume_classification`) — bulk volume classification
  (Easley, López de Prado & O'Hara, 2012): the buy fraction of a *volume
  bar* via the standardized-price-change normal CDF.  This is the
  VPIN-native method — it labels volume, not individual trades.

:func:`classify_trade_sign` is the per-trade entry point (tick / Lee–Ready);
:func:`~ob_analytics.flow_toxicity.compute_vpin` and
:func:`~ob_analytics.flow_toxicity.order_flow_imbalance` call it as a
fallback when the trades frame has no native ``direction``.

Sign convention (matching the pipeline's trades ``direction``): ``+1`` /
``"buy"`` = buyer-initiated (the taker lifted the ask), ``-1`` / ``"sell"``
= seller-initiated (the taker hit the bid).
"""

from __future__ import annotations

import warnings
from math import erf, sqrt

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from ob_analytics._utils import validate_columns, validate_non_empty
from ob_analytics.depth import _BID_ASK_COLUMNS, _bid_ask_pair, readable_quotes
from ob_analytics.exceptions import ConfigError
from ob_analytics.schemas import time_order_keys

_SQRT2 = sqrt(2.0)

#: The two aggressor sides.  Every consumer of a ``direction`` column tests it
#: as ``== "buy"`` and treats the rest as a sell, so anything else in the
#: column is not a missing value to them -- it is the wrong side.
_SIDES: tuple[str, str] = ("buy", "sell")

#: The trades schema's ``direction`` dtype, which every classifier here writes.
_DIRECTION_DTYPE = pd.CategoricalDtype(list(_SIDES), ordered=True)

#: Most volume buckets :func:`bulk_volume_classification` and
#: :func:`~ob_analytics.flow_toxicity.compute_vpin` build.  Both hold every
#: bucket in memory, so a ``bucket_volume`` given in the base asset against
#: sizes in integer lots (10^8 times too small for BTC) would otherwise ask for
#: hundreds of millions of buckets and run out of memory.  Fifty buckets a day
#: for fifty years is under a million.
MAX_VOLUME_BUCKETS = 1_000_000

# Accepted mid-column spellings for the Lee–Ready midpoint, most specific
# first.  A mid column is used as-is; otherwise a bid/ask pair (in one of the
# spellings ``depth._BID_ASK_COLUMNS`` lists) is averaged.
_MID_COLUMNS: tuple[str, ...] = ("mid", "midprice", "mid_price")


# ── Normal CDF (no scipy dependency) ─────────────────────────────────


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard-normal CDF Φ(z), vectorised over :func:`math.erf`.

    Evaluated per *bar* (bucket counts are small), so the Python-level
    ``erf`` loop is negligible and keeps scipy out of the dependency set.
    """
    erf_vec = np.vectorize(erf, otypes=[np.float64])
    return 0.5 * (1.0 + erf_vec(np.asarray(z, dtype=np.float64) / _SQRT2))


# ── Tick rule ────────────────────────────────────────────────────────


def tick_rule(prices: npt.ArrayLike) -> np.ndarray:
    """Classify trade signs by the tick rule.

    Signs each trade from the sign of its price change relative to the
    previous trade: an uptick is buyer-initiated (``+1``), a downtick
    seller-initiated (``-1``).  A **zero tick** (unchanged price) inherits
    the last non-zero sign — the classic Lee–Ready convention.

    *prices* must already be in trade order (chronological).  A leading run
    of zero ticks (before the first price move) is back-filled from the
    first determinable sign; a perfectly flat series has no information and
    defaults to ``+1``.

    Parameters
    ----------
    prices : array-like
        Trade prices in chronological order. Anything
        :func:`numpy.asarray` accepts: an ndarray, a Series, or a sequence.

    Returns
    -------
    numpy.ndarray
        ``int8`` array of ``+1`` (buy) / ``-1`` (sell), one per trade.
    """
    p = np.asarray(prices, dtype=np.float64)
    n = p.size
    if n == 0:
        return np.empty(0, dtype=np.int8)

    # Raw tick: -1 / 0 / +1.  The first element has no predecessor → 0.
    raw = np.sign(np.diff(p, prepend=p[0])).astype(np.int8)

    nonzero = np.flatnonzero(raw)
    if nonzero.size == 0:
        # Perfectly flat: no directional information.  Default to buy.
        return np.ones(n, dtype=np.int8)

    # Carry the last non-zero sign forward across zero ticks; back-fill the
    # leading zero run from the first non-zero sign.
    src = np.where(raw != 0, np.arange(n), -1)
    np.maximum.accumulate(src, out=src)
    src[src == -1] = nonzero[0]
    return raw[src]


# ── Lee–Ready ────────────────────────────────────────────────────────


def lee_ready(
    prices: npt.ArrayLike,
    mid: npt.ArrayLike,
) -> np.ndarray:
    """Classify trade signs by the Lee–Ready quote-midpoint test.

    A trade above the prevailing mid is buyer-initiated (``+1``), below it
    seller-initiated (``-1``).  A trade **at** the mid — or one with no
    prevailing quote (``mid`` is ``NaN``) — falls back to the
    :func:`tick_rule`.

    *prices* and *mid* must be equal-length and in chronological trade
    order; *mid* is the midpoint of the quote each trade arrived into, the
    last one strictly before it (:func:`mid_before`).  The quote stamped at
    the trade's own instant can be the book after the trade, and reading it
    flips signs.  :func:`classify_trade_sign` aligns quotes to trades this
    way.

    Parameters
    ----------
    prices : array-like
        Trade prices in chronological order.
    mid : array-like
        Prevailing quote midpoint per trade (``NaN`` where unknown).

    Returns
    -------
    numpy.ndarray
        ``int8`` array of ``+1`` (buy) / ``-1`` (sell), one per trade.
    """
    p = np.asarray(prices, dtype=np.float64)
    m = np.asarray(mid, dtype=np.float64)
    if p.shape != m.shape:
        raise ConfigError(
            f"lee_ready: prices and mid must be the same length, "
            f"got {p.shape} and {m.shape}"
        )

    signs = np.zeros(p.size, dtype=np.int8)
    above = p > m  # NaN mid → False
    below = p < m  # NaN mid → False
    signs[above] = 1
    signs[below] = -1

    # At-mid or unknown-quote trades: defer to the tick rule.
    undecided = ~(above | below)
    if undecided.any():
        signs[undecided] = tick_rule(p)[undecided]
    return signs


# ── Unified per-trade entry point ────────────────────────────────────


def classify_trade_sign(
    trades: pd.DataFrame,
    method: str = "lee_ready",
    quotes: pd.DataFrame | None = None,
) -> pd.Series:
    """Classify the aggressor side of each trade.

    A drop-in source of the ``direction`` column for feeds that don't label
    the aggressor.  Sorts *trades* chronologically, applies the chosen
    per-trade classifier, and returns the result realigned to the original
    index.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with at least ``timestamp`` and ``price``.
    method : str, optional
        ``"tick"`` (:func:`tick_rule`) or ``"lee_ready"``
        (:func:`lee_ready`, the default).  ``"bvc"`` is **not** a per-trade
        classifier — it labels volume bars; use
        :func:`bulk_volume_classification` or
        :func:`~ob_analytics.flow_toxicity.compute_vpin` with
        ``sign_method="bvc"`` instead.
    quotes : pandas.DataFrame, optional
        Required for ``method="lee_ready"``.  A quote frame with
        ``timestamp`` plus either a mid column (``mid`` / ``midprice`` /
        ``mid_price``) or a bid/ask pair (``best_bid_price`` /
        ``best_ask_price``, ``best_bid`` / ``best_ask``, or ``bid`` /
        ``ask``) — e.g. a pipeline ``depth_summary``.  Each trade is
        measured against the midpoint of the quote it arrived into: the last
        readable quote strictly before it (:func:`mid_before`).  The quote
        stamped at the trade's own instant is not used, because on a feed
        whose trades and book share one clock it is the book after the trade.

    Returns
    -------
    pandas.Series
        Ordered categorical ``"buy"`` / ``"sell"`` values named
        ``"direction"``, the trades schema's dtype, indexed like *trades*.

    Raises
    ------
    ConfigError
        If *method* is unknown, if ``"bvc"`` is requested here, if required
        columns are missing, or if ``lee_ready`` is requested without usable
        *quotes*.
    ObAnalyticsError
        If *trades* is empty.
    """
    validate_non_empty(trades, "classify_trade_sign")
    method = method.lower()
    if method == "bvc":
        raise ConfigError(
            "classify_trade_sign: BVC labels volume bars, not individual "
            "trades. Use bulk_volume_classification() or "
            "compute_vpin(..., sign_method='bvc')."
        )
    if method not in {"tick", "lee_ready"}:
        raise ConfigError(
            f"classify_trade_sign: unknown method {method!r}; "
            "expected 'tick', 'lee_ready', or 'bvc'."
        )
    validate_columns(trades, {"timestamp", "price"}, "classify_trade_sign")

    # Stable chronological order; remember it to restore the caller's index.
    order = np.argsort(trades["timestamp"].to_numpy(), kind="stable")
    prices = trades["price"].to_numpy(dtype=np.float64)[order]

    if method == "tick":
        signs_sorted = tick_rule(prices)
    else:  # lee_ready
        if quotes is None:
            raise ConfigError(
                "classify_trade_sign: method='lee_ready' requires quotes "
                "(a frame with timestamp + mid or bid/ask columns)."
            )
        mid = mid_before(trades, quotes, "classify_trade_sign")
        signs_sorted = lee_ready(prices, mid[order])

    signs = np.empty(len(trades), dtype=np.int8)
    signs[order] = signs_sorted
    side = np.where(signs > 0, "buy", "sell")
    return pd.Series(
        pd.Categorical(side, dtype=_DIRECTION_DTYPE),
        index=trades.index,
        name="direction",
    )


def quote_before(rows: pd.DataFrame, quotes: pd.DataFrame) -> np.ndarray:
    """Find the quote each row arrived into: the last one strictly before it.

    This is the package's one rule for "the quote standing when something
    arrived".  Lee-Ready (:func:`classify_trade_sign`), the effective spread
    (:func:`~ob_analytics.cost.transaction_costs`), the spread a hidden trade
    printed inside (:func:`~ob_analytics.hidden_liquidity.hidden_trades`) and
    order aggressiveness (:func:`~ob_analytics.analytics.order_aggressiveness`)
    all read their quote through it.

    "Before" is the canonical total order
    (:func:`~ob_analytics.schemas.time_order_keys`): ``timestamp`` first, then
    the tie-break keys that both frames carry, such as ``event_id``.  A quote
    that carries a row's own keys is the book *after* that row: on a feed
    whose trades and book events share one clock, the quote stamped at a
    trade's own instant already shows the touch the trade took.  So it does
    not count.  ``event_id`` is not a clock (a loader may number its events
    in another order, such as by order id), so it only decides between rows at
    the same instant.  A trades frame carries no ``event_id``, so a trade reads
    the last quote stamped strictly before its own instant.  A row with a
    missing tie-break key is read the same way.

    The quotes are taken as they are.  To read only quotes that make sense as
    a price, pass :func:`~ob_analytics.depth.readable_quotes` of them, as
    :func:`mid_before` does.

    Parameters
    ----------
    rows : pandas.DataFrame
        The rows to look up, with ``timestamp`` and any of the tie-break keys.
    quotes : pandas.DataFrame
        Quote frame with ``timestamp``, such as a pipeline ``depth_summary``.

    Returns
    -------
    numpy.ndarray
        For each row of *rows*, in order, the position (as for ``iloc``) of
        its quote in *quotes*, or ``-1`` when no quote stands before it or the
        row has no timestamp.

    Raises
    ------
    ConfigError
        If either frame lacks ``timestamp``, or one frame's timestamps carry a
        time zone and the other's do not.
    """
    validate_columns(rows, {"timestamp"}, "quote_before(rows)")
    validate_columns(quotes, {"timestamp"}, "quote_before(quotes)")
    keys = [k for k in time_order_keys(quotes) if k in rows.columns]
    standing = np.full(len(rows), -1, dtype=np.int64)
    if not len(rows) or quotes.empty:
        return standing
    zones = [getattr(f["timestamp"].dtype, "tz", None) for f in (rows, quotes)]
    if (zones[0] is None) != (zones[1] is None):
        raise ConfigError(
            "quote_before: the rows and the quotes must both have time-zone "
            f"aware timestamps, or both naive ones; got {rows['timestamp'].dtype} "
            f"and {quotes['timestamp'].dtype}."
        )
    row_keys = rows[keys]
    has_time = row_keys["timestamp"].notna().to_numpy()
    complete = row_keys.notna().all(axis=1).to_numpy()
    # A row missing a tie-break key cannot be placed among the quotes at its
    # own instant, so it is placed ahead of all of them.
    partial = has_time & ~complete
    if complete.any():
        standing[complete] = _last_quote_before(row_keys[complete], quotes, keys)
    if partial.any():
        standing[partial] = _last_quote_before(
            row_keys.loc[partial, ["timestamp"]], quotes, ["timestamp"]
        )
    return standing


def _last_quote_before(
    rows: pd.DataFrame, quotes: pd.DataFrame, keys: list[str]
) -> np.ndarray:
    """Position of the last quote strictly before each row, ordered by *keys*.

    Every row has a value in every key.  ``-1`` where no quote comes first.
    """
    if keys == ["timestamp"]:
        both_times = [f["timestamp"] for f in (rows, quotes)]
        if all(pd.api.types.is_datetime64_any_dtype(t) for t in both_times):
            return _last_quote_before_instant(*both_times)
    n_quotes = len(quotes)
    # At equal keys a row sorts ahead of the quotes, so the quotes written by
    # its own event are not read as standing before it.
    both = pd.concat(
        [
            quotes[keys].reset_index(drop=True).assign(_quote_last=1),
            rows[keys].reset_index(drop=True).assign(_quote_last=0),
        ],
        ignore_index=True,
    )
    both = both.sort_values([*keys, "_quote_last"], kind="stable")
    source = both.index.to_numpy()
    is_row = source >= n_quotes

    # For each position in the merged order, the source position of the last
    # quote at or before it (-1 while there is none yet).  A quote with no
    # timestamp sorts after every row, so it is never read.
    last = np.maximum.accumulate(np.where(is_row, -1, np.arange(len(both))))
    last = np.where(last >= 0, source[last], -1)
    standing = np.full(len(rows), -1, dtype=np.int64)
    standing[source[is_row] - n_quotes] = last[is_row]
    return standing


def _ns(times: pd.Series) -> np.ndarray:
    """*times* as int64 nanoseconds since the epoch (UTC for a tz-aware clock)."""
    index = pd.DatetimeIndex(times).as_unit("ns")
    return index.to_numpy(dtype="datetime64[ns]").view("int64")


def _last_quote_before_instant(at: pd.Series, stamps: pd.Series) -> np.ndarray:
    """:func:`_last_quote_before` on timestamps alone, by binary search.

    Equal stamps keep their input order, so the last of them is the one read,
    as in the sorted merge.  A quote with no timestamp is never read.
    """
    stamp_ns = _ns(stamps)
    valid = np.flatnonzero(~pd.isna(stamps).to_numpy())
    order = valid[np.argsort(stamp_ns[valid], kind="stable")]
    if not len(order):
        return np.full(len(at), -1, dtype=np.int64)
    before = np.searchsorted(stamp_ns[order], _ns(at), side="left")
    return np.where(before > 0, order[np.maximum(before - 1, 0)], -1)


def mid_before(
    rows: pd.DataFrame,
    quotes: pd.DataFrame,
    context: str = "mid_before",
    *,
    mid_column: str | None = None,
) -> np.ndarray:
    """Midpoint of the quote each row arrived into.

    Reads the last *readable* quote strictly before each row, by
    :func:`quote_before`.  A quote that is not readable
    (:func:`~ob_analytics.depth.readable_quotes`: an empty side or a crossed
    book) is skipped, so the row reaches back to the last quote that was.  A
    row with no such quote before it gets ``NaN``: Lee-Ready falls back to the
    tick rule there, and the cost measures leave the row unmeasured.

    Parameters
    ----------
    rows : pandas.DataFrame
        The rows to price, usually trades, with ``timestamp``.  They need not
        be sorted.
    quotes : pandas.DataFrame
        Quote frame with ``timestamp`` plus a mid column (``mid`` /
        ``midprice`` / ``mid_price``) or a bid/ask pair (``best_bid_price`` /
        ``best_ask_price``, ``best_bid`` / ``best_ask``, or ``bid`` / ``ask``).
        A pipeline ``depth_summary`` can be passed straight in.
    context : str, optional
        Caller name, used in the error message.
    mid_column : str, optional
        Column to read the midpoint from, e.g. ``"micro_price"`` for the
        size-weighted mid (:func:`~ob_analytics.depth.micro_price`).  ``None``
        (default) takes the first of the accepted mid spellings present, and
        otherwise averages a bid/ask pair.

    Returns
    -------
    numpy.ndarray
        The midpoint per row of *rows*, in order, ``NaN`` where none exists.

    Raises
    ------
    ConfigError
        If *quotes* lacks ``timestamp`` or any recognised price columns.
    """
    validate_columns(quotes, {"timestamp"}, f"{context}(quotes)")
    readable, mid = _readable_mids(quotes, mid_column=mid_column, context=context)
    standing = quote_before(rows, readable)
    out = np.full(len(standing), np.nan, dtype=np.float64)
    known = standing >= 0
    out[known] = mid[standing[known]]
    return out


def prevailing_mid(
    timestamps: np.ndarray,
    quotes: pd.DataFrame,
    context: str = "prevailing_mid",
    *,
    mid_column: str | None = None,
    require_covered: bool = False,
) -> np.ndarray:
    """Midpoint standing at each (sorted) instant, the instant included.

    A backward as-of join of *timestamps* against the readable quotes
    (:func:`~ob_analytics.depth.readable_quotes`): each instant gets the
    midpoint of the last readable quote published at or before it.  This is
    the mid *at an instant*, such as the mid one horizon after a trade.  The
    mid a trade arrived into is a different question, with a different answer
    on a feed whose trades and book share one clock: see :func:`mid_before`.
    Instants before the first readable quote get ``NaN``.

    Any of the accepted quote-column spellings works: a mid column (``mid`` /
    ``midprice`` / ``mid_price``) or a bid/ask pair (``best_bid_price`` /
    ``best_ask_price``, ``best_bid`` / ``best_ask``, or ``bid`` / ``ask``).  So
    a pipeline ``depth_summary`` can be passed straight in.

    Parameters
    ----------
    timestamps : numpy.ndarray
        Instants to price, **sorted ascending** (``merge_asof`` requires it).
    quotes : pandas.DataFrame
        Quote frame with ``timestamp`` plus a mid or bid/ask pair.
    context : str, optional
        Caller name, used in the error message.
    mid_column : str, optional
        Column to read the midpoint from, e.g. ``"micro_price"`` for the
        size-weighted mid (:func:`~ob_analytics.depth.micro_price`).  ``None``
        (default) takes the first of the accepted mid spellings present, and
        otherwise averages a bid/ask pair.
    require_covered : bool, optional
        Whether an instant past the newest readable quote is ``NaN`` rather
        than that quote's mid.  Default ``False`` returns the last mid known.
        A backward join cannot tell "the state at this instant" from "the last
        state before the data ran out", and a caller measuring over a fixed
        wait needs to: reusing the final quote reports a shorter reach as
        though it were the full one.

    Returns
    -------
    numpy.ndarray
        The midpoint per instant, ``NaN`` where none exists yet.

    Raises
    ------
    ConfigError
        If *quotes* lacks ``timestamp`` or any recognised price columns.
    """
    validate_columns(quotes, {"timestamp"}, f"{context}(quotes)")
    readable, mid = _readable_mids(quotes, mid_column=mid_column, context=context)
    q = pd.DataFrame(
        {"timestamp": readable["timestamp"].to_numpy(), "_mid": mid}
    ).sort_values("timestamp", kind="stable")
    if q.empty:
        # No quote to join against: every instant is "no mid yet".  Returned
        # here because merge_asof on an empty frame raises on dtype instead.
        return np.full(len(timestamps), np.nan, dtype=np.float64)
    left = pd.DataFrame({"timestamp": timestamps})
    merged = pd.merge_asof(left, q, on="timestamp", direction="backward")
    mid_out = merged["_mid"].to_numpy(dtype=np.float64)
    if require_covered:
        beyond = (left["timestamp"] > q["timestamp"].max()).to_numpy()
        mid_out = np.where(beyond, np.nan, mid_out)
    return mid_out


def _readable_mids(
    quotes: pd.DataFrame,
    *,
    mid_column: str | None,
    context: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """The readable quotes that have a timestamp and a mid, and those mids.

    *mid_column* names the column to read; ``None`` falls back to the known
    mid spellings, then to the average of a bid/ask pair.  Readability is
    tested on the bid/ask pair whenever the frame carries one, even when the
    value itself comes from a mid column: a frame can hold both
    (:func:`~ob_analytics.depth.depth_signals` adds ``mid_price`` beside the
    touch), and reading the mid from one column must not skip a test the
    other columns can answer.
    """
    if mid_column is not None and mid_column not in quotes.columns:
        raise ConfigError(
            f"{context}: quotes have no column {mid_column!r}. "
            f"Available columns: {sorted(quotes.columns)}"
        )
    # Only the columns read here, so the filter below does not copy a whole
    # depth summary with its depth bins.
    wanted = {
        *time_order_keys(quotes),
        *_MID_COLUMNS,
        *(column for pair in _BID_ASK_COLUMNS for column in pair),
        *([mid_column] if mid_column is not None else []),
    }
    readable = readable_quotes(quotes[[c for c in quotes.columns if c in wanted]])
    mid: np.ndarray | None = None
    if mid_column is not None:
        mid = readable[mid_column].to_numpy(dtype=np.float64)
    else:
        for col in _MID_COLUMNS:
            if col in quotes.columns:
                mid = readable[col].to_numpy(dtype=np.float64)
                break
    if mid is None:
        pair = _bid_ask_pair(readable)
        if pair is None:
            raise ConfigError(
                f"{context}: quotes need a mid column "
                f"({' / '.join(_MID_COLUMNS)}) or a bid/ask pair "
                f"({', '.join('/'.join(p) for p in _BID_ASK_COLUMNS)}). "
                f"Available columns: {sorted(quotes.columns)}"
            )
        bid = readable[pair[0]].to_numpy(dtype=np.float64)
        ask = readable[pair[1]].to_numpy(dtype=np.float64)
        mid = 0.5 * (bid + ask)
    # Dropped rather than kept as NaN, so a reader reaches the last quote that
    # had a midpoint instead of reporting "no mid" for the instant.
    usable = ~np.isnan(mid) & readable["timestamp"].notna().to_numpy()
    return readable[usable], mid[usable]


def resolve_direction(
    trades: pd.DataFrame,
    sign_method: str | None,
    quotes: pd.DataFrame | None,
    context: str,
    *,
    fallback: str | None = "tick",
) -> pd.DataFrame:
    """Return *trades* with a ``buy``/``sell`` ``direction`` on every trade.

    This is the package's one rule for filling a trade's missing aggressor
    side.  The pipeline uses it for trades the venue left unlabelled, and the
    signed-flow analytics use it for theirs.  L3 feeds state the side; L2 and
    aggregated feeds often do not, so a trade-sign classifier
    (:func:`classify_trade_sign`) estimates it.

    * ``sign_method=None``: keep a ``direction`` the trade already has, and
      classify every trade whose value is neither ``"buy"`` nor ``"sell"``
      (or every trade, when there is no ``direction`` column).  With
      *quotes*, the classifier is Lee-Ready, against the quote each trade
      arrived into (:func:`mid_before`); a trade with no quote before it falls
      back to the tick rule there.  With *quotes* ``None``, the classifier is
      *fallback*.
    * ``sign_method="tick"`` / ``"lee_ready"``: always classify every trade
      with that method, replacing any existing ``direction``.

    The frame is only copied when a ``direction`` column is written, so a feed
    that already labels every trade is passed straight through.  The column
    written is the trades schema's ordered ``buy``/``sell`` categorical.

    With the default *fallback*, the returned column holds only ``"buy"`` and
    ``"sell"``.  That matters because every consumer reads it as ``== "buy"``
    and treats everything else as a sell: an unlabelled trade left in place is
    not dropped by them, it is counted on the wrong side.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with at least ``timestamp`` and ``price``.
    sign_method : str or None
        ``None``, ``"tick"`` or ``"lee_ready"``. See above.
    quotes : pandas.DataFrame or None
        Quote frame for Lee-Ready (e.g. a pipeline ``depth_summary``).
    context : str
        Caller name, used in the messages.
    fallback : str or None, optional
        What ``sign_method=None`` does when *quotes* is ``None``.  ``"tick"``
        (the default) classifies the unlabelled trades by the tick rule.
        ``None`` leaves them empty and warns: this is the pipeline's rule for
        a run that produced no quotes at all.

    Returns
    -------
    pandas.DataFrame
        *trades* with a ``direction`` column.

    Raises
    ------
    ConfigError
        If *sign_method* is ``"bvc"``, which labels volume rather than
        individual trades, or *fallback* is not ``"tick"`` or ``None``.
    """
    if fallback not in ("tick", None):
        raise ConfigError(
            f"{context}: fallback must be 'tick' or None, got {fallback!r}."
        )
    if sign_method == "bvc":
        raise ConfigError(
            f"{context}: sign_method='bvc' labels volume bars, not individual "
            "trades, and is only supported by compute_vpin."
        )
    if sign_method is not None:
        out = trades.copy()
        out["direction"] = classify_trade_sign(
            trades, method=sign_method, quotes=quotes
        )
        return out

    if "direction" in trades.columns:
        unlabelled = ~trades["direction"].isin(_SIDES).to_numpy()
        if not unlabelled.any():
            return trades
        known = trades["direction"].astype(object).where(~unlabelled)
    else:
        unlabelled = np.ones(len(trades), dtype=bool)
        known = pd.Series(np.nan, index=trades.index, dtype=object)

    out = trades.copy()
    method = "lee_ready" if quotes is not None else fallback
    if method is None:
        warnings.warn(
            f"{context}: {int(unlabelled.sum())} of {len(trades)} trades carry "
            "no aggressor side and there are no quotes to classify them "
            "against; they are left empty.",
            UserWarning,
            stacklevel=2,
        )
        out["direction"] = pd.Categorical(known, dtype=_DIRECTION_DTYPE)
        return out

    inferred = classify_trade_sign(trades, method=method, quotes=quotes)
    out["direction"] = pd.Categorical(
        known.where(~unlabelled, inferred.astype(object)), dtype=_DIRECTION_DTYPE
    )
    if "direction" in trades.columns:
        logger.info(
            "{}: {} of {} trades carry no usable direction; inferred with the "
            "{!r} rule.",
            context,
            int(unlabelled.sum()),
            len(trades),
            method,
        )
    return out


_UNITS_ADVICE = (
    "bucket_volume is in the units of trades['volume'], which are integer lots "
    "on a pipeline result (size = lots * lot_size). Size it from the data, "
    "e.g. trades['volume'].sum() / 60."
)


def check_bucket_count(
    trades: pd.DataFrame,
    bucket_volume: float,
    context: str,
    *,
    advice: str | None = None,
) -> None:
    """Refuse a *bucket_volume* that cuts the trades into too many buckets.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with a ``volume`` column.
    bucket_volume : float
        Volume per bucket, in the units of ``trades["volume"]``.  Must be
        positive.
    context : str
        Name of the caller, used in the error message.
    advice : str, optional
        What to do about it, ending the error message.  ``None`` (default)
        says which units *bucket_volume* is in, the usual cause.

    Raises
    ------
    ValueError
        If the trades' total volume divided by *bucket_volume* is more than
        :data:`MAX_VOLUME_BUCKETS`.
    """
    total = float(trades["volume"].sum())
    count = total / bucket_volume
    if count > MAX_VOLUME_BUCKETS:
        raise ValueError(
            f"{context}: bucket_volume={bucket_volume:g} cuts the trades' total "
            f"volume of {total:g} into about {count:.3g} buckets, more than "
            f"MAX_VOLUME_BUCKETS ({MAX_VOLUME_BUCKETS:,}). "
            f"{_UNITS_ADVICE if advice is None else advice}"
        )


# ── Bulk volume classification (BVC) ─────────────────────────────────


def bulk_volume_classification(
    trades: pd.DataFrame,
    bucket_volume: float,
    *,
    sigma: float | None = None,
) -> pd.DataFrame:
    """Split volume bars into buy / sell fractions (Easley–LdP–O'Hara BVC).

    Partitions cumulative trade volume into equal-sized *buckets* (the same
    volume bars :func:`~ob_analytics.flow_toxicity.compute_vpin` uses; a
    trade straddling a boundary is split proportionally) and estimates each
    bucket's buy fraction as

    ``buy_fraction = Φ(ΔP / σ)``

    where ``ΔP`` is the bucket's close-to-close price change and ``σ`` the
    standard deviation of those changes.  Unlike a per-trade classifier this
    labels *volume*, so it needs neither the aggressor side nor quotes — the
    VPIN-native method for feeds that carry only trade prints.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with ``timestamp``, ``price``, and ``volume``.
    bucket_volume : float
        Total volume per bucket (instrument-specific), in the units of
        ``trades["volume"]``: integer lots on a pipeline result.
    sigma : float, optional
        Standard deviation of bucketed price changes.  Estimated from the
        data (sample std of the bucket ΔP series) when omitted.

    Returns
    -------
    pandas.DataFrame
        One row per completed bucket with columns ``bucket``,
        ``timestamp_start``, ``timestamp_end``, ``close``, ``delta_price``,
        ``buy_fraction``, ``buy_volume``, ``sell_volume``.  Empty (no rows)
        if the trades don't fill a single bucket.

    Raises
    ------
    ConfigError
        If required columns are missing.
    ObAnalyticsError
        If *trades* is empty.
    ValueError
        If *bucket_volume* is not positive or would make more than
        :data:`MAX_VOLUME_BUCKETS` buckets, or *sigma* is not positive.
    """
    validate_columns(
        trades, {"timestamp", "price", "volume"}, "bulk_volume_classification"
    )
    validate_non_empty(trades, "bulk_volume_classification")
    if bucket_volume <= 0:
        raise ValueError(f"bucket_volume must be positive, got {bucket_volume}")
    check_bucket_count(trades, bucket_volume, "bulk_volume_classification")
    if sigma is not None and sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    df = trades.sort_values("timestamp").reset_index(drop=True)
    prices = df["price"].to_numpy(dtype=np.float64)
    volumes = df["volume"].to_numpy(dtype=np.float64)
    timestamps = df["timestamp"].to_numpy()

    # Walk trades, closing a bucket every `bucket_volume` units.  A single
    # trade can fill several buckets; each carries the price at its close.
    starts: list = []
    ends: list = []
    closes: list[float] = []
    bucket_start_ts = timestamps[0]
    bucket_remaining = bucket_volume

    for i in range(len(df)):
        remaining_trade = volumes[i]
        while remaining_trade > 0:
            alloc = min(remaining_trade, bucket_remaining)
            remaining_trade -= alloc
            bucket_remaining -= alloc
            if bucket_remaining <= 1e-12:
                starts.append(bucket_start_ts)
                ends.append(timestamps[i])
                closes.append(prices[i])
                bucket_remaining = bucket_volume
                bucket_start_ts = timestamps[i]

    if not closes:
        return pd.DataFrame(
            columns=[
                "bucket",
                "timestamp_start",
                "timestamp_end",
                "close",
                "delta_price",
                "buy_fraction",
                "buy_volume",
                "sell_volume",
            ]
        )

    close = np.asarray(closes, dtype=np.float64)
    # Close-to-close change; the first bucket is anchored to the first price.
    delta_price = np.diff(close, prepend=prices[0])

    if sigma is None:
        sigma = float(np.std(delta_price, ddof=1)) if delta_price.size > 1 else 0.0
    if not sigma > 0:
        # Degenerate (flat prices / single bucket): no information → 50/50.
        buy_fraction = np.full(close.size, 0.5)
    else:
        buy_fraction = _norm_cdf(delta_price / sigma)

    return pd.DataFrame(
        {
            "bucket": np.arange(close.size),
            "timestamp_start": starts,
            "timestamp_end": ends,
            "close": close,
            "delta_price": delta_price,
            "buy_fraction": buy_fraction,
            "buy_volume": buy_fraction * bucket_volume,
            "sell_volume": (1.0 - buy_fraction) * bucket_volume,
        }
    )
