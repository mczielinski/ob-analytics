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
  fallback for at-the-mid trades.  Needs the prevailing BBO / mid, e.g. the
  ``best_bid_price`` / ``best_ask_price`` columns of a pipeline
  ``depth_summary``.
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

from math import erf, sqrt

import numpy as np
import pandas as pd

from ob_analytics._utils import validate_columns, validate_non_empty
from ob_analytics.exceptions import ConfigError

_SQRT2 = sqrt(2.0)

# Accepted quote-column spellings for the Lee–Ready midpoint, most specific
# first.  A ``(bid, ask)`` pair is averaged; a single mid column is used as-is.
_MID_COLUMNS: tuple[str, ...] = ("mid", "midprice", "mid_price")
_BID_ASK_COLUMNS: tuple[tuple[str, str], ...] = (
    ("best_bid_price", "best_ask_price"),
    ("best_bid", "best_ask"),
    ("bid", "ask"),
)


# ── Normal CDF (no scipy dependency) ─────────────────────────────────


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard-normal CDF Φ(z), vectorised over :func:`math.erf`.

    Evaluated per *bar* (bucket counts are small), so the Python-level
    ``erf`` loop is negligible and keeps scipy out of the dependency set.
    """
    erf_vec = np.vectorize(erf, otypes=[np.float64])
    return 0.5 * (1.0 + erf_vec(np.asarray(z, dtype=np.float64) / _SQRT2))


# ── Tick rule ────────────────────────────────────────────────────────


def tick_rule(prices: np.ndarray | pd.Series) -> np.ndarray:
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
    prices : numpy.ndarray or pandas.Series
        Trade prices in chronological order.

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
    prices: np.ndarray | pd.Series,
    mid: np.ndarray | pd.Series,
) -> np.ndarray:
    """Classify trade signs by the Lee–Ready quote-midpoint test.

    A trade above the prevailing mid is buyer-initiated (``+1``), below it
    seller-initiated (``-1``).  A trade **at** the mid — or one with no
    prevailing quote (``mid`` is ``NaN``) — falls back to the
    :func:`tick_rule`.

    *prices* and *mid* must be equal-length and in chronological trade
    order; *mid* is the quote midpoint prevailing at (or just before) each
    trade — see :func:`classify_trade_sign` for aligning quotes to trades.

    Parameters
    ----------
    prices : numpy.ndarray or pandas.Series
        Trade prices in chronological order.
    mid : numpy.ndarray or pandas.Series
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
        ``ask``) — e.g. a pipeline ``depth_summary``.  The midpoint
        prevailing at or before each trade is used (a backward as-of join).

    Returns
    -------
    pandas.Series
        Categorical ``"buy"`` / ``"sell"`` values named ``"direction"``,
        indexed like *trades*.

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
        mid_sorted = _prevailing_mid(trades["timestamp"].to_numpy()[order], quotes)
        signs_sorted = lee_ready(prices, mid_sorted)

    signs = np.empty(len(trades), dtype=np.int8)
    signs[order] = signs_sorted
    side = np.where(signs > 0, "buy", "sell")
    return pd.Series(
        pd.Categorical(side, categories=["buy", "sell"]),
        index=trades.index,
        name="direction",
    )


def _prevailing_mid(
    trade_timestamps: np.ndarray,
    quotes: pd.DataFrame,
) -> np.ndarray:
    """Midpoint prevailing at or before each (sorted) trade timestamp.

    Backward as-of join of *trade_timestamps* against *quotes*.  Trades
    before the first quote get ``NaN`` (Lee–Ready then falls back to the
    tick rule).

    *trade_timestamps* must be sorted ascending.
    """
    validate_columns(quotes, {"timestamp"}, "classify_trade_sign(quotes)")
    mid = _quote_mid(quotes)
    q = (
        pd.DataFrame({"timestamp": quotes["timestamp"].to_numpy(), "_mid": mid})
        .dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="stable")
    )
    left = pd.DataFrame({"timestamp": trade_timestamps})
    merged = pd.merge_asof(left, q, on="timestamp", direction="backward")
    return merged["_mid"].to_numpy(dtype=np.float64)


def _quote_mid(quotes: pd.DataFrame) -> np.ndarray:
    """Extract a midpoint array from a quote frame's known column spellings."""
    for col in _MID_COLUMNS:
        if col in quotes.columns:
            return quotes[col].to_numpy(dtype=np.float64)
    for bid_col, ask_col in _BID_ASK_COLUMNS:
        if bid_col in quotes.columns and ask_col in quotes.columns:
            bid = quotes[bid_col].to_numpy(dtype=np.float64)
            ask = quotes[ask_col].to_numpy(dtype=np.float64)
            return 0.5 * (bid + ask)
    raise ConfigError(
        "classify_trade_sign: quotes need a mid column "
        f"({' / '.join(_MID_COLUMNS)}) or a bid/ask pair "
        f"({', '.join('/'.join(p) for p in _BID_ASK_COLUMNS)}). "
        f"Available columns: {sorted(quotes.columns)}"
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
        Total volume per bucket (instrument-specific).
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
        If *bucket_volume* is not positive, or *sigma* is not positive.
    """
    validate_columns(
        trades, {"timestamp", "price", "volume"}, "bulk_volume_classification"
    )
    validate_non_empty(trades, "bulk_volume_classification")
    if bucket_volume <= 0:
        raise ValueError(f"bucket_volume must be positive, got {bucket_volume}")
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
