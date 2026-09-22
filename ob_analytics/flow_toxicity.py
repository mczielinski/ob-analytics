"""Order flow toxicity and imbalance metrics.

Implements three market microstructure measures for detecting informed
trading and quantifying price impact:

* :func:`compute_vpin` — Volume-Synchronized Probability of Informed
  Trading (Easley, López de Prado & O'Hara, 2012).
* :func:`compute_kyle_lambda` — Kyle's Lambda price-impact coefficient
  (Kyle, 1985).
* :func:`order_flow_imbalance` — Normalised buy/sell volume imbalance
  per time window.

All functions accept a trades DataFrame from the pipeline (or any
DataFrame with the required columns).

VPIN and Kyle's λ were designed for markets that trade thousands of times a
minute.  On a thin tape they still return a number, so both results say when
that number rests on too little data: :class:`KyleLambdaResult` carries
``significant`` and ``diagnostics``, and the VPIN frame carries
``attrs["diagnostics"]``.  The thresholds are the module constants
:data:`KYLE_MIN_T_STAT` and :data:`KYLE_MIN_WINDOWS`; for VPIN the threshold is
the ``n_buckets`` argument itself.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ob_analytics._utils import validate_columns, validate_non_empty
from ob_analytics.trade_sign import (
    bulk_volume_classification,
    resolve_direction,
)

#: Smallest ``|t|`` at which λ counts as distinguishable from zero.  The
#: usual rule of thumb, about a 5% two-sided test on a large sample.
KYLE_MIN_T_STAT = 2.0

#: Fewest regression windows for λ to mean much.  Below this the t-statistic
#: itself is unstable, whatever its value.
KYLE_MIN_WINDOWS = 30

#: Buckets in one average day of volume, for :func:`vpin_bucket_volume`.
#: Fifty is the rule used by Easley, López de Prado and O'Hara (2012), and
#: matches the default ``n_buckets`` of :func:`compute_vpin`: the trailing
#: average then covers about one day.
VPIN_BUCKETS_PER_DAY = 50

# Most window values the λ bootstrap holds in one array at a time.
_BOOTSTRAP_CHUNK = 1 << 20


@dataclass(frozen=True)
class KyleLambdaResult:
    """Result of a Kyle's λ OLS regression.

    Attributes
    ----------
    lambda_ : float
        Slope — price change per unit signed order flow (higher = less liquid).
    t_stat : float
        t-statistic for ``lambda_``.
    r_squared : float
        Fraction of ΔPrice variance explained by signed order flow.
    n_windows : int
        Number of time windows in the regression.
    regression_df : pandas.DataFrame
        Per-window ``timestamp``/``delta_price``/``signed_volume`` data.
    ci_low, ci_high : float
        Bounds of a block-bootstrap confidence interval for ``lambda_``.
        ``NaN`` when the bootstrap was turned off or the fit is undefined.
    ci_level : float
        Coverage of that interval, for example ``0.95``.
    """

    lambda_: float
    t_stat: float
    r_squared: float
    n_windows: int
    regression_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    ci_level: float = 0.95

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Reasons ``lambda_`` should not be relied on; empty when there are none.

        Checks the fit is defined, that there are at least
        :data:`KYLE_MIN_WINDOWS` windows, and that ``|t_stat|`` reaches
        :data:`KYLE_MIN_T_STAT`.
        """
        if not np.isfinite(self.lambda_):
            undefined = (
                "λ is undefined: fewer than 2 windows, or the signed volume "
                "is the same in every window"
            )
            return (undefined,)
        reasons = []
        if self.n_windows < KYLE_MIN_WINDOWS:
            reasons.append(
                f"only {self.n_windows} windows; at least {KYLE_MIN_WINDOWS} "
                "are needed for the fit to mean much"
            )
        if not np.isfinite(self.t_stat):
            reasons.append("the t-statistic is undefined")
        elif abs(self.t_stat) < KYLE_MIN_T_STAT:
            reasons.append(
                f"|t| = {abs(self.t_stat):.2f} is below {KYLE_MIN_T_STAT:g}; "
                "λ is not distinguishable from zero"
            )
        return tuple(reasons)

    @property
    def significant(self) -> bool:
        """``True`` when :attr:`diagnostics` found nothing wrong."""
        return not self.diagnostics


# ── VPIN ─────────────────────────────────────────────────────────────


def vpin_bucket_volume(
    trades: pd.DataFrame,
    buckets_per_day: int = VPIN_BUCKETS_PER_DAY,
    trading_day: str | pd.Timedelta = "24h",
) -> float:
    """Pick a VPIN ``bucket_volume`` from the trades: average daily volume ÷ 50.

    Average daily volume is the traded volume per unit of time, scaled to one
    *trading_day*::

        daily volume = total volume × trading_day / (last timestamp − first timestamp)

    The same formula covers every session length.  A 30-minute capture is
    scaled up to a full day at the rate it traded; a week-long one is averaged
    down to a day.  On a short capture the result is therefore a large bucket,
    and VPIN will fill only a few of them, which :func:`compute_vpin` then
    reports in its diagnostics.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with at least ``timestamp`` and ``volume``.
    buckets_per_day : int, optional
        How many buckets one average day of volume fills.  Default 50.
    trading_day : str or pandas.Timedelta, optional
        Length of one trading day, on the same clock as the span between the
        first and last trade.  Default ``"24h"``, which is right for venues
        that trade around the clock and for any capture that runs over several
        days, closed hours included.  Use a session length, for example
        ``"6.5h"`` for US equities, only when the capture falls inside a
        single session.

    Returns
    -------
    float
        The bucket volume, in the units of ``trades["volume"]``.

    Raises
    ------
    ConfigError
        If required columns are missing.
    ObAnalyticsError
        If *trades* is empty.
    ValueError
        If the trades span no time, or *buckets_per_day* or *trading_day* is
        not positive.
    """
    validate_columns(trades, {"timestamp", "volume"}, "vpin_bucket_volume")
    validate_non_empty(trades, "vpin_bucket_volume")
    if buckets_per_day <= 0:
        raise ValueError(f"buckets_per_day must be positive, got {buckets_per_day}")
    day = pd.Timedelta(trading_day)
    if day <= pd.Timedelta(0):
        raise ValueError(f"trading_day must be positive, got {trading_day!r}")
    span = trades["timestamp"].max() - trades["timestamp"].min()
    if span <= pd.Timedelta(0):
        raise ValueError(
            "vpin_bucket_volume: the trades span no time, so there is no rate "
            "to scale to a day; pass bucket_volume explicitly."
        )
    daily_volume = float(trades["volume"].sum()) * (day / span)
    return daily_volume / buckets_per_day


def _empty_vpin_frame(
    timestamp_dtype: np.dtype | pd.api.extensions.ExtensionDtype,
) -> pd.DataFrame:
    """Zero-row :func:`compute_vpin` result with the standard columns and dtypes.

    *timestamp_dtype* is the trades' ``timestamp`` dtype, so the bucket
    bounds keep the same time zone as a non-empty result.
    """
    return pd.DataFrame(
        {
            "bucket": pd.Series(dtype="int64"),
            "timestamp_start": pd.Series(dtype=timestamp_dtype),
            "timestamp_end": pd.Series(dtype=timestamp_dtype),
            "buy_volume": pd.Series(dtype="float64"),
            "sell_volume": pd.Series(dtype="float64"),
            "vpin": pd.Series(dtype="float64"),
            "vpin_avg": pd.Series(dtype="float64"),
        }
    )


def compute_vpin(
    trades: pd.DataFrame,
    bucket_volume: float | None = None,
    n_buckets: int = 50,
    sign_method: str | None = None,
    quotes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the Volume-Synchronized Probability of Informed Trading.

    Partitions cumulative trade volume into equal-sized *buckets* and
    measures the normalised buy/sell imbalance within each bucket.  The
    trailing average of ``vpin`` over *n_buckets* is the headline VPIN
    metric.

    Works on feeds without a native aggressor side.  When the trades frame
    has no ``direction``, the buy/sell split is inferred with a trade-sign
    classifier (see *sign_method*), so VPIN runs on L2 / aggregated captures
    too.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with at least ``timestamp``, ``price``, and ``volume``.  A
        ``direction`` column (``"buy"`` / ``"sell"``, the taker side) is used
        when present; otherwise it is inferred — see *sign_method*.
    bucket_volume : float, optional
        Total volume per bucket.  This is highly instrument-specific.  When
        left out, it is picked by :func:`vpin_bucket_volume` (average daily
        volume ÷ 50, with a 24-hour trading day).
    n_buckets : int, optional
        Window length (in buckets) for the trailing VPIN average.
        Default is 50, following the original paper.  Fewer complete buckets
        than this is reported in ``attrs["diagnostics"]``.
    sign_method : str, optional
        How to obtain the buy/sell split when there is no native
        ``direction``.  ``None`` (default) uses an existing ``direction`` if
        present, else falls back to a per-trade classifier (Lee–Ready when
        *quotes* are given, otherwise the tick rule).  ``"tick"`` /
        ``"lee_ready"`` force a per-trade classifier
        (:func:`~ob_analytics.trade_sign.classify_trade_sign`), overriding
        any native ``direction``.  ``"bvc"`` splits each *volume bar* with
        bulk volume classification
        (:func:`~ob_analytics.trade_sign.bulk_volume_classification`) — the
        VPIN-native estimator, which needs no per-trade sign at all.
    quotes : pandas.DataFrame, optional
        Quote frame for ``sign_method="lee_ready"`` (or the ``None`` fallback
        when quotes are available) — passed through to
        :func:`~ob_analytics.trade_sign.classify_trade_sign`.

    Returns
    -------
    pandas.DataFrame
        One row per completed bucket (zero rows, same columns and dtypes,
        when the trades fill no bucket) with columns:

        * ``bucket`` — zero-based bucket index
        * ``timestamp_start`` — first trade timestamp in the bucket
        * ``timestamp_end`` — last trade timestamp in the bucket
        * ``buy_volume`` — total buy volume in the bucket
        * ``sell_volume`` — total sell volume in the bucket
        * ``vpin`` — ``|buy_volume - sell_volume| / bucket_volume``
        * ``vpin_avg`` — trailing mean of ``vpin`` over *n_buckets*

        The frame's ``attrs`` record how it was computed, so the settings can
        be reported next to the number:

        * ``attrs["bucket_volume"]`` — the bucket size used
        * ``attrs["bucket_volume_rule"]`` — ``"given"`` when passed in,
          ``"adv/50"`` when picked by :func:`vpin_bucket_volume`
        * ``attrs["n_buckets"]`` — the trailing window, in buckets
        * ``attrs["diagnostics"]`` — a tuple of reasons the result should not
          be relied on; empty when there are none.  Today the one check is
          whether there are at least *n_buckets* complete buckets, since
          ``vpin_avg`` is not a full trailing average before that.  The same
          condition also raises a :class:`UserWarning`, since ``diagnostics``
          is easy to miss on a frame that otherwise looks fine.

    Raises
    ------
    ConfigError
        If required columns are missing.
    ObAnalyticsError
        If *trades* is empty.
    ValueError
        If *bucket_volume* is not positive, or it is left out and the trades
        span no time (see :func:`vpin_bucket_volume`).
    """
    validate_columns(trades, {"timestamp", "price", "volume"}, "compute_vpin")
    validate_non_empty(trades, "compute_vpin")
    rule = "given"
    if bucket_volume is None:
        bucket_volume = vpin_bucket_volume(trades)
        rule = f"adv/{VPIN_BUCKETS_PER_DAY}"
    if bucket_volume <= 0:
        raise ValueError(f"bucket_volume must be positive, got {bucket_volume}")

    if sign_method == "bvc":
        result = _vpin_from_bvc(trades, bucket_volume, n_buckets)
    else:
        result = _vpin_from_signs(trades, bucket_volume, n_buckets, sign_method, quotes)

    diagnostics: tuple[str, ...] = ()
    if len(result) < n_buckets:
        plural = "" if len(result) == 1 else "s"
        too_few = (
            f"only {len(result)} complete bucket{plural}, fewer than "
            f"n_buckets={n_buckets}; vpin_avg never averages a full window"
        )
        diagnostics = (too_few,)
        if rule == "given":
            advice = "Pass a smaller bucket_volume or a smaller n_buckets."
        else:
            advice = (
                "The default bucket_volume (average daily volume / "
                f"{VPIN_BUCKETS_PER_DAY}) is too large for a capture that runs "
                "well under a day; pass a smaller bucket_volume (see "
                "vpin_bucket_volume) or a smaller n_buckets."
            )
        warnings.warn(f"compute_vpin: {too_few}. {advice}", stacklevel=2)
    result.attrs["bucket_volume"] = float(bucket_volume)
    result.attrs["bucket_volume_rule"] = rule
    result.attrs["n_buckets"] = n_buckets
    result.attrs["diagnostics"] = diagnostics
    return result


def _vpin_from_signs(
    trades: pd.DataFrame,
    bucket_volume: float,
    n_buckets: int,
    sign_method: str | None,
    quotes: pd.DataFrame | None,
) -> pd.DataFrame:
    """VPIN from a per-trade buy/sell sign (every ``sign_method`` but ``"bvc"``)."""
    trades = resolve_direction(trades, sign_method, quotes, "compute_vpin")
    df = trades.sort_values("timestamp").reset_index(drop=True)

    # Assign signed volume
    is_buy = df["direction"] == "buy"
    buy_vol = df["volume"].where(is_buy, 0.0).to_numpy(dtype=np.float64)
    sell_vol = df["volume"].where(~is_buy, 0.0).to_numpy(dtype=np.float64)
    timestamps = df["timestamp"].to_numpy()

    # Walk through trades, splitting volume into equal-sized buckets.
    # A single trade can be split across two buckets if it straddles a
    # boundary.
    buckets: list[dict] = []
    bucket_buy = 0.0
    bucket_sell = 0.0
    bucket_start_ts = timestamps[0]
    bucket_remaining = bucket_volume

    for i in range(len(df)):
        trade_buy = buy_vol[i]
        trade_sell = sell_vol[i]
        trade_total = trade_buy + trade_sell

        while trade_total > 0:
            alloc = min(trade_total, bucket_remaining)
            # Proportionally split the buy/sell within this trade
            frac = alloc / trade_total

            bucket_buy += trade_buy * frac
            bucket_sell += trade_sell * frac
            trade_buy -= trade_buy * frac
            trade_sell -= trade_sell * frac
            trade_total -= alloc
            bucket_remaining -= alloc

            if bucket_remaining <= 1e-12:
                # Bucket is full
                buckets.append(
                    {
                        "bucket": len(buckets),
                        "timestamp_start": bucket_start_ts,
                        "timestamp_end": timestamps[i],
                        "buy_volume": bucket_buy,
                        "sell_volume": bucket_sell,
                        "vpin": abs(bucket_buy - bucket_sell) / bucket_volume,
                    }
                )
                bucket_buy = 0.0
                bucket_sell = 0.0
                bucket_remaining = bucket_volume
                # Next bucket starts at the same trade timestamp
                bucket_start_ts = timestamps[i]

    if not buckets:
        return _empty_vpin_frame(df["timestamp"].dtype)
    result = pd.DataFrame(buckets)
    result["vpin_avg"] = result["vpin"].rolling(n_buckets, min_periods=1).mean()
    return result


def _vpin_from_bvc(
    trades: pd.DataFrame,
    bucket_volume: float,
    n_buckets: int,
) -> pd.DataFrame:
    """VPIN from bulk volume classification (``sign_method="bvc"``).

    Splits each volume bar with
    :func:`~ob_analytics.trade_sign.bulk_volume_classification` instead of a
    per-trade sign, then measures the same normalised bucket imbalance.
    Returns the standard :func:`compute_vpin` schema.
    """
    bvc = bulk_volume_classification(trades, bucket_volume)
    if bvc.empty:
        return _empty_vpin_frame(trades["timestamp"].dtype)
    result = bvc[
        ["bucket", "timestamp_start", "timestamp_end", "buy_volume", "sell_volume"]
    ].copy()
    result["vpin"] = (
        result["buy_volume"] - result["sell_volume"]
    ).abs() / bucket_volume
    result["vpin_avg"] = result["vpin"].rolling(n_buckets, min_periods=1).mean()
    return result


# ── Kyle's Lambda ────────────────────────────────────────────────────


def compute_kyle_lambda(
    trades: pd.DataFrame,
    window: str = "5min",
    *,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int | np.random.Generator | None = 0,
) -> KyleLambdaResult:
    """Estimate Kyle's Lambda via OLS regression.

    For each time *window*, computes:

    * **ΔPrice** = last trade price − first trade price
    * **signed_volume** = Σ(buy volume) − Σ(sell volume)

    Then regresses ΔPrice on signed_volume across all windows.  The
    slope (λ) measures how much the price moves per unit of net order
    flow — a proxy for market illiquidity and adverse selection.

    ΔPrice is read directly from ``trades["price"]``, which is an integer tick
    count, so λ is in **ticks** per unit volume.  It scales with the price
    unit — a run at a finer ``tick_size`` reports a proportionally larger λ;
    multiply by ``tick_size`` to express it in the quote currency.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with ``timestamp``, ``price``, ``volume``, ``direction``.
        Unlike :func:`compute_vpin` and :func:`order_flow_imbalance`, the
        aggressor side is required rather than inferred, so this needs a feed
        that labels it (or a ``direction`` attached beforehand with
        :func:`~ob_analytics.trade_sign.classify_trade_sign`).  Individual
        rows the venue left blank are filled with the tick rule.
    window : str, optional
        Pandas frequency string for grouping trades.  Default ``"5min"``.
    n_boot : int, optional
        Number of bootstrap resamples for the confidence interval.  Default
        1000; ``0`` skips the interval.  Each resample draws blocks of
        consecutive windows with replacement (a moving block bootstrap, with
        blocks of ``ceil(n_windows ** (1/3))`` windows), so correlation
        between neighbouring windows is kept, and refits the slope.  The
        interval is the percentile range of those slopes.
    ci_level : float, optional
        Coverage of the interval, strictly between 0 and 1.  Default 0.95.
    seed : int or numpy.random.Generator, optional
        Seed or generator for the bootstrap.  Default ``0``, so the same
        trades always give the same interval; pass ``None`` for fresh
        randomness.

    Returns
    -------
    KyleLambdaResult
        Frozen dataclass with ``lambda_``, ``t_stat``, ``r_squared``,
        ``n_windows``, ``regression_df``, the interval ``ci_low`` /
        ``ci_high``, and ``significant`` / ``diagnostics``, which say when λ
        rests on too little data to rely on.

    Raises
    ------
    ConfigError
        If required columns are missing.
    ObAnalyticsError
        If *trades* is empty.
    ValueError
        If *ci_level* is not between 0 and 1, or *n_boot* is negative.
    """
    validate_columns(
        trades,
        {"timestamp", "price", "volume", "direction"},
        "compute_kyle_lambda",
    )
    validate_non_empty(trades, "compute_kyle_lambda")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be between 0 and 1, got {ci_level}")
    if n_boot < 0:
        raise ValueError(f"n_boot must not be negative, got {n_boot}")

    # `direction` is required here, but a column being present does not make
    # every row in it usable: the signed volume below reads it as == "buy" and
    # takes the rest as a sell, so a blank would be counted on the wrong side
    # rather than skipped.  resolve_direction infers those rows instead.
    trades = resolve_direction(trades, None, None, "compute_kyle_lambda")
    df = trades.sort_values("timestamp").copy()
    df["signed_volume"] = df["volume"].where(df["direction"] == "buy", -df["volume"])

    # Group by time window
    grouped = df.groupby(pd.Grouper(key="timestamp", freq=window))
    rows = []
    for ts, group in grouped:
        if group.empty:
            continue
        dp = group["price"].iloc[-1] - group["price"].iloc[0]
        sv = group["signed_volume"].sum()
        rows.append({"timestamp": ts, "delta_price": dp, "signed_volume": sv})

    reg_df = pd.DataFrame(rows)

    def _nan_result(n: int) -> KyleLambdaResult:
        return KyleLambdaResult(
            lambda_=float("nan"),
            t_stat=float("nan"),
            r_squared=float("nan"),
            n_windows=n,
            regression_df=reg_df,
            ci_level=ci_level,
        )

    n = len(reg_df)
    if n < 2:
        return _nan_result(n)

    # OLS fit y = α + λ·x via least squares (numerically safer than the
    # explicit normal-equations inverse it replaces; rank flags singularity).
    x = reg_df["signed_volume"].to_numpy(dtype=np.float64)
    y = reg_df["delta_price"].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(n), x])  # design matrix [1, x]

    beta, _residuals, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    if rank < 2:  # singular design (e.g. all signed_volume equal)
        return _nan_result(n)
    lambda_ = float(beta[1])

    y_hat = X @ beta
    residuals = y - y_hat
    ss_res = float(residuals @ residuals)
    ss_tot = float((y - y.mean()) @ (y - y.mean()))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")

    if n > 2:
        cov = np.linalg.inv(X.T @ X)  # safe: rank == 2 guarantees invertible
        mse = ss_res / (n - 2)
        se_lambda = float(np.sqrt(mse * cov[1, 1]))
        t_stat = lambda_ / se_lambda if se_lambda > 1e-15 else float("nan")
    else:
        t_stat = float("nan")

    ci_low, ci_high = _block_bootstrap_slope_ci(
        x, y, n_boot, ci_level, np.random.default_rng(seed)
    )

    return KyleLambdaResult(
        lambda_=lambda_,
        t_stat=t_stat,
        r_squared=r_squared,
        n_windows=n,
        regression_df=reg_df,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci_level,
    )


def _block_bootstrap_slope_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile interval for the OLS slope of *y* on *x*, by moving blocks.

    Returns ``(nan, nan)`` when there are fewer than three points, no
    resamples were asked for, or fewer than two resamples had any spread in
    *x* to fit a slope to.
    """
    nan = float("nan")
    n = len(x)
    if n_boot == 0 or n < 3:
        return nan, nan
    block = int(np.ceil(n ** (1 / 3)))
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    # Resamples are handled a chunk of rows at a time, so memory stays near
    # _BOOTSTRAP_CHUNK values however many windows there are.
    rows_per_chunk = max(1, _BOOTSTRAP_CHUNK // n)
    chunks = []
    for first in range(0, n_boot, rows_per_chunk):
        part = starts[first : first + rows_per_chunk]
        # Each row is n window indices, built from n_blocks runs of `block`
        # consecutive windows, cut back to n.
        idx = (part[:, :, None] + np.arange(block)).reshape(len(part), -1)[:, :n]
        xs, ys = x[idx], y[idx]
        usable = np.ptp(xs, axis=1) > 0  # one x value in a resample: no slope
        xs, ys = xs[usable], ys[usable]
        xc = xs - xs.mean(axis=1, keepdims=True)
        yc = ys - ys.mean(axis=1, keepdims=True)
        chunks.append((xc * yc).sum(axis=1) / (xc * xc).sum(axis=1))
    slopes = np.concatenate(chunks)
    if len(slopes) < 2:
        return nan, nan
    tail = (1.0 - ci_level) / 2.0
    low, high = np.quantile(slopes, [tail, 1.0 - tail])
    return float(low), float(high)


# ── Order Flow Imbalance ─────────────────────────────────────────────


def order_flow_imbalance(
    trades: pd.DataFrame,
    window: str = "1min",
    sign_method: str | None = None,
    quotes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute normalised order flow imbalance per time window.

    For each *window*:

    * ``ofi = (buy_volume − sell_volume) / (buy_volume + sell_volume)``

    Values range from −1 (all sells) to +1 (all buys).  Zero indicates
    balanced flow.

    Works on feeds without a native aggressor side: when the trades frame
    has no ``direction``, it is inferred with a per-trade trade-sign
    classifier (see *sign_method*).

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with ``timestamp`` and ``volume``.  A ``direction`` column
        (``"buy"`` / ``"sell"``) is used when present; otherwise it is
        inferred — see *sign_method* (which additionally requires
        ``price``).
    window : str, optional
        Pandas frequency string.  Default ``"1min"``.
    sign_method : str, optional
        How to obtain the buy/sell split when there is no native
        ``direction``.  ``None`` (default) uses an existing ``direction`` if
        present, else falls back to a per-trade classifier (Lee–Ready when
        *quotes* are given, otherwise the tick rule).  ``"tick"`` /
        ``"lee_ready"`` force a per-trade classifier, overriding any native
        ``direction``.  (``"bvc"`` is VPIN-native; use
        :func:`compute_vpin`.)
    quotes : pandas.DataFrame, optional
        Quote frame for ``sign_method="lee_ready"`` — passed through to
        :func:`~ob_analytics.trade_sign.classify_trade_sign`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp``, ``buy_volume``, ``sell_volume``,
        ``net_volume``, ``ofi``.

    Raises
    ------
    ConfigError
        If required columns are missing.
    ObAnalyticsError
        If *trades* is empty.
    """
    validate_columns(trades, {"timestamp", "volume"}, "order_flow_imbalance")
    validate_non_empty(trades, "order_flow_imbalance")

    trades = resolve_direction(trades, sign_method, quotes, "order_flow_imbalance")
    df = trades.sort_values("timestamp").copy()
    df["buy_vol"] = df["volume"].where(df["direction"] == "buy", 0.0)
    df["sell_vol"] = df["volume"].where(df["direction"] != "buy", 0.0)

    grouped = df.groupby(pd.Grouper(key="timestamp", freq=window)).agg(
        buy_volume=("buy_vol", "sum"),
        sell_volume=("sell_vol", "sum"),
    )
    grouped = grouped[(grouped["buy_volume"] > 0) | (grouped["sell_volume"] > 0)]
    grouped = grouped.reset_index()

    total = grouped["buy_volume"] + grouped["sell_volume"]
    grouped["net_volume"] = grouped["buy_volume"] - grouped["sell_volume"]
    grouped["ofi"] = grouped["net_volume"] / total.replace(0, np.nan)

    return grouped
