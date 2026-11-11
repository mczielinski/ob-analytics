"""Order flow toxicity and imbalance metrics.

Implements market microstructure measures for detecting informed trading and
quantifying price impact:

* :func:`vpin` — Volume-Synchronized Probability of Informed
  Trading (Easley, López de Prado & O'Hara, 2012).
* :func:`kyle_lambda` — Kyle's Lambda price-impact coefficient
  (Kyle, 1985).
* :func:`order_flow_imbalance` — Normalised buy/sell volume imbalance
  per time window.
* :func:`ofi_by_window` — the same imbalance over several trailing windows
  on one time grid.

All functions accept a trades DataFrame from the pipeline (or any
DataFrame with the required columns).  Every result is in the instrument's
own units — prices in the quote currency, sizes in BTC, shares or contracts —
whether the trades hold integer ticks and lots (with ``tick_size`` and
``lot_size`` on the frame's ``attrs``, as on a pipeline result) or display
prices and sizes.

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
from typing import Literal

import numpy as np
import pandas as pd

from ob_analytics._utils import (
    fixed_window,
    price_scale,
    size_scale,
    validate_non_empty,
    window_start,
)
from ob_analytics.bars import bars
from ob_analytics.exceptions import ConfigError
from ob_analytics.schemas import validate_columns
from ob_analytics.trade_sign import (
    _bucket_bounds,
    _bulk_volume_classification,
    _rows_at,
    resolve_direction,
    volume_buckets,
)

__all__ = [
    "KYLE_MIN_T_STAT",
    "KYLE_MIN_WINDOWS",
    "OFI_WINDOWS",
    "VPIN_BUCKETS_PER_DAY",
    "KyleLambdaResult",
    "kyle_lambda",
    "ofi_by_window",
    "order_flow_imbalance",
    "vpin",
    "vpin_bucket_volume",
]

#: Smallest ``|t|`` at which λ counts as distinguishable from zero.  The
#: usual rule of thumb, about a 5% two-sided test on a large sample.
KYLE_MIN_T_STAT = 2.0

#: Fewest regression windows for λ to mean much.  Below this the t-statistic
#: itself is unstable, whatever its value.
KYLE_MIN_WINDOWS = 30

#: Buckets in one average day of volume, for :func:`vpin_bucket_volume`.
#: Fifty is the rule used by Easley, López de Prado and O'Hara (2012), and
#: matches the default ``n_buckets`` of :func:`vpin`: the trailing
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
        Slope — price change in the quote currency per unit of signed order
        flow in the instrument's units (higher = less liquid).
    t_stat : float
        t-statistic for ``lambda_``.
    r_squared : float
        Fraction of ΔPrice variance explained by signed order flow.
    n_windows : int
        Number of time windows in the regression.
    regression_df : pandas.DataFrame
        Per-window ``timestamp``/``delta_price``/``signed_volume`` data, in
        the quote currency and the instrument's units.
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
    and VPIN will fill only a few of them, which :func:`vpin` then
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
        The bucket volume, in the instrument's own units (BTC, shares,
        contracts), ready to pass to :func:`vpin` as
        ``bucket_volume``.

    Raises
    ------
    ConfigError
        If required columns are missing, or the sizes are integer lots and
        the frame carries no ``lot_size``.
        Also if *trades* is empty.
        Also if the trades span no time, or *buckets_per_day* or *trading_day* is
        not positive.
    """
    validate_columns(trades, {"timestamp", "volume"}, "vpin_bucket_volume")
    validate_non_empty(trades, "vpin_bucket_volume")
    scale = size_scale(trades, "vpin_bucket_volume")
    if buckets_per_day <= 0:
        raise ConfigError(f"buckets_per_day must be positive, got {buckets_per_day}")
    day = pd.Timedelta(trading_day)
    if day <= pd.Timedelta(0):
        raise ConfigError(f"trading_day must be positive, got {trading_day!r}")
    span = trades["timestamp"].max() - trades["timestamp"].min()
    if span <= pd.Timedelta(0):
        raise ConfigError(
            "vpin_bucket_volume: the trades span no time, so there is no rate "
            "to scale to a day; pass bucket_volume explicitly."
        )
    daily_volume = float(trades["volume"].sum()) / scale * (day / span)
    return daily_volume / buckets_per_day


def _empty_vpin_frame(
    timestamp_dtype: np.dtype | pd.api.extensions.ExtensionDtype,
) -> pd.DataFrame:
    """Zero-row :func:`vpin` result with the standard columns and dtypes.

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


def vpin(
    trades: pd.DataFrame,
    bucket_volume: float | Literal["fit"] | None = None,
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
        Total volume per bucket, in the instrument's own units (BTC, shares,
        contracts).  It means the same amount whether ``trades["volume"]``
        holds integer lots (with ``trades.attrs["lot_size"]``, as on a
        pipeline result) or display sizes (see
        :func:`~ob_analytics.visualization.display_result`).  This is highly
        instrument-specific, so size it from the data.  When left out, it is
        picked by :func:`vpin_bucket_volume` (average daily volume ÷ 50, with
        a 24-hour trading day).  ``"fit"`` picks it the same way, unless that
        bucket would fill fewer than *n_buckets* buckets, as it does on a
        capture shorter than a day; then each bucket is the trades' total
        volume ÷ (2 × *n_buckets*), so they fill about twice *n_buckets*.
        The gallery uses ``"fit"``, so a short capture still plots.
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
        * ``buy_volume`` — total buy volume in the bucket, in the
          instrument's own units
        * ``sell_volume`` — total sell volume in the bucket, likewise
        * ``vpin`` — ``|buy_volume - sell_volume| / bucket_volume``
        * ``vpin_avg`` — trailing mean of ``vpin`` over *n_buckets*

        The frame's ``attrs`` record how it was computed, so the settings can
        be reported next to the number:

        * ``attrs["bucket_volume"]`` — the bucket size used, in the
          instrument's own units
        * ``attrs["bucket_volume_rule"]`` — ``"given"`` when passed in,
          ``"adv/50"`` when picked by :func:`vpin_bucket_volume`, and
          ``"total/100"`` (for *n_buckets* 50) when ``"fit"`` sized it from
          the total volume
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
        If required columns are missing, or the sizes are integer lots and
        the frame carries no ``lot_size``.
        Also if *trades* is empty.
        Also if *bucket_volume* is a string other than ``"fit"``, is not
        positive, would make more than
        :data:`~ob_analytics.trade_sign.MAX_VOLUME_BUCKETS` buckets, or is
        left out and the trades span no time (see :func:`vpin_bucket_volume`).
    """
    validate_columns(trades, {"timestamp", "price", "volume"}, "vpin")
    validate_non_empty(trades, "vpin")
    rule = "given"
    advice: str | None = None
    if isinstance(bucket_volume, str) and bucket_volume != "fit":
        raise ConfigError(
            f"vpin: bucket_volume must be a number, None or 'fit', "
            f"got {bucket_volume!r}"
        )
    if bucket_volume is None or bucket_volume == "fit":
        fit = bucket_volume == "fit"
        bucket_volume = vpin_bucket_volume(trades)
        rule = f"adv/{VPIN_BUCKETS_PER_DAY}"
        if fit:
            total = float(trades["volume"].sum()) / size_scale(trades, "vpin")
            if total < n_buckets * bucket_volume:
                # A day's volume ÷ 50 fills fewer than n_buckets buckets: size
                # them from the total so the trades fill about 2 × n_buckets.
                bucket_volume = total / (2 * n_buckets)
                rule = f"total/{2 * n_buckets}"
        # The default rule makes about 50 buckets per day the trades span, so
        # only a span of decades trips the cap: a timestamp problem, not units.
        span = trades["timestamp"].max() - trades["timestamp"].min()
        advice = (
            f"vpin_bucket_volume sized the bucket from the trades' time span "
            f"of {span}; check the timestamps, or pass bucket_volume."
        )

    if sign_method == "bvc":
        result = _vpin_from_bvc(trades, bucket_volume, n_buckets, advice)
    else:
        result = _vpin_from_signs(
            trades, bucket_volume, n_buckets, sign_method, quotes, advice
        )

    diagnostics: tuple[str, ...] = ()
    if len(result) < n_buckets:
        plural = "" if len(result) == 1 else "s"
        too_few = (
            f"only {len(result)} complete bucket{plural}, fewer than "
            f"n_buckets={n_buckets}; vpin_avg never averages a full window"
        )
        diagnostics = (too_few,)
        # Both a given and a defaulted bucket_volume are named here — the
        # default rule sizes buckets from a fixed 50-per-day constant, not
        # from n_buckets, so a caller who raises n_buckets above 50 can see
        # this even on a capture that runs well over a day; naming a cause
        # (e.g. "too large for a capture under a day") would be wrong there.
        see_rule = " (see vpin_bucket_volume)" if rule != "given" else ""
        advice = f"Pass a smaller bucket_volume{see_rule} or a smaller n_buckets."
        warnings.warn(f"vpin: {too_few}. {advice}", stacklevel=2)
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
    advice: str | None,
) -> pd.DataFrame:
    """VPIN from a per-trade buy/sell sign (every ``sign_method`` but ``"bvc"``).

    The bucket volumes are summed in the frame's units, as
    :func:`~ob_analytics.trade_sign.volume_buckets` gives them, and reported in
    the instrument's.
    """
    scale = size_scale(trades, "vpin")
    trades = resolve_direction(trades, sign_method, quotes, "vpin")
    df = trades.sort_values("timestamp").reset_index(drop=True)
    buckets = volume_buckets(df, bucket_volume, "vpin", advice=advice)
    if buckets.empty:
        return _empty_vpin_frame(df["timestamp"].dtype)

    # A trade is all buy or all sell, so each part of it in a bucket is too.
    is_buy = (df["direction"] == "buy").to_numpy()[buckets["row"].to_numpy()]
    volume = buckets["volume"]
    buy = volume.where(is_buy, 0.0).groupby(buckets["bucket"]).sum()
    sell = volume.where(~is_buy, 0.0).groupby(buckets["bucket"]).sum()
    starts, ends = _bucket_bounds(buckets)
    result = pd.DataFrame(
        {
            "bucket": buy.index.to_numpy(),
            "timestamp_start": _rows_at(df["timestamp"], starts),
            "timestamp_end": _rows_at(df["timestamp"], ends),
            "buy_volume": buy.to_numpy() / scale,
            "sell_volume": sell.to_numpy() / scale,
            "vpin": (buy - sell).abs().to_numpy() / (bucket_volume * scale),
        }
    )
    result["vpin_avg"] = result["vpin"].rolling(n_buckets, min_periods=1).mean()
    return result


def _vpin_from_bvc(
    trades: pd.DataFrame,
    bucket_volume: float,
    n_buckets: int,
    advice: str | None,
) -> pd.DataFrame:
    """VPIN from bulk volume classification (``sign_method="bvc"``).

    Splits each volume bar with
    :func:`~ob_analytics.trade_sign.bulk_volume_classification` instead of a
    per-trade sign, then measures the same normalised bucket imbalance.
    Returns the standard :func:`vpin` schema.
    """
    bvc = _bulk_volume_classification(trades, bucket_volume, None, "vpin", advice)
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


def kyle_lambda(
    trades: pd.DataFrame,
    window: str = "5min",
    *,
    sign_method: str | None = None,
    quotes: pd.DataFrame | None = None,
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

    λ is in the quote currency per unit of size (for example USD per BTC),
    whether the trades hold integer ticks and lots (with ``tick_size`` and
    ``lot_size`` on the frame's ``attrs``, as on a pipeline result) or
    display prices and sizes.  ``regression_df`` is in the same units.

    Works on feeds without a native aggressor side.  When the trades frame
    has no ``direction``, the side is inferred with a trade-sign classifier
    (see *sign_method*), as for :func:`vpin` and
    :func:`order_flow_imbalance`.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with ``timestamp``, ``price`` and ``volume``.  A ``direction``
        column (``"buy"`` / ``"sell"``, the taker side) is used when present;
        otherwise it is inferred — see *sign_method*.
    window : str, optional
        Length of each regression window, a fixed duration such as ``"30s"``
        or ``"5min"``.  Default ``"5min"``.  Windows are the time bars of
        :func:`~ob_analytics.bars.bars`: they count from the Unix epoch, and a
        window in which nothing traded is left out.
    sign_method : str, optional
        How to obtain the buy/sell split when there is no native
        ``direction``.  ``None`` (default) uses an existing ``direction`` if
        present, else falls back to a per-trade classifier (Lee–Ready when
        *quotes* are given, otherwise the tick rule).  ``"tick"`` /
        ``"lee_ready"`` force a per-trade classifier, overriding any native
        ``direction``.  (``"bvc"`` is VPIN-native; use :func:`vpin`.)
    quotes : pandas.DataFrame, optional
        Quote frame for ``sign_method="lee_ready"`` — passed through to
        :func:`~ob_analytics.trade_sign.classify_trade_sign`.
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
        If required columns are missing, *window* is not a fixed duration,
        or the prices or sizes are integers and the frame carries no
        ``tick_size`` or ``lot_size``.
        Also if *trades* is empty.
        Also if *ci_level* is not between 0 and 1, or *n_boot* is negative.
    """
    validate_columns(trades, {"timestamp", "price", "volume"}, "kyle_lambda")
    validate_non_empty(trades, "kyle_lambda")
    prices = price_scale(trades, "kyle_lambda")
    sizes = size_scale(trades, "kyle_lambda")
    if not 0.0 < ci_level < 1.0:
        raise ConfigError(f"ci_level must be between 0 and 1, got {ci_level}")
    if n_boot < 0:
        raise ConfigError(f"n_boot must not be negative, got {n_boot}")
    step = fixed_window(window, "kyle_lambda: window")

    trades = resolve_direction(trades, sign_method, quotes, "kyle_lambda")
    # One time bar per window: ΔPrice is the bar's close minus its open.  The
    # bars keep the frame's units, so both columns are converted here.
    windows = bars(trades, "time", step)
    reg_df = pd.DataFrame(
        {
            "timestamp": window_start(windows["timestamp_start"], step),
            "delta_price": (windows["close"] - windows["open"]) / prices,
            "signed_volume": windows["signed_volume"] / sizes,
        }
    )

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

    Each window is one time bar of :func:`~ob_analytics.bars.bars`, so
    ``ofi`` is that bar's ``signed_volume / volume``.  Windows count from the
    Unix epoch, and a window in which nothing traded is left out.  A window
    whose trades all have zero size is kept, with ``ofi`` set to ``NaN``.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with ``timestamp``, ``price`` and ``volume``.  A ``direction``
        column (``"buy"`` / ``"sell"``) is used when present; otherwise it is
        inferred — see *sign_method*.
    window : str, optional
        Length of each window, a fixed duration such as ``"7s"`` or
        ``"1min"``.  Default ``"1min"``.
    sign_method : str, optional
        How to obtain the buy/sell split when there is no native
        ``direction``.  ``None`` (default) uses an existing ``direction`` if
        present, else falls back to a per-trade classifier (Lee–Ready when
        *quotes* are given, otherwise the tick rule).  ``"tick"`` /
        ``"lee_ready"`` force a per-trade classifier, overriding any native
        ``direction``.  (``"bvc"`` is VPIN-native; use
        :func:`vpin`.)
    quotes : pandas.DataFrame, optional
        Quote frame for ``sign_method="lee_ready"`` — passed through to
        :func:`~ob_analytics.trade_sign.classify_trade_sign`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp`` (the start of the window), ``buy_volume``,
        ``sell_volume``, ``signed_volume`` (in the instrument's own units, BTC,
        shares or contracts, whether the trades hold integer lots or display
        sizes), ``ofi``.

    Raises
    ------
    ConfigError
        If required columns are missing, *window* is not a fixed duration,
        or the sizes are integer lots and the frame carries no ``lot_size``.
        Also if *trades* is empty.
    """
    return _ofi_windows(trades, window, sign_method, quotes)[0]


def _ofi_windows(
    trades: pd.DataFrame,
    window: str,
    sign_method: str | None,
    quotes: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the :func:`order_flow_imbalance` table and the time bars it came from.

    The two frames share a row order, so a caller that needs more of each
    window (its closing price, say) reads it from the bars.  The table is in
    the instrument's units; the bars keep the units of *trades*.
    """
    validate_columns(trades, {"timestamp", "price", "volume"}, "order_flow_imbalance")
    validate_non_empty(trades, "order_flow_imbalance")
    step = fixed_window(window, "order_flow_imbalance: window")
    sizes = size_scale(trades, "order_flow_imbalance")

    trades = resolve_direction(trades, sign_method, quotes, "order_flow_imbalance")
    windows = bars(trades, "time", step)
    out = pd.DataFrame(
        {
            "timestamp": window_start(windows["timestamp_start"], step),
            "buy_volume": windows["buy_volume"] / sizes,
            "sell_volume": windows["sell_volume"] / sizes,
            "signed_volume": windows["signed_volume"] / sizes,
        }
    )
    # A ratio of two sizes in the bars' own units, so it needs no conversion.
    out["ofi"] = windows["signed_volume"] / windows["volume"].replace(0, np.nan)
    return out, windows


#: Trailing windows :func:`ofi_by_window` measures by default, shortest first.
OFI_WINDOWS: tuple[str, ...] = ("5s", "15s", "60s", "300s")


def ofi_by_window(
    trades: pd.DataFrame,
    windows: tuple[str, ...] = OFI_WINDOWS,
    grid: str = "5s",
    sign_method: str | None = None,
    quotes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute order flow imbalance over several trailing windows on one grid.

    :func:`order_flow_imbalance` measures one window at a time.  This measures
    each of *windows* at every step of a common time *grid*, so pressure over short
    and long windows can be compared at the same instant.  Each value is a
    *trailing* window ending at the grid step, not a block of the grid, so a
    long window changes slowly and a short one changes quickly.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with ``timestamp`` and ``volume``.  A ``direction`` column is
        used when present; otherwise it is inferred — see *sign_method*.
    windows : tuple of str, optional
        Fixed durations, one per window.  Each is rounded to a whole number
        of *grid* steps, with a minimum of one.  Default
        :data:`OFI_WINDOWS`.
    grid : str, optional
        Step of the time grid, a fixed duration.  Default ``"5s"``.  Grid
        steps count from the Unix epoch, as the windows of
        :func:`order_flow_imbalance` do.
    sign_method, quotes : optional
        How to obtain the buy/sell split, as for :func:`order_flow_imbalance`.

    Returns
    -------
    pandas.DataFrame
        One row per grid step, with ``timestamp`` and one column per window,
        named by its offset string.  Each value is OFI in ``[-1, +1]`` (buy
        pressure positive), or ``NaN`` when nothing traded in that window.

    Raises
    ------
    ConfigError
        If required columns are missing, or *grid* or a window is not a
        fixed duration.
        Also if *trades* is empty.
    """
    validate_columns(trades, {"timestamp", "volume"}, "ofi_by_window")
    validate_non_empty(trades, "ofi_by_window")
    trades = resolve_direction(trades, sign_method, quotes, "ofi_by_window")

    step = fixed_window(grid, "ofi_by_window: grid")
    timestamps = trades["timestamp"]
    # Signed volume binned onto the fine grid, then trailing-summed per window.
    floored = window_start(timestamps, step)
    # The grid runs to the step at or after the last trade.
    last = floored.max()
    end = last if timestamps.max() == last else last + step
    index = pd.date_range(floored.min(), end, freq=step)
    is_buy = trades["direction"] == "buy"
    volume = trades["volume"]
    buy = (
        volume.where(is_buy, 0.0).groupby(floored).sum().reindex(index, fill_value=0.0)
    )
    sell = (
        volume.where(~is_buy, 0.0).groupby(floored).sum().reindex(index, fill_value=0.0)
    )

    out = pd.DataFrame({"timestamp": index})
    for window in windows:
        length = fixed_window(window, "ofi_by_window: window")
        k = max(round(length / step), 1)
        b = buy.rolling(k, min_periods=1).sum()
        s = sell.rolling(k, min_periods=1).sum()
        total = (b + s).replace(0.0, np.nan)
        out[window] = ((b - s) / total).to_numpy()
    return out
