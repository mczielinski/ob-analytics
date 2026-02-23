"""Order flow toxicity and imbalance metrics.

Implements three market microstructure measures for detecting informed
trading and quantifying price impact:

* :func:`compute_vpin` — Volume-Synchronized Probability of Informed
  Trading (Easley, López de Prado & O'Hara, 2012).
* :func:`compute_kyle_lambda` — Kyle's Lambda price-impact coefficient
  (Kyle, 1985).
* :func:`order_flow_imbalance` — Normalised buy/sell volume imbalance
  per time window.

All functions accept a trades DataFrame produced by
:class:`~ob_analytics.trades.DefaultTradeInferrer` (or any DataFrame
with the required columns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ob_analytics._utils import validate_columns, validate_non_empty
from ob_analytics.models import KyleLambdaResult


# ── VPIN ─────────────────────────────────────────────────────────────


def compute_vpin(
    trades: pd.DataFrame,
    bucket_volume: float,
    n_buckets: int = 50,
) -> pd.DataFrame:
    """Compute the Volume-Synchronized Probability of Informed Trading.

    Partitions cumulative trade volume into equal-sized *buckets* and
    measures the normalised buy/sell imbalance within each bucket.  The
    trailing average of ``vpin`` over *n_buckets* is the headline VPIN
    metric.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with at least ``timestamp``, ``price``, ``volume``, and
        ``direction`` columns.  ``direction`` must contain ``"buy"`` or
        ``"sell"`` values.
    bucket_volume : float
        Total volume per bucket.  This is highly instrument-specific —
        a reasonable starting point is average daily volume / 50.
    n_buckets : int, optional
        Window length (in buckets) for the trailing VPIN average.
        Default is 50, following the original paper.

    Returns
    -------
    pandas.DataFrame
        One row per completed bucket with columns:

        * ``bucket`` — zero-based bucket index
        * ``timestamp_start`` — first trade timestamp in the bucket
        * ``timestamp_end`` — last trade timestamp in the bucket
        * ``buy_volume`` — total buy volume in the bucket
        * ``sell_volume`` — total sell volume in the bucket
        * ``vpin`` — ``|buy_volume - sell_volume| / bucket_volume``
        * ``vpin_avg`` — trailing mean of ``vpin`` over *n_buckets*

    Raises
    ------
    InvalidDataError
        If required columns are missing.
    InsufficientDataError
        If *trades* is empty.
    ValueError
        If *bucket_volume* is not positive.
    """
    validate_columns(
        trades,
        {"timestamp", "price", "volume", "direction"},
        "compute_vpin",
    )
    validate_non_empty(trades, "compute_vpin")
    if bucket_volume <= 0:
        raise ValueError(f"bucket_volume must be positive, got {bucket_volume}")

    df = trades.sort_values("timestamp").reset_index(drop=True)

    # Assign signed volume
    is_buy = df["direction"] == "buy"
    buy_vol = df["volume"].where(is_buy, 0.0).to_numpy(dtype=np.float64)
    sell_vol = df["volume"].where(~is_buy, 0.0).to_numpy(dtype=np.float64)
    cum_vol = df["volume"].cumsum().to_numpy(dtype=np.float64)
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
            if trade_total > 0:
                frac = alloc / trade_total
            else:
                frac = 0.0

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

    result = pd.DataFrame(buckets)
    if not result.empty:
        result["vpin_avg"] = result["vpin"].rolling(n_buckets, min_periods=1).mean()
    return result


# ── Kyle's Lambda ────────────────────────────────────────────────────


def compute_kyle_lambda(
    trades: pd.DataFrame,
    window: str = "5min",
) -> KyleLambdaResult:
    """Estimate Kyle's Lambda via OLS regression.

    For each time *window*, computes:

    * **ΔPrice** = last trade price − first trade price
    * **signed_volume** = Σ(buy volume) − Σ(sell volume)

    Then regresses ΔPrice on signed_volume across all windows.  The
    slope (λ) measures how much the price moves per unit of net order
    flow — a proxy for market illiquidity and adverse selection.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with ``timestamp``, ``price``, ``volume``, ``direction``.
    window : str, optional
        Pandas frequency string for grouping trades.  Default ``"5min"``.

    Returns
    -------
    KyleLambdaResult
        Frozen dataclass with ``lambda_``, ``t_stat``, ``r_squared``,
        ``n_windows``, and ``regression_df``.

    Raises
    ------
    InvalidDataError
        If required columns are missing.
    InsufficientDataError
        If *trades* is empty.
    """
    validate_columns(
        trades,
        {"timestamp", "price", "volume", "direction"},
        "compute_kyle_lambda",
    )
    validate_non_empty(trades, "compute_kyle_lambda")

    df = trades.sort_values("timestamp").copy()
    df["signed_volume"] = df["volume"].where(
        df["direction"] == "buy", -df["volume"]
    )

    # Group by time window
    grouped = df.groupby(pd.Grouper(key="timestamp", freq=window))
    rows = []
    for ts, group in grouped:
        if group.empty:
            continue
        dp = group["price"].iloc[-1] - group["price"].iloc[0]
        sv = group["signed_volume"].sum()
        rows.append(
            {"timestamp": ts, "delta_price": dp, "signed_volume": sv}
        )

    reg_df = pd.DataFrame(rows)

    # Need at least 2 points for a regression
    if len(reg_df) < 2:
        return KyleLambdaResult(
            lambda_=float("nan"),
            t_stat=float("nan"),
            r_squared=float("nan"),
            n_windows=len(reg_df),
            regression_df=reg_df,
        )

    # OLS via normal equations: y = α + λ·x
    x = reg_df["signed_volume"].to_numpy(dtype=np.float64)
    y = reg_df["delta_price"].to_numpy(dtype=np.float64)
    n = len(x)

    X = np.column_stack([np.ones(n), x])  # [1, x] design matrix
    # β = (X'X)^{-1} X'y
    XtX = X.T @ X
    det = XtX[0, 0] * XtX[1, 1] - XtX[0, 1] * XtX[1, 0]
    if abs(det) < 1e-15:
        return KyleLambdaResult(
            lambda_=float("nan"),
            t_stat=float("nan"),
            r_squared=float("nan"),
            n_windows=n,
            regression_df=reg_df,
        )

    XtX_inv = (
        np.array([[XtX[1, 1], -XtX[0, 1]], [-XtX[1, 0], XtX[0, 0]]]) / det
    )
    beta = XtX_inv @ (X.T @ y)
    lambda_ = float(beta[1])

    # Residuals and statistics
    y_hat = X @ beta
    residuals = y - y_hat
    ss_res = float(residuals @ residuals)
    ss_tot = float((y - y.mean()) @ (y - y.mean()))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")

    # Standard error of lambda
    if n > 2:
        mse = ss_res / (n - 2)
        se_lambda = float(np.sqrt(mse * XtX_inv[1, 1]))
        t_stat = lambda_ / se_lambda if se_lambda > 1e-15 else float("nan")
    else:
        t_stat = float("nan")

    return KyleLambdaResult(
        lambda_=lambda_,
        t_stat=t_stat,
        r_squared=r_squared,
        n_windows=n,
        regression_df=reg_df,
    )


# ── Order Flow Imbalance ─────────────────────────────────────────────


def order_flow_imbalance(
    trades: pd.DataFrame,
    window: str = "1min",
) -> pd.DataFrame:
    """Compute normalised order flow imbalance per time window.

    For each *window*:

    * ``ofi = (buy_volume − sell_volume) / (buy_volume + sell_volume)``

    Values range from −1 (all sells) to +1 (all buys).  Zero indicates
    balanced flow.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with ``timestamp``, ``volume``, ``direction``.
    window : str, optional
        Pandas frequency string.  Default ``"1min"``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp``, ``buy_volume``, ``sell_volume``,
        ``net_volume``, ``ofi``.

    Raises
    ------
    InvalidDataError
        If required columns are missing.
    InsufficientDataError
        If *trades* is empty.
    """
    validate_columns(
        trades,
        {"timestamp", "volume", "direction"},
        "order_flow_imbalance",
    )
    validate_non_empty(trades, "order_flow_imbalance")

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
