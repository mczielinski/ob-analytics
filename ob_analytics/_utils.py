"""Internal utility functions for ob-analytics.

Array helpers, DataFrame validation, timestamp conversions, and other
shared internals.  Nothing in this module is part of the public API.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ob_analytics.exceptions import ConfigError, ObAnalyticsError

# ---------------------------------------------------------------------------
# DataFrame validation
# ---------------------------------------------------------------------------


def validate_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    context: str,
) -> None:
    """Raise :class:`ConfigError` if *required* columns are missing."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ConfigError(
            f"{context}: missing required columns {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


def validate_non_empty(df: pd.DataFrame, context: str) -> None:
    """Raise :class:`ObAnalyticsError` if *df* is empty."""
    if df.empty:
        raise ObAnalyticsError(
            f"{context}: received empty DataFrame ({len(df.columns)} columns, 0 rows)"
        )


# ---------------------------------------------------------------------------
# Trades schema
# ---------------------------------------------------------------------------

# Canonical trades columns, verified identical against the bitstamp.py and
# lobster.py trade readers. Carries BOTH the event-id attribution
# (maker_event_id / taker_event_id, required by trade_impacts /
# order_aggressiveness in analytics.py) AND the order-id / original-number
# columns. Named EMPTY_TRADES_COLUMNS — not "TRADE_COLUMNS" — to leave room
# for a smaller *required* validation subset under a different name later.
EMPTY_TRADES_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "price",
    "volume",
    "direction",
    "maker_event_id",
    "taker_event_id",
    "maker",
    "taker",
    "maker_og",
    "taker_og",
)


def empty_trades() -> pd.DataFrame:
    """Return an empty trades DataFrame with the canonical column set.

    Columns default to ``object`` dtype, matching the inline empty frames
    this helper replaces in the Bitstamp and LOBSTER trade readers.
    """
    return pd.DataFrame(columns=list(EMPTY_TRADES_COLUMNS))


# ---------------------------------------------------------------------------
# Array / DataFrame helpers
# ---------------------------------------------------------------------------


def reverse_matrix(m: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """
    Reverse the rows of a DataFrame or 2D NumPy array.

    Parameters
    ----------
    m : pandas.DataFrame or numpy.ndarray
        A pandas DataFrame or 2D NumPy array.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        A DataFrame or 2D NumPy array with the rows reversed.

    Raises
    ------
    TypeError
        If the input `m` is not a pandas DataFrame or a NumPy array.
    """
    if isinstance(m, pd.DataFrame):
        return m.iloc[::-1].reset_index(drop=True)
    elif isinstance(m, np.ndarray):
        return m[::-1, :]
    else:
        raise TypeError("Input must be a pandas DataFrame or a NumPy array.")


# ---------------------------------------------------------------------------
# Timestamp conversions
# ---------------------------------------------------------------------------

# Nanoseconds per unit: ``datetime_to_epoch`` takes a Timedelta's int64 ns
# count and divides by this to land in the requested unit.
_EPOCH_DIVISORS: dict[str, int] = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}


def epoch_to_datetime(series: pd.Series, unit: str) -> pd.Series:
    """Convert numeric epoch timestamps to :class:`pandas.Timestamp`.

    Parameters
    ----------
    series : pandas.Series
        Numeric timestamps (integers or floats).
    unit : str
        Epoch unit — one of ``"s"``, ``"ms"``, ``"us"``, or ``"ns"``.

    Returns
    -------
    pandas.Series
        Datetime series (dtype ``datetime64[ns]``).
    """
    return pd.to_datetime(series, unit=unit)  # ty: ignore[no-matching-overload]


def datetime_to_epoch(series: pd.Series, unit: str) -> pd.Series:
    """Convert a datetime :class:`pandas.Series` back to numeric epoch values.

    Parameters
    ----------
    series : pandas.Series
        Datetime series (dtype ``datetime64[ns]``).
    unit : str
        Target epoch unit — one of ``"s"``, ``"ms"``, ``"us"``, or ``"ns"``.

    Returns
    -------
    pandas.Series
        Integer epoch values in the requested unit.
    """
    epoch = pd.Timestamp("1970-01-01")
    delta = series - epoch
    divisor = _EPOCH_DIVISORS[unit]
    return (delta.astype("int64") // divisor).astype("int64")


def seconds_after_midnight_to_datetime(
    series: pd.Series, date: pd.Timestamp
) -> pd.Series:
    """Convert seconds-after-midnight floats to :class:`pandas.Timestamp`.

    LOBSTER message files record timestamps as fractional seconds elapsed
    since the start of the trading day (midnight local time).

    Parameters
    ----------
    series : pandas.Series
        Seconds after midnight (float).
    date : pandas.Timestamp
        Calendar date of the trading session.  Must be normalised to
        midnight (``date.normalize()``).

    Returns
    -------
    pandas.Series
        Absolute datetime series anchored to *date*.
    """
    return date + pd.to_timedelta(series, unit="s")


def datetime_to_seconds_after_midnight(
    series: pd.Series, date: pd.Timestamp
) -> pd.Series:
    """Convert absolute datetimes to seconds elapsed since midnight.

    Parameters
    ----------
    series : pandas.Series
        Absolute datetime series.
    date : pandas.Timestamp
        Calendar date of the trading session (midnight anchor).

    Returns
    -------
    pandas.Series
        Float seconds after midnight.
    """
    return (series - date).dt.total_seconds()
