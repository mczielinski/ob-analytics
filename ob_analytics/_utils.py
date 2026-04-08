"""Internal utility functions for ob-analytics.

Array helpers, DataFrame validation, timestamp conversions, and other
shared internals.  Nothing in this module is part of the public API.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ob_analytics.exceptions import InsufficientDataError, InvalidDataError

# ---------------------------------------------------------------------------
# DataFrame validation
# ---------------------------------------------------------------------------


def validate_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    context: str,
) -> None:
    """Raise :class:`InvalidDataError` if *required* columns are missing."""
    missing = set(required) - set(df.columns)
    if missing:
        raise InvalidDataError(
            f"{context}: missing required columns {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


def validate_non_empty(df: pd.DataFrame, context: str) -> None:
    """Raise :class:`InsufficientDataError` if *df* is empty."""
    if df.empty:
        raise InsufficientDataError(
            f"{context}: received empty DataFrame ({len(df.columns)} columns, 0 rows)"
        )


# ---------------------------------------------------------------------------
# Array / DataFrame helpers
# ---------------------------------------------------------------------------


def vector_diff(v: np.ndarray) -> np.ndarray:
    """
    Calculate the difference between consecutive elements of a vector.

    Parameters
    ----------
    v : numpy.ndarray
        A NumPy array of numerical values.

    Returns
    -------
    numpy.ndarray
        A NumPy array with the same length as `v`, where the first element is 0
        and the remaining elements are the differences between consecutive elements
        of `v`.
    """
    return np.diff(v, prepend=0)


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


def norml(
    v: np.ndarray, minv: float | None = None, maxv: float | None = None
) -> np.ndarray:
    """
    Normalize a vector to the range [0, 1].

    Parameters
    ----------
    v : numpy.ndarray
        A NumPy array of numerical values.
    minv : float or None, optional
        The minimum value for normalization. If None, the minimum of `v` is used.
    maxv : float or None, optional
        The maximum value for normalization. If None, the maximum of `v` is used.

    Returns
    -------
    numpy.ndarray
        A NumPy array with the same shape as `v`, where each element is normalized
        to the range [0, 1].
    """
    minv = np.min(v) if minv is None else minv
    maxv = np.max(v) if maxv is None else maxv
    return (v - minv) / (maxv - minv)


def interval_sum_breaks(v: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """
    Calculate the sum of values in each interval defined by breaks.

    Parameters
    ----------
    v : numpy.ndarray
        A NumPy array of numerical values.
    breaks : numpy.ndarray
        A NumPy array of indices that define the intervals.

    Returns
    -------
    numpy.ndarray
        A NumPy array with the sum of values in each interval.
    """
    cs = np.cumsum(v)
    intervals = cs[breaks]
    return np.concatenate((np.array([intervals[0]]), np.diff(intervals)))


def vwap(price: np.ndarray, volume: np.ndarray) -> float:
    """
    Calculate the volume-weighted average price (VWAP).

    Parameters
    ----------
    price : numpy.ndarray
        A NumPy array of prices.
    volume : numpy.ndarray
        A NumPy array of volumes.

    Returns
    -------
    float
        The VWAP as a float.
    """
    return np.average(price, weights=volume)


def interval_vwap(
    price: np.ndarray, volume: np.ndarray, breaks: np.ndarray
) -> np.ndarray:
    """
    Calculate the VWAP for each interval defined by breaks.

    Parameters
    ----------
    price : numpy.ndarray
        A NumPy array of prices.
    volume : numpy.ndarray
        A NumPy array of volumes.
    breaks : numpy.ndarray
        A NumPy array of indices that define the intervals.

    Returns
    -------
    numpy.ndarray
        A NumPy array with the VWAP for each interval.
    """
    return interval_sum_breaks(price * volume, breaks) / interval_sum_breaks(
        volume, breaks
    )


def interval_price_level_gaps(volume: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """
    Calculate the number of price level gaps in each interval.

    Parameters
    ----------
    volume : numpy.ndarray
        A NumPy array of volumes.
    breaks : numpy.ndarray
        A NumPy array of indices that define the intervals.

    Returns
    -------
    numpy.ndarray
        A NumPy array with the number of price level gaps in each interval.
    """
    return interval_sum_breaks(np.where(volume == 0, 1, 0), breaks)


# ---------------------------------------------------------------------------
# Timestamp conversions
# ---------------------------------------------------------------------------

_EPOCH_DIVISORS: dict[str, int] = {"ms": 1_000_000, "us": 1_000, "ns": 1}


def epoch_to_datetime(series: pd.Series, unit: str) -> pd.Series:
    """Convert numeric epoch timestamps to :class:`pandas.Timestamp`.

    Parameters
    ----------
    series : pandas.Series
        Numeric timestamps (integers or floats).
    unit : str
        Epoch unit — one of ``"ms"``, ``"us"``, or ``"ns"``.

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
        Target epoch unit — one of ``"ms"``, ``"us"``, or ``"ns"``.

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
