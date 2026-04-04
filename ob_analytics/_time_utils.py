"""Shared timestamp conversion utilities for exchange format loaders/writers.

Each exchange serialises timestamps differently.  This module provides the
named conversion helpers used by the built-in Bitstamp and LOBSTER formats.
New exchange format modules should import whichever helpers they need, or
add new ones here for shared use.

Conversion pairs
----------------
Bitstamp (epoch milliseconds / microseconds / nanoseconds):
    :func:`epoch_to_datetime`   — load path (numeric → pd.Timestamp)
    :func:`datetime_to_epoch`   — write path (pd.Timestamp → numeric)

LOBSTER (seconds after midnight, date-anchored):
    :func:`seconds_after_midnight_to_datetime`   — load path
    :func:`datetime_to_seconds_after_midnight`   — write path
"""

from __future__ import annotations

import pandas as pd

# Divisors for converting pandas internal nanosecond int64 values to
# the target epoch unit.
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
    return pd.to_datetime(series, unit=unit)


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
    # Use astype("int64") to get nanoseconds; avoid deprecated .view()
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
