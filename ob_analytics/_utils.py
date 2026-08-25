"""Internal utility functions for ob-analytics.

Array helpers, DataFrame validation, timestamp conversions, and other
shared internals.  Nothing in this module is part of the public API.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ob_analytics.exceptions import ConfigError, ObAnalyticsError
from ob_analytics.schemas import INGEST_SEQ_COLUMN

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

# Canonical trades columns shared by the Bitstamp and LOBSTER trade readers:
# event-id attribution (maker_event_id / taker_event_id, required by
# analytics.py) plus order-id / original-number provenance.
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
# Events schema
# ---------------------------------------------------------------------------

# The canonical per-order event columns (schemas.EVENT_COLUMNS) plus the
# provenance columns every loader carries. Typed so a zero-row frame still
# satisfies ``validate_events_df`` and dtype-sensitive consumers.
_EMPTY_EVENT_DTYPES: dict[str, str] = {
    "event_id": "int64",
    "id": "int64",
    "timestamp": "datetime64[ns, UTC]",
    "exchange_timestamp": "datetime64[ns, UTC]",
    "price": "float64",
    "volume": "float64",
    "direction": "object",
    "action": "object",
    "fill": "float64",
    "type": "object",
    "original_number": "int64",
    "raw_event_type": "object",
}


def empty_events() -> pd.DataFrame:
    """Return a schema-valid, zero-row events DataFrame.

    The L2 (price-level) pipeline path has no per-order events, but the rest
    of the system still expects a :func:`~ob_analytics.schemas.validate_events_df`-valid
    ``events`` frame on the :class:`~ob_analytics.pipeline.PipelineResult`.
    This is that frame: every canonical event column present, correctly typed,
    with no rows — the explicit "the per-order stages did not apply here"
    marker (see :class:`~ob_analytics.protocols.Level`).
    """
    return pd.DataFrame(
        {
            name: pd.Series([], dtype=dtype)
            for name, dtype in _EMPTY_EVENT_DTYPES.items()
        }
    )


def attach_ingest_seq(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the local monotonic ingest counter, in place, and return *frame*.

    ``ingest_seq`` is a 0-based ``int64`` index over *frame*'s rows in their
    current order — a deterministic ordering / replay key that never depends on
    a venue-supplied number (see :mod:`ob_analytics.schemas`).  Loaders call this
    while the frame is still in source (arrival) order, so the counter records
    arrival order even when the frame is sorted differently before it is
    returned.  Called only when ``track_sequence`` is enabled on the
    :class:`~ob_analytics.config.PipelineConfig`, so the default output is
    unchanged.
    """
    frame[INGEST_SEQ_COLUMN] = np.arange(len(frame), dtype="int64")
    return frame


# ---------------------------------------------------------------------------
# Timestamp conversions
# ---------------------------------------------------------------------------
#
# The canonical time model (issue #154) is tz-aware UTC nanoseconds:
# ``datetime64[ns, UTC]`` (Arrow ``timestamp[ns, tz=UTC]``).  Every loader
# builds its clocks through these helpers, so frames from different venues sit
# on one clock.  The conversion functions below take a source's native
# representation (integer epoch, or session-relative seconds after midnight)
# and land it on that shared clock; the inverses reverse the process for the
# round-trip writers.

# The canonical timestamp dtype for every schema timestamp column.
UTC_NS_DTYPE = "datetime64[ns, UTC]"

# Nanoseconds per unit: ``datetime_to_epoch`` takes a Timedelta's int64 ns
# count and divides by this to land in the requested unit.
_EPOCH_DIVISORS: dict[str, int] = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}


def epoch_to_datetime(series: pd.Series, unit: str) -> pd.Series:
    """Convert numeric epoch timestamps to tz-aware UTC nanosecond datetimes.

    Epoch integers count from the Unix epoch in UTC, so the values are already
    on the shared clock; this attaches the UTC zone and fixes the unit at
    nanoseconds (see the canonical time model, issue #154).

    Parameters
    ----------
    series : pandas.Series
        Numeric timestamps (integers or floats).
    unit : str
        Epoch unit of the input — one of ``"s"``, ``"ms"``, ``"us"``, or
        ``"ns"``.

    Returns
    -------
    pandas.Series
        Datetime series (dtype ``datetime64[ns, UTC]``).
    """
    return pd.to_datetime(series, unit=unit, utc=True).astype(  # ty: ignore[no-matching-overload]
        UTC_NS_DTYPE
    )


def datetime_to_epoch(series: pd.Series, unit: str) -> pd.Series:
    """Convert a datetime :class:`pandas.Series` back to numeric epoch values.

    Accepts the canonical tz-aware UTC series and, for robustness, a tz-naive
    series (read as UTC); the returned integers are epoch counts in *unit*.

    Parameters
    ----------
    series : pandas.Series
        Datetime series (canonically ``datetime64[ns, UTC]``).
    unit : str
        Target epoch unit — one of ``"s"``, ``"ms"``, ``"us"``, or ``"ns"``.

    Returns
    -------
    pandas.Series
        Integer epoch values in the requested unit.
    """
    utc = pd.to_datetime(series, utc=True).astype(UTC_NS_DTYPE)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    delta = (utc - epoch).astype("timedelta64[ns]")
    divisor = _EPOCH_DIVISORS[unit]
    return (delta.astype("int64") // divisor).astype("int64")


def seconds_after_midnight_to_datetime(
    series: pd.Series, date: pd.Timestamp, tz: str
) -> pd.Series:
    """Convert session-relative seconds-after-midnight to tz-aware UTC datetimes.

    LOBSTER message files record timestamps as fractional seconds elapsed since
    the start of the trading day (midnight *local* time), with no receive clock
    and no time zone.  Placing them on the shared UTC clock therefore needs both
    the session date and the venue's time zone: the seconds are anchored to
    *date* in *tz*, then converted to UTC (see issue #154).

    Parameters
    ----------
    series : pandas.Series
        Seconds after midnight (float).
    date : pandas.Timestamp
        Calendar date of the trading session, tz-naive and normalised to
        midnight (``date.normalize()``).
    tz : str
        The venue's local time zone (e.g. ``"America/New_York"``), used to place
        the session's midnight on the UTC clock.

    Returns
    -------
    pandas.Series
        Absolute datetime series (dtype ``datetime64[ns, UTC]``).
    """
    local = date + pd.to_timedelta(series, unit="s")
    return local.dt.tz_localize(tz).dt.tz_convert("UTC").astype(UTC_NS_DTYPE)


def datetime_to_seconds_after_midnight(
    series: pd.Series, date: pd.Timestamp, tz: str
) -> pd.Series:
    """Convert absolute tz-aware datetimes back to session-relative seconds.

    The inverse of :func:`seconds_after_midnight_to_datetime`: the UTC series is
    converted to the venue's local time *tz* and measured from local midnight
    *date*.

    Parameters
    ----------
    series : pandas.Series
        Absolute datetime series (canonically ``datetime64[ns, UTC]``).
    date : pandas.Timestamp
        Calendar date of the trading session (tz-naive local midnight anchor).
    tz : str
        The venue's local time zone (e.g. ``"America/New_York"``).

    Returns
    -------
    pandas.Series
        Float seconds after midnight.
    """
    local = pd.to_datetime(series, utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    return (local - date).dt.total_seconds()
