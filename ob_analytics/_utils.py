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
# Price / tick conversions (issue #155)
# ---------------------------------------------------------------------------
#
# Canonical prices are stored as a whole number of ticks (``int64``) plus a
# per-instrument ``tick_size`` (see ``ob_analytics.config.PipelineConfig`` and
# ``docs/schema.md``).  Loaders convert a raw quote-currency price to ticks on
# the way in; writers and the display layer convert back.  Keeping the exact
# integer here — rather than a float in the quote currency — is what removes the
# float rounding that made small-tick and 0-1 instruments show crossed levels
# that were not real (issue #155).


def tick_multiplier(tick_size: float) -> int | None:
    """Return ``round(1 / tick_size)`` when the tick is a reciprocal integer.

    A tick such as ``0.01`` (``1/100``), ``0.05`` (``1/20``) or ``0.25``
    (``1/4``) has an exact integer inverse, so a price is converted to ticks by
    an integer multiply-and-round — the same operation
    :class:`~ob_analytics.depth.DepthMetricsEngine` used internally before this
    change, so the tick integers reproduce it bit-for-bit.  A tick with no
    integer inverse returns ``None``; :func:`price_to_ticks` then divides.
    """
    inv = 1.0 / tick_size
    nearest = round(inv)
    if nearest > 0 and abs(inv - nearest) <= 1e-9 * nearest:
        return int(nearest)
    return None


def price_to_ticks(price: object, tick_size: float) -> np.ndarray:
    """Convert quote-currency *price* to an ``int64`` whole number of ticks.

    ``ticks = round(price / tick_size)``, computed through an exact integer
    multiplier when the tick has one (:func:`tick_multiplier`) so the default
    grid reproduces the engine's former internal integers exactly.  *price* is
    any array-like of finite floats (a price column is non-null by contract).
    """
    arr = np.asarray(price, dtype=np.float64)
    multiplier = tick_multiplier(tick_size)
    ticks = (
        np.round(arr * multiplier)
        if multiplier is not None
        else np.round(arr / tick_size)
    )
    return ticks.astype(np.int64)


def ticks_to_price(
    ticks: object, tick_size: float, *, decimals: int | None = None
) -> np.ndarray:
    """Convert integer *ticks* back to a quote-currency ``float64`` price.

    ``price = ticks * tick_size``.  Pass *decimals* to round the result to that
    many places — the display layer does this so a reconstructed float matches
    the quote-currency value exactly (e.g. ``25001 * 0.01`` rounds to
    ``250.01`` rather than ``250.01000000000002``).
    """
    price = np.asarray(ticks, dtype=np.float64) * tick_size
    return price if decimals is None else np.round(price, decimals)


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
    "price": "int64",
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
