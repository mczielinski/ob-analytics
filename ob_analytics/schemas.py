"""Column contracts for the DataFrames flowing through the pipeline.

This module is the single source of truth for which columns each pipeline
DataFrame must carry. It replaces the parallel Pydantic models that lived
in the deleted ``models.py``: the pipeline runs on DataFrames, so the
contract is a set of required column names plus validators that raise on
violation — not a row model the pipeline never instantiated.

Validators check for REQUIRED columns only; extra columns are allowed.

Canonical per-order volume semantics (every loader must satisfy these;
consumers — ``price_level_volume``, ``order_book``, the L3 faces — assume
them):

* ``volume`` — the order's **outstanding size after the event** for
  ``created``/``changed`` rows, and the **size removed** (outstanding
  immediately before the delete) for ``deleted`` rows.
* ``fill`` — the **executed** delta at this event (0 when nothing traded).
  A ``changed`` row is either an execution (``fill > 0``, outstanding drops
  by exactly ``fill``) or a non-executed reduction (``fill == 0``, e.g. a
  LOBSTER partial cancel) — never both in one event.
* Orders first seen mid-stream (a pre-existing opening book, LOBSTER hidden
  executions sharing the native ``id=0``) have no submission to anchor the
  outstanding size; loaders keep the venue's raw per-event quantity for
  those rows, and depth/lifecycle consumers exclude them (no ``created``
  row).  Format loaders may carry the venue's raw per-event quantity in a
  ``raw_size`` column for round-trip writers.  The classifier labels these
  orders ``pre-existing`` — structurally unclassifiable, not failures.

Timestamp policy (issue #154): both clocks are **tz-aware UTC nanoseconds**
(``datetime64[ns, UTC]``, Arrow ``timestamp[ns, tz=UTC]``).  ``timestamp`` is
the local receive time and ``exchange_timestamp`` the venue's matching-engine
time (identical for LOBSTER, where only exchange time exists).  Because every
frame sits on one UTC clock, frames from different venues **are** comparable:
they can be joined or concatenated directly.  A source in a venue-local clock
(LOBSTER, US/Eastern) is converted to UTC on load; a source already in epoch-UTC
(Bitstamp) keeps its wall-clock instants and only gains the zone and the
nanosecond unit.

Same-instant order: a timestamp alone cannot order events that share an
instant, so the deterministic **total order** is ``timestamp``, then — as
tie-breaks when present — the venue ``sequence`` (:data:`SEQUENCE_COLUMN`), then
``event_id`` (the dense per-order key on the L3 path), then the local
``ingest_seq`` (:data:`INGEST_SEQ_COLUMN`).  :func:`time_order_keys` returns that
key list for a frame, and the per-order reconstructions (``queue_positions``)
sort by it.  Every loader-built frame is produced by fixed, stable sorts, so the
rebuild is deterministic run-to-run.  On a price-level (L2) feed there is no
``event_id``; ties there fall to ``sequence`` / ``ingest_seq`` when tracked,
otherwise to the loader's stable arrival order.  Enforcing the full key inside
the price-level depth engine — so an alternate engine reproduces it bit-for-bit
— lands with the engine separation and rewrite (#136 / #104 / #138), when it can
be checked against that second backend.

Ordering keys (both **optional**, so neither is in :data:`EVENT_COLUMNS`;
consumers read them when present):

* ``sequence`` (:data:`SEQUENCE_COLUMN`) — the venue's own per-event sequence
  number, a nullable ``Int64``.  Carried only by sources that publish one (the
  CCXT ``nonce`` on the price-level path; a ``sequence`` column in a Bitstamp or
  L2 capture when the file recorded one).  It is **absent** for sources that do
  not number their events (LOBSTER, and the public Bitstamp diff feed, which
  publishes a microtimestamp rather than a sequence).  Consecutive values on one
  channel should rise by exactly one; a skip means a dropped message and a step
  that does not rise means a reordered one — see
  :func:`ob_analytics.analytics.detect_sequence_gaps`.
* ``ingest_seq`` (:data:`INGEST_SEQ_COLUMN`) — a local monotonic ingest counter,
  a 0-based ``int64`` index in the order the rows were read from the source
  (arrival order).  It is captured before any reordering, so it stays monotonic
  in arrival order even when the frame is later sorted for other reasons (e.g.
  the id-ordered Bitstamp events); rows dropped as duplicates leave it monotonic
  but not necessarily contiguous.  This is the deterministic ordering / replay
  key that never depends on a venue-supplied number, so it is defined even when
  ``sequence`` is missing (the common case on L2 diff feeds), and it is the
  order :func:`~ob_analytics.analytics.detect_sequence_gaps` scans ``sequence``
  along.  The loaders attach it only when ``track_sequence`` is enabled on
  :class:`~ob_analytics.config.PipelineConfig`, so the default pipeline output is
  byte-for-byte unchanged.

On-disk schema version: the canonical Parquet output is versioned.
:data:`SCHEMA_VERSION` is written into every Parquet file's key-value
metadata by :func:`ob_analytics.data.save_data`, and checked on load by
:func:`ob_analytics.data.load_data`.  The full column-by-column spec — Arrow
type, unit, nullability, and meaning for every table — is in ``docs/schema.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from ob_analytics.exceptions import ConfigError

if TYPE_CHECKING:
    from pandas.api.typing import DataFrameGroupBy

# ── Schema version ────────────────────────────────────────────────────
#
# The canonical Parquet/Arrow output is versioned so the on-disk format is a
# contract, not an implementation detail.  ``save_data`` writes SCHEMA_VERSION
# into each Parquet file's key-value metadata under SCHEMA_VERSION_KEY;
# ``load_data`` reads it back and calls ``check_schema_version`` before
# returning the frame.  The written spec is in ``docs/schema.md``.
#
# Bump SCHEMA_VERSION when a change would make an older reader misread a newer
# file: a renamed or removed column, a changed unit or dtype, or a new required
# column.  List every version this build can still read in
# ``_SUPPORTED_SCHEMA_VERSIONS``.

SCHEMA_VERSION: str = "2.0"
"""Version of the canonical Parquet/Arrow schema written to file metadata.

``2.0`` (issue #154) makes both timestamp clocks tz-aware UTC nanoseconds
(``timestamp[ns, tz=UTC]``); ``1.0`` wrote them tz-naive in each venue's native
clock.  Both are still read (a ``1.0`` file is self-describing and loads as the
tz-naive frame it stored)."""

SCHEMA_VERSION_KEY: bytes = b"ob_analytics_schema_version"
"""Parquet metadata key under which :data:`SCHEMA_VERSION` is stored.

Bytes, because Arrow file-metadata keys and values are raw bytes."""

_SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0", "2.0"})
"""Schema versions this build can read.  A file tagged with anything else
raises; an untagged (legacy) file loads with a warning.  ``1.0`` (pre-#154,
tz-naive timestamps) still reads: Parquet is self-describing, so pyarrow returns
the exact dtype the file stored."""


def check_schema_version(version: str | None, *, source: str = "<parquet>") -> None:
    """Validate an on-disk schema *version* against what this build reads.

    Parameters
    ----------
    version : str or None
        The value stored under :data:`SCHEMA_VERSION_KEY` in a Parquet file's
        metadata, or ``None`` when the file carries no such key.
    source : str
        Short identifier for the file, used only in messages.

    Raises
    ------
    ConfigError
        If *version* is a string this build does not support.

    Notes
    -----
    ``version is None`` is a **legacy/unversioned** file — data written before
    the schema carried a version, and the bundled sample.  Such files still
    load, with a warning, so existing captures and pre-existing outputs keep
    working.
    """
    if version is None:
        logger.warning(
            "{}: no schema version in Parquet metadata; treating as legacy "
            "(pre-{}) data. Re-save with save_data() to tag it.",
            source,
            SCHEMA_VERSION,
        )
        return
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ConfigError(
            f"{source}: unsupported schema version {version!r}. This build "
            f"reads {sorted(_SUPPORTED_SCHEMA_VERSIONS)} (current "
            f"{SCHEMA_VERSION!r}). Upgrade ob-analytics or re-export the data."
        )


# Optional ordering-key columns (never in EVENT_COLUMNS — see the module
# docstring). `sequence` is the venue's per-event number (nullable Int64) and
# `ingest_seq` the local monotonic ingest counter (int64).
SEQUENCE_COLUMN: str = "sequence"
"""Name of the optional venue per-event sequence column (nullable ``Int64``)."""

INGEST_SEQ_COLUMN: str = "ingest_seq"
"""Name of the optional local monotonic ingest-counter column (``int64``)."""

TIMESTAMP_COLUMN: str = "timestamp"
"""Name of the primary (receive-clock) timestamp column."""

# Tie-break keys applied after ``timestamp``, in priority order, to totally
# order events that share an instant (issue #154).  The venue ``sequence`` is
# the authoritative order when a source publishes one; ``event_id`` is the dense
# per-order key always present on the L3 path; ``ingest_seq`` is the local
# arrival counter that backs the price-level (L2) path when tracking is on.
_TIME_ORDER_TIEBREAKS: tuple[str, ...] = (
    SEQUENCE_COLUMN,
    "event_id",
    INGEST_SEQ_COLUMN,
)


def time_order_keys(df: pd.DataFrame) -> list[str]:
    """Return the canonical same-instant sort keys present in *df* (issue #154).

    The deterministic total order is ``timestamp`` followed by whichever of the
    tie-break keys (:data:`SEQUENCE_COLUMN`, ``event_id``,
    :data:`INGEST_SEQ_COLUMN`) *df* carries, in that priority order.  Sorting a
    frame by these keys reproduces the same event order on every run and in
    every engine implementation, so the rebuilt book does not depend on
    incidental input order at a shared instant.

    Parameters
    ----------
    df : pandas.DataFrame
        A frame carrying at least a ``timestamp`` column.

    Returns
    -------
    list of str
        ``["timestamp", ...]`` with the tie-break columns that are present.
    """
    return [TIMESTAMP_COLUMN, *(k for k in _TIME_ORDER_TIEBREAKS if k in df.columns)]


# Required by price_level_volume / set_order_types (see depth.py).
# `volume`/`fill` semantics: see the module docstring.
EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "id",
    "timestamp",
    "exchange_timestamp",
    "price",
    "volume",
    "direction",
    "action",
    "fill",
    "type",
)

# Canonical trades schema: taker-side direction + maker/taker attribution.
TRADE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "price",
    "volume",
    "direction",
    "maker_event_id",
    "taker_event_id",
)

# Depth uses 'direction' (bid/ask) to match DepthMetricsEngine.compute.
DEPTH_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "price",
    "volume",
    "direction",
)


# ── Instrument identity (optional, additive) ──────────────────────────
#
# Two optional per-row columns identify the instrument and the source venue,
# so one frame can hold rows from more than one instrument or venue and still
# be told apart (issue #147, additive stage).  They are deliberately NOT part
# of the required column sets above: loaders add them only when a symbol or
# venue is supplied for the run, so an existing single-instrument frame keeps
# its schema and consumers read the columns only when present.
#
# * ``venue``  — the source venue as a short string (e.g. "bitstamp",
#   "lobster", or a CCXT exchange id).  NA when unknown.
# * ``symbol`` — the instrument as a readable string (e.g. "BTC/USD").  NA when
#   unknown.
#
# Both are held as the pandas nullable ``string`` dtype: it carries NA cleanly
# and concatenates across venues without dtype surprises.  A Parquet writer may
# dictionary-encode either column for near-zero storage.

VENUE_COLUMN: str = "venue"
"""Name of the optional per-row source-venue column."""

SYMBOL_COLUMN: str = "symbol"
"""Name of the optional per-row instrument-symbol column."""

INSTRUMENT_ID_COLUMNS: tuple[str, str] = (VENUE_COLUMN, SYMBOL_COLUMN)
"""The optional instrument-identity columns, in ``(venue, symbol)`` order."""


def attach_instrument_identity(
    df: pd.DataFrame,
    *,
    venue: str | None,
    symbol: str | None,
    default_venue: str | None = None,
) -> pd.DataFrame:
    """Add the optional ``venue`` / ``symbol`` identity columns to *df*.

    Identity is opt-in.  When neither *venue* nor *symbol* is supplied the
    frame is returned unchanged, so a default single-instrument run keeps the
    existing schema.  When either is supplied, both columns are added in place:

    * ``venue`` takes *venue* if given, else *default_venue* (the loader's own
      source name), else NA;
    * ``symbol`` takes *symbol* if given, else NA.

    Parameters
    ----------
    df : pandas.DataFrame
        The frame to tag (the loader's freshly built output).
    venue, symbol : str or None
        The run's venue / instrument, or ``None`` when not supplied.
    default_venue : str or None
        The loader's own source name, used as the ``venue`` value when *venue*
        is ``None`` but *symbol* is given.  ``None`` for a generic loader that
        does not know its venue.

    Returns
    -------
    pandas.DataFrame
        *df*, with the two identity columns added when identity was supplied.
    """
    if venue is None and symbol is None:
        return df
    resolved_venue = venue if venue is not None else default_venue
    df[VENUE_COLUMN] = pd.array([resolved_venue] * len(df), dtype="string")
    df[SYMBOL_COLUMN] = pd.array([symbol] * len(df), dtype="string")
    return df


def group_by_instrument(df: pd.DataFrame) -> DataFrameGroupBy:
    """Group *df* by whichever identity columns (``venue``, ``symbol``) it has.

    A convenience for splitting a combined, multi-venue frame back into its
    per-instrument pieces.  ``dropna=False`` keeps rows whose identity is NA as
    their own group.

    Raises
    ------
    ConfigError
        If *df* carries neither identity column — a single-instrument frame
        that was never tagged.  Load it with a symbol/venue (for example
        ``RunContext(symbol=..., venue=...)``) to tag identity first.
    """
    keys = [c for c in INSTRUMENT_ID_COLUMNS if c in df.columns]
    if not keys:
        raise ConfigError(
            "group_by_instrument: frame carries no instrument-identity columns "
            f"{INSTRUMENT_ID_COLUMNS}. Load it with a symbol/venue "
            "(RunContext(symbol=..., venue=...)) to tag identity first."
        )
    return df.groupby(keys, dropna=False, observed=True)


def _require(df: pd.DataFrame, cols: tuple[str, ...], who: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ConfigError(
            f"{who}: DataFrame is missing required columns {missing}. "
            f"Present: {list(df.columns)}"
        )


def validate_events_df(df: pd.DataFrame) -> None:
    """Raise :class:`ConfigError` unless *df* has all :data:`EVENT_COLUMNS`."""
    _require(df, EVENT_COLUMNS, "validate_events_df")


def validate_trades_df(df: pd.DataFrame) -> None:
    """Raise :class:`ConfigError` unless *df* has all :data:`TRADE_COLUMNS`."""
    _require(df, TRADE_COLUMNS, "validate_trades_df")


def validate_depth_df(df: pd.DataFrame) -> None:
    """Raise :class:`ConfigError` unless *df* has all :data:`DEPTH_COLUMNS`."""
    _require(df, DEPTH_COLUMNS, "validate_depth_df")
