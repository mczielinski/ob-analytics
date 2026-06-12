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
  ``raw_size`` column for round-trip writers.
"""

from __future__ import annotations

import pandas as pd

from ob_analytics.exceptions import ConfigError

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
