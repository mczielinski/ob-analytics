"""Column contracts for the DataFrames flowing through the pipeline.

This module is the single source of truth for which columns each pipeline
DataFrame must carry. It replaces the parallel Pydantic models that lived
in the deleted ``models.py``: the pipeline runs on DataFrames, so the
contract is a set of required column names plus validators that raise on
violation — not a row model the pipeline never instantiated.

Validators check for REQUIRED columns only; extra columns are allowed.
"""

from __future__ import annotations

import pandas as pd

from ob_analytics.exceptions import InvalidDataError

# Required by price_level_volume / set_order_types (see depth.py).
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
        raise InvalidDataError(
            f"{who}: DataFrame is missing required columns {missing}. "
            f"Present: {list(df.columns)}"
        )


def validate_events_df(df: pd.DataFrame) -> None:
    """Raise :class:`InvalidDataError` unless *df* has all :data:`EVENT_COLUMNS`."""
    _require(df, EVENT_COLUMNS, "validate_events_df")


def validate_trades_df(df: pd.DataFrame) -> None:
    """Raise :class:`InvalidDataError` unless *df* has all :data:`TRADE_COLUMNS`."""
    _require(df, TRADE_COLUMNS, "validate_trades_df")


def validate_depth_df(df: pd.DataFrame) -> None:
    """Raise :class:`InvalidDataError` unless *df* has all :data:`DEPTH_COLUMNS`."""
    _require(df, DEPTH_COLUMNS, "validate_depth_df")
