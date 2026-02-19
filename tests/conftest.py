"""Shared fixtures for ob-analytics tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "inst" / "extdata" / "orders.csv"


@pytest.fixture(scope="module")
def sample_csv_path() -> Path:
    """Path to the bundled sample orders.csv."""
    if not SAMPLE_CSV.exists():
        pytest.skip(f"Sample data not found: {SAMPLE_CSV}")
    return SAMPLE_CSV


@pytest.fixture
def tiny_events() -> pd.DataFrame:
    """Minimal valid events DataFrame (4 rows, 2 bid + 2 ask fills)."""
    ts = pd.Timestamp("2015-10-10 21:32:00", tz="UTC")
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "id": [100, 200, 300, 400],
            "timestamp": [
                ts,
                ts + pd.Timedelta(milliseconds=10),
                ts + pd.Timedelta(seconds=10),
                ts + pd.Timedelta(seconds=10, milliseconds=10),
            ],
            "exchange_timestamp": [
                ts - pd.Timedelta(seconds=1),
                ts - pd.Timedelta(seconds=1),
                ts + pd.Timedelta(seconds=9),
                ts + pd.Timedelta(seconds=9),
            ],
            "price": [236.50, 236.50, 236.75, 236.75],
            "volume": [1000.0, 1000.0, 500.0, 500.0],
            "action": pd.Categorical(
                ["changed", "changed", "changed", "changed"],
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "direction": pd.Categorical(
                ["bid", "ask", "bid", "ask"],
                categories=["bid", "ask"],
                ordered=True,
            ),
            "fill": [1234.0, 1234.0, 1234.0, 1234.0],
            "original_number": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def tiny_depth() -> pd.DataFrame:
    """Minimal depth DataFrame for testing depth_metrics."""
    ts_base = pd.Timestamp("2015-05-01 00:00:00")
    return pd.DataFrame(
        {
            "timestamp": [
                ts_base,
                ts_base + pd.Timedelta(seconds=1),
                ts_base + pd.Timedelta(seconds=2),
                ts_base + pd.Timedelta(seconds=3),
            ],
            "price": [236.50, 237.00, 236.45, 237.10],
            "volume": [10000, 5000, 8000, 3000],
            "direction": pd.Categorical(
                ["bid", "ask", "bid", "ask"],
                categories=["bid", "ask"],
                ordered=True,
            ),
        }
    )


@pytest.fixture
def matched_events(tiny_events: pd.DataFrame) -> pd.DataFrame:
    """Events with matching_event column populated."""
    df = tiny_events.copy()
    df["matching_event"] = [2, 1, 4, 3]
    return df
