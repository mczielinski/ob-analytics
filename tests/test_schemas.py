"""Tests for the schemas.py column contract and on-disk schema version."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ob_analytics import schemas
from ob_analytics.data import load_data, save_data
from ob_analytics.exceptions import ConfigError
from ob_analytics.schemas import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    check_schema_version,
)


def test_validate_events_df_accepts_valid(tiny_events):
    # tiny_events lacks 'type'/'id' in some fixtures; add the required set.
    df = tiny_events.assign(type="limit")
    if "id" not in df.columns:
        df = df.assign(id=df["event_id"])
    schemas.validate_events_df(df)  # must not raise


def test_validate_events_df_rejects_missing():
    with pytest.raises(ConfigError, match="missing required columns"):
        schemas.validate_events_df(pd.DataFrame({"price": [1.0]}))


def test_validate_depth_df_uses_direction(tiny_depth):
    schemas.validate_depth_df(tiny_depth)  # tiny_depth has 'direction'


def test_validate_trades_df_rejects_missing():
    with pytest.raises(ConfigError, match="missing required columns"):
        schemas.validate_trades_df(pd.DataFrame({"price": [1.0]}))


# ── On-disk schema version ────────────────────────────────────────────


def _canonical_events() -> pd.DataFrame:
    """A tiny events frame in the canonical Arrow-native dtypes.

    Covers every core on-disk type — ``int64``, ``timestamp[ns]``,
    ``double``, and ordered categorical (dictionary) — so a write/read
    round trip is lossless and exact dtype checks are meaningful.
    """
    ts = pd.Series(
        pd.to_datetime(["2026-01-05 10:00:00.000", "2026-01-05 10:00:01.500"])
    ).astype("datetime64[ns]")
    return pd.DataFrame(
        {
            "event_id": pd.array([1, 2], dtype="int64"),
            "id": pd.array([10, 11], dtype="int64"),
            "timestamp": ts,
            "exchange_timestamp": ts.copy(),
            "price": pd.array([100.0, 101.0], dtype="float64"),
            "volume": pd.array([2.0, 3.0], dtype="float64"),
            "direction": pd.Categorical(
                ["bid", "ask"], categories=["bid", "ask"], ordered=True
            ),
            "action": pd.Categorical(
                ["created", "created"],
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "fill": pd.array([0.0, 0.0], dtype="float64"),
            "type": pd.Categorical(
                ["resting-limit", "resting-limit"],
                categories=[
                    "unknown",
                    "pre-existing",
                    "flashed-limit",
                    "resting-limit",
                    "market-limit",
                    "market",
                ],
                ordered=True,
            ),
        }
    )


def test_parquet_round_trip_preserves_columns_and_dtypes(tmp_path: Path):
    events = _canonical_events()
    save_data({"events": events}, tmp_path / "out")
    loaded = load_data(tmp_path / "out")["events"]

    assert list(loaded.columns) == list(events.columns)
    assert list(loaded.dtypes) == list(events.dtypes)
    pd.testing.assert_frame_equal(loaded, events)


def test_schema_version_written_to_parquet_metadata(tmp_path: Path):
    save_data({"events": _canonical_events()}, tmp_path / "out")

    table = pq.read_table(tmp_path / "out" / "events.parquet")
    metadata = table.schema.metadata or {}
    assert metadata.get(SCHEMA_VERSION_KEY) == SCHEMA_VERSION.encode()


def test_load_rejects_unsupported_schema_version(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    table = pa.Table.from_pandas(_canonical_events(), preserve_index=False)
    table = table.replace_schema_metadata({SCHEMA_VERSION_KEY: b"999.0"})
    pq.write_table(table, out / "events.parquet")

    with pytest.raises(ConfigError, match="unsupported schema version"):
        load_data(out)


def test_load_accepts_legacy_unversioned_file(tmp_path: Path):
    """A Parquet file with no version key (legacy data) still loads."""
    out = tmp_path / "out"
    out.mkdir()
    events = _canonical_events()
    # A plain pandas write carries no ob-analytics version key.
    events.to_parquet(out / "events.parquet", index=False)

    loaded = load_data(out)["events"]  # must not raise
    assert list(loaded.columns) == list(events.columns)


def test_check_schema_version_current_passes():
    check_schema_version(SCHEMA_VERSION, source="test")  # must not raise


def test_check_schema_version_none_is_legacy():
    check_schema_version(None, source="test")  # warns, does not raise


def test_check_schema_version_unknown_raises():
    with pytest.raises(ConfigError, match="unsupported schema version"):
        check_schema_version("3.0", source="test")
