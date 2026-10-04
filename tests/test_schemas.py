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
        check_schema_version("99.0", source="test")


class TestCanonicalSizeDtype:
    """Schema 4.0 stores every size as a whole number of lots (``int64``).

    A float slipping into one of these columns does not raise anywhere: it
    reads as a base-asset size, sums like one, and only shows up much later as
    a wrong number in a face.  That is how ``fill`` stayed ``float64`` on the
    LOBSTER path -- a ``0.0`` literal in the expression that built it widened
    the whole column -- through a full run and out to Parquet.
    """

    SIZE_COLUMNS = ("volume", "fill")

    def _assert_lots(self, events: pd.DataFrame, who: str) -> None:
        for column in self.SIZE_COLUMNS:
            assert events[column].dtype == "int64", (
                f"{who}: {column!r} is {events[column].dtype}, not integer lots"
            )

    def test_bitstamp_loader_emits_integer_lots(self, tiny_bitstamp_orders_csv):
        from ob_analytics.bitstamp import BitstampSource
        from ob_analytics.pipeline import Pipeline

        result = Pipeline(source=BitstampSource()).run(str(tiny_bitstamp_orders_csv))
        self._assert_lots(result.events, "bitstamp")
        assert result.trades["volume"].dtype == "int64"

    def test_lobster_loader_emits_integer_lots(self, tmp_path):
        from ob_analytics.lobster import LobsterLoader

        # time,event_type,id,volume,price,direction — a creation, an
        # execution against it, and a delete on the other side.
        msg = tmp_path / "AAPL_2024-01-01_34200000_57600000_message_1.csv"
        msg.write_text(
            "34200.0,1,1,100,1000000,1\n"
            "34200.2,1,2,100,1010000,-1\n"
            "34200.3,4,1,50,1000000,1\n"
            "34200.5,3,2,100,1010000,-1\n"
        )
        events = LobsterLoader(trading_date="2024-01-01").load(msg)
        self._assert_lots(events, "lobster")

    def test_synthetic_stream_emits_integer_lots(self):
        from ob_analytics.synth import SynthConfig, generate_session

        session = generate_session(SynthConfig(seed=0))
        self._assert_lots(session.events, "synth")
