"""Tests for ``ob_analytics.scan``: reading and aggregating a saved run in parts.

The claim under test is the one issue #206 makes: reading a saved run one batch
at a time gives the same numbers as loading it whole and computing in memory.
Each test therefore compares the batched result with the in-memory one on the
same rows.  Batches are kept small, so every table is read in many parts and
the places where two batches meet are exercised.
"""

from __future__ import annotations

import shutil
import warnings
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from ob_analytics import Pipeline, scan
from ob_analytics._utils import window_start
from ob_analytics.data import load_data, load_result, save_data
from ob_analytics.depth import summarize_depth
from ob_analytics.exceptions import ConfigError
from ob_analytics.scan import (
    AGGREGATIONS,
    aggregate_by_time,
    read_table,
    scan_table,
    summarize_depth_file,
)
from ob_analytics.synth import SynthLoader, SynthTradeReader, generate_session

#: Small enough that every table is read in many batches.
BATCH = 997

TABLES = ("events", "trades", "depth", "depth_summary")


@pytest.fixture(scope="module")
def synth_result():
    session = generate_session(seed=11, duration=600.0)
    return Pipeline(
        loader=SynthLoader(session), trade_reader=SynthTradeReader(session)
    ).run(None)


@pytest.fixture(scope="module")
def saved(synth_result, tmp_path_factory):
    """A synthetic run saved as Parquet, and the folder read back whole."""
    folder = tmp_path_factory.mktemp("saved")
    save_data(synth_result, folder)
    return folder, load_data(folder)


def _bucketed(frame: pd.DataFrame, every: str, how, columns=None) -> pd.DataFrame:
    """The in-memory reduction aggregate_by_time must match."""
    bucket = window_start(frame["timestamp"], every).rename("timestamp")
    grouped = frame.groupby(bucket)
    out = grouped[columns].agg(how) if columns is not None else grouped.agg(how)
    return out.reset_index()


# ── Reading ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("table", TABLES)
def test_batches_together_are_the_whole_table(saved, table):
    folder, whole = saved
    parts = list(scan_table(folder, table, batch_rows=BATCH))
    assert all(len(part) <= BATCH for part in parts)
    joined = pd.concat(parts, ignore_index=True)
    pd.testing.assert_frame_equal(joined, whole[table])
    assert parts[0].attrs["tick_size"] == whole[table].attrs["tick_size"]
    assert parts[0].attrs["lot_size"] == whole[table].attrs["lot_size"]


def test_read_table_is_the_in_memory_slice(saved):
    folder, whole = saved
    summary = whole["depth_summary"]
    start, end = summary["timestamp"].quantile([0.25, 0.6])
    columns = ["timestamp", "best_bid_price", "best_ask_vol"]
    got = read_table(folder, "depth_summary", columns=columns, start=start, end=end)
    expected = summary.loc[
        (summary["timestamp"] >= start) & (summary["timestamp"] < end), columns
    ].reset_index(drop=True)
    assert 0 < len(got) < len(summary)
    pd.testing.assert_frame_equal(got, expected)


def test_a_time_with_no_zone_is_read_as_utc(saved):
    folder, _ = saved
    start = read_table(folder, "depth", columns=["timestamp"])["timestamp"].iloc[500]
    aware = read_table(folder, "depth", start=start)
    naive = read_table(folder, "depth", start=start.tz_localize(None))
    pd.testing.assert_frame_equal(naive, aware)


def test_a_row_filter_is_passed_to_the_reader(saved):
    folder, whole = saved
    depth = whole["depth"]
    got = read_table(folder, "depth", filter=ds.field("direction") == "bid")
    expected = depth[depth["direction"] == "bid"].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected)


def test_an_empty_selection_keeps_the_columns_and_dtypes(saved):
    folder, whole = saved
    got = read_table(folder, "trades", start="2100-01-01")
    assert got.empty
    pd.testing.assert_series_equal(got.dtypes, whole["trades"].dtypes)


def test_files_are_written_in_row_groups(saved, monkeypatch, tmp_path):
    # Parquet keeps the time range of each row group, which is what lets a
    # time filter skip the rest of a file.
    _, whole = saved
    monkeypatch.setattr("ob_analytics.data.ROW_GROUP_ROWS", 1000)
    save_data({"depth": whole["depth"]}, tmp_path)
    groups = pq.ParquetFile(tmp_path / "depth.parquet").metadata.num_row_groups
    assert groups == -(-len(whole["depth"]) // 1000)


# ── Aggregating ───────────────────────────────────────────────────────


@pytest.mark.parametrize("how", AGGREGATIONS)
def test_aggregate_matches_the_in_memory_groupby(saved, how, monkeypatch):
    folder, whole = saved
    # Combine the partial results often, so that path runs too.
    monkeypatch.setattr(scan, "_COMBINE_EVERY", 3)
    got = aggregate_by_time(folder, "depth_summary", "10s", how, batch_rows=BATCH)
    columns = [c for c in got.columns if c != "timestamp"]
    assert "event_id" not in columns
    assert "best_bid_price" in columns
    pd.testing.assert_frame_equal(
        got, _bucketed(whole["depth_summary"], "10s", how, columns)
    )
    assert got.attrs["tick_size"] == whole["depth_summary"].attrs["tick_size"]


def test_aggregate_by_mapping_and_time_range(saved):
    folder, whole = saved
    trades = whole["trades"]
    start, end = trades["timestamp"].quantile([0.1, 0.9])
    how = {"price": "max", "volume": "sum", "maker_event_id": "count"}
    got = aggregate_by_time(
        folder, "trades", "1min", how, start=start, end=end, batch_rows=7
    )
    inside = trades[(trades["timestamp"] >= start) & (trades["timestamp"] < end)]
    pd.testing.assert_frame_equal(got, _bucketed(inside, "1min", how))


def test_aggregate_with_no_rows_keeps_the_column_types(saved):
    folder, _ = saved
    got = aggregate_by_time(
        folder,
        "depth_summary",
        "1min",
        {"best_bid_price": "last", "best_bid_vol": "mean", "best_ask_vol": "count"},
        start="2100-01-01",
    )
    assert got.empty
    assert got.dtypes.astype(str).to_dict() == {
        "timestamp": "datetime64[ns, UTC]",
        "best_bid_price": "int64",
        "best_bid_vol": "float64",
        "best_ask_vol": "int64",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"how": "median"}, "cannot reduce by"),
        ({"how": {"price": "sum"}, "columns": ["price"]}, "only with a single"),
        ({"how": "sum", "columns": ["nope"]}, "no column"),
        ({"how": {"timestamp": "max"}}, "cannot reduce that column"),
        ({"every": "1M"}, "duration"),
        ({"every": 5}, "bare number"),
    ],
)
def test_aggregate_rejects_bad_arguments(saved, kwargs, message):
    folder, _ = saved
    arguments: dict[str, Any] = {"every": "1min", "how": "last", **kwargs}
    with pytest.raises(ConfigError, match=message):
        aggregate_by_time(folder, "trades", **arguments)


# ── Errors at the call ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("table", "kwargs", "message"),
    [
        ("nope", {}, "has no 'nope' table"),
        ("depth", {"columns": ["nope"]}, "no column"),
        ("depth", {"batch_rows": 0}, "at least 1"),
        ("depth", {"start": "not a time"}, "not a time"),
    ],
)
def test_scan_table_fails_at_the_call(saved, table, kwargs, message):
    folder, _ = saved
    # Nothing is iterated: the error comes from the call itself.
    with pytest.raises(ConfigError, match=message):
        scan_table(folder, table, **kwargs)


def test_a_missing_folder_is_named(tmp_path):
    with pytest.raises(ConfigError, match="is not a folder"):
        read_table(tmp_path / "nope", "depth")


def test_a_file_with_no_version_warns_at_the_users_call(saved, tmp_path):
    _, whole = saved
    whole["trades"].to_parquet(tmp_path / "trades.parquet")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scan_table(tmp_path, "trades")
    legacy = [w for w in caught if "schema version" in str(w.message).lower()]
    assert legacy, [str(w.message) for w in caught]
    assert legacy[0].filename == __file__


# ── Rebuilding the depth summary ──────────────────────────────────────


def test_summarize_depth_file_matches_the_run(saved, tmp_path):
    folder, whole = saved
    written = summarize_depth_file(folder, tmp_path, batch_rows=BATCH)
    assert written == tmp_path / "depth_summary.parquet"
    rebuilt = load_data(tmp_path)["depth_summary"]
    pd.testing.assert_frame_equal(rebuilt, whole["depth_summary"])


def test_summarize_depth_file_with_other_bins(saved, tmp_path):
    folder, whole = saved
    summarize_depth_file(folder, tmp_path, depth_bps=10, depth_bins=4, batch_rows=BATCH)
    rebuilt = load_data(tmp_path)["depth_summary"]
    expected = summarize_depth(whole["depth"], depth_bps=10, depth_bins=4)
    pd.testing.assert_frame_equal(rebuilt, expected)
    assert "bid_vol40bps" in rebuilt.columns
    assert "bid_vol50bps" not in rebuilt.columns


def test_summarize_depth_file_can_replace_the_runs_summary(
    saved, synth_result, tmp_path
):
    _, whole = saved
    tables = {"depth": whole["depth"], "depth_summary": whole["depth_summary"]}
    save_data(tables, tmp_path, config=synth_result.config)
    summarize_depth_file(tmp_path, tmp_path, depth_bins=3, batch_rows=BATCH)
    rebuilt = load_data(tmp_path)
    assert "bid_vol75bps" in rebuilt["depth_summary"].columns
    assert "bid_vol100bps" not in rebuilt["depth_summary"].columns
    pd.testing.assert_frame_equal(rebuilt["depth"], whole["depth"])


def test_summarize_depth_file_needs_time_order(saved, synth_result, tmp_path):
    _, whole = saved
    source = tmp_path / "source"
    shuffled = whole["depth"].iloc[::-1].reset_index(drop=True)
    save_data({"depth": shuffled}, source, config=synth_result.config)
    dest = tmp_path / "dest"
    with pytest.raises(ConfigError, match="not in time order"):
        summarize_depth_file(source, dest, batch_rows=BATCH)
    # The working file is removed, and nothing is moved into place.
    assert list(dest.iterdir()) == []


def test_an_empty_sequence_at_a_batch_boundary(saved, synth_result, tmp_path):
    # Whole seconds make rows share a time, so the order check falls through to
    # the sequence, which is empty on every row.
    _, whole = saved
    depth = whole["depth"].copy()
    depth["timestamp"] = depth["timestamp"].dt.floor("s")
    depth["sequence"] = pd.array([pd.NA] * len(depth), dtype="Int64")
    source = tmp_path / "source"
    save_data({"depth": depth}, source, config=synth_result.config)
    summarize_depth_file(source, tmp_path / "dest", batch_rows=BATCH)
    rebuilt = load_data(tmp_path / "dest")["depth_summary"]
    expected = summarize_depth(load_data(source)["depth"])
    pd.testing.assert_frame_equal(rebuilt, expected)


def test_a_rebuilt_summary_records_its_own_config(saved, tmp_path):
    folder, _ = saved
    run = tmp_path / "run"
    shutil.copytree(folder, run)
    summarize_depth_file(run, run, depth_bps=10, depth_bins=3, batch_rows=BATCH)
    result = load_result(run)
    assert (result.config.depth_bps, result.config.depth_bins) == (10, 3)
    assert "bid_vol30bps" in result.depth_summary.columns
