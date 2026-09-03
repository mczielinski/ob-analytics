"""Tests for the Arrow and Polars accessors on :class:`PipelineResult`.

The public seam is what comes back from ``to_arrow()`` / ``to_polars()``: the
tables a run produces, the types the schema pins on their columns, and the
metadata that travels with them (issue #104, ``adr/0002-dataframe-library.md``).
How the conversion is performed inside is not asserted anywhere here.
"""

import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ob_analytics.data import save_data
from ob_analytics.datasets import toy_l2_depth, toy_l2_trades
from ob_analytics.depth_l2 import DepthCsvWriter
from ob_analytics.pipeline import Pipeline
from ob_analytics.schemas import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    TICK_SIZE_KEY,
    decode_tick_sizes,
    resolve_tick_size,
)

TABLE_NAMES = {"events", "trades", "depth", "depth_summary"}


@pytest.fixture(scope="module")
def result(tiny_bitstamp_orders_csv):
    """A small L3 ``PipelineResult`` shared by the tests in this module."""
    return Pipeline().run(tiny_bitstamp_orders_csv)


@pytest.fixture
def l2_result(tmp_path):
    """A price-level ``PipelineResult``, whose ``events`` frame is empty."""
    DepthCsvWriter().write(
        {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, tmp_path
    )
    return Pipeline.from_source("depth_csv").run(tmp_path)


class TestToArrow:
    def test_returns_every_table_keyed_by_name(self, result):
        tables = result.to_arrow()

        assert set(tables) == TABLE_NAMES
        assert all(isinstance(t, pa.Table) for t in tables.values())

    def test_pins_the_schema_types(self, result):
        tables = result.to_arrow()
        utc_ns = pa.timestamp("ns", tz="UTC")

        for name in ("events", "trades", "depth"):
            schema = tables[name].schema
            assert schema.field("timestamp").type == utc_ns, name
            assert schema.field("price").type == pa.int64(), name

        events = tables["events"].schema
        assert events.field("exchange_timestamp").type == utc_ns

        summary = tables["depth_summary"].schema
        assert summary.field("timestamp").type == utc_ns
        assert summary.field("best_bid_price").type == pa.int64()
        assert summary.field("best_ask_price").type == pa.int64()

    def test_carries_the_schema_version_and_tick_size(self, result):
        tables = result.to_arrow()

        for name, table in tables.items():
            metadata = table.schema.metadata or {}
            assert metadata.get(SCHEMA_VERSION_KEY) == SCHEMA_VERSION.encode(), name
            tick_sizes = decode_tick_sizes(metadata.get(TICK_SIZE_KEY))
            assert resolve_tick_size(tick_sizes) == result.config.tick_size, name

    def test_metadata_matches_what_a_saved_file_carries(self, result, tmp_path):
        """A reader handed a table in memory is no worse off than one reading
        the files."""
        tables = result.to_arrow()
        save_data(
            {
                "events": result.events,
                "trades": result.trades,
                "depth": result.depth,
                "depth_summary": result.depth_summary,
            },
            tmp_path,
            config=result.config,
        )

        for name, table in tables.items():
            on_disk = pq.read_table(tmp_path / f"{name}.parquet").schema.metadata
            in_memory = table.schema.metadata
            for key in (SCHEMA_VERSION_KEY, TICK_SIZE_KEY):
                assert in_memory[key] == on_disk[key], name

    def test_l2_run_has_the_same_keys_with_an_empty_events_table(self, l2_result):
        """The keys do not change with the run's level (issue #104)."""
        tables = l2_result.to_arrow()

        assert set(tables) == TABLE_NAMES
        assert tables["events"].num_rows == 0
        assert tables["depth"].num_rows > 0


class TestToPolars:
    def test_missing_polars_explains_how_to_install_it(self, result, monkeypatch):
        """Without polars the call says what is missing, not ``ModuleNotFound``."""
        monkeypatch.setitem(sys.modules, "polars", None)

        with pytest.raises(ImportError) as raised:
            result.to_polars()

        message = str(raised.value)
        assert "to_polars" in message
        assert "pip install polars" in message

    def test_returns_every_table_with_the_schema_types(self, result):
        pl = pytest.importorskip("polars")
        frames = result.to_polars()

        assert set(frames) == TABLE_NAMES
        assert all(isinstance(f, pl.DataFrame) for f in frames.values())

        for name in ("events", "trades", "depth"):
            schema = frames[name].schema
            assert schema["timestamp"] == pl.Datetime("ns", "UTC"), name
            assert schema["price"] == pl.Int64, name

    def test_holds_the_same_rows_as_the_pandas_frames(self, result):
        pytest.importorskip("polars")
        frames = result.to_polars()

        assert frames["trades"].height == len(result.trades)
        assert frames["depth"].height == len(result.depth)
        assert frames["trades"]["price"].to_list() == result.trades["price"].tolist()
