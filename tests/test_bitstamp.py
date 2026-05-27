"""Coverage for the Bitstamp format (loader, trade reader, writer, format).

Pipeline.run on the bundled sample data is slow (minutes). The Plan-4 spec
asked for round-trip and pipeline-via-format tests; we get those without
the runtime cost by reusing the ``tiny_bitstamp_orders_csv`` fixture
(programmatic minimal dataset, sub-second pipeline) and sharing a single
Pipeline result across the writer tests via a module-scoped fixture.

The end-to-end pipeline / metadata / extras assertions live in
``test_pipeline.py`` and ``test_pipeline_extras.py``; we don't re-test
them here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ob_analytics import sample_csv_path
from ob_analytics.bitstamp import (
    BitstampFormat,
    BitstampLoader,
    BitstampTradeReader,
    BitstampWriter,
)
from ob_analytics.config import PipelineConfig
from ob_analytics.exceptions import InvalidDataError
from ob_analytics.pipeline import Pipeline
from ob_analytics.protocols import RunContext


# ---------------------------------------------------------------------------
# Shared Pipeline result for the writer tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_pipeline_result(tiny_bitstamp_orders_csv: Path):
    """Single Pipeline.run() shared across the writer-test module.

    Uses the tiny programmatic fixture (not the bundled sample data) so
    the run finishes in well under a second.
    """
    return Pipeline(format=BitstampFormat()).run(tiny_bitstamp_orders_csv)


# ---------------------------------------------------------------------------
# BitstampLoader
# ---------------------------------------------------------------------------


class TestBitstampLoader:
    def test_loads_sample_data(self):
        loader = BitstampLoader()
        events = loader.load(sample_csv_path())
        assert len(events) > 0
        for col in (
            "event_id",
            "id",
            "timestamp",
            "exchange_timestamp",
            "price",
            "volume",
            "action",
            "direction",
        ):
            assert col in events.columns, f"missing column: {col}"

    def test_timestamps_are_datetime(self):
        events = BitstampLoader().load(sample_csv_path())
        # BitstampLoader converts epoch -> tz-naive datetime64 (the loader uses
        # pd.to_datetime without utc=True). Pin that contract.
        assert pd.api.types.is_datetime64_any_dtype(events["timestamp"])
        assert pd.api.types.is_datetime64_any_dtype(events["exchange_timestamp"])

    def test_direction_is_ordered_categorical(self):
        events = BitstampLoader().load(sample_csv_path())
        assert events["direction"].dtype.name == "category"
        assert list(events["direction"].cat.categories) == ["bid", "ask"]
        assert events["direction"].cat.ordered

    def test_action_is_ordered_categorical(self):
        events = BitstampLoader().load(sample_csv_path())
        assert events["action"].dtype.name == "category"
        assert list(events["action"].cat.categories) == [
            "created",
            "changed",
            "deleted",
        ]
        assert events["action"].cat.ordered

    def test_event_ids_unique_and_sequential(self):
        events = BitstampLoader().load(sample_csv_path())
        assert events["event_id"].is_unique
        sorted_events = events.sort_values("event_id")
        assert (sorted_events["event_id"].diff().dropna() >= 1).all()

    def test_missing_file_raises(self, tmp_path):
        loader = BitstampLoader()
        with pytest.raises((FileNotFoundError, InvalidDataError)):
            loader.load(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# BitstampTradeReader
# ---------------------------------------------------------------------------


class TestBitstampTradeReader:
    def test_loads_companion_trades(self, bitstamp_sample_dir):
        events = BitstampLoader().load(bitstamp_sample_dir / "orders.csv")
        trades = BitstampTradeReader().load(events, bitstamp_sample_dir / "orders.csv")
        assert len(trades) > 0
        for col in (
            "timestamp",
            "price",
            "volume",
            "direction",
            "maker",
            "taker",
            "maker_event_id",
            "taker_event_id",
        ):
            assert col in trades.columns

    def test_missing_companion_returns_empty_or_raises(
        self, bitstamp_sample_orders_only
    ):
        """No companion trades.csv -> either empty DataFrame or clean error."""
        events = BitstampLoader().load(bitstamp_sample_orders_only)
        reader = BitstampTradeReader()
        try:
            trades = reader.load(events, bitstamp_sample_orders_only)
            assert isinstance(trades, pd.DataFrame)
        except (FileNotFoundError, InvalidDataError):
            pass


# ---------------------------------------------------------------------------
# BitstampWriter (round-trip)
# ---------------------------------------------------------------------------


class TestBitstampWriter:
    def test_round_trip_event_count(self, tmp_path, tiny_pipeline_result):
        result = tiny_pipeline_result
        rt_csv = tmp_path / "orders.csv"
        BitstampWriter().write(
            {
                "events": result.events,
                "trades": result.trades,
                "depth": result.depth,
                "depth_summary": result.depth_summary,
            },
            rt_csv,
        )
        # Synthesise companion trades.csv the way the demo does
        from ob_analytics._demos import _write_trades_csv_for_reader

        _write_trades_csv_for_reader(result.trades, rt_csv.parent / "trades.csv")

        rt_events = BitstampLoader().load(rt_csv)
        assert len(rt_events) == len(result.events)

    def test_writer_creates_file(self, tmp_path, tiny_pipeline_result):
        target = tmp_path / "orders.csv"
        BitstampWriter().write({"events": tiny_pipeline_result.events}, target)
        assert target.exists()
        assert target.stat().st_size > 0


# ---------------------------------------------------------------------------
# BitstampFormat
# ---------------------------------------------------------------------------


class TestBitstampFormat:
    def test_name(self):
        assert BitstampFormat().name == "bitstamp"

    def test_config_defaults_present(self):
        defaults = BitstampFormat().config_defaults()
        assert "price_decimals" in defaults
        assert "timestamp_unit" in defaults
        assert defaults["timestamp_unit"] == "ms"

    def test_constructs_loader_and_trade_source(self):
        fmt = BitstampFormat()
        cfg = PipelineConfig(**fmt.config_defaults())
        ctx = RunContext()
        loader = fmt.create_loader(cfg, ctx)
        ts = fmt.create_trade_source(cfg, ctx)
        writer = fmt.create_writer(cfg, ctx)
        assert isinstance(loader, BitstampLoader)
        assert isinstance(ts, BitstampTradeReader)
        assert isinstance(writer, BitstampWriter)

    def test_collect_extras_is_empty(self):
        fmt = BitstampFormat()
        cfg = PipelineConfig(**fmt.config_defaults())
        ctx = RunContext()
        loader = fmt.create_loader(cfg, ctx)
        events = loader.load(sample_csv_path())
        extras = fmt.collect_extras(loader, events, sample_csv_path(), ctx)
        assert extras == {}
