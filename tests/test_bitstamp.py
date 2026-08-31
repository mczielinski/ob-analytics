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

from ob_analytics.bitstamp import (
    BitstampLoader,
    BitstampSource,
    BitstampTradeReader,
    BitstampWriter,
)
from ob_analytics.config import PipelineConfig
from ob_analytics.exceptions import ConfigError
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
    return Pipeline(source=BitstampSource()).run(tiny_bitstamp_orders_csv)


# ---------------------------------------------------------------------------
# BitstampLoader
# ---------------------------------------------------------------------------


class TestBitstampLoader:
    # These six tests share one session-scoped load of the bundled sample
    # (the `sample_events` fixture in conftest.py) instead of paying the
    # ~15s load each.

    def test_loads_sample_data(self, sample_events: pd.DataFrame):
        events = sample_events
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

    def test_timestamps_are_datetime(self, sample_events: pd.DataFrame):
        events = sample_events
        # BitstampLoader lands both clocks on the canonical tz-aware UTC
        # nanosecond model (issue #154): epoch-ms already counts from the UTC
        # epoch, so the instants keep their wall clock and only gain the zone
        # and the ns unit. Pin that contract.
        assert events["timestamp"].dtype == "datetime64[ns, UTC]"
        assert events["exchange_timestamp"].dtype == "datetime64[ns, UTC]"

    def test_direction_is_ordered_categorical(self, sample_events: pd.DataFrame):
        events = sample_events
        assert events["direction"].dtype.name == "category"
        assert list(events["direction"].cat.categories) == ["bid", "ask"]
        assert events["direction"].cat.ordered

    def test_action_is_ordered_categorical(self, sample_events: pd.DataFrame):
        events = sample_events
        assert events["action"].dtype.name == "category"
        assert list(events["action"].cat.categories) == [
            "created",
            "changed",
            "deleted",
        ]
        assert events["action"].cat.ordered

    def test_event_ids_unique_and_sequential(self, sample_events: pd.DataFrame):
        events = sample_events
        assert events["event_id"].is_unique
        sorted_events = events.sort_values("event_id")
        assert (sorted_events["event_id"].diff().dropna() >= 1).all()

    def test_original_number_tracks_source_row(self, sample_events: pd.DataFrame):
        """original_number is the 1-based source CSV row, not an event_id alias.

        The reference convention the LOBSTER loader now mirrors: original_number
        is assigned before the [id, volume, action, timestamp] sort and carried
        through it, so it stays distinct from the post-sort event_id surrogate.
        """
        events = sample_events
        assert events["original_number"].is_unique
        assert events["original_number"].min() >= 1
        assert (
            not events["original_number"]
            .reset_index(drop=True)
            .equals(events["event_id"].reset_index(drop=True))
        )

    def test_missing_file_raises(self, tmp_path):
        loader = BitstampLoader()
        with pytest.raises((FileNotFoundError, ConfigError)):
            loader.load(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# BitstampTradeReader
# ---------------------------------------------------------------------------


class TestBitstampTradeReader:
    def test_loads_companion_trades(self, bitstamp_sample_dir):
        events = BitstampLoader().load(bitstamp_sample_dir / "orders.csv.gz")
        trades = BitstampTradeReader().load(
            events, bitstamp_sample_dir / "orders.csv.gz"
        )
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
        except (FileNotFoundError, ConfigError):
            pass


class TestTradeReaderOrderIdTypes:
    """The reader keys maker/taker attribution on the order id. Bitstamp
    publishes integers, but the same schema carries captures from venues that
    publish UUIDs, and a public trade tape carries no maker/taker id at all --
    neither may crash the reader."""

    @staticmethod
    def _capture(tmp_path, order_ids, *, buy_id, sell_id):
        ts = 1_700_000_000_000
        rows = []
        for i, oid in enumerate(order_ids):
            rows.append(
                {
                    "id": oid,
                    "timestamp": ts + i,
                    "exchange_timestamp": ts + i,
                    "price": 100.0 + i,
                    "volume": 2.0,
                    "action": "created",
                    "direction": "bid" if i % 2 == 0 else "ask",
                }
            )
        for i, oid in enumerate(order_ids):
            rows.append(
                {
                    "id": oid,
                    "timestamp": ts + 50 + i,
                    "exchange_timestamp": ts + 50 + i,
                    "price": 100.0 + i,
                    "volume": 2.0,
                    "action": "deleted",
                    "direction": "bid" if i % 2 == 0 else "ask",
                }
            )
        orders = tmp_path / "orders.csv"
        pd.DataFrame(rows).to_csv(orders, index=False)
        pd.DataFrame(
            [
                {
                    "trade_id": "t1",
                    "timestamp": ts + 10,
                    "exchange_timestamp": ts + 10,
                    "price": 100.0,
                    "amount": 0.5,
                    "buy_order_id": buy_id,
                    "sell_order_id": sell_id,
                    "side": "buy",
                }
            ]
        ).to_csv(tmp_path / "trades.csv", index=False)
        return orders

    def test_a_tape_without_order_ids_resolves_to_no_attribution(self, tmp_path):
        """cryptofeed and CCXT publish a public tape with no maker/taker ids."""
        orders = self._capture(tmp_path, [11, 21], buy_id="", sell_id="")
        events = BitstampLoader().load(orders)
        trades = BitstampTradeReader().load(events, orders)
        assert len(trades) == 1
        assert pd.isna(trades["maker_event_id"].iloc[0])
        assert pd.isna(trades["taker_event_id"].iloc[0])

    def test_uuid_order_ids_do_not_raise(self, tmp_path):
        orders = self._capture(tmp_path, ["3f2b-aa", "9c1d-bb"], buy_id="", sell_id="")
        events = BitstampLoader().load(orders)
        trades = BitstampTradeReader().load(events, orders)
        assert len(trades) == 1
        assert set(events["id"].astype(str)) == {"3f2b-aa", "9c1d-bb"}

    def test_integer_attribution_still_resolves(self, tmp_path):
        """The integer path must behave exactly as it did before."""
        orders = self._capture(tmp_path, [11, 21], buy_id=11, sell_id=21)
        events = BitstampLoader().load(orders)
        trades = BitstampTradeReader().load(events, orders)
        assert len(trades) == 1
        assert trades["maker"].iloc[0] == 21
        assert trades["taker"].iloc[0] == 11


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
        # The writer emits a companion trades.csv automatically whenever the
        # payload carries a "trades" frame, so a full re-read round-trips
        # without any demo-side shim.
        assert (rt_csv.parent / "trades.csv").exists()

        rt = Pipeline(source=BitstampSource()).run(str(rt_csv))
        assert len(rt.events) == len(result.events)

    def test_writer_creates_file(self, tmp_path, tiny_pipeline_result):
        target = tmp_path / "orders.csv"
        BitstampWriter().write({"events": tiny_pipeline_result.events}, target)
        assert target.exists()
        assert target.stat().st_size > 0
        # No "trades" key -> no companion trades.csv is written.
        assert not (target.parent / "trades.csv").exists()


# ---------------------------------------------------------------------------
# BitstampSource
# ---------------------------------------------------------------------------


class TestBitstampSource:
    def test_name(self):
        assert BitstampSource().name == "bitstamp"

    def test_config_defaults_present(self):
        defaults = BitstampSource().config_defaults()
        assert "price_decimals" in defaults
        assert "timestamp_unit" in defaults
        assert defaults["timestamp_unit"] == "ms"

    def test_constructs_loader_and_trade_source(self):
        source = BitstampSource()
        cfg = PipelineConfig(**source.config_defaults())
        ctx = RunContext()
        loader = source.create_loader(cfg, ctx)
        ts = source.create_trade_source(cfg, ctx)
        writer = source.create_writer(cfg, ctx)
        assert isinstance(loader, BitstampLoader)
        assert isinstance(ts, BitstampTradeReader)
        assert isinstance(writer, BitstampWriter)
