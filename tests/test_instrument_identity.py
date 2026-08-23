"""Instrument identity: the optional per-row ``venue`` / ``symbol`` columns.

Additive first stage of issue #147.  Each loader can tag its output with the
source venue and the instrument symbol so a single frame can hold rows from
more than one instrument or venue and still be told apart.  Identity is
opt-in: a default run adds no columns and existing single-instrument flows are
unchanged.

Covered here:

* every loader (Bitstamp, LOBSTER, L2 depth CSV) tags ``venue`` / ``symbol``
  with the right dtype and values when a symbol/venue is supplied, and adds no
  columns when none is;
* the CCXT capturer stamps identity onto every event it emits;
* the tags survive the full pipeline (through ``set_order_types`` /
  ``order_aggressiveness`` for L3, and the L2 path);
* a combined two-venue frame splits back by ``venue`` via
  :func:`~ob_analytics.schemas.group_by_instrument`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from ob_analytics import (
    SYMBOL_COLUMN,
    VENUE_COLUMN,
    BitstampFormat,
    ConfigError,
    Pipeline,
    RunContext,
    group_by_instrument,
)
from ob_analytics.bitstamp import BitstampLoader
from ob_analytics.depth_l2 import L2DepthLoader
from ob_analytics.live._base import CaptureConfig
from ob_analytics.live.ccxt_source import CcxtCapturer
from ob_analytics.lobster import LobsterFormat, LobsterLoader

_BASE_MS = 1_700_000_000_000  # arbitrary epoch-ms anchor


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_lobster_message(directory: Path) -> Path:
    """Write a tiny headerless LOBSTER message file and return its path.

    Columns: time, event_type, id, size, price, direction.  Two submissions,
    one execution, one deletion — enough for a valid, non-empty events frame.
    """
    rows = [
        (34200.0, 1, 1, 100, 2459800, 1),  # submit bid  id 1
        (34200.5, 1, 2, 100, 2460000, -1),  # submit ask  id 2
        (34201.0, 4, 1, 40, 2459800, 1),  # execute      id 1
        (34202.0, 3, 2, 100, 2460000, -1),  # delete       id 2
    ]
    path = directory / "AMZN_2012-06-21_message.csv"
    pd.DataFrame(rows).to_csv(path, index=False, header=False)
    return path


@pytest.fixture
def lobster_message_file(tmp_path) -> Path:
    return _write_lobster_message(tmp_path)


def _write_l2_depth(directory: Path) -> Path:
    """Write a tiny canonical L2 depth.csv and return the directory."""
    pd.DataFrame(
        [
            (_BASE_MS, "bid", 99.0, 2.0),
            (_BASE_MS, "ask", 101.0, 3.0),
            (_BASE_MS + 1000, "bid", 100.0, 1.0),
        ],
        columns=["timestamp", "side", "price", "volume"],
    ).to_csv(directory / "depth.csv", index=False)
    return directory


@pytest.fixture
def l2_depth_dir(tmp_path) -> Path:
    return _write_l2_depth(tmp_path)


def _assert_identity(df: pd.DataFrame, *, venue, symbol) -> None:
    """Assert *df* carries string ``venue`` / ``symbol`` columns with values.

    ``venue`` / ``symbol`` of ``None`` asserts the column is entirely NA.
    """
    assert VENUE_COLUMN in df.columns
    assert SYMBOL_COLUMN in df.columns
    assert df[VENUE_COLUMN].dtype == "string"
    assert df[SYMBOL_COLUMN].dtype == "string"
    assert len(df) > 0
    for col, value in ((VENUE_COLUMN, venue), (SYMBOL_COLUMN, symbol)):
        if value is None:
            assert df[col].isna().all()
        else:
            assert (df[col] == value).all()


# ---------------------------------------------------------------------------
# RunContext carries the identity
# ---------------------------------------------------------------------------


class TestRunContextIdentity:
    def test_defaults_are_none(self):
        ctx = RunContext()
        assert ctx.symbol is None
        assert ctx.venue is None

    def test_carries_values(self):
        ctx = RunContext(symbol="BTC/USD", venue="bitstamp")
        assert ctx.symbol == "BTC/USD"
        assert ctx.venue == "bitstamp"


# ---------------------------------------------------------------------------
# Bitstamp (L3 events)
# ---------------------------------------------------------------------------


class TestBitstampIdentity:
    def test_loader_tags_venue_and_symbol(self, tiny_bitstamp_orders_csv: Path):
        events = BitstampLoader(venue="bitstamp", symbol="BTC/USD").load(
            tiny_bitstamp_orders_csv
        )
        _assert_identity(events, venue="bitstamp", symbol="BTC/USD")

    def test_venue_defaults_to_source_name(self, tiny_bitstamp_orders_csv: Path):
        # Only a symbol supplied -> venue falls back to the loader's source name.
        events = BitstampLoader(symbol="BTC/USD").load(tiny_bitstamp_orders_csv)
        _assert_identity(events, venue="bitstamp", symbol="BTC/USD")

    def test_explicit_venue_overrides_source_name(self, tiny_bitstamp_orders_csv: Path):
        events = BitstampLoader(venue="kraken", symbol="BTC/USD").load(
            tiny_bitstamp_orders_csv
        )
        _assert_identity(events, venue="kraken", symbol="BTC/USD")

    def test_default_load_is_untagged(self, tiny_bitstamp_orders_csv: Path):
        events = BitstampLoader().load(tiny_bitstamp_orders_csv)
        assert VENUE_COLUMN not in events.columns
        assert SYMBOL_COLUMN not in events.columns

    def test_identity_survives_full_pipeline(self, tiny_bitstamp_orders_csv: Path):
        # End-to-end: the tags must survive set_order_types + order_aggressiveness.
        result = Pipeline(
            format=BitstampFormat(), ctx=RunContext(symbol="BTC/USD")
        ).run(tiny_bitstamp_orders_csv)
        _assert_identity(result.events, venue="bitstamp", symbol="BTC/USD")

    def test_default_pipeline_events_untagged(self, tiny_bitstamp_orders_csv: Path):
        result = Pipeline(format=BitstampFormat()).run(tiny_bitstamp_orders_csv)
        assert VENUE_COLUMN not in result.events.columns
        assert SYMBOL_COLUMN not in result.events.columns


# ---------------------------------------------------------------------------
# LOBSTER (L3 events)
# ---------------------------------------------------------------------------


class TestLobsterIdentity:
    def test_loader_tags_venue_and_symbol(self, lobster_message_file: Path):
        events = LobsterLoader(
            trading_date="2012-06-21", venue="nasdaq", symbol="AMZN"
        ).load(lobster_message_file)
        _assert_identity(events, venue="nasdaq", symbol="AMZN")

    def test_venue_defaults_to_source_name(self, lobster_message_file: Path):
        events = LobsterLoader(trading_date="2012-06-21", symbol="AMZN").load(
            lobster_message_file
        )
        _assert_identity(events, venue="lobster", symbol="AMZN")

    def test_default_load_is_untagged(self, lobster_message_file: Path):
        events = LobsterLoader(trading_date="2012-06-21").load(lobster_message_file)
        assert VENUE_COLUMN not in events.columns
        assert SYMBOL_COLUMN not in events.columns

    def test_format_forwards_ctx_identity(self, lobster_message_file: Path):
        # The format's create_loader must pass ctx.symbol / ctx.venue through.
        from ob_analytics import PipelineConfig

        fmt = LobsterFormat()
        ctx = RunContext(trading_date="2012-06-21", symbol="AMZN", venue="nasdaq")
        loader = fmt.create_loader(PipelineConfig(**fmt.config_defaults()), ctx)
        events = loader.load(lobster_message_file)
        _assert_identity(events, venue="nasdaq", symbol="AMZN")


# ---------------------------------------------------------------------------
# L2 depth CSV
# ---------------------------------------------------------------------------


class TestL2DepthIdentity:
    def test_loader_tags_venue_and_symbol(self, l2_depth_dir: Path):
        depth = L2DepthLoader(venue="binance", symbol="BTC/USDT").load(l2_depth_dir)
        _assert_identity(depth, venue="binance", symbol="BTC/USDT")

    def test_generic_venue_is_na_when_only_symbol_given(self, l2_depth_dir: Path):
        # A generic price-level CSV has no venue of its own: venue stays NA.
        depth = L2DepthLoader(symbol="BTC/USDT").load(l2_depth_dir)
        _assert_identity(depth, venue=None, symbol="BTC/USDT")

    def test_default_load_is_untagged(self, l2_depth_dir: Path):
        depth = L2DepthLoader().load(l2_depth_dir)
        assert VENUE_COLUMN not in depth.columns
        assert SYMBOL_COLUMN not in depth.columns

    def test_identity_survives_l2_pipeline(self, l2_depth_dir: Path):
        result = Pipeline.from_format(
            "depth_csv", ctx=RunContext(venue="binance", symbol="BTC/USDT")
        ).run(l2_depth_dir)
        _assert_identity(result.depth, venue="binance", symbol="BTC/USDT")


# ---------------------------------------------------------------------------
# CCXT capturer stamps identity on every emitted event
# ---------------------------------------------------------------------------


class _FakeExchange:
    """Minimal duck-typed ccxt exchange: one REST snapshot, no websockets."""

    id = "binance"

    def __init__(self) -> None:
        self.has = {"watchOrderBook": False, "watchTrades": False}

    async def fetch_order_book(self, symbol, limit=None):
        return {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": _BASE_MS}


class TestCcxtIdentity:
    def test_map_trade_stamps_identity(self):
        cap = CcxtCapturer()
        cap.exchange_id = "binance"
        cap._symbol = "BTC/USDT"
        ev = cap._map_trade(
            {
                "id": "t1",
                "timestamp": _BASE_MS,
                "price": 100.5,
                "amount": 0.5,
                "side": "buy",
            }
        )
        assert ev["venue"] == "binance"
        assert ev["symbol"] == "BTC/USDT"

    def test_diff_book_stamps_identity_on_every_row(self):
        cap = CcxtCapturer()
        cap.exchange_id = "kraken"
        cap._symbol = "ETH/USD"
        cap._last = {"bid": {100.0: 5.0}, "ask": {}}
        book = {"bids": [[100.0, 7.0], [99.0, 2.0]], "asks": [[101.0, 1.0]], "ts": 0}
        rows = [r for r, _raw in cap._diff_book(book, pd.Timestamp.now(tz="UTC"))]
        assert rows  # the book changed, so at least one row is emitted
        assert all(r["venue"] == "kraken" and r["symbol"] == "ETH/USD" for r in rows)

    def test_snapshot_stamps_identity(self, tmp_path):
        cap = CcxtCapturer()
        cfg = CaptureConfig(
            pair="BTC/USDT",
            out_dir=tmp_path,
            minutes=0.01,
            extras={"exchange": _FakeExchange()},
        )

        async def _collect():
            return [ev async for ev in cap.snapshot(cfg)]

        rows = asyncio.run(_collect())
        assert rows
        assert all(r["venue"] == "binance" and r["symbol"] == "BTC/USDT" for r in rows)


# ---------------------------------------------------------------------------
# Splitting a combined, multi-venue frame back apart
# ---------------------------------------------------------------------------


class TestGroupByInstrument:
    def test_two_venue_frame_splits_by_venue(self, l2_depth_dir: Path):
        binance = L2DepthLoader(venue="binance", symbol="BTC/USDT").load(l2_depth_dir)
        kraken = L2DepthLoader(venue="kraken", symbol="BTC/USD").load(l2_depth_dir)
        combined = pd.concat([binance, kraken], ignore_index=True)

        # The identity columns concatenate cleanly (stay string dtype).
        assert combined[VENUE_COLUMN].dtype == "string"

        # A plain groupby on venue splits the rows back.
        by_venue = combined.groupby(VENUE_COLUMN, observed=True).size()
        assert set(by_venue.index) == {"binance", "kraken"}
        assert (by_venue == len(binance)).all()

        # The helper groups by (venue, symbol) together.
        groups = dict(list(group_by_instrument(combined)))
        assert set(groups) == {("binance", "BTC/USDT"), ("kraken", "BTC/USD")}
        for key, part in groups.items():
            assert (part[VENUE_COLUMN] == key[0]).all()
            assert (part[SYMBOL_COLUMN] == key[1]).all()

    def test_group_by_instrument_raises_on_untagged_frame(self, l2_depth_dir: Path):
        depth = L2DepthLoader().load(l2_depth_dir)  # no identity columns
        with pytest.raises(ConfigError, match="no instrument-identity columns"):
            group_by_instrument(depth)
