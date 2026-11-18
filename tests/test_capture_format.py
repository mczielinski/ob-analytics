"""The capture file format: its versions, times and instrument (issue #494).

Format version 2 writes times in nanoseconds, names the side and size
``direction`` and ``volume`` in every file, and records ``venue`` and
``symbol`` in ``meta.json``.  Version 1 (no ``capture_format`` in
``meta.json``) wrote milliseconds and named them ``side`` and ``amount`` in
``trades.csv`` and ``depth.csv``.  Both must load.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ob_analytics import Pipeline, RunContext
from ob_analytics.bitstamp import (
    BitstampLoader,
    BitstampSource,
    BitstampTradeReader,
    BitstampWriter,
)
from ob_analytics.capture_files import FileCaptureSink, repair_files
from ob_analytics.capture_record import (
    CURRENT_FORMAT,
    read_fields,
    read_format,
    write_record,
)
from ob_analytics.depth_csv import (
    DepthCsvLoader,
    DepthCsvSettings,
    DepthCsvTradeReader,
    DepthCsvWriter,
)
from ob_analytics.exceptions import ConfigError
from ob_analytics.live import CaptureConfig, CaptureResult
from ob_analytics.live._runner import run_capturer
from ob_analytics.protocols import Level

# Bitstamp's microtimestamps: the two events are 250 microseconds apart, in
# the same millisecond.
_SNAPSHOT_US = 1_777_689_380_521_321
_ORDER_US = _SNAPSHOT_US + 1_000_100
_TRADE_US = _ORDER_US + 250


def _order_frame(order_id: int, microtimestamp: int) -> str:
    return json.dumps(
        {
            "channel": "live_orders_btcusd",
            "event": "order_created",
            "data": {
                "id": order_id,
                "price": "100.0",
                "amount": "1.0",
                "order_type": 1,
                "microtimestamp": str(microtimestamp),
            },
        }
    )


def _trade_frame(microtimestamp: int) -> str:
    return json.dumps(
        {
            "channel": "live_trades_btcusd",
            "event": "trade",
            "data": {
                "id": 5,
                "price": "99.0",
                "amount": "0.5",
                "buy_order_id": 9,
                "sell_order_id": 8,
                "type": 0,
                "microtimestamp": str(microtimestamp),
            },
        }
    )


def _fields(folder: Path) -> dict[str, Any]:
    fields = read_fields(folder)
    assert fields is not None
    return fields


class _FakeWs:
    def __init__(self, frames: list[str]) -> None:
        self._frames = list(frames)

    async def recv(self) -> str:
        await asyncio.sleep(0.001)
        if self._frames:
            return self._frames.pop(0)
        raise TimeoutError


def _bitstamp_capture(monkeypatch, folder: Path) -> Path:
    """Run a Bitstamp capture into *folder* from a fake WebSocket and REST book."""
    pytest.importorskip("websockets")
    from ob_analytics.live import bitstamp as live_bitstamp
    from ob_analytics.live.bitstamp import BitstampCapturer

    book = {
        "microtimestamp": str(_SNAPSHOT_US),
        "timestamp": str(_SNAPSHOT_US // 1_000_000),
        "bids": [["99.0", "1.0", "8"]],
        "asks": [],
    }
    frames = [_order_frame(9, _ORDER_US), _trade_frame(_TRADE_US)]

    async def open_ws(self, channels: list[str]) -> None:
        self._ws = _FakeWs(frames)

    monkeypatch.setattr(live_bitstamp, "_fetch_book_snapshot", lambda pair: book)
    monkeypatch.setattr(live_bitstamp, "SNAPSHOT_RETRY_SECONDS", 0.01)
    monkeypatch.setattr(BitstampCapturer, "_open_ws", open_ws)
    source = BitstampSource()
    source._engine = BitstampCapturer()
    config = CaptureConfig(pair="btcusd", out_dir=folder, minutes=0.002)
    asyncio.run(run_capturer(source, config, stop=asyncio.Event()))
    return folder


class TestBitstampMicroseconds:
    def test_the_capture_keeps_the_venues_microseconds(self, monkeypatch, tmp_path):
        cap = _bitstamp_capture(monkeypatch, tmp_path / "cap")
        assert _fields(cap)["capture_format"] == CURRENT_FORMAT

        events = BitstampLoader().load(cap / "orders.csv")
        exchange = set(events["exchange_timestamp"])
        assert pd.Timestamp(_SNAPSHOT_US, unit="us", tz="UTC") in exchange
        assert pd.Timestamp(_ORDER_US, unit="us", tz="UTC") in exchange

        trades = pd.read_csv(cap / "trades.csv")
        assert list(trades["exchange_timestamp"]) == [_TRADE_US * 1_000]
        assert list(trades["direction"]) == ["buy"]
        assert list(trades["volume"]) == [0.5]

    def test_a_trade_and_an_order_in_one_millisecond_stay_apart(
        self, monkeypatch, tmp_path
    ):
        cap = _bitstamp_capture(monkeypatch, tmp_path / "cap")
        orders = pd.read_csv(cap / "orders.csv")
        trades = pd.read_csv(cap / "trades.csv")
        order_ns = orders.loc[orders["id"] == 9, "exchange_timestamp"].iloc[0]
        assert trades["exchange_timestamp"].iloc[0] - order_ns == 250_000


class TestInstrumentFromTheRecord:
    def test_a_replayed_capture_names_its_instrument(self, monkeypatch, tmp_path):
        cap = _bitstamp_capture(monkeypatch, tmp_path / "cap")
        fields = _fields(cap)
        assert (fields["venue"], fields["symbol"]) == ("bitstamp", "btcusd")

        events = BitstampLoader().load(cap / "orders.csv")
        assert set(events["venue"]) == {"bitstamp"}
        assert set(events["symbol"]) == {"btcusd"}

    def test_the_pipeline_needs_no_run_context(self, monkeypatch, tmp_path):
        cap = _bitstamp_capture(monkeypatch, tmp_path / "cap")
        result = Pipeline(source=BitstampSource()).run(cap / "orders.csv")
        assert set(result.events["symbol"]) == {"btcusd"}
        assert set(result.events["venue"]) == {"bitstamp"}

    def test_the_run_context_wins(self, monkeypatch, tmp_path):
        cap = _bitstamp_capture(monkeypatch, tmp_path / "cap")
        result = Pipeline(source=BitstampSource()).run(
            cap / "orders.csv", ctx=RunContext(symbol="BTC/USD")
        )
        assert set(result.events["symbol"]) == {"BTC/USD"}
        assert set(result.events["venue"]) == {"bitstamp"}

    def test_an_l2_capture_names_its_instrument(self, tmp_path):
        cap = _l2_capture(tmp_path / "cap")
        depth = DepthCsvLoader().load(cap)
        assert set(depth["venue"]) == {"kraken"}
        assert set(depth["symbol"]) == {"BTC/USD"}


# ---------------------------------------------------------------------------
# One capture written in each format version
# ---------------------------------------------------------------------------

_T0 = pd.Timestamp("2026-11-17 12:00", tz="UTC")
# Two book rows 400 microseconds apart, so only version 2 tells them apart.
_L2_ROWS = [
    ("bid", 100.0, 2.0, _T0),
    ("ask", 101.0, 3.0, _T0 + pd.Timedelta(microseconds=400)),
]


def _l2_capture(folder: Path) -> Path:
    """An L2 capture in format version 2, written by the capture's own sink."""
    sink = FileCaptureSink(folder, keep_raw=False, level=Level.L2)
    for side, price, volume, ts in _L2_ROWS:
        sink.write_depth(
            {"timestamp": ts, "exchange_timestamp": ts, "direction": side,
             "price": price, "volume": volume, "venue": "kraken",
             "symbol": "BTC/USD"}
        )  # fmt: skip
    sink.write_trade(
        {"trade_id": "t1", "timestamp": _T0, "exchange_timestamp": _T0,
         "price": 101.0, "volume": 1.0, "buy_order_id": "", "sell_order_id": "",
         "direction": "buy"}
    )  # fmt: skip
    sink.finalize(_result(folder))
    return folder


def _v1_l2_capture(folder: Path) -> Path:
    """The same capture as a version 1 capture wrote it: ms, ``side``, ``amount``."""
    folder.mkdir(parents=True)
    ms = [int(ts.value // 1_000_000) for *_, ts in _L2_ROWS]
    pd.DataFrame(
        {
            "timestamp": ms,
            "exchange_timestamp": ms,
            "side": [r[0] for r in _L2_ROWS],
            "price": [r[1] for r in _L2_ROWS],
            "volume": [r[2] for r in _L2_ROWS],
        }
    ).to_csv(folder / "depth.csv", index=False)
    t0_ms = int(_T0.value // 1_000_000)
    pd.DataFrame(
        {
            "trade_id": ["t1"],
            "timestamp": [t0_ms],
            "exchange_timestamp": [t0_ms],
            "price": [101.0],
            "amount": [1.0],
            "buy_order_id": [""],
            "sell_order_id": [""],
            "side": ["buy"],
        }
    ).to_csv(folder / "trades.csv", index=False)
    write_record(folder, {"source": "ccxt", "level": "L2"})
    return folder


def _result(folder: Path) -> CaptureResult:
    return CaptureResult(
        out_dir=folder,
        n_order_events=0,
        n_trade_events=1,
        n_raw_frames=0,
        started=_T0,
        ended=_T0,
    )


class TestBothVersionsLoad:
    def test_version_1_reads_as_milliseconds(self, tmp_path):
        cap = _v1_l2_capture(tmp_path / "v1")
        assert read_format(cap).version == 1
        depth = DepthCsvLoader().load(cap)
        assert list(depth["timestamp"]) == [_T0, _T0]
        assert list(depth["direction"]) == ["bid", "ask"]
        assert "venue" not in depth.columns
        trades = DepthCsvTradeReader().load(depth, cap)
        assert list(trades["timestamp"]) == [_T0]
        assert list(trades["direction"]) == ["buy"]

    def test_version_2_keeps_the_microseconds(self, tmp_path):
        cap = _l2_capture(tmp_path / "v2")
        assert read_format(cap).version == 2
        depth = DepthCsvLoader().load(cap)
        assert list(depth["timestamp"]) == [ts for *_, ts in _L2_ROWS]
        trades = DepthCsvTradeReader().load(depth, cap)
        assert list(trades["timestamp"]) == [_T0]
        assert list(trades["direction"]) == ["buy"]

    def test_the_two_versions_hold_the_same_book(self, tmp_path):
        v1 = DepthCsvLoader().load(_v1_l2_capture(tmp_path / "v1"))
        v2 = DepthCsvLoader().load(_l2_capture(tmp_path / "v2"))
        columns = ["direction", "price", "volume"]
        pd.testing.assert_frame_equal(v1[columns], v2[columns])

    def test_the_bundled_version_1_sample_still_loads(self):
        from ob_analytics import sample_csv_path, sample_data_dir

        assert read_format(sample_data_dir()).version == 1
        events = BitstampLoader().load(sample_csv_path())
        trades = BitstampTradeReader().load(events, sample_data_dir())
        # The sample was captured on 2026-05-02; read as nanoseconds, its
        # millisecond times would land in 1970.
        assert events["timestamp"].min().year == 2026
        assert trades["timestamp"].min().year == 2026
        assert trades["direction"].notna().all()

    def test_a_newer_format_is_refused(self, tmp_path):
        cap = _l2_capture(tmp_path / "cap")
        write_record(cap, {**_fields(cap), "capture_format": CURRENT_FORMAT + 1})
        with pytest.raises(ConfigError, match="Upgrade ob-analytics"):
            DepthCsvLoader().load(cap)

    def test_a_version_that_is_not_a_number_is_read_as_1(self, tmp_path):
        cap = _v1_l2_capture(tmp_path / "cap")
        write_record(cap, {"capture_format": "two"})
        with pytest.warns(UserWarning, match="not a format version"):
            assert read_format(cap).version == 1


class TestWriters:
    def test_the_bitstamp_writer_records_the_format(self, tmp_path):
        from ob_analytics import sample_csv_path

        events = BitstampLoader().load(sample_csv_path()).head(200)
        out = tmp_path / "out"
        out.mkdir()
        write_record(out, {"pair": "btcusd"})
        BitstampWriter().write({"events": events}, out)
        assert read_fields(out) == {"pair": "btcusd", "capture_format": CURRENT_FORMAT}
        again = BitstampLoader().load(out / "orders.csv")
        assert sorted(again["timestamp"]) == sorted(events["timestamp"])

    def test_the_depth_writer_round_trips_nanoseconds(self, tmp_path):
        depth = DepthCsvLoader().load(_l2_capture(tmp_path / "cap"))
        out = tmp_path / "out"
        DepthCsvWriter().write({"depth": depth}, out)
        assert read_format(out).version == CURRENT_FORMAT
        assert "direction" in pd.read_csv(out / "depth.csv").columns
        again = DepthCsvLoader().load(out)
        assert list(again["timestamp"]) == list(depth["timestamp"])

    def test_a_unit_set_in_the_settings_is_used_as_it_is(self, tmp_path):
        depth = DepthCsvLoader().load(_l2_capture(tmp_path / "cap"))
        out = tmp_path / "out"
        settings = DepthCsvSettings(timestamp_unit="us")
        DepthCsvWriter(settings=settings).write({"depth": depth}, out)
        assert read_fields(out) is None
        again = DepthCsvLoader(settings=settings).load(out)
        assert list(again["timestamp"]) == list(depth["timestamp"])


class TestRepair:
    def test_an_open_version_2_capture_is_repaired_in_nanoseconds(self, tmp_path):
        folder = tmp_path / "cap"
        sink = FileCaptureSink(folder, keep_raw=False, level=Level.L3)
        ts = _T0 + pd.Timedelta(microseconds=7)
        sink.write_order(
            {"id": 1, "timestamp": ts, "exchange_timestamp": ts, "price": 100.0,
             "volume": 1.0, "action": "created", "direction": "bid"}
        )  # fmt: skip
        sink.flush()
        # The process dies here: no finalize, so meta.json is the first one.
        assert _fields(folder)["provisional"] is True

        files = repair_files(folder)
        assert files.last_timestamp == ts
        assert files.orders_closed == 1
        events = BitstampLoader().load(folder / "orders.csv")
        assert list(events["timestamp"]) == [ts, ts]

    def test_an_open_version_1_capture_is_repaired_in_milliseconds(self, tmp_path):
        folder = tmp_path / "cap"
        folder.mkdir()
        (folder / "orders.csv").write_text(
            "id,timestamp,exchange_timestamp,price,volume,action,direction\n"
            "1,1000,1000,100.0,1.0,created,bid\n"
        )
        (folder / "trades.csv").write_text("")

        files = repair_files(folder)
        assert files.last_timestamp == pd.Timestamp(1000, unit="ms", tz="UTC")
        assert (
            (folder / "trades.csv")
            .read_text()
            .startswith("trade_id,timestamp,exchange_timestamp,price,amount,")
        )


class TestOldEventKeys:
    @pytest.mark.parametrize(
        ("level", "write", "event"),
        [
            (Level.L2, "write_depth", {"side": "bid", "volume": 1.0}),
            (Level.L3, "write_trade", {"amount": 1.0, "direction": "buy"}),
        ],
        ids=["depth side", "trade amount"],
    )
    def test_a_source_that_yields_the_old_names_is_refused(
        self, tmp_path, level, write, event
    ):
        sink = FileCaptureSink(tmp_path / "cap", keep_raw=False, level=level)
        row = {"timestamp": _T0, "exchange_timestamp": _T0, "price": 1.0, **event}
        with pytest.raises(ConfigError, match="names the side 'direction'"):
            getattr(sink, write)(row)


class TestProcessKeepsTheFormat:
    """`process` copies the input's record; the CSV writer's version must stay."""

    def test_a_version_1_capture_processed_to_csv_reads_back(
        self, cli_runner, tmp_path
    ):
        cap = _v1_l2_capture(tmp_path / "v1")
        out = tmp_path / "out"
        r = cli_runner(
            "process", str(cap), "--source", "depth_csv", "--format", "depth_csv",
            "-o", str(out),
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        fields = _fields(out)
        assert fields["source"] == "ccxt"
        assert fields["capture_format"] == CURRENT_FORMAT
        depth = DepthCsvLoader().load(out)
        assert depth["timestamp"].min() == _T0
