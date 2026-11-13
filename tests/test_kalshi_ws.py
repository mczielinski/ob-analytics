"""The kalshi live source (Kalshi's WebSocket feed) against recorded frames -- no network.

``fixtures/kalshi_ws.jsonl`` holds frames recorded from Kalshi on 2026-11-12
for one market: the two subscription replies, the opening book, 40 book
changes and two trades.
"""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("websockets")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
from pydantic import SecretStr

from ob_analytics import Pipeline
from ob_analytics.exceptions import ConfigError
from ob_analytics.live import (
    CaptureConfig,
    KalshiSettings,
    KalshiSource,
    UnmappedMessageError,
    kalshi as live_kalshi,
)
from ob_analytics.live._runner import run_capturer
from ob_analytics.quality import detect_sequence_gaps
from ob_analytics.schemas import validate_depth_df, validate_trades_df

FIXTURE = Path(__file__).parent / "fixtures" / "kalshi_ws.jsonl"
TICKER = "KXNCAAWBTOTAL-26NOV12HCBRWN-106"
KEY_ID = "0a1b2c3d-1111-2222-3333-444455556666"
MARKET = {
    "ticker": TICKER,
    "price_ranges": [{"start": "0.0000", "end": "1.0000", "step": "0.0100"}],
}


@pytest.fixture(autouse=True)
def _no_key_env(monkeypatch):
    """Keep a developer's own Kalshi key out of every test."""
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)


def _frames() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


def _pem(key: Any) -> str:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


@pytest.fixture(scope="module")
def ed_key() -> ed25519.Ed25519PrivateKey:
    return ed25519.Ed25519PrivateKey.generate()


@pytest.fixture
def settings(ed_key) -> KalshiSettings:
    return KalshiSettings(
        api_key_id=SecretStr(KEY_ID), private_key=SecretStr(_pem(ed_key))
    )


def _delta(
    seq: int, side: str, price: str, change: str, sid: int = 1
) -> dict[str, Any]:
    return {
        "type": "orderbook_delta",
        "sid": sid,
        "seq": seq,
        "msg": {
            "market_ticker": TICKER,
            "price_dollars": price,
            "delta_fp": change,
            "side": side,
            "ts_ms": 1794521486000,
        },
    }


def _trade(seq: int, taker_side: str = "yes", **body: Any) -> dict[str, Any]:
    msg = {
        "trade_id": f"t{seq}",
        "market_ticker": TICKER,
        "yes_price_dollars": "0.8200",
        "no_price_dollars": "0.1800",
        "count_fp": "0.25",
        "taker_side": taker_side,
        "ts_ms": 1794521522602,
    }
    msg.update(body)
    return {"type": "trade", "sid": 2, "seq": seq, "msg": msg}


def _seeded(yes=(("0.4000", "10.00"),), no=(("0.5500", "7.00"),)) -> KalshiSource:
    """A source whose book holds *yes* and *no* bids, as after an opening book."""
    source = KalshiSource()
    source._ticker = TICKER
    source._apply_snapshot(
        {
            "type": "orderbook_snapshot",
            "sid": 1,
            "seq": 1,
            "sending_ts_ms": 1794521483541,
            "msg": {
                "yes_dollars_fp": [list(x) for x in yes],
                "no_dollars_fp": [list(x) for x in no],
            },
        }
    )
    return source


class TestYesBook:
    def test_a_no_bid_is_a_yes_ask_at_one_minus_the_price(self):
        source = KalshiSource()
        rows = source._apply_snapshot(
            {
                "sid": 1,
                "seq": 1,
                "msg": {
                    "yes_dollars_fp": [["0.4000", "10.00"]],
                    "no_dollars_fp": [["0.5500", "7.00"]],
                },
            }
        )
        assert [(r["side"], r["price"], r["volume"]) for r in rows] == [
            ("bid", 0.40, 10.0),
            ("ask", 0.45, 7.0),
        ]

    def test_the_recorded_opening_book_is_never_crossed(self):
        source = KalshiSource()
        source._ticker = TICKER
        snapshot = next(f for f in _frames() if f["type"] == "orderbook_snapshot")
        rows = source._apply_snapshot(snapshot)
        assert len(rows) == source.snapshot_levels == 69 + 11
        best_bid = max(r["price"] for r in rows if r["side"] == "bid")
        best_ask = min(r["price"] for r in rows if r["side"] == "ask")
        assert best_bid < best_ask
        assert {r["sequence"] for r in rows} == {1}

    def test_the_opening_book_carries_the_time_kalshi_sent_it(self):
        rows = _seeded()._apply_snapshot(
            {
                "sid": 1,
                "seq": 1,
                "sending_ts_ms": 1794521483541,
                "msg": {"yes_dollars_fp": [["0.4000", "1.00"]]},
            }
        )
        assert rows[0]["exchange_timestamp"] == pd.Timestamp(
            1794521483541, unit="ms", tz="UTC"
        )


class TestDeltas:
    def test_a_change_writes_the_new_absolute_size(self):
        source = _seeded()
        row = source._apply_delta(_delta(2, "yes", "0.4000", "-3.50"))
        assert (row["side"], row["price"], row["volume"], row["sequence"]) == (
            "bid",
            0.40,
            6.5,
            2,
        )

    def test_a_no_change_moves_the_yes_ask(self):
        source = _seeded()
        row = source._apply_delta(_delta(2, "no", "0.5500", "2.00"))
        assert (row["side"], row["price"], row["volume"]) == ("ask", 0.45, 9.0)

    def test_a_change_to_zero_removes_the_level(self):
        source = _seeded()
        row = source._apply_delta(_delta(2, "yes", "0.4000", "-10.00"))
        assert row["volume"] == 0.0
        assert source._book["bid"] == {}

    def test_a_change_at_a_new_price_adds_a_level(self):
        source = _seeded()
        row = source._apply_delta(_delta(2, "yes", "0.3900", "4.00"))
        assert (row["price"], row["volume"]) == (0.39, 4.0)

    def test_a_change_carries_kalshis_time(self):
        row = _seeded()._apply_delta(_delta(2, "yes", "0.4000", "1.00"))
        assert row["exchange_timestamp"] == pd.Timestamp(
            1794521486000, unit="ms", tz="UTC"
        )

    def test_a_skipped_sequence_number_stops_the_segment(self):
        source = _seeded()
        with pytest.raises(UnmappedMessageError, match="missed 1 book change"):
            source._apply_delta(_delta(3, "yes", "0.4000", "1.00"))

    def test_a_repeated_sequence_number_stops_the_segment(self):
        source = _seeded()
        source._apply_delta(_delta(2, "yes", "0.4000", "1.00"))
        with pytest.raises(UnmappedMessageError, match="again or out of order"):
            source._apply_delta(_delta(2, "yes", "0.4000", "1.00"))

    def test_a_change_on_another_subscription_stops_the_segment(self):
        with pytest.raises(UnmappedMessageError, match="subscription"):
            _seeded()._apply_delta(_delta(2, "yes", "0.4000", "1.00", sid=9))

    def test_a_change_below_zero_stops_the_segment(self):
        with pytest.raises(UnmappedMessageError, match="negative size"):
            _seeded()._apply_delta(_delta(2, "yes", "0.4000", "-11.00"))

    def test_an_unparseable_change_stops_the_segment(self):
        bad = _delta(2, "yes", "0.4000", "1.00")
        del bad["msg"]["delta_fp"]
        with pytest.raises(UnmappedMessageError, match="cannot parse"):
            _seeded()._apply_delta(bad)

    @pytest.mark.parametrize("bad", ["NaN", "Infinity", "-inf"])
    def test_a_change_that_is_not_a_finite_number_stops_the_segment(self, bad):
        with pytest.raises(UnmappedMessageError, match="cannot parse"):
            _seeded()._apply_delta(_delta(2, "yes", bad, "1.00"))

    def test_a_change_on_an_unknown_side_stops_the_segment(self):
        with pytest.raises(UnmappedMessageError, match="unknown side"):
            _seeded()._apply_delta(_delta(2, "maybe", "0.4000", "1.00"))

    def test_the_recorded_changes_apply_in_order(self):
        source = KalshiSource()
        source._ticker = TICKER
        frames = _frames()
        source._apply_snapshot(
            next(f for f in frames if f["type"] == "orderbook_snapshot")
        )
        kinds = [
            source.parse(f)[0] for f in frames if f["type"] != "orderbook_snapshot"
        ]
        assert kinds.count("depth") == source.book_updates == 40
        assert kinds.count("trade") == 2
        assert source._book_seq == 41


class TestTrades:
    def test_a_trade_is_priced_in_yes_with_the_takers_side(self):
        source = _seeded()
        kind, buy, _ = source.parse(_trade(1, "yes"))
        _, sell, _ = source.parse(_trade(2, "no"))
        assert kind == "trade"
        assert (buy["price"], buy["amount"], buy["side"]) == (0.82, 0.25, "buy")
        assert sell["side"] == "sell"
        assert buy["exchange_timestamp"] == pd.Timestamp(
            1794521522602, unit="ms", tz="UTC"
        )

    def test_an_unparseable_trade_is_dropped_and_counted(self):
        source = _seeded()
        kind, event, raw = source.parse(_trade(1, taker_side="neither"))
        assert (kind, event) == ("raw", {})
        assert raw["type"] == "trade"
        assert source.dropped == 1

    def test_a_skipped_trade_number_is_counted_as_missed(self):
        source = _seeded()
        source.parse(_trade(1))
        source.parse(_trade(4))
        assert source.trades_missed == 2

    def test_a_trade_with_kalshis_time_in_seconds_only_is_kept(self):
        frame = _trade(1)
        del frame["msg"]["ts_ms"]
        frame["msg"]["ts"] = 1794521522
        _, event, _ = _seeded().parse(frame)
        assert event["exchange_timestamp"] == pd.Timestamp(
            1794521522, unit="s", tz="UTC"
        )

    def test_a_second_opening_book_ends_the_segment(self):
        snapshot = next(f for f in _frames() if f["type"] == "orderbook_snapshot")
        with pytest.raises(UnmappedMessageError, match="new opening book"):
            _seeded().parse(snapshot)

    def test_an_error_frame_ends_the_stream(self):
        with pytest.raises(ConnectionError, match="Kalshi sent an error"):
            _seeded().parse({"type": "error", "msg": {"code": 9, "msg": "boom"}})


class TestGrid:
    def test_the_tick_is_the_finest_step_of_the_market(self):
        market = {
            "price_ranges": [
                {"start": "0.0000", "end": "0.1000", "step": "0.0010"},
                {"start": "0.1000", "end": "0.9000", "step": "0.0100"},
            ]
        }
        assert live_kalshi._tick_size(market) == 0.001

    def test_a_market_without_ranges_has_no_tick(self):
        assert live_kalshi._tick_size({}) is None

    def test_a_price_between_ticks_makes_the_tick_finer(self):
        source = _seeded()
        source.tick_size = 0.01
        source._apply_delta(_delta(2, "yes", "0.4050", "1.00"))
        assert source.tick_size == 0.001
        assert source.tick_size_changes == 1


class TestSigning:
    def test_an_ed25519_signature_covers_time_method_and_path(self, ed_key):
        headers = live_kalshi._sign_request(
            ed_key, KEY_ID, "GET", "/trade-api/ws/v2?x=1", 1700000000000
        )
        assert headers["KALSHI-ACCESS-KEY"] == KEY_ID
        assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"
        ed_key.public_key().verify(
            base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
            b"1700000000000GET/trade-api/ws/v2",
        )

    def test_an_rsa_key_signs_with_pss(self):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        headers = live_kalshi._sign_request(
            key, KEY_ID, "GET", "/trade-api/ws/v2", 1700000000000
        )
        key.public_key().verify(
            base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
            b"1700000000000GET/trade-api/ws/v2",
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256(),
        )

    def test_a_file_that_is_not_a_key_is_refused_without_quoting_it(self):
        text = "not a key but a secret-looking line"
        source = KalshiSource(
            KalshiSettings(api_key_id=SecretStr(KEY_ID), private_key=SecretStr(text))
        )
        with pytest.raises(ConfigError, match="KALSHI_PRIVATE_KEY_PATH") as info:
            source.preflight()
        assert text not in str(info.value)

    def test_the_settings_read_both_keys_from_the_environment(
        self, monkeypatch, tmp_path, ed_key
    ):
        key_file = tmp_path / "kalshi.pem"
        key_file.write_text(_pem(ed_key))
        monkeypatch.setenv("KALSHI_API_KEY_ID", KEY_ID)
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key_file))
        settings = KalshiSettings()
        assert (
            settings.api_key_id is not None
            and settings.api_key_id.get_secret_value() == KEY_ID
        )
        assert settings.private_key is not None
        assert settings.private_key.get_secret_value() == _pem(ed_key)


class _FakeWs:
    """A WebSocket that replays *frames*, then waits for the deadline."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self._frames = [json.dumps(f) for f in frames]
        self.sent: list[dict[str, Any]] = []

    async def send(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def recv(self) -> str:
        await asyncio.sleep(0)
        if self._frames:
            return self._frames.pop(0)
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


class _FakeConnect:
    """Stands in for websockets' ``connect``; records the URL and headers."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self.ws = _FakeWs(frames)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, *, additional_headers: dict[str, str], **_: Any):
        self.calls.append((url, additional_headers))

        @asynccontextmanager
        async def connection():
            yield self.ws

        return connection()


def _fake_venue(monkeypatch, frames: list[dict[str, Any]]) -> _FakeConnect:
    import websockets.asyncio.client

    connect = _FakeConnect(frames)
    monkeypatch.setattr(websockets.asyncio.client, "connect", connect)
    monkeypatch.setattr(live_kalshi, "_fetch_market", lambda ticker: MARKET)
    return connect


class TestSubscribe:
    def test_an_error_before_the_opening_book_is_a_config_error(
        self, monkeypatch, settings
    ):
        _fake_venue(
            monkeypatch,
            [{"type": "error", "id": 1, "msg": {"code": 8, "msg": "Unknown market"}}],
        )
        source = KalshiSource(settings)

        async def collect():
            return [
                row
                async for row in source.snapshot(
                    CaptureConfig(pair=TICKER, out_dir=Path("x"))
                )
            ]

        with pytest.raises(ConfigError, match="Unknown market"):
            asyncio.run(collect())
        assert source._ws is None

    def test_trades_before_the_opening_book_are_skipped(self, monkeypatch, settings):
        snapshot = next(f for f in _frames() if f["type"] == "orderbook_snapshot")
        _fake_venue(monkeypatch, [_trade(1), snapshot])
        source = KalshiSource(settings)

        async def collect():
            return [
                row
                async for row in source.snapshot(
                    CaptureConfig(pair=TICKER, out_dir=Path("x"))
                )
            ]

        assert len(asyncio.run(collect())) == 80
        assert source.pre_snapshot_trades_skipped == 1
        asyncio.run(source._close())


@pytest.fixture
def capture(monkeypatch, tmp_path, ed_key, settings) -> tuple[Path, _FakeConnect]:
    """The recorded frames run through a whole capture into *tmp_path*."""
    connect = _fake_venue(monkeypatch, _frames())
    out = tmp_path / "cap"
    config = CaptureConfig(pair=TICKER, out_dir=out, minutes=0.02)
    asyncio.run(run_capturer(KalshiSource(settings), config))
    return out, connect


class TestCapture:
    def test_the_connection_is_signed_and_subscribes_to_the_market(
        self, capture, ed_key
    ):
        _, connect = capture
        [(url, headers)] = connect.calls
        assert url == live_kalshi.WS_URL
        assert headers["KALSHI-ACCESS-KEY"] == KEY_ID
        ed_key.public_key().verify(
            base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
            f"{headers['KALSHI-ACCESS-TIMESTAMP']}GET{live_kalshi.WS_SIGN_PATH}".encode(),
        )
        [subscribe] = connect.ws.sent
        assert subscribe["params"] == {
            "channels": ["orderbook_delta", "trade"],
            "market_ticker": TICKER,
        }

    def test_the_files_validate_and_the_sequence_has_no_gaps(self, capture):
        out, _ = capture
        depth = pd.read_csv(out / "depth.csv")
        trades = pd.read_csv(out / "trades.csv")
        assert len(depth) == 80 + 40
        assert len(trades) == 2
        gaps = detect_sequence_gaps(depth)
        assert gaps.empty
        assert gaps.attrs["n_sequenced"] == 120

    def test_raw_jsonl_holds_every_frame_the_opening_book_included(self, capture):
        out, _ = capture
        lines = (out / "raw.jsonl").read_text().splitlines()
        assert len(lines) == len(_frames())
        assert '"orderbook_snapshot"' in lines[1]

    def test_meta_records_the_grid_and_what_the_source_declares(self, capture):
        out, _ = capture
        meta = json.loads((out / "meta.json").read_text())
        assert meta["source"] == "kalshi"
        assert (meta["tick_size"], meta["lot_size"]) == (0.01, 0.01)
        assert meta["sequence_kind"] == "contiguous"
        assert meta["clocks"] == "both"
        assert (meta["last_book_seq"], meta["trades_missed"], meta["dropped"]) == (
            41,
            0,
            0,
        )

    def test_the_capture_replays_through_depth_csv(self, capture):
        out, _ = capture
        result = Pipeline.from_source("depth_csv").run(out)
        validate_depth_df(result.depth)
        validate_trades_df(result.trades)
        assert result.config.tick_size == 0.01
        assert len(result.trades) == 2

    def test_no_file_holds_the_key(self, capture, ed_key):
        out, _ = capture
        secrets = [
            KEY_ID,
            *[line for line in _pem(ed_key).splitlines() if "-----" not in line],
        ]
        for path in out.rglob("*"):
            if path.is_file():
                text = path.read_text(errors="replace")
                for secret in secrets:
                    assert secret not in text, f"{path.name} holds a key"
