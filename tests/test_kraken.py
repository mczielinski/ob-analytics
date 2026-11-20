"""The kraken live source (Kraken's WebSocket v2 level3 feed) -- no network.

``fixtures/kraken_l3_checksum_example.json`` is the opening book from Kraken's
L3 checksum guide (docs.kraken.com, "L3 checksum (WebSocket v2)", read on
2026-11-19), with the checksum the guide gives for it.  The other frames are
built here from the message shapes in Kraken's level3 and trade channel
documentation.  Issue #555.
"""

from __future__ import annotations

import asyncio
import binascii
import json
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("websockets")

from pydantic import SecretStr

from ob_analytics import _secrets
from ob_analytics.bitstamp import BitstampSource
from ob_analytics.exceptions import ConfigError
from ob_analytics.live import (
    CaptureConfig,
    KrakenSettings,
    KrakenSource,
    UnmappedMessageError,
    kraken as live_kraken,
)
from ob_analytics.live._runner import run_capturer
from ob_analytics.pipeline import Pipeline
from ob_analytics.schemas import validate_trades_df

FIXTURE = Path(__file__).parent / "fixtures" / "kraken_l3_checksum_example.json"
SYMBOL = "BTC/USD"
# Shaped like a Kraken key and private key; not real ones.
API_KEY = "Kk3yT3stK3yAAAAbbbbCCCCddddEEEEffffGGGGhhhhIIIIjjjjKKKK"
API_SECRET = "kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRzBHCd3pd5nE9qa99HAZtuZuj6F1huXg=="
TOKEN = "1Dwc4lzSwNWOAwkMdqhssNNFhs1ed606d1WcF3XfEMw"
T0 = "2026-11-19T12:00:00.000000000Z"
T1 = "2026-11-19T12:00:01.250000000Z"


@pytest.fixture(autouse=True)
def _no_key_env(monkeypatch):
    """Keep a developer's own Kraken key out of every test."""
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_API_SECRET", raising=False)


def _settings(**fields: Any) -> KrakenSettings:
    return KrakenSettings(
        api_key=SecretStr(API_KEY), api_secret=SecretStr(API_SECRET), **fields
    )


# -- frames ---------------------------------------------------------------------

Book = dict[str, list[tuple[str, str, str]]]


def _checksum(book: Book) -> int:
    """Kraken's checksum of *book*, worked out as the guide spells it.

    *book* lists each side's orders as ``(order_id, price, qty)``, best price
    first and in queue order within a price.
    """

    def text(number: str) -> str:
        return number.replace(".", "").lstrip("0")

    parts = []
    for side in ("ask", "bid"):
        prices: list[str] = []
        for _, price, _ in book[side]:
            if price not in prices:
                prices.append(price)
        for price in prices[:10]:
            for _, at, qty in book[side]:
                if at == price:
                    parts.append(text(price) + text(qty))
    return binascii.crc32("".join(parts).encode()) & 0xFFFFFFFF


def _order(order_id: str, price: str, qty: str, event: str | None = None) -> dict:
    order = {
        "order_id": order_id,
        "limit_price": price,
        "order_qty": qty,
        "timestamp": T0,
    }
    if event is not None:
        order["event"] = event
    return order


def _snapshot(book: Book, checksum: int | None = None) -> dict[str, Any]:
    return {
        "channel": "level3",
        "type": "snapshot",
        "data": [
            {
                "symbol": SYMBOL,
                "checksum": _checksum(book) if checksum is None else checksum,
                "bids": [_order(*o) for o in book["bid"]],
                "asks": [_order(*o) for o in book["ask"]],
                "timestamp": T0,
            }
        ],
    }


def _update(
    after: Book,
    bids: Sequence[dict] = (),
    asks: Sequence[dict] = (),
    checksum: int | None = None,
    timestamp: str | None = T1,
) -> dict[str, Any]:
    """An update frame; *after* is Kraken's book once it is applied."""
    entry: dict[str, Any] = {
        "symbol": SYMBOL,
        "checksum": _checksum(after) if checksum is None else checksum,
        "bids": list(bids),
        "asks": list(asks),
    }
    if timestamp is not None:
        entry["timestamp"] = timestamp
    return {"channel": "level3", "type": "update", "data": [entry]}


def _trade(**fields: Any) -> dict[str, Any]:
    trade = {
        "symbol": SYMBOL,
        "side": "buy",
        "price": "100.5",
        "qty": "0.25",
        "ord_type": "market",
        "trade_id": 4665906,
        "timestamp": T1,
    }
    trade.update(fields)
    return {"channel": "trade", "type": "update", "data": [trade]}


BOOK: Book = {
    "bid": [("B1", "100.0", "1.5"), ("B2", "100.0", "0.5"), ("B3", "99.5", "2.0")],
    "ask": [("A1", "100.5", "1.0"), ("A2", "101.0", "3.0")],
}


def _seeded(book: Book = BOOK, **fields: Any) -> tuple[KrakenSource, list[dict]]:
    """A source holding *book* as its opening book, and the rows it wrote."""
    src = KrakenSource(_settings(**fields))
    src._symbol = SYMBOL
    rows = src._apply_snapshot(_snapshot(book)["data"][0])
    return src, rows


def _book_events(src: KrakenSource, frame: dict) -> list[dict]:
    return [event for kind, event, _ in src.parse(frame) if kind == "order"]


def _next_message(src: KrakenSource, book: Book) -> list[dict]:
    """The rows of an update that changes nothing, which writes held deletes."""
    return _book_events(src, _update(book))


# -- signing and the token ------------------------------------------------------


class TestSigning:
    def test_the_signature_matches_krakens_worked_example(self):
        """The example in Kraken's "Spot REST authentication" guide."""
        nonce = "1616492376594"
        post_data = (
            f"nonce={nonce}&ordertype=limit&pair=XBTUSD&price=37500&type=buy"
            "&volume=1.25"
        )
        assert live_kraken._sign(
            "/0/private/AddOrder", nonce, post_data, API_SECRET
        ) == (
            "4/dpxb3iT4tp/ZCVEwSnEsLxx0bqyhLpdfOpc6fn7OR8+UClSV5n9E6aSS8MPtnRfp32bAb0nmbRn6H8ndwLUQ=="
        )

    def test_a_secret_that_is_not_base64_is_refused_without_quoting_it(self):
        src = KrakenSource(
            KrakenSettings(
                api_key=SecretStr(API_KEY), api_secret=SecretStr("not base64 !!")
            )
        )
        with pytest.raises(ConfigError, match="not base64") as info:
            src.preflight()
        assert "not base64 !!" not in str(info.value)

    def test_each_nonce_is_larger_than_the_last(self):
        nonces = [int(live_kraken._nonce()) for _ in range(5)]
        assert nonces == sorted(set(nonces))

    def test_the_token_request_is_signed_with_the_key(self, monkeypatch):
        sent: dict[str, Any] = {}

        class _Reply:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({"error": [], "result": {"token": TOKEN}}).encode()

        def urlopen(req, timeout):
            sent["url"] = req.full_url
            sent["headers"] = dict(req.header_items())
            sent["body"] = req.data.decode()
            return _Reply()

        monkeypatch.setattr(live_kraken.urllib.request, "urlopen", urlopen)
        assert live_kraken._fetch_token(API_KEY, API_SECRET) == TOKEN
        assert sent["url"] == "https://api.kraken.com/0/private/GetWebSocketsToken"
        nonce = sent["body"].removeprefix("nonce=")
        assert sent["headers"]["Api-key"] == API_KEY
        assert sent["headers"]["Api-sign"] == live_kraken._sign(
            live_kraken.TOKEN_PATH, nonce, sent["body"], API_SECRET
        )

    @pytest.mark.parametrize(
        ("error", "hint"),
        [
            ("EAPI:Invalid key", "refused the API key"),
            ("EGeneral:Permission denied", "WebSocket interface - On"),
            ("EAPI:Invalid nonce", "nonce"),
            ("EService:Unavailable", "EService:Unavailable"),
        ],
    )
    def test_a_refused_token_says_what_to_change(self, error, hint):
        with pytest.raises(ConfigError, match=hint):
            live_kraken._token_from({"error": [error], "result": {}})

    def test_the_token_is_kept_out_of_written_files(self):
        token = "TokenNeverSeenBefore0123456789abcdefghijk"
        live_kraken._token_from({"error": [], "result": {"token": token}})
        assert _secrets.redact(f"x{token}y") == f"x{_secrets.MASK}y"

    def test_the_settings_read_both_keys_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("KRAKEN_API_KEY", API_KEY)
        monkeypatch.setenv("KRAKEN_API_SECRET", API_SECRET)
        settings = KrakenSettings()
        assert settings.api_key is not None
        assert settings.api_key.get_secret_value() == API_KEY
        assert settings.depth == 1000

    def test_a_missing_key_names_the_variables(self):
        with pytest.raises(ConfigError, match="KRAKEN_API_KEY") as info:
            KrakenSettings().check_credentials("kraken")
        assert "KRAKEN_API_SECRET" in str(info.value)


# -- the checksum ---------------------------------------------------------------


class TestChecksum:
    def test_the_opening_book_of_krakens_example_matches_its_checksum(self):
        entry = json.loads(FIXTURE.read_text())["data"][0]
        src = KrakenSource(_settings())
        src._symbol = SYMBOL
        rows = src._apply_snapshot(entry)
        assert src._book.checksum() == entry["checksum"] == 1063832831
        assert src.checksum_checks == 1
        assert src.checksum_failures == 0
        assert len(rows) == 35

    def test_trailing_zeros_count(self):
        """A size of 0.10000000 and one of 0.1 give different checksums, so
        the frame must be read with its digits kept."""
        entry = json.loads(FIXTURE.read_text())["data"][0]
        entry["bids"][2]["order_qty"] = "0.1"
        src = KrakenSource(_settings())
        src._symbol = SYMBOL
        with pytest.raises(UnmappedMessageError, match="checksum"):
            src._apply_snapshot(entry)

    def test_a_number_frame_keeps_its_digits(self):
        """Kraken may send prices and sizes as JSON numbers, not strings."""
        text = json.dumps(_snapshot(BOOK)).replace('"1.5"', "1.50000000")
        exact = json.loads(text, parse_float=live_kraken.Decimal)
        qty = exact["data"][0]["bids"][0]["order_qty"]
        assert live_kraken._checksum_text(live_kraken._number(qty)) == "150000000"

    def test_a_mismatch_ends_the_segment_and_is_counted(self):
        src, _ = _seeded()
        after = {**BOOK, "ask": [("A1", "100.5", "0.4"), ("A2", "101.0", "3.0")]}
        frame = _update(
            after, asks=[_order("A1", "100.5", "0.4", "modify")], checksum=12345
        )
        with pytest.raises(UnmappedMessageError, match="12345"):
            src.parse(frame)
        assert src.checksum_failures == 1
        assert src.diagnostics()["checksum_failures"] == 1

    def test_in_count_mode_a_mismatch_is_counted_and_the_capture_goes_on(self):
        src, _ = _seeded(on_checksum_failure="count")
        after = {**BOOK, "ask": [("A1", "100.5", "0.4"), ("A2", "101.0", "3.0")]}
        bad = _update(
            after, asks=[_order("A1", "100.5", "0.4", "modify")], checksum=12345
        )
        with pytest.warns(UserWarning, match="checksum_failures"):
            events = _book_events(src, bad)
        assert [e["action"] for e in events] == ["changed"]
        assert src.checksum_failures == 1
        assert src.checksum_checks == 2


# -- the opening book and order events ------------------------------------------


class TestOpeningBook:
    def test_each_order_is_a_created_row_with_krakens_id(self):
        _, rows = _seeded()
        assert [(r["id"], r["direction"], r["price"], r["volume"]) for r in rows] == [
            ("B1", "bid", 100.0, 1.5),
            ("B2", "bid", 100.0, 0.5),
            ("B3", "bid", 99.5, 2.0),
            ("A1", "ask", 100.5, 1.0),
            ("A2", "ask", 101.0, 3.0),
        ]
        assert {r["action"] for r in rows} == {"created"}
        assert {r["venue"] for r in rows} == {"kraken"}

    def test_the_rows_carry_the_time_kraken_made_the_book(self):
        _, rows = _seeded()
        assert rows[0]["exchange_timestamp"] == pd.Timestamp(T0)

    def test_an_unparseable_order_ends_the_snapshot(self):
        src = KrakenSource(_settings())
        src._symbol = SYMBOL
        entry = _snapshot(BOOK)["data"][0]
        entry["bids"][0]["limit_price"] = "NaN"
        with pytest.raises(UnmappedMessageError, match="opening book"):
            src._apply_snapshot(entry)


class TestOrderEvents:
    def test_add_is_created(self):
        src, _ = _seeded()
        after = {**BOOK, "bid": [*BOOK["bid"], ("B4", "99.0", "0.7")]}
        events = _book_events(
            src, _update(after, bids=[_order("B4", "99.0", "0.7", "add")])
        )
        assert [(e["id"], e["action"], e["direction"]) for e in events] == [
            ("B4", "created", "bid")
        ]
        assert events[0]["exchange_timestamp"] == pd.Timestamp(T1)
        assert src.adds == 1

    def test_modify_is_changed_and_keeps_the_order_in_its_queue(self):
        src, _ = _seeded()
        # B1 is first at 100.0; a fill leaves it first, which the checksum
        # (order by order) would catch if the capture moved it.
        after = {**BOOK, "bid": [("B1", "100.0", "0.2"), *BOOK["bid"][1:]]}
        events = _book_events(
            src, _update(after, bids=[_order("B1", "100.0", "0.2", "modify")])
        )
        assert [(e["id"], e["action"], e["volume"]) for e in events] == [
            ("B1", "changed", 0.2)
        ]
        assert src.checksum_failures == 0

    def test_a_modify_that_raises_the_size_sends_the_order_to_the_back(self):
        """Seen in a live capture: Kraken's checksum puts an order whose size
        grew behind the others at its price."""
        src, _ = _seeded()
        after = {
            **BOOK,
            "bid": [("B2", "100.0", "0.5"), ("B1", "100.0", "2.5"), BOOK["bid"][2]],
        }
        events = _book_events(
            src, _update(after, bids=[_order("B1", "100.0", "2.5", "modify")])
        )
        assert [(e["id"], e["action"], e["volume"]) for e in events] == [
            ("B1", "changed", 2.5)
        ]
        assert src.checksum_failures == 0

    def test_delete_is_deleted_once_the_next_message_does_not_add_it_again(self):
        src, _ = _seeded()
        after = {**BOOK, "ask": BOOK["ask"][1:]}
        events = _book_events(
            src, _update(after, asks=[_order("A1", "100.5", "1.0", "delete")])
        )
        assert events == []
        events = _next_message(src, after)
        assert [(e["id"], e["action"], e["direction"]) for e in events] == [
            ("A1", "deleted", "ask")
        ]
        assert events[0]["exchange_timestamp"] == pd.Timestamp(T1)
        assert "A1" not in src._open

    def test_a_delete_then_an_add_of_the_same_id_is_one_amend(self):
        """Kraken amends an order's price as a delete and, in the next
        message, an add with the same id (seen in live captures)."""
        src, _ = _seeded()
        gone = {**BOOK, "ask": BOOK["ask"][1:]}
        assert (
            _book_events(
                src, _update(gone, asks=[_order("A1", "100.5", "1.0", "delete")])
            )
            == []
        )
        moved = {**BOOK, "ask": [BOOK["ask"][1], ("A1", "101.0", "1.0")]}
        events = _book_events(
            src, _update(moved, asks=[_order("A1", "101.0", "1.0", "add")])
        )
        assert [(e["id"], e["action"], e["price"]) for e in events] == [
            ("A1", "changed", 101.0)
        ]
        assert src.amends == 1
        # The checksum passed, so A1 joined the back of the 101.0 level.
        assert src.checksum_failures == 0

    def test_an_amend_that_moves_and_shrinks_is_a_cancel_and_a_new_order(self):
        """A replay reads a smaller size as a fill at the old price, so such
        an amend is written under a new id instead (seen in live captures:
        without this the replayed book crossed)."""
        src, _ = _seeded()
        gone = {**BOOK, "ask": BOOK["ask"][1:]}
        _book_events(src, _update(gone, asks=[_order("A1", "100.5", "1.0", "delete")]))
        moved = {**BOOK, "ask": [BOOK["ask"][1], ("A1", "101.0", "0.4")]}
        events = _book_events(
            src, _update(moved, asks=[_order("A1", "101.0", "0.4", "add")])
        )
        assert [(e["id"], e["action"], e["price"]) for e in events] == [
            ("A1", "deleted", 100.5),
            ("A1#1", "created", 101.0),
        ]
        assert src.amends_split == 1
        assert "A1" not in src._open
        assert src.checksum_failures == 0

    def test_an_amend_that_shrinks_at_the_same_price_is_split_too(self):
        src, _ = _seeded()
        gone = {**BOOK, "ask": BOOK["ask"][1:]}
        _book_events(src, _update(gone, asks=[_order("A1", "100.5", "1.0", "delete")]))
        back = {**BOOK, "ask": [("A1", "100.5", "0.4"), BOOK["ask"][1]]}
        events = _book_events(
            src, _update(back, asks=[_order("A1", "100.5", "0.4", "add")])
        )
        assert [(e["id"], e["action"]) for e in events] == [
            ("A1", "deleted"),
            ("A1#1", "created"),
        ]

    def test_rows_carry_the_time_the_frame_arrived(self):
        """A frame can wait in the reader queue; its rows keep its arrival."""
        src, _ = _seeded()
        arrived = pd.Timestamp("2026-11-19T12:00:01.300Z").as_unit("ns")
        after = {**BOOK, "bid": [*BOOK["bid"], ("B4", "99.0", "0.7")]}
        frame = _update(after, bids=[_order("B4", "99.0", "0.7", "add")])
        ((_, row, _),) = src.parse(frame, arrived)
        assert row["timestamp"] == arrived
        ((_, trade, _),) = src.parse(_trade(), arrived)
        assert trade["timestamp"] == arrived

    def test_a_delete_and_an_add_in_one_message_is_one_amend(self):
        src, _ = _seeded()
        moved = {**BOOK, "ask": [BOOK["ask"][1], ("A1", "101.0", "1.0")]}
        frame = _update(
            moved,
            asks=[
                _order("A1", "100.5", "1.0", "delete"),
                _order("A1", "101.0", "1.0", "add"),
            ],
        )
        assert [(e["id"], e["action"]) for e in _book_events(src, frame)] == [
            ("A1", "changed")
        ]

    def test_an_id_added_again_later_gets_a_new_id(self):
        src, _ = _seeded()
        gone = {**BOOK, "ask": BOOK["ask"][1:]}
        _book_events(src, _update(gone, asks=[_order("A1", "100.5", "1.0", "delete")]))
        _next_message(src, gone)
        back = {**BOOK, "ask": [("A1", "100.5", "1.0"), BOOK["ask"][1]]}
        events = _book_events(
            src, _update(back, asks=[_order("A1", "100.5", "1.0", "add")])
        )
        assert [(e["id"], e["action"]) for e in events] == [("A1#1", "created")]
        assert src.readded_after_delete == 1

    def test_a_held_delete_is_written_at_shutdown(self):
        src, _ = _seeded()
        after = {**BOOK, "ask": BOOK["ask"][1:]}
        _book_events(src, _update(after, asks=[_order("A1", "100.5", "1.0", "delete")]))

        async def collect():
            return [e async for e in src.shutdown_synthetic_events()]

        closing = asyncio.run(collect())
        a1 = [e for e in closing if e["id"] == "A1"]
        assert len(a1) == 1
        assert a1[0]["exchange_timestamp"] == pd.Timestamp(T1)
        assert a1[0]["origin"] == "stream"

    def test_several_events_in_one_frame_archive_the_frame_once(self):
        src, _ = _seeded()
        after = {
            "bid": [*BOOK["bid"], ("B4", "99.0", "0.7")],
            "ask": [*BOOK["ask"], ("A3", "101.5", "1")],
        }
        frame = _update(
            after,
            bids=[_order("B4", "99.0", "0.7", "add")],
            asks=[_order("A3", "101.5", "1", "add")],
        )
        items = src.parse(frame)
        assert [kind for kind, _, _ in items] == ["order", "order"]
        assert [raw is frame for _, _, raw in items] == [True, False]

    def test_without_the_message_time_an_add_uses_its_own(self):
        src, _ = _seeded()
        after = {**BOOK, "bid": [*BOOK["bid"], ("B4", "99.0", "0.7")]}
        frame = _update(
            after, bids=[_order("B4", "99.0", "0.7", "add")], timestamp=None
        )
        (event,) = _book_events(src, frame)
        assert event["exchange_timestamp"] == pd.Timestamp(T0)
        assert src.messages_without_time == 1

    @pytest.mark.parametrize("event", ["modify", "delete"])
    def test_an_event_for_an_order_not_held_ends_the_segment(self, event):
        src, _ = _seeded()
        with pytest.raises(UnmappedMessageError, match="does not hold"):
            src.parse(_update(BOOK, bids=[_order("NOPE", "100.0", "1", event)]))

    def test_an_event_of_unknown_kind_ends_the_segment(self):
        src, _ = _seeded()
        with pytest.raises(UnmappedMessageError, match="unknown kind"):
            src.parse(_update(BOOK, bids=[_order("B1", "100.0", "1", "amend")]))

    def test_a_second_opening_book_ends_the_segment(self):
        src, _ = _seeded()
        with pytest.raises(UnmappedMessageError, match="new opening book"):
            src.parse(_snapshot(BOOK))

    def test_a_refused_subscription_is_a_config_error(self):
        src, _ = _seeded()
        reply = {
            "method": "subscribe",
            "success": False,
            "error": "Invalid token",
            "time_in": T0,
            "time_out": T0,
        }
        with pytest.raises(ConfigError, match="WebSocket interface - On"):
            src.parse(reply)

    def test_other_frames_are_only_archived(self):
        src, _ = _seeded()
        heartbeat = {"channel": "heartbeat"}
        assert src.parse(heartbeat) == [("raw", {}, heartbeat)]


# -- the depth window ------------------------------------------------------------


def _deep_book(levels: int = 10) -> Book:
    """Bids at 100.0 down in steps of 1, one order each; one ask."""
    return {
        "bid": [(f"B{i}", f"{100 - i}.0", "1.0") for i in range(levels)],
        "ask": [("A1", "101.0", "1.0")],
    }


class TestDepthWindow:
    """At depth 10, Kraken sends ten levels a side and no delete for an
    order pushed out of them."""

    def _pushed_out(self) -> tuple[KrakenSource, Book]:
        """B9 (the tenth level) is pushed out by a new best bid."""
        src, _ = _seeded(_deep_book(), depth=10)
        book = _deep_book()
        after = {"bid": [("N1", "100.5", "1.0"), *book["bid"][:9]], "ask": book["ask"]}
        events = _book_events(
            src, _update(after, bids=[_order("N1", "100.5", "1.0", "add")])
        )
        assert [(e["id"], e["action"]) for e in events] == [("N1", "created")]
        return src, after

    def test_an_order_pushed_out_is_kept_open_and_not_written(self):
        src, _ = self._pushed_out()
        assert "B9" in src._parked
        assert "B9" in src._open
        assert src.window_exits == 1
        assert src.checksum_failures == 0

    def test_an_order_that_comes_back_unchanged_is_not_written(self):
        src, before = self._pushed_out()
        after = {
            "bid": before["bid"][1:] + [("B9", "91.0", "1.0")],
            "ask": before["ask"],
        }
        frame = _update(
            after,
            bids=[
                _order("N1", "100.5", "1.0", "delete"),
                _order("B9", "91.0", "1.0", "add"),
            ],
        )
        events = _book_events(src, frame)
        # N1's delete is written with the next message.
        assert events == []
        assert src.window_returns == 1
        assert "B9" not in src._parked

    def test_an_order_that_comes_back_smaller_is_changed(self):
        src, before = self._pushed_out()
        after = {
            "bid": before["bid"][1:] + [("B9", "91.0", "0.3")],
            "ask": before["ask"],
        }
        frame = _update(
            after,
            bids=[
                _order("N1", "100.5", "1.0", "delete"),
                _order("B9", "91.0", "0.3", "add"),
            ],
        )
        events = _book_events(src, frame)
        assert [(e["id"], e["action"], e["volume"]) for e in events] == [
            ("B9", "changed", 0.3),
        ]

    def test_an_order_that_does_not_come_back_is_deleted_when_the_window_reaches_it(
        self,
    ):
        src, before = self._pushed_out()
        # B9 left the book while outside the window: Kraken sends the next
        # level down (X1 at 90.0) instead.
        after = {
            "bid": before["bid"][1:] + [("X1", "90.0", "2.0")],
            "ask": before["ask"],
        }
        frame = _update(
            after,
            bids=[
                _order("N1", "100.5", "1.0", "delete"),
                _order("X1", "90.0", "2.0", "add"),
            ],
        )
        events = _book_events(src, frame)
        assert [(e["id"], e["action"]) for e in events] == [
            ("X1", "created"),
            ("B9", "deleted"),
        ]
        assert events[-1]["price"] == 91.0
        assert src.window_unseen_deletes == 1
        assert "B9" not in src._open
        assert src.checksum_failures == 0

    def test_an_order_deleted_unseen_that_comes_back_gets_a_new_id(self):
        src, before = self._pushed_out()
        gone = {
            "bid": before["bid"][1:] + [("X1", "90.0", "2.0")],
            "ask": before["ask"],
        }
        _book_events(
            src,
            _update(
                gone,
                bids=[
                    _order("N1", "100.5", "1.0", "delete"),
                    _order("X1", "90.0", "2.0", "add"),
                ],
            ),
        )
        back = {"bid": gone["bid"][:9] + [("B9", "91.0", "1.0")], "ask": gone["ask"]}
        events = _book_events(
            src, _update(back, bids=[_order("B9", "91.0", "1.0", "add")])
        )
        assert [(e["id"], e["action"]) for e in events] == [
            ("N1", "deleted"),
            ("B9#1", "created"),
        ]
        assert src.readded_after_unseen_delete == 1
        delete = _update(gone, bids=[_order("B9", "91.0", "1.0", "delete")])
        # X1 was pushed out by B9's return and comes back with B9's delete.
        delete["data"][0]["bids"].append(_order("X1", "90.0", "2.0", "add"))
        _book_events(src, delete)
        events = _next_message(src, gone)
        assert ("B9#1", "deleted") in [(e["id"], e["action"]) for e in events]

    def test_a_short_side_shows_the_whole_book(self):
        """With fewer levels than the window, a parked order whose price is
        on the book is gone."""
        src, before = self._pushed_out()
        after = {"bid": before["bid"][:5], "ask": before["ask"]}
        frame = _update(
            after,
            bids=[
                _order(oid, price, qty, "delete")
                for oid, price, qty in before["bid"][5:]
            ],
        )
        events = _book_events(src, frame)
        assert ("B9", "deleted") in [(e["id"], e["action"]) for e in events]

    def test_parked_orders_get_their_closing_delete_at_shutdown(self):
        src, _ = self._pushed_out()

        async def collect():
            return [e async for e in src.shutdown_synthetic_events()]

        closing = asyncio.run(collect())
        assert "B9" in {e["id"] for e in closing}
        assert {e["action"] for e in closing} == {"deleted"}
        assert src.diagnostics()["parked_at_end"] == 1


# -- trades ------------------------------------------------------------------------


class TestTrades:
    def test_a_trade_has_the_takers_side_and_names_no_orders(self):
        src, _ = _seeded()
        ((kind, event, _),) = src.parse(_trade(side="sell"))
        assert kind == "trade"
        assert event["direction"] == "sell"
        assert (event["price"], event["volume"]) == (100.5, 0.25)
        assert (event["buy_order_id"], event["sell_order_id"]) == ("", "")
        assert event["trade_id"] == 4665906
        assert event["exchange_timestamp"] == pd.Timestamp(T1)

    @pytest.mark.parametrize(
        "bad", [{"price": "x"}, {"side": "up"}, {"timestamp": None}, {"qty": None}]
    )
    def test_an_unparseable_trade_is_dropped_and_counted(self, bad):
        src, _ = _seeded()
        assert src.parse(_trade(**bad))[0][0] == "raw"
        assert src.dropped == 1

    def test_a_trade_before_the_opening_book_is_skipped(self):
        src, _ = _seeded()
        assert src.parse(_trade(timestamp="2026-11-19T11:59:59Z"))[0][0] == "raw"
        assert src.pre_snapshot_trades_skipped == 1

    def test_a_trade_on_another_symbol_is_only_archived(self):
        src, _ = _seeded()
        assert src.parse(_trade(symbol="ETH/USD"))[0][0] == "raw"


# -- a whole capture against a fake venue ----------------------------------------


class _FakeWs:
    """One connection: records what is sent, yields *frames*, then stays open."""

    def __init__(self, frames: list[str]) -> None:
        self.frames = frames
        self.sent: list[dict[str, Any]] = []

    async def send(self, text: str) -> None:
        self.sent.append(json.loads(text))

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        for frame in self.frames:
            yield frame
            await asyncio.sleep(0)
        await asyncio.Event().wait()


class _FakeVenue:
    def __init__(self, l3: list[str], trades: list[str]) -> None:
        self.ws = {
            live_kraken.L3_WS_URL: _FakeWs(l3),
            live_kraken.TRADE_WS_URL: _FakeWs(trades),
        }

    def connect(self, url: str, **_: Any):
        @asynccontextmanager
        async def connection():
            yield self.ws[url]

        return connection()


def _subscribed(channel: str) -> str:
    return json.dumps(
        {
            "method": "subscribe",
            "result": {"channel": channel, "symbol": SYMBOL, "snapshot": True},
            "success": True,
            "time_in": T0,
            "time_out": T0,
        }
    )


def _run_capture(
    monkeypatch, out: Path, l3: list[str], trades: Sequence[str] = ()
) -> tuple[_FakeVenue, Any]:
    """Run one capture segment against a fake Kraken sending *l3* and *trades*."""
    import websockets

    venue = _FakeVenue(l3=l3, trades=[_subscribed("trade"), *trades])
    monkeypatch.setattr(websockets, "connect", venue.connect)

    def fetch_token(api_key: str, api_secret: str) -> str:
        assert (api_key, api_secret) == (API_KEY, API_SECRET)
        return live_kraken._token_from({"error": [], "result": {"token": TOKEN}})

    monkeypatch.setattr(live_kraken, "_fetch_token", fetch_token)
    config = CaptureConfig(pair=SYMBOL, out_dir=out, minutes=0.01)
    return venue, asyncio.run(run_capturer(KrakenSource(_settings()), config))


def _orphans(out: Path) -> set[str]:
    """Ids in orders.csv with a ``deleted`` row but no ``created`` row."""
    orders = pd.read_csv(out / "orders.csv")
    created = set(orders.loc[orders["action"] == "created", "id"])
    return set(orders.loc[orders["action"] == "deleted", "id"]) - created


@pytest.fixture
def capture(monkeypatch, tmp_path) -> tuple[Path, _FakeVenue]:
    # A1's size is sent as the JSON number 1.00000000, as Kraken's level3
    # page shows numbers, so the capture must keep its trailing zeros for the
    # checksum to match.
    book = {**BOOK, "ask": [("A1", "100.5", "1.00000000"), BOOK["ask"][1]]}
    after_add = {**book, "bid": [*book["bid"], ("B4", "99.0", "0.7")]}
    after_fill = {**after_add, "ask": [("A1", "100.5", "0.75"), book["ask"][1]]}

    def as_numbers(frame: dict) -> str:
        return json.dumps(frame).replace('"1.00000000"', "1.00000000")

    out = tmp_path / "cap"
    venue, result = _run_capture(
        monkeypatch,
        out,
        l3=[
            _subscribed("level3"),
            as_numbers(_snapshot(book)),
            as_numbers(_update(after_add, bids=[_order("B4", "99.0", "0.7", "add")])),
            as_numbers(
                _update(after_fill, asks=[_order("A1", "100.5", "0.75", "modify")])
            ),
            json.dumps({"channel": "heartbeat"}),
        ],
        trades=[json.dumps(_trade(price="100.5", qty="0.25"))],
    )
    assert result.capture_error is None, result.capture_error
    return out, venue


class TestBrokenBook:
    """A checksum failure ends the segment, but every row the capture applied
    is still written, so no order is closed without being opened."""

    def test_the_rows_of_the_message_that_failed_are_written(
        self, monkeypatch, tmp_path
    ):
        after = {**BOOK, "bid": [*BOOK["bid"], ("B4", "99.0", "0.7")]}
        bad = _update(after, bids=[_order("B4", "99.0", "0.7", "add")], checksum=1)
        out = tmp_path / "cap"
        _, result = _run_capture(
            monkeypatch,
            out,
            l3=[_subscribed("level3"), json.dumps(_snapshot(BOOK)), json.dumps(bad)],
        )
        assert "checksum" in (result.capture_error or "")
        orders = pd.read_csv(out / "orders.csv")
        assert "B4" in set(orders.loc[orders["action"] == "created", "id"])
        assert _orphans(out) == set()
        raw = (out / "raw.jsonl").read_text()
        assert '"checksum":1,' in raw.replace(" ", "")

    def test_a_held_delete_is_written_when_the_next_message_fails(
        self, monkeypatch, tmp_path
    ):
        gone = {**BOOK, "ask": BOOK["ask"][1:]}
        delete = _update(gone, asks=[_order("A1", "100.5", "1.0", "delete")])
        bad = _update(gone, checksum=1)
        out = tmp_path / "cap"
        _run_capture(
            monkeypatch,
            out,
            l3=[
                _subscribed("level3"),
                json.dumps(_snapshot(BOOK)),
                json.dumps(delete),
                json.dumps(bad),
            ],
        )
        orders = pd.read_csv(out / "orders.csv")
        a1 = orders[orders["id"] == "A1"]
        assert list(a1["action"]) == ["created", "deleted"]
        assert list(a1["origin"]) == ["snapshot", "stream"]

    def test_an_opening_book_that_fails_is_written_and_closed(
        self, monkeypatch, tmp_path
    ):
        out = tmp_path / "cap"
        _, result = _run_capture(
            monkeypatch,
            out,
            l3=[_subscribed("level3"), json.dumps(_snapshot(BOOK, checksum=1))],
        )
        assert result.capture_error_phase == "snapshot"
        orders = pd.read_csv(out / "orders.csv")
        assert set(orders.loc[orders["action"] == "created", "id"]) == {
            "B1",
            "B2",
            "B3",
            "A1",
            "A2",
        }
        assert _orphans(out) == set()

    def test_an_opening_book_that_cannot_be_read_leaves_no_order_open(
        self, monkeypatch, tmp_path
    ):
        snapshot = _snapshot(BOOK)
        snapshot["data"][0]["asks"][1]["limit_price"] = "NaN"
        out = tmp_path / "cap"
        _run_capture(monkeypatch, out, l3=[_subscribed("level3"), json.dumps(snapshot)])
        orders_csv = out / "orders.csv"
        assert not orders_csv.exists() or _orphans(out) == set()


class TestCapture:
    def test_both_channels_are_subscribed_and_the_book_gets_the_token(self, capture):
        _, venue = capture
        (l3,) = venue.ws[live_kraken.L3_WS_URL].sent
        assert l3["params"] == {
            "channel": "level3",
            "symbol": [SYMBOL],
            "depth": 1000,
            "snapshot": True,
            "token": TOKEN,
        }
        (trade,) = venue.ws[live_kraken.TRADE_WS_URL].sent
        assert trade["params"]["channel"] == "trade"
        assert "token" not in trade["params"]

    def test_the_files_hold_the_order_lifecycles_and_the_trade(self, capture):
        out, _ = capture
        orders = pd.read_csv(out / "orders.csv")
        assert orders.groupby("action").size().to_dict() == {
            "changed": 1,
            "created": 6,
            "deleted": 6,
        }
        trades = pd.read_csv(out / "trades.csv")
        assert len(trades) == 1
        assert (out / "raw.jsonl").exists()

    def test_meta_records_the_checks_and_what_the_source_declares(self, capture):
        out, _ = capture
        meta = json.loads((out / "meta.json").read_text())
        assert meta["source"] == "kraken"
        assert meta["replay_source"] == "bitstamp"
        assert meta["level"] == "L3"
        assert meta["feed_type"] == "matched_book"
        assert meta["trade_attribution"] == "maker_only"
        assert meta["checksum_checks"] == 3
        assert meta["checksum_failures"] == 0
        # Every price sits on the default cent grid, so it stays.
        assert meta["tick_size"] == 0.01

    def test_the_capture_replays_through_the_pipeline(self, capture):
        out, _ = capture
        result = Pipeline(source=BitstampSource()).run(out / "orders.csv")
        assert set(result.events["id"]) >= {"B1", "B4", "A1"}
        validate_trades_df(result.trades)

    def test_the_capture_passes_audit(self, capture, cli_runner):
        out, _ = capture
        r = cli_runner("audit", str(out / "orders.csv"), "--json")
        assert r.returncode == 0, r.stdout + r.stderr
        assert json.loads(r.stdout)["ok"] is True

    def test_no_file_holds_the_key_or_the_token(self, capture):
        out, _ = capture
        for path in out.rglob("*"):
            if path.is_file():
                text = path.read_text(errors="replace")
                for secret in (API_KEY, API_SECRET, TOKEN):
                    assert secret not in text, path.name
