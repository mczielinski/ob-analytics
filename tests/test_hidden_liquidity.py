"""Tests for hidden_liquidity.py — iceberg refills and trades inside the spread."""

import hashlib
import uuid
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ob_analytics import LobsterSource, Pipeline, RunContext
from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader
from ob_analytics.exceptions import ConfigError
from ob_analytics.hidden_liquidity import (
    ICEBERG_MAX_DELAY,
    detect_icebergs,
    hidden_trades,
    iceberg_slices,
)
from ob_analytics.synth import generate_session
from ob_analytics.visualization import prepare

BASE = pd.Timestamp("2024-01-02 14:30:00", tz="UTC")


def _us(microseconds: float) -> pd.Timestamp:
    return BASE + pd.Timedelta(microseconds=microseconds)


class _Stream:
    """Build an events frame and its trades one row at a time."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.makers: list[int] = []

    def add(
        self, oid, us, action, volume, fill=0, *, price=100, side="ask", maker=False
    ):
        eid = len(self.events) + 1
        self.events.append(
            {
                "event_id": eid,
                "id": oid,
                "timestamp": _us(us),
                "price": price,
                "volume": volume,
                "action": action,
                "direction": side,
                "fill": fill,
            }
        )
        if maker:
            self.makers.append(eid)
        return self

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        events = pd.DataFrame(self.events)
        events["action"] = pd.Categorical(
            events["action"], categories=["created", "changed", "deleted"], ordered=True
        )
        events["direction"] = pd.Categorical(
            events["direction"], categories=["bid", "ask"], ordered=True
        )
        trades = pd.DataFrame({"maker_event_id": self.makers})
        return events, trades


# ── detect_icebergs ──────────────────────────────────────────────────


class TestDetectIcebergs:
    def test_three_slice_iceberg_is_one_chain(self):
        s = _Stream()
        s.add(1, 0, "created", 100)
        s.add(1, 1_000, "changed", 0, 100, maker=True)  # peak filled
        s.add(2, 1_250, "created", 100)  # refill, 0.25 ms later
        s.add(2, 5_000, "changed", 40, 60, maker=True)
        s.add(2, 5_000, "changed", 0, 40, maker=True)  # filled out in two fills
        s.add(3, 5_200, "created", 100)  # refill
        s.add(3, 9_000, "deleted", 100)  # cancelled: the chain ends
        events, trades = s.frames()

        det = detect_icebergs(events, trades)
        det_slices = iceberg_slices(events, trades)

        assert len(det) == 1
        row = det.iloc[0]
        assert row["slices"] == 3
        assert row["refills"] == 2
        assert row["same_size_refills"] == 2
        assert row["peak"] == 100
        assert row["executed"] == 200
        assert row["confidence"] == "high"
        assert row["direction"] == "ask"
        assert row["price"] == 100
        assert row["timestamp_start"] == _us(0)
        assert row["timestamp_end"] == _us(9_000)
        assert row["median_delay_s"] == pytest.approx(225e-6)
        assert det_slices["id"].tolist() == [1, 2, 3]
        assert det_slices["slice"].tolist() == [1, 2, 3]
        assert det_slices["delay_s"].iloc[1:].tolist() == pytest.approx(
            [250e-6, 200e-6]
        )

    def test_uuid_order_ids_are_kept_as_written(self):
        # Independent Reserve writes order ids as UUID strings.
        a, b, c = (
            "0b6c8f1e-3a52-4c1e-9d0a-5f2e7c41a001",
            "0b6c8f1e-3a52-4c1e-9d0a-5f2e7c41a002",
            "0b6c8f1e-3a52-4c1e-9d0a-5f2e7c41a003",
        )
        s = _Stream()
        s.add(a, 0, "created", 100)
        s.add(a, 1_000, "changed", 0, 100, maker=True)
        s.add(b, 1_250, "created", 100)
        s.add(b, 5_000, "changed", 0, 100, maker=True)
        s.add(c, 5_200, "created", 100)
        s.add(c, 9_000, "deleted", 100)
        events, trades = s.frames()

        det = detect_icebergs(events, trades)
        det_slices = iceberg_slices(events, trades)

        assert len(det) == 1
        assert det.iloc[0]["slices"] == 3
        assert det.iloc[0]["confidence"] == "high"
        assert det_slices["id"].tolist() == [a, b, c]

        # String ids pass through the gallery's overlay.
        overlay = prepare.hidden_liquidity_overlay(
            det_slices, pd.DataFrame(columns=["timestamp", "price"])
        )
        assert overlay["iceberg_lines"]["id"].tolist() == [a, b, c]
        assert overlay["iceberg_refills"]["id"].tolist() == [b, c]

    def test_new_order_after_a_cancel_is_not_a_refill(self):
        s = _Stream()
        s.add(1, 0, "created", 100)
        s.add(1, 1_000, "deleted", 100)
        s.add(2, 1_100, "created", 100)
        events, trades = s.frames()

        assert detect_icebergs(events, trades).empty

    def test_filled_aggressor_is_not_a_slice(self):
        # Order 1 crossed and was filled to zero as the taker, not the maker.
        s = _Stream()
        s.add(1, 0, "created", 50, side="bid")
        s.add(1, 0, "deleted", 0, 50, side="bid")
        s.add(2, 100, "created", 50, side="bid")
        events, trades = s.frames()

        assert detect_icebergs(events, trades).empty

    def test_refill_later_than_max_delay_is_ignored(self):
        s = _Stream()
        s.add(1, 0, "created", 100)
        s.add(1, 1_000, "changed", 0, 100, maker=True)
        s.add(2, 3_000, "created", 100)  # 2 ms later
        events, trades = s.frames()

        assert detect_icebergs(events, trades).empty
        found_slices = iceberg_slices(events, trades, max_delay="5ms")
        assert found_slices["id"].tolist() == [1, 2]

    def test_refill_must_match_price_and_side(self):
        s = _Stream()
        s.add(1, 0, "created", 100)
        s.add(1, 1_000, "changed", 0, 100, maker=True)
        s.add(2, 1_100, "created", 100, price=101)
        s.add(3, 1_100, "created", 100, side="bid")
        events, trades = s.frames()

        assert detect_icebergs(events, trades).empty

    def test_new_order_is_matched_to_the_peak_it_equals(self):
        # Two peaks filled by one sweep; only the 30-lot one refills.
        s = _Stream()
        s.add(1, 0, "created", 70)
        s.add(2, 10, "created", 30)
        s.add(1, 1_000, "deleted", 0, 70, maker=True)
        s.add(2, 1_000, "deleted", 0, 30, maker=True)
        s.add(3, 1_200, "created", 30)
        events, trades = s.frames()

        det = detect_icebergs(events, trades)
        det_slices = iceberg_slices(events, trades)
        assert det_slices["id"].tolist() == [2, 3]
        assert det["confidence"].tolist() == ["medium"]

    def test_different_size_refill_is_low_confidence(self):
        s = _Stream()
        s.add(1, 0, "created", 100)
        s.add(1, 1_000, "changed", 0, 100, maker=True)
        s.add(2, 1_200, "created", 60)
        events, trades = s.frames()

        det = detect_icebergs(events, trades)
        assert det["confidence"].tolist() == ["low"]
        assert det["same_size_refills"].tolist() == [0]

    def test_order_follows_time_not_event_id(self):
        # Bitstamp numbers events by order id, not by time.  Order 2 was placed
        # 20 ms before order 1 was filled out, so it is not a refill, even
        # though its event ids come after the fill's.
        s = _Stream()
        s.add(1, 0, "created", 100)
        s.add(1, 30_000, "changed", 0, 100, maker=True)
        s.add(2, 10_000, "created", 100)
        events, trades = s.frames()

        found = detect_icebergs(events, trades, max_delay="1s")

        assert found.empty

    def test_refill_found_when_event_ids_are_out_of_time_order(self):
        s = _Stream()
        s.add(2, 1_200, "created", 100)  # the refill, numbered first
        s.add(1, 0, "created", 100)
        s.add(1, 1_000, "changed", 0, 100, maker=True)
        events, trades = s.frames()

        found = detect_icebergs(events, trades)
        found_slices = iceberg_slices(events, trades)

        assert found_slices["id"].tolist() == [1, 2]
        assert found_slices["delay_s"].iloc[1] == pytest.approx(200e-6)
        assert found["timestamp_start"].tolist() == [_us(0)]

    def test_peak_is_na_when_first_slice_was_already_resting(self):
        s = _Stream()
        s.add(1, 1_000, "changed", 0, 100, maker=True)  # no created row
        s.add(2, 1_200, "created", 100)
        events, trades = s.frames()

        found = detect_icebergs(events, trades)
        found_slices = iceberg_slices(events, trades)

        assert found["peak"].isna().tolist() == [True]
        assert found_slices["volume"].isna().tolist() == [True, False]

    def test_hidden_order_id_never_forms_a_slice(self):
        s = _Stream()
        s.add(0, 1_000, "changed", 0, 100, maker=True)  # LOBSTER hidden execution
        s.add(2, 1_200, "created", 100)
        events, trades = s.frames()

        assert detect_icebergs(events, trades).empty

    def test_empty_result_keeps_its_columns(self):
        s = _Stream()
        s.add(1, 0, "created", 100)
        events, trades = s.frames()

        det = detect_icebergs(events, trades)
        det_slices = iceberg_slices(events, trades)
        assert "confidence" in det.columns
        assert "delay_s" in det_slices.columns

    def test_negative_max_delay_raises(self):
        events, trades = _Stream().add(1, 0, "created", 1).frames()
        with pytest.raises(ConfigError, match="max_delay"):
            detect_icebergs(events, trades, max_delay="-1ms")

    def test_missing_column_raises(self):
        events, trades = _Stream().add(1, 0, "created", 1).frames()
        with pytest.raises(ConfigError, match="fill"):
            detect_icebergs(events.drop(columns="fill"), trades)

    def test_default_max_delay_is_one_millisecond(self):
        assert ICEBERG_MAX_DELAY == pd.Timedelta("1ms")


class TestSyntheticGroundTruth:
    """The synth generator labels every iceberg slice, so scores are exact."""

    @staticmethod
    def _links(frame: pd.DataFrame, group: str) -> set[tuple[int, int]]:
        links = set()
        for _, g in frame.sort_values([group, "id"]).groupby(group):
            ids = g["id"].tolist()
            links |= set(pairwise(ids))
        return links

    @pytest.mark.parametrize("process", ["poisson", "hawkes"])
    def test_recovers_every_refill(self, process):
        session = generate_session(
            seed=3, duration=300, iceberg_fraction=0.1, arrival_process=process
        )
        det_slices = iceberg_slices(session.events, session.trades)

        true_links = self._links(session.icebergs, "iceberg")
        found = self._links(det_slices, "iceberg")
        hits = len(true_links & found)
        assert true_links, "the session should hold refilled icebergs"
        assert hits == len(true_links)  # recall 1.0
        assert hits / len(found) >= 0.97  # a few coincidental new orders

    def test_ground_truth_is_empty_without_icebergs(self):
        session = generate_session(seed=1, duration=30)
        assert session.icebergs.empty
        assert list(session.icebergs.columns) == ["id", "iceberg"]


class TestOrderIdsDoNotMatter:
    """Detection follows time, not how the venue's order ids sort."""

    @pytest.mark.parametrize(
        "remap",
        [
            lambda i: int(hashlib.sha1(str(i).encode()).hexdigest()[:12], 16),
            lambda i: str(uuid.uuid5(uuid.NAMESPACE_OID, str(i))),
        ],
        ids=["hashed-integers", "uuids"],
    )
    def test_ids_unrelated_to_time_give_the_same_icebergs(
        self, bitstamp_sample_dir, tmp_path, remap
    ):
        # Bitstamp's ids rise with time; Independent Reserve writes the same
        # format with UUIDs, which do not, and neither do integers from a hash
        # (#456: a refill placed in the millisecond its old slice filled was
        # numbered before the fill and lost).
        orders = pd.read_csv(bitstamp_sample_dir / "orders.csv.gz")
        trades = pd.read_csv(bitstamp_sample_dir / "trades.csv")
        sides = ["buy_order_id", "sell_order_id"]
        ids = pd.unique(pd.concat([orders["id"], *(trades[c] for c in sides)]))
        new_id = {i: remap(i) for i in ids}
        assert len(set(new_id.values())) == len(ids)
        orders["id"] = orders["id"].map(new_id)
        for c in sides:
            trades[c] = trades[c].map(new_id)
        orders.to_csv(tmp_path / "orders.csv", index=False)
        trades.to_csv(tmp_path / "trades.csv", index=False)

        def detect(orders_csv):
            events = BitstampLoader().load(orders_csv)
            linked = BitstampTradeReader().load(events, orders_csv)
            return detect_icebergs(events, linked), iceberg_slices(events, linked)

        expected, expected_slices = detect(bitstamp_sample_dir / "orders.csv.gz")
        got, got_slices = detect(tmp_path / "orders.csv")

        old_id = {v: k for k, v in new_id.items()}
        assert len(expected) == 2
        assert got_slices["id"].map(old_id).tolist() == expected_slices["id"].tolist()
        pd.testing.assert_frame_equal(got, expected)


# ── hidden_trades ────────────────────────────────────────────────────


def _summary(rows):
    return pd.DataFrame(
        rows, columns=["timestamp", "best_bid_price", "best_ask_price"]
    ).assign(timestamp=lambda f: [_us(t) for t in f["timestamp"]])


def _trade_frame(rows):
    """Trades with no maker event, so each is read at its own timestamp."""
    return pd.DataFrame(rows, columns=["timestamp", "price"]).assign(
        timestamp=lambda f: [_us(t) for t in f["timestamp"]],
        maker_event_id=pd.array([pd.NA] * len(rows), dtype="Int64"),
    )


#: One event that no trade names as its maker, so each trade is read at its
#: own timestamp.  hidden_trades refuses an events frame with no rows, which
#: is what an L2 run has.
UNNAMED_EVENTS = pd.DataFrame(
    {
        "event_id": pd.Series([10**9], dtype="int64"),
        "timestamp": pd.Series([pd.Timestamp(0, tz="UTC")]),
    }
)


class TestHiddenTrades:
    def test_trade_inside_the_spread_is_flagged(self):
        summary = _summary([(0, 100, 104)])
        trades = _trade_frame([(10, 102), (20, 104), (30, 100)])

        out = hidden_trades(UNNAMED_EVENTS, trades, summary)

        assert out.index.tolist() == [0]
        assert out["best_bid_price"].tolist() == [100]
        assert out["best_ask_price"].tolist() == [104]

    def test_book_at_the_trade_instant_is_not_used(self):
        # The sweep at t=10 emptied the 104 level; the print still met it.
        summary = _summary([(0, 100, 104), (10, 100, 106)])
        trades = _trade_frame([(10, 104)])

        assert hidden_trades(UNNAMED_EVENTS, trades, summary).empty

    def test_one_sided_or_crossed_book_flags_nothing(self):
        summary = _summary([(0, 0, 104), (10, 105, 104)])
        trades = _trade_frame([(5, 50), (15, 104)])

        assert hidden_trades(UNNAMED_EVENTS, trades, summary).empty

    def test_an_unreadable_spread_is_not_replaced_by_an_older_one(self):
        # The bid side emptied at t=5.  The spread at t=0 no longer stood, so
        # a print inside it says nothing about hidden orders.
        summary = _summary([(0, 100, 104), (5, 0, 104)])
        trades = _trade_frame([(10, 102)])

        assert hidden_trades(UNNAMED_EVENTS, trades, summary).empty

    def test_trade_before_any_book_flags_nothing(self):
        summary = _summary([(10, 100, 104)])
        trades = _trade_frame([(5, 102)])

        assert hidden_trades(UNNAMED_EVENTS, trades, summary).empty

    def test_input_order_and_index_are_kept(self):
        summary = _summary([(0, 100, 104)])
        trades = _trade_frame([(30, 101), (10, 104), (20, 103)])
        trades.index = [7, 8, 7]

        out = hidden_trades(UNNAMED_EVENTS, trades, summary)

        assert out["price"].tolist() == [101, 103]
        assert out.index.tolist() == [7, 7]

    def test_best_prices_keep_depth_summarys_own_dtype(self):
        # depth_summary carries whatever price representation the caller
        # already has -- integer ticks on the canonical schema, but a caller
        # holding display-unit floats must get floats back, not a value
        # silently truncated to int64.
        summary = _summary([(0, 100, 104)]).astype(
            {"best_bid_price": "float64", "best_ask_price": "float64"}
        )
        summary.loc[0, ["best_bid_price", "best_ask_price"]] = [100.5, 104.5]
        trades = _trade_frame([(10, 102.75)])

        out = hidden_trades(UNNAMED_EVENTS, trades, summary)

        assert out["best_bid_price"].dtype == np.float64
        assert out["best_ask_price"].dtype == np.float64
        assert out["best_bid_price"].tolist() == [100.5]
        assert out["best_ask_price"].tolist() == [104.5]

    def test_book_is_read_before_the_maker_fill(self):
        # The order stream reported the fill at t=10, and the print arrived at
        # t=30.  By t=30 the maker had left the book and the spread widened.
        summary = _summary([(0, 100, 104), (10, 100, 106)])
        events = pd.DataFrame({"event_id": [5], "timestamp": [_us(10)]})
        trades = _trade_frame([(30, 104)])
        trades["maker_event_id"] = pd.array([5], dtype="Int64")

        assert hidden_trades(events, trades, summary).empty
        assert len(hidden_trades(UNNAMED_EVENTS, trades, summary)) == 1

    def test_book_is_read_just_before_the_maker_fill_at_one_instant(self):
        # At t=10 the maker was placed at 104 (event 4) and filled (event 5).
        # The book just before the fill already showed it, so the print at
        # 104 met a visible order.  The book before t=10 did not show it.
        summary = _summary([(0, 100, 106), (10, 100, 104), (10, 100, 106)])
        summary.insert(1, "event_id", [1, 4, 5])
        events = pd.DataFrame({"event_id": [4, 5], "timestamp": [_us(10), _us(10)]})
        trades = _trade_frame([(10, 104)])
        trades["maker_event_id"] = pd.array([5], dtype="Int64")

        assert hidden_trades(events, trades, summary).empty

    def test_no_trades_returns_an_empty_frame(self):
        # Pipeline tables are tz-aware UTC nanoseconds; set both sides to it,
        # since an empty frame has no values to infer the unit from.
        summary = _summary([(0, 100, 104)])
        summary["timestamp"] = summary["timestamp"].astype("datetime64[ns, UTC]")
        trades = _trade_frame([])
        trades["timestamp"] = trades["timestamp"].astype("datetime64[ns, UTC]")

        out = hidden_trades(UNNAMED_EVENTS, trades, summary)

        assert out.empty
        assert {"best_bid_price", "best_ask_price"} <= set(out.columns)

    def test_missing_column_raises(self):
        with pytest.raises(ConfigError, match="best_ask_price"):
            hidden_trades(
                UNNAMED_EVENTS,
                _trade_frame([]),
                _summary([]).drop(columns="best_ask_price"),
            )


def _write_lobster(directory: Path) -> Path:
    """One visible bid and ask, then two type-5 executions.

    The first hidden execution prints at 100.02, inside the 100.00 / 100.05
    spread.  The second prints at the ask, 100.05, so it is not flagged.
    """
    stem = "TEST_2012-06-21_34200000_57600000"
    (directory / f"{stem}_message_1.csv").write_text(
        "34200.000000000,1,11,100,1000000,1\n"
        "34200.100000000,1,22,100,1000500,-1\n"
        "34201.000000000,5,0,30,1000200,-1\n"
        "34202.000000000,5,0,40,1000500,-1\n"
        "34203.000000000,3,11,100,1000000,1\n"
    )
    (directory / f"{stem}_orderbook_1.csv").write_text(
        "9999999999,0,1000000,100\n"
        "1000500,100,1000000,100\n"
        "1000500,100,1000000,100\n"
        "1000500,100,1000000,100\n"
        "1000500,100,-9999999999,0\n"
    )
    return directory


def test_hidden_trades_matches_lobster_type_5(tmp_path):
    result = Pipeline(
        source=LobsterSource(), ctx=RunContext(trading_date="2012-06-21")
    ).run(_write_lobster(tmp_path))

    flagged = hidden_trades(result.events, result.trades, result.depth_summary)

    events = result.events
    hidden_eids = set(events.loc[events["raw_event_type"] == 5, "event_id"])
    assert len(hidden_eids) == 2
    assert flagged["maker_event_id"].isin(hidden_eids).all()
    assert flagged["price"].tolist() == [10002]
