"""Tests for LOBSTER format components.

Focused on behaviors that are easy to break during performance
refactors: ``LobsterTradeReader._find_takers`` (taker identification
heuristic), the orderbook file ``LobsterWriter`` writes, and the depth read
from an orderbook file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics._utils import price_to_ticks
from ob_analytics.lobster import (
    LobsterLoader,
    LobsterTradeReader,
    LobsterWriter,
)

# The reconstruction/depth tests below build canonical events by hand; canonical
# prices are integer ticks (issue #155), so the fixtures quantise their
# quote-currency prices at the default cent grid before feeding the writer.
_TEST_TICK_SIZE = 0.01

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _events(rows: list[dict]) -> pd.DataFrame:
    """Build an events DataFrame matching the LOBSTER pipeline schema."""
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "original_number" not in df.columns:
        df["original_number"] = df["event_id"]
    return df


# ---------------------------------------------------------------------------
# LobsterTradeReader._find_takers
# ---------------------------------------------------------------------------


class TestFindTakers:
    def test_no_submissions_returns_all_na(self):
        execs = _events(
            [
                {
                    "event_id": 10,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                }
            ]
        )
        all_events = execs.copy()  # only execs; no type-1 rows
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert len(result) == 1
        assert pd.isna(result[0])

    def test_empty_execs_returns_empty_array(self):
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 1,
                }
            ]
        )
        execs = all_events.iloc[0:0]
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert len(result) == 0

    def test_marketable_bid_exec_matched_to_recent_ask_submission(self):
        """A bid execution with a prior ask submission at price <= exec price matches."""
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                {
                    "event_id": 2,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00.001",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert int(result[0]) == 1

    def test_non_marketable_ask_submission_does_not_match_bid_exec(self):
        """ask submission priced ABOVE the bid exec is non-marketable -> NA."""
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 200.0,  # > exec price 100
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                {
                    "event_id": 2,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00.001",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert pd.isna(result[0])

    def test_marketable_ask_exec_matched_to_recent_bid_submission(self):
        """An ask execution with a prior bid submission at price >= exec price matches."""
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 100.0,  # >= exec ask price 100
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 1,
                },
                {
                    "event_id": 2,
                    "id": 200,
                    "timestamp": "2024-01-01 09:30:00.001",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert int(result[0]) == 1

    def test_uses_most_recent_opposite_side_submission(self):
        """When multiple opposite-side submissions exist, the latest is picked."""
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                {
                    "event_id": 2,
                    "id": 2,
                    "timestamp": "2024-01-01 09:30:00.005",  # more recent
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                {
                    "event_id": 3,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00.010",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert int(result[0]) == 2  # most recent ask submission

    def test_handles_mixed_directions_and_preserves_order(self):
        """Multiple execs on both sides, each gets its own taker (or NA)."""
        all_events = _events(
            [
                # Ask sub before bid exec[0] (matchable)
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                # Bid sub before ask exec[1] (matchable)
                {
                    "event_id": 2,
                    "id": 2,
                    "timestamp": "2024-01-01 09:30:00.001",
                    "price": 101.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 1,
                },
                # bid exec  -> should match event_id 1
                {
                    "event_id": 3,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00.002",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
                # ask exec  -> should match event_id 2
                {
                    "event_id": 4,
                    "id": 200,
                    "timestamp": "2024-01-01 09:30:00.003",
                    "price": 101.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        # execs preserves order: bid exec first, then ask exec
        assert int(result[0]) == 1
        assert int(result[1]) == 2

    def test_sparse_matches_keep_position_alignment(self):
        """An unmatched exec sits between matched ones; result indexed by exec position."""
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                # bid exec[0]: matchable
                {
                    "event_id": 2,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00.001",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
                # bid exec[1]: no fresh ask sub at >= this price -> NA
                {
                    "event_id": 3,
                    "id": 101,
                    "timestamp": "2024-01-01 09:30:00.002",
                    "price": 50.0,  # ask sub @100 is NOT marketable for buy@50
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert int(result[0]) == 1
        assert pd.isna(result[1])


# ---------------------------------------------------------------------------
# LobsterWriter: the orderbook file
# ---------------------------------------------------------------------------


class TestWriterOrderbookFile:
    """The orderbook file holds the book after each event, in LOBSTER's
    encoding: raw integer prices, and dummy prices with zero size for an
    empty level."""

    @pytest.fixture
    def writer(self) -> LobsterWriter:
        from ob_analytics.config import PipelineConfig

        config = PipelineConfig(lot_size=1.0, volume_decimals=0)
        return LobsterWriter(config, trading_date="2024-01-01", price_divisor=10000)

    def _book_events(self, rows: list[dict]) -> pd.DataFrame:
        """Canonical events: ``volume`` is the outstanding size after the
        event (the size removed, on a ``deleted`` row) and ``fill`` the size
        executed."""
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
        df["exchange_timestamp"] = df["timestamp"]
        # Canonical events carry integer-tick prices (issue #155); the rows give
        # quote-currency dollars, so quantise them at the cent grid.
        df["price"] = price_to_ticks(df["price"].to_numpy(dtype=float), _TEST_TICK_SIZE)
        df["volume"] = df["volume"].astype(np.int64)
        fill = df["fill"] if "fill" in df.columns else pd.Series(0, index=df.index)
        df["fill"] = fill.fillna(0).astype(np.int64)
        # A row with no ``id`` is a new order, numbered by its row.
        df["event_id"] = np.arange(1, len(df) + 1)
        ids = df["id"] if "id" in df.columns else pd.Series(np.nan, index=df.index)
        df["id"] = ids.fillna(df["event_id"]).astype(np.int64)
        df["action"] = pd.Categorical(
            df["action"], categories=["created", "changed", "deleted"], ordered=True
        )
        df["direction"] = pd.Categorical(
            df["direction"], categories=["bid", "ask"], ordered=True
        )
        df["type"] = "resting-limit"
        return df

    def _orderbook(self, writer, events, tmp_path, num_levels) -> pd.DataFrame:
        _msg, ob_path = writer.write(
            {"events": events}, tmp_path, num_levels=num_levels
        )
        names = []
        for i in range(1, num_levels + 1):
            names += [
                f"ask_price_{i}",
                f"ask_size_{i}",
                f"bid_price_{i}",
                f"bid_size_{i}",
            ]
        return pd.read_csv(ob_path, header=None, names=names)

    def test_single_creation_populates_first_level(self, writer, tmp_path):
        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "bid",
                    "price": 100.0,
                    "volume": 5,
                }
            ]
        )
        ob = self._orderbook(writer, events, tmp_path, num_levels=2)
        assert len(ob) == 1
        # Bid level 1 = (price * 10000, size)
        assert ob.iloc[0]["bid_price_1"] == 1000000
        assert ob.iloc[0]["bid_size_1"] == 5
        # No asks yet -> dummy ask price and 0 size
        assert ob.iloc[0]["ask_price_1"] == 9999999999
        assert ob.iloc[0]["ask_size_1"] == 0
        assert ob.iloc[0]["bid_price_2"] == -9999999999

    def test_creates_then_deletes_clears_level(self, writer, tmp_path):
        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "ask",
                    "price": 100.0,
                    "volume": 3,
                },
                {
                    "timestamp": "2024-01-01 09:30:01",
                    "action": "deleted",
                    "direction": "ask",
                    "price": 100.0,
                    "volume": 3,
                    "id": 1,
                },
            ]
        )
        ob = self._orderbook(writer, events, tmp_path, num_levels=1)
        assert len(ob) == 2
        assert ob.iloc[0]["ask_size_1"] == 3
        assert ob.iloc[1]["ask_size_1"] == 0
        assert ob.iloc[1]["ask_price_1"] == 9999999999

    def test_an_execution_reduces_the_level(self, writer, tmp_path):
        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "bid",
                    "price": 100.0,
                    "volume": 10,
                },
                {
                    "timestamp": "2024-01-01 09:30:01",
                    "action": "changed",
                    "direction": "bid",
                    "price": 100.0,
                    "volume": 6,
                    "fill": 4,
                    "id": 1,
                },
            ]
        )
        ob = self._orderbook(writer, events, tmp_path, num_levels=1)
        assert ob.iloc[0]["bid_size_1"] == 10
        assert ob.iloc[1]["bid_size_1"] == 6

    def test_a_partial_cancel_reduces_the_level(self, writer, tmp_path):
        """A ``changed`` row with no fill and a smaller size is a partial
        cancel, whatever the source; no LOBSTER event type is needed."""
        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "ask",
                    "price": 100.0,
                    "volume": 10,
                },
                {
                    "timestamp": "2024-01-01 09:30:01",
                    "action": "changed",
                    "direction": "ask",
                    "price": 100.0,
                    "volume": 4,
                    "id": 1,
                },
            ]
        )
        ob = self._orderbook(writer, events, tmp_path, num_levels=1)
        assert ob.iloc[0]["ask_size_1"] == 10
        assert ob.iloc[1]["ask_size_1"] == 4

    def test_multiple_levels_sorted_correctly(self, writer, tmp_path):
        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "ask",
                    "price": 101.0,
                    "volume": 1,
                },
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "ask",
                    "price": 100.0,  # better ask
                    "volume": 2,
                },
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "bid",
                    "price": 99.0,
                    "volume": 3,
                },
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "bid",
                    "price": 98.0,
                    "volume": 4,
                },
            ]
        )
        ob = self._orderbook(writer, events, tmp_path, num_levels=2)
        last = ob.iloc[-1]
        # Asks ascending
        assert last["ask_price_1"] == 1000000  # 100.0
        assert last["ask_size_1"] == 2
        assert last["ask_price_2"] == 1010000  # 101.0
        assert last["ask_size_2"] == 1
        # Bids descending
        assert last["bid_price_1"] == 990000  # 99.0
        assert last["bid_size_1"] == 3
        assert last["bid_price_2"] == 980000  # 98.0
        assert last["bid_size_2"] == 4

    def test_row_per_event(self, writer, tmp_path):
        events = self._book_events(
            [
                {
                    "timestamp": f"2024-01-01 09:30:0{i}",
                    "action": "created",
                    "direction": "bid",
                    "price": 100.0 - i,
                    "volume": 1,
                }
                for i in range(5)
            ]
        )
        ob = self._orderbook(writer, events, tmp_path, num_levels=3)
        assert len(ob) == 5
        # Only the three best bids fit in the file.
        assert list(ob.iloc[-1][["bid_price_1", "bid_price_2", "bid_price_3"]]) == [
            1000000,
            990000,
            980000,
        ]

    def test_a_depth_table_naming_other_events_is_refused(self, writer, tmp_path):
        from ob_analytics.depth import price_level_volume
        from ob_analytics.exceptions import ConfigError

        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "bid",
                    "price": 100.0,
                    "volume": 5,
                }
            ]
        )
        depth = price_level_volume(events).assign(event_id=99)
        with pytest.raises(ConfigError, match="depth"):
            writer.write({"events": events, "depth": depth}, tmp_path)

    def _one_bid(self) -> pd.DataFrame:
        return self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "bid",
                    "price": 100.0,
                    "volume": 5,
                }
            ]
        )

    def test_events_with_no_type_and_no_depth_are_refused(self, writer, tmp_path):
        from ob_analytics.exceptions import ConfigError

        events = self._one_bid().drop(columns="type")
        with pytest.raises(ConfigError, match="set_order_types"):
            writer.write({"events": events}, tmp_path)

    def test_a_run_with_no_events_is_refused(self, writer, tmp_path):
        from ob_analytics.exceptions import ConfigError

        events = self._one_bid().iloc[0:0]
        with pytest.raises(ConfigError, match="no events"):
            writer.write({"events": events}, tmp_path)

    def test_a_depth_table_in_display_units_is_refused(self, writer, tmp_path):
        from ob_analytics.depth import price_level_volume
        from ob_analytics.exceptions import ConfigError

        events = self._one_bid()
        depth = price_level_volume(events)
        depth = depth.assign(price=depth["price"] * 0.01, volume=depth["volume"] * 0.5)
        with pytest.raises(ConfigError, match="display units"):
            writer.write({"events": events, "depth": depth}, tmp_path)

    def test_a_depth_table_with_float_prices_is_refused(self, writer, tmp_path):
        """A float price is a display price, even when it is a whole number:
        100.0 dollars is 10000 ticks at a cent tick, not 100."""
        from ob_analytics.depth import price_level_volume
        from ob_analytics.exceptions import ConfigError

        events = self._one_bid()
        depth = price_level_volume(events)
        depth = depth.assign(price=depth["price"] * 0.01)
        with pytest.raises(ConfigError, match="price"):
            self._orderbook_from(writer, events, depth, tmp_path)

    def test_a_writer_with_no_config_uses_lobster_defaults(self, tmp_path):
        """A cent tick, prices in ten-thousandths of a dollar, whole shares."""
        writer = LobsterWriter(trading_date="2024-01-01")
        _msg, ob_path = writer.write(
            {"events": self._one_bid()}, tmp_path, num_levels=1
        )
        assert ob_path.read_text() == "9999999999,0,1000000,5\n"

    def test_prices_are_exact_on_a_half_cent_tick(self, tmp_path):
        """Adjacent ticks stay apart even when price_decimals is too coarse."""
        from ob_analytics.config import PipelineConfig

        config = PipelineConfig(
            tick_size=0.005, price_decimals=2, price_divisor=10_000, lot_size=1.0
        )
        writer = LobsterWriter(config, trading_date="2024-01-01")
        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "bid",
                    "price": 0.0,
                    "volume": 5,
                },
                {
                    "timestamp": "2024-01-01 09:30:01",
                    "action": "created",
                    "direction": "bid",
                    "price": 0.0,
                    "volume": 5,
                },
            ]
        )
        events["price"] = np.array([20001, 20002], dtype=np.int64)
        ob = self._orderbook(writer, events, tmp_path, num_levels=2)
        last = ob.iloc[-1]
        assert (last["bid_price_1"], last["bid_price_2"]) == (1000100, 1000050)

    def _orderbook_from(self, writer, events, depth, tmp_path) -> pd.DataFrame:
        _msg, ob_path = writer.write(
            {"events": events, "depth": depth}, tmp_path, num_levels=1
        )
        return pd.read_csv(
            ob_path,
            header=None,
            names=["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"],
        )

    def test_a_price_divisor_too_coarse_for_the_tick_is_refused(self, tmp_path):
        """With a cent tick, a divisor of 1 would write 100.01 and 100.02 as
        the same raw price 100."""
        from ob_analytics.config import PipelineConfig
        from ob_analytics.exceptions import ConfigError

        writer = LobsterWriter(
            PipelineConfig(tick_size=0.01, price_divisor=1, lot_size=1.0),
            trading_date="2024-01-01",
        )
        with pytest.raises(ConfigError, match="price_divisor"):
            writer.write({"events": self._one_bid()}, tmp_path)

    def test_events_with_repeated_index_labels_are_written(self, tmp_path):
        """Events joined with ``pd.concat`` keep repeated index labels."""
        from ob_analytics.analytics import set_order_types
        from ob_analytics.config import PipelineConfig
        from ob_analytics.datasets import toy_events, toy_trades

        events = set_order_types(toy_events(), toy_trades())
        events.index = [0] * len(events)
        writer = LobsterWriter(
            PipelineConfig(tick_size=1.0, lot_size=1.0, price_divisor=10_000),
            trading_date="2026-01-05",
        )
        ob = self._orderbook(writer, events, tmp_path, num_levels=1)
        assert len(ob) == len(events)
        # After event 10: bids 4 @ 99, asks 2 @ 101.
        assert ob.iloc[9][["bid_price_1", "bid_size_1"]].tolist() == [990000, 4]
        assert ob.iloc[9][["ask_price_1", "ask_size_1"]].tolist() == [1010000, 2]

    def test_fewer_than_one_level_is_refused(self, writer, tmp_path):
        from ob_analytics.exceptions import ConfigError

        with pytest.raises(ConfigError, match="num_levels"):
            writer.write({"events": self._one_bid()}, tmp_path, num_levels=0)

    def test_events_with_float_sizes_are_refused(self, writer, tmp_path):
        """Event sizes are always integer lots; a float size is a display size,
        even when it is a whole number."""
        from ob_analytics.depth import price_level_volume
        from ob_analytics.exceptions import ConfigError

        events = self._one_bid()
        depth = price_level_volume(events)
        shown = events.assign(volume=events["volume"].astype(float))
        with pytest.raises(ConfigError, match="display units"):
            writer.write({"events": shown, "depth": depth}, tmp_path)

    def test_a_coarse_price_divisor_is_refused_for_the_message_file(self):
        from ob_analytics.config import PipelineConfig
        from ob_analytics.exceptions import ConfigError

        writer = LobsterWriter(
            PipelineConfig(tick_size=0.01, price_divisor=1, lot_size=1.0),
            trading_date="2024-01-01",
        )
        with pytest.raises(ConfigError, match="price_divisor"):
            writer._events_to_message(self._one_bid())

    def test_events_in_display_units_are_refused(self, writer, tmp_path):
        from ob_analytics.depth import price_level_volume
        from ob_analytics.exceptions import ConfigError

        events = self._one_bid()
        depth = price_level_volume(events)
        shown = events.assign(price=events["price"] * 0.01)
        with pytest.raises(ConfigError, match="display units"):
            writer.write({"events": shown, "depth": depth}, tmp_path)

    def test_events_missing_a_column_are_refused(self, writer, tmp_path):
        from ob_analytics.depth import price_level_volume
        from ob_analytics.exceptions import ConfigError

        events = self._one_bid()
        depth = price_level_volume(events)
        with pytest.raises(ConfigError, match="timestamp"):
            writer.write(
                {"events": events.drop(columns="timestamp"), "depth": depth}, tmp_path
            )

    def test_repeated_event_ids_are_refused(self, writer, tmp_path):
        from ob_analytics.exceptions import ConfigError

        events = pd.concat([self._one_bid(), self._one_bid()], ignore_index=True)
        with pytest.raises(ConfigError, match="event_id"):
            writer.write({"events": events}, tmp_path)


# ---------------------------------------------------------------------------
# original_number convention (cross-format parity)
# ---------------------------------------------------------------------------


class TestOriginalNumberConvention:
    """``original_number`` tracks the 1-based source-file row, not ``event_id``.

    The Bitstamp and LOBSTER loaders historically diverged: Bitstamp carried
    the original CSV row through its ``[id, volume, action, timestamp]`` sort,
    while LOBSTER simply aliased ``original_number = event_id`` and discarded
    the source position of filtered rows (halts / cross trades). They now share
    one convention so trade provenance (``maker_og`` / ``taker_og``) is
    comparable across formats. This pins the LOBSTER side of that contract.
    """

    def test_original_number_tracks_source_rows(self, tmp_path):
        # LOBSTER message schema: time,event_type,id,volume,price,direction.
        # Row 2 (halt, type 7) and row 5 (cross, type 6) are filtered out, so
        # the surviving events keep their 1-based source-file row in
        # original_number while event_id is renumbered contiguously.
        msg = tmp_path / "AAPL_2024-01-01_34200000_57600000_message_1.csv"
        msg.write_text(
            "34200.0,1,1,100,1000000,1\n"  # row 1 -> kept (created bid)
            "34200.1,7,0,0,0,1\n"  # row 2 -> halt (dropped)
            "34200.2,1,2,100,1010000,-1\n"  # row 3 -> kept (created ask)
            "34200.3,4,1,50,1000000,1\n"  # row 4 -> kept (execution)
            "34200.4,6,0,0,0,1\n"  # row 5 -> cross (dropped)
            "34200.5,3,2,100,1010000,-1\n"  # row 6 -> kept (deleted ask)
        )
        events = LobsterLoader(trading_date="2024-01-01").load(msg)

        assert list(events["event_id"]) == [1, 2, 3, 4]
        assert list(events["original_number"]) == [1, 3, 4, 6]
        # The two columns are distinct concepts once any source row is filtered.
        assert not events["original_number"].equals(events["event_id"])
        assert events["original_number"].is_unique


class TestLobsterDepthFromOrderbook:
    """Oracle test: the vectorized per-side diff must match a dict-diff
    reference (the algorithm it replaced) modulo within-event row order,
    which is now deterministic (asks then bids, ascending price)."""

    def test_matches_dict_diff_reference(self, tmp_path):
        from ob_analytics.config import PipelineConfig
        from ob_analytics.lobster import (
            _DUMMY_ASK_PRICE,
            _DUMMY_BID_PRICE,
            lobster_depth_from_orderbook,
        )

        rng = np.random.default_rng(20260611)
        n, levels, divisor, dec = 60, 3, 10_000, 2

        # Random walk of a tiny book: occasional empty levels (dummy price),
        # occasional zero sizes, occasional unchanged consecutive rows.
        rows = []
        for i in range(n):
            row = []
            base = 5_000_000 + int(rng.integers(-3, 4)) * 100
            for lv in range(levels):
                ap = base + (lv + 1) * 100
                bp = base - (lv + 1) * 100
                av = int(rng.integers(0, 4)) * 100
                bv = int(rng.integers(0, 4)) * 100
                if rng.random() < 0.15:
                    ap, av = _DUMMY_ASK_PRICE, 0
                if rng.random() < 0.15:
                    bp, bv = _DUMMY_BID_PRICE, 0
                row.extend([ap, av, bp, bv])
            rows.append(row)
            if rng.random() < 0.2 and i + 1 < n:
                rows.append(list(row))  # unchanged consecutive row
        ob = pd.DataFrame(rows[:n])
        ob_path = tmp_path / "TEST_orderbook_3.csv"
        ob.to_csv(ob_path, index=False, header=False)

        events = pd.DataFrame(
            {
                "event_id": np.arange(1, n + 1),
                "original_number": np.arange(1, n + 1),
                "timestamp": pd.date_range("2012-06-21 09:30", periods=n, freq="s"),
                "raw_event_type": np.ones(n, dtype=int),
            }
        )

        # LOBSTER's own lot (one share), so the reference's share counts are
        # the depth's lots.
        cfg = PipelineConfig(
            price_decimals=dec, price_divisor=divisor, lot_size=1.0, volume_decimals=0
        )
        depth, _summary = lobster_depth_from_orderbook(events, ob_path, cfg)

        # Dict-diff reference (previous implementation, verbatim semantics).
        arr = ob.to_numpy()
        ref_rows, prev = [], {}
        for i in range(n):
            curr: dict[tuple[str, float], float] = {}
            for j in range(levels):
                b = j * 4
                ap, av, bp, bv = arr[i, b], arr[i, b + 1], arr[i, b + 2], arr[i, b + 3]
                if ap != _DUMMY_ASK_PRICE and av > 0:
                    # Canonical depth prices are integer ticks (issue #155).
                    pr = int(price_to_ticks(np.array([ap / divisor]), cfg.tick_size)[0])
                    curr[("ask", pr)] = curr.get(("ask", pr), 0) + av
                if bp != _DUMMY_BID_PRICE and bv > 0:
                    pr = int(price_to_ticks(np.array([bp / divisor]), cfg.tick_size)[0])
                    curr[("bid", pr)] = curr.get(("bid", pr), 0) + bv
            for key in set(prev) | set(curr):
                pv, cv = prev.get(key, 0.0), curr.get(key, 0.0)
                if pv != cv:
                    ref_rows.append(
                        {
                            "event_id": events["event_id"].iloc[i],
                            "timestamp": events["timestamp"].iloc[i],
                            "price": key[1],
                            "volume": float(cv),
                            "direction": key[0],
                        }
                    )
            prev = curr
        ref = pd.DataFrame(ref_rows)

        def canon(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out["direction"] = out["direction"].astype(str)
            return out.sort_values(
                ["event_id", "direction", "price"], kind="stable"
            ).reset_index(drop=True)[
                ["event_id", "timestamp", "price", "volume", "direction"]
            ]

        pd.testing.assert_frame_equal(canon(ref), canon(depth), check_dtype=False)

    def test_within_event_order_is_asks_then_bids_ascending(self, tmp_path):
        from ob_analytics.config import PipelineConfig
        from ob_analytics.lobster import lobster_depth_from_orderbook

        # One row: two asks + two bids -> first event emits everything.
        ob = pd.DataFrame([[5001000, 100, 4999000, 200, 5002000, 300, 4998000, 400]])
        ob_path = tmp_path / "TEST_orderbook_2.csv"
        ob.to_csv(ob_path, index=False, header=False)
        events = pd.DataFrame(
            {
                "event_id": [1],
                "original_number": [1],
                "timestamp": pd.to_datetime(["2012-06-21 09:30"]),
                "raw_event_type": [1],
            }
        )
        cfg = PipelineConfig(price_decimals=2, price_divisor=10_000)
        depth, _ = lobster_depth_from_orderbook(events, ob_path, cfg)
        assert list(depth["direction"].astype(str)) == ["ask", "ask", "bid", "bid"]
        # Integer ticks at the cent grid (issue #155): 500.1 -> 50010, etc.
        assert list(depth["price"]) == [50010, 50020, 49980, 49990]


# ---------------------------------------------------------------------------
# Depth from the orderbook file: rows keyed by message row
# ---------------------------------------------------------------------------

_STEM = "TEST_2024-01-02_34200000_57600000"


def _write_pair(directory, messages: str, orderbook: str):
    (directory / f"{_STEM}_message_1.csv").write_text(messages)
    (directory / f"{_STEM}_orderbook_1.csv").write_text(orderbook)
    return directory


def _run_lobster(directory, config=None):
    from ob_analytics.lobster import LobsterSource
    from ob_analytics.pipeline import Pipeline
    from ob_analytics.protocols import RunContext

    return Pipeline(
        source=LobsterSource(),
        config=config,
        ctx=RunContext(trading_date="2024-01-02"),
    ).run(directory)


class TestOrderbookRowsFollowMessageRows:
    """A type 6 (cross trade) or type 7 (halt) message has its own row in the
    orderbook file but no row in the events, so the two files are paired by
    the event's message row, not by position."""

    # Bid 100 @ 100.00, a cross trade, ask 200 @ 101.00, a halt, delete the bid.
    MESSAGES = (
        "34200.0,1,1,100,1000000,1\n"
        "34201.0,6,0,50,1005000,-1\n"
        "34202.0,1,2,200,1010000,-1\n"
        "34202.5,7,0,0,-1,-1\n"
        "34203.0,3,1,100,1000000,1\n"
    )
    ORDERBOOK = (
        "9999999999,0,1000000,100\n"
        "9999999999,0,1000000,100\n"
        "1010000,200,1000000,100\n"
        "1010000,200,1000000,100\n"
        "1010000,200,-9999999999,0\n"
    )

    def test_each_book_state_is_stamped_with_its_own_event(self, tmp_path):
        result = _run_lobster(_write_pair(tmp_path, self.MESSAGES, self.ORDERBOOK))

        depth = result.depth
        assert list(depth["event_id"]) == [1, 2, 3]
        assert list(depth["direction"].astype(str)) == ["bid", "ask", "bid"]
        assert list(depth["price"]) == [10000, 10100, 10000]
        assert list(depth["volume"]) == [100, 200, 0]
        stamps = result.events.set_index("event_id")["timestamp"]
        assert (
            depth["timestamp"].to_numpy() == stamps[depth["event_id"]].to_numpy()
        ).all()

    def test_the_last_book_state_is_kept(self, tmp_path):
        result = _run_lobster(_write_pair(tmp_path, self.MESSAGES, self.ORDERBOOK))

        last = result.depth_summary.iloc[-1]
        assert last["best_bid_vol"] == 0
        assert last["best_ask_price"] == 10100
        assert last["best_ask_vol"] == 200

    def test_an_orderbook_file_shorter_than_the_messages_is_refused(self, tmp_path):
        from ob_analytics.exceptions import ConfigError

        short = "".join(self.ORDERBOOK.splitlines(keepends=True)[:4])
        with pytest.raises(ConfigError, match="orderbook"):
            _run_lobster(_write_pair(tmp_path, self.MESSAGES, short))

    def test_an_orderbook_file_longer_than_the_messages_is_refused(self, tmp_path):
        """A longer file is the orderbook of another message file."""
        from ob_analytics.exceptions import ConfigError

        longer = self.ORDERBOOK + self.ORDERBOOK.splitlines(keepends=True)[-1]
        with pytest.raises(ConfigError, match="orderbook"):
            _run_lobster(_write_pair(tmp_path, self.MESSAGES, longer))

    def test_events_in_another_order_give_the_same_depth(self, tmp_path):
        from ob_analytics.config import PipelineConfig
        from ob_analytics.lobster import LobsterLoader, lobster_depth_from_orderbook

        _write_pair(tmp_path, self.MESSAGES, self.ORDERBOOK)
        loader = LobsterLoader(trading_date="2024-01-02")
        events = loader.load(tmp_path)
        assert loader.orderbook_path is not None
        config = PipelineConfig(price_divisor=10_000, lot_size=1.0)

        depth, summary = lobster_depth_from_orderbook(
            events, loader.orderbook_path, config
        )
        reversed_depth, reversed_summary = lobster_depth_from_orderbook(
            events.iloc[::-1], loader.orderbook_path, config
        )

        pd.testing.assert_frame_equal(reversed_depth, depth)
        pd.testing.assert_frame_equal(reversed_summary, summary)

    def test_a_message_row_number_below_one_is_refused(self, tmp_path):
        from ob_analytics.config import PipelineConfig
        from ob_analytics.exceptions import ConfigError
        from ob_analytics.lobster import LobsterLoader, lobster_depth_from_orderbook

        _write_pair(tmp_path, self.MESSAGES, self.ORDERBOOK)
        loader = LobsterLoader(trading_date="2024-01-02")
        events = loader.load(tmp_path)
        assert loader.orderbook_path is not None
        # Counted from 0, so the first event would read the last book row.
        events["original_number"] -= 1
        with pytest.raises(ConfigError, match="original_number"):
            lobster_depth_from_orderbook(
                events, loader.orderbook_path, PipelineConfig(lot_size=1.0)
            )

    def test_a_missing_message_row_number_is_refused(self, tmp_path):
        from ob_analytics.config import PipelineConfig
        from ob_analytics.exceptions import ConfigError
        from ob_analytics.lobster import LobsterLoader, lobster_depth_from_orderbook

        _write_pair(tmp_path, self.MESSAGES, self.ORDERBOOK)
        loader = LobsterLoader(trading_date="2024-01-02")
        events = loader.load(tmp_path)
        assert loader.orderbook_path is not None
        events["original_number"] = events["original_number"].astype("Int64")
        events.loc[[0, 1], "original_number"] = pd.NA
        with pytest.raises(ConfigError, match="missing"):
            lobster_depth_from_orderbook(
                events, loader.orderbook_path, PipelineConfig(lot_size=1.0)
            )

    def test_a_fractional_message_row_number_is_refused(self, tmp_path):
        from ob_analytics.config import PipelineConfig
        from ob_analytics.exceptions import ConfigError
        from ob_analytics.lobster import LobsterLoader, lobster_depth_from_orderbook

        _write_pair(tmp_path, self.MESSAGES, self.ORDERBOOK)
        loader = LobsterLoader(trading_date="2024-01-02")
        events = loader.load(tmp_path)
        assert loader.orderbook_path is not None
        events["original_number"] = events["original_number"] + 0.5
        with pytest.raises(ConfigError, match="original_number"):
            lobster_depth_from_orderbook(
                events, loader.orderbook_path, PipelineConfig(lot_size=1.0)
            )


class TestOrderbookSizesAreLots:
    """Sizes read from the orderbook file are integer lots, like the events."""

    MESSAGES = TestOrderbookRowsFollowMessageRows.MESSAGES
    ORDERBOOK = TestOrderbookRowsFollowMessageRows.ORDERBOOK

    def test_depth_and_summary_sizes_are_int64(self, tmp_path):
        result = _run_lobster(_write_pair(tmp_path, self.MESSAGES, self.ORDERBOOK))

        assert result.depth["volume"].dtype == np.int64
        size_columns = [c for c in result.depth_summary.columns if "vol" in c]
        assert size_columns
        assert all(result.depth_summary[c].dtype == np.int64 for c in size_columns)

    def test_sizes_honour_lot_size(self, tmp_path):
        from ob_analytics.config import PipelineConfig

        # A lot of 100 shares: the 100-share bid is 1 lot, the 200-share ask 2.
        config = PipelineConfig(
            tick_size=0.01,
            price_decimals=2,
            price_divisor=10_000,
            lot_size=100.0,
            volume_decimals=0,
        )
        result = _run_lobster(
            _write_pair(tmp_path, self.MESSAGES, self.ORDERBOOK), config
        )

        assert list(result.depth["volume"]) == [1, 2, 0]
        created = result.events[result.events["action"] == "created"]
        assert list(created["volume"]) == [1, 2]
        assert result.depth_summary["best_ask_vol"].iloc[-1] == 2


# ---------------------------------------------------------------------------
# LobsterWriter: the orderbook file is the run's own depth
# ---------------------------------------------------------------------------


def _book_after_each_event(summary: pd.DataFrame, event_ids) -> pd.DataFrame:
    """The depth summary after each of *event_ids*, in that order.

    An event that changed no price level keeps the book of the event before
    it; before the first change the book is empty, which the summary writes
    as zeros.
    """
    columns = [c for c in summary.columns if c not in ("timestamp", "event_id")]
    last = summary.groupby("event_id", sort=False)[columns].last()
    book = last.reindex(list(event_ids)).ffill().fillna(0).astype(np.int64)
    return book.reset_index(drop=True)


def _write_and_read_back(result, config, tmp_path, trading_date):
    """Write *result* as LOBSTER files and run the pipeline on them."""
    from ob_analytics.data import save_data
    from ob_analytics.lobster import LobsterSource
    from ob_analytics.pipeline import Pipeline
    from ob_analytics.protocols import RunContext

    ctx = RunContext(trading_date=trading_date, session_tz="UTC")
    # Enough levels to hold every price the run ever rests at, so the file
    # holds the whole book and not only its top.
    num_levels = int(result.depth["price"].nunique())
    save_data(
        {"events": result.events, "depth": result.depth},
        tmp_path,
        fmt="lobster",
        config=config,
        ctx=ctx,
        num_levels=num_levels,
    )
    return Pipeline(source=LobsterSource(), config=config, ctx=ctx).run(tmp_path)


def _assert_read_back_gives_the_runs_depth(result, read_back) -> None:
    # The files hold every event in time order, so the read-back run's events
    # are the run's events in time order, renumbered 1..N.
    from ob_analytics.schemas import time_order_keys

    assert len(read_back.events) == len(result.events)
    in_time_order = result.events.sort_values(
        time_order_keys(result.events), kind="stable"
    )
    want = _book_after_each_event(result.depth_summary, in_time_order["event_id"])
    got = _book_after_each_event(read_back.depth_summary, read_back.events["event_id"])
    pd.testing.assert_frame_equal(got, want)


class TestLobsterWriterWritesTheRunsDepth:
    """Reading the written files back gives the depth of the run that was
    written, whatever source the run came from."""

    def test_toy_run(self, tmp_path):
        from ob_analytics.analytics import set_order_types
        from ob_analytics.config import PipelineConfig
        from ob_analytics.datasets import LOT_SIZE, TICK_SIZE, toy_events, toy_trades
        from ob_analytics.depth import depth_metrics, price_level_volume
        from ob_analytics.pipeline import PipelineResult

        config = PipelineConfig(
            tick_size=TICK_SIZE,
            price_decimals=0,
            lot_size=LOT_SIZE,
            volume_decimals=0,
            price_divisor=10_000,
        )
        # As built: ``raw_event_type`` is all missing, as on every frame that
        # was not read from LOBSTER.
        events = set_order_types(toy_events(), toy_trades())
        depth = price_level_volume(events)
        result = PipelineResult(
            events=events,
            trades=toy_trades(),
            depth=depth,
            depth_summary=depth_metrics(depth),
            config=config,
        )

        read_back = _write_and_read_back(result, config, tmp_path, "2026-01-05")

        _assert_read_back_gives_the_runs_depth(result, read_back)
        # The toy book after event 10 (Gus's ask at 103): bids 4 @ 99 and
        # 4 @ 98, asks 2 @ 101, 2 @ 102 and 2 @ 103.
        after_10 = read_back.depth_summary[read_back.depth_summary["event_id"] <= 10]
        last = after_10.iloc[-1]
        assert (last["best_bid_price"], last["best_bid_vol"]) == (99, 4)
        assert (last["best_ask_price"], last["best_ask_vol"]) == (101, 2)

    def test_events_out_of_time_order_are_written_in_time_order(self, tmp_path):
        """A LOBSTER file is in time order, and the depth summary replays the
        depth in time order, so rows given out of time order are written in
        it."""
        from ob_analytics.analytics import set_order_types
        from ob_analytics.config import PipelineConfig
        from ob_analytics.datasets import LOT_SIZE, TICK_SIZE, toy_events, toy_trades
        from ob_analytics.depth import depth_metrics, price_level_volume
        from ob_analytics.pipeline import PipelineResult

        config = PipelineConfig(
            tick_size=TICK_SIZE,
            price_decimals=0,
            lot_size=LOT_SIZE,
            volume_decimals=0,
            price_divisor=10_000,
        )
        events = set_order_types(toy_events(), toy_trades())
        depth = price_level_volume(events)
        result = PipelineResult(
            events=events.iloc[::-1].reset_index(drop=True),
            trades=toy_trades(),
            depth=depth,
            depth_summary=depth_metrics(depth),
            config=config,
        )

        read_back = _write_and_read_back(result, config, tmp_path, "2026-01-05")

        _assert_read_back_gives_the_runs_depth(result, read_back)
        assert read_back.events["timestamp"].is_monotonic_increasing

    def test_synthetic_run(self, tmp_path):
        from ob_analytics.config import PipelineConfig
        from ob_analytics.depth import depth_metrics, price_level_volume
        from ob_analytics.pipeline import PipelineResult
        from ob_analytics.synth import generate_session

        session = generate_session(seed=3, duration=20)
        config = PipelineConfig(
            tick_size=session.config.tick_size,
            price_decimals=2,
            lot_size=session.config.lot_size,
            volume_decimals=8,
            price_divisor=10_000,
        )
        depth = price_level_volume(session.events)
        result = PipelineResult(
            events=session.events,
            trades=session.trades,
            depth=depth,
            depth_summary=depth_metrics(depth),
            config=config,
        )

        trading_date = session.events["timestamp"].iloc[0].strftime("%Y-%m-%d")
        read_back = _write_and_read_back(result, config, tmp_path, trading_date)

        _assert_read_back_gives_the_runs_depth(result, read_back)

    def test_bitstamp_run(self, tmp_path, tiny_bitstamp_orders_csv):
        from ob_analytics.config import PipelineConfig
        from ob_analytics.pipeline import Pipeline

        config = PipelineConfig(
            tick_size=0.01,
            price_decimals=2,
            lot_size=1e-8,
            volume_decimals=8,
            price_divisor=10_000,
        )
        result = Pipeline(config=config).run(tiny_bitstamp_orders_csv)

        trading_date = result.events["timestamp"].iloc[0].strftime("%Y-%m-%d")
        read_back = _write_and_read_back(result, config, tmp_path, trading_date)

        _assert_read_back_gives_the_runs_depth(result, read_back)

    def test_a_lobster_run_is_written_back_as_read(self, tmp_path):
        """The orderbook file of a LOBSTER run, written at its own number of
        levels, is the file the run was read from."""
        from ob_analytics.lobster import LobsterSource
        from ob_analytics.pipeline import Pipeline
        from ob_analytics.protocols import RunContext

        source_dir = tmp_path / "in"
        source_dir.mkdir()
        messages = (
            "34200.0,1,1,100,1000000,1\n"
            "34201.0,1,2,200,1010000,-1\n"
            "34202.0,1,3,300,999900,1\n"
            "34203.0,4,1,40,1000000,1\n"
            "34204.0,2,2,50,1010000,-1\n"
            "34205.0,3,3,300,999900,1\n"
        )
        orderbook = (
            "9999999999,0,1000000,100,9999999999,0,-9999999999,0\n"
            "1010000,200,1000000,100,9999999999,0,-9999999999,0\n"
            "1010000,200,1000000,100,9999999999,0,999900,300\n"
            "1010000,200,1000000,60,9999999999,0,999900,300\n"
            "1010000,150,1000000,60,9999999999,0,999900,300\n"
            "1010000,150,1000000,60,9999999999,0,-9999999999,0\n"
        )
        (source_dir / f"{_STEM}_message_2.csv").write_text(messages)
        (source_dir / f"{_STEM}_orderbook_2.csv").write_text(orderbook)
        ctx = RunContext(trading_date="2024-01-02")
        result = Pipeline(source=LobsterSource(), ctx=ctx).run(source_dir)

        writer = LobsterWriter(result.config, trading_date="2024-01-02")
        msg_path, ob_path = writer.write(
            {"events": result.events, "depth": result.depth},
            tmp_path / "out",
            num_levels=2,
        )

        assert ob_path.read_text() == orderbook
        assert len(msg_path.read_text().splitlines()) == 6
