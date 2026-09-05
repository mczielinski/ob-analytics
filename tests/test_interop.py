"""Export to backtesting engines (issue #113).

The expected shapes are written out literally here rather than imported from
``nautilus_trader`` or ``hftbacktest``: neither engine is a dependency, and a
test that asked the library what it wants would agree with the writer by
construction instead of pinning the contract.  The values were read off
nautilus-trader 1.221.0 and hftbacktest 2.4.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics import save_data
from ob_analytics.config import PipelineConfig
from ob_analytics.datasets import toy_events
from ob_analytics.interop import HFT_ARRAY_KEY

# ── The contracts, as the engines publish them ────────────────────────

#: ``hftbacktest.types.event_dtype`` (hftbacktest 2.4.4).
HFT_DTYPE = np.dtype(
    {
        "names": ["ev", "exch_ts", "local_ts", "px", "qty", "order_id", "ival", "fval"],
        "formats": ["<u8", "<i8", "<i8", "<f8", "<f8", "<u8", "<i8", "<f8"],
        "offsets": [0, 8, 16, 24, 32, 40, 48, 56],
        "itemsize": 64,
        "aligned": True,
    }
)

# ``hftbacktest`` event-type codes and the flag bits packed into ``ev``.
HFT_DEPTH_EVENT = 1
HFT_TRADE_EVENT = 2
HFT_ADD_ORDER_EVENT = 10
HFT_CANCEL_ORDER_EVENT = 11
HFT_MODIFY_ORDER_EVENT = 12
HFT_FILL_EVENT = 13
HFT_EXCH_EVENT = 1 << 31
HFT_LOCAL_EVENT = 1 << 30
HFT_BUY_EVENT = 1 << 29
HFT_SELL_EVENT = 1 << 28

#: The columns ``OrderBookDeltaDataWrangler.process`` indexes into, in its order.
WRANGLER_COLUMNS = ["action", "side", "price", "size", "order_id", "flags", "sequence"]


def _events(rows: list[tuple[str, str, int, float, int]]) -> pd.DataFrame:
    """A minimal canonical events frame: (action, direction, ticks, volume, id)."""
    ts = pd.Timestamp("2026-01-05 10:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "event_id": range(1, len(rows) + 1),
            "id": [r[4] for r in rows],
            "timestamp": ts,
            "exchange_timestamp": ts,
            "price": [r[2] for r in rows],
            "volume": [r[3] for r in rows],
            "direction": [r[1] for r in rows],
            "action": [r[0] for r in rows],
            "fill": 0.0,
            "type": "flashed-limit",
        }
    )


@pytest.fixture
def events() -> pd.DataFrame:
    return toy_events()


class TestHftbacktestWriter:
    """``save_data(fmt="hftbacktest")`` writes hftbacktest's own npz layout."""

    def test_writes_an_npz_holding_one_row_per_event_in_the_engine_dtype(
        self, tmp_path, events
    ):
        dest = tmp_path / "session.npz"

        save_data({"events": events}, dest, fmt="hftbacktest", config=PipelineConfig())

        # hftbacktest reads ``np.load(path)["data"]``.
        array = np.load(dest)["data"]
        assert array.dtype == HFT_DTYPE
        assert len(array) == len(events)

    def test_packs_the_event_type_and_side_into_the_ev_flags(self, tmp_path):
        # Hand-computed against hftbacktest's published constants:
        #   ADD(10)    | EXCH | LOCAL | BUY  = 10 + 2147483648 + 1073741824 + 536870912
        #   MODIFY(12) | EXCH | LOCAL | BUY  = same base, type 12
        #   CANCEL(11) | EXCH | LOCAL | SELL = 11 + 2147483648 + 1073741824 + 268435456
        events = _events(
            [
                ("created", "bid", 99, 2.0, 7),
                ("changed", "bid", 99, 1.0, 7),
                ("deleted", "ask", 101, 3.0, 8),
            ]
        )
        dest = tmp_path / "session.npz"

        save_data(
            {"events": events},
            dest,
            fmt="hftbacktest",
            config=PipelineConfig(tick_size=0.5),
        )

        array = np.load(dest)[HFT_ARRAY_KEY]
        assert list(array["ev"]) == [3758096394, 3758096396, 3489660939]

    def test_scales_integer_ticks_to_the_quote_currency(self, tmp_path):
        events = _events(
            [("created", "bid", 99, 2.0, 7), ("created", "ask", 101, 3.0, 8)]
        )
        dest = tmp_path / "session.npz"

        save_data(
            {"events": events},
            dest,
            fmt="hftbacktest",
            config=PipelineConfig(tick_size=0.5),
        )

        array = np.load(dest)[HFT_ARRAY_KEY]
        assert list(array["px"]) == [49.5, 50.5]
        assert list(array["qty"]) == [2.0, 3.0]
        assert list(array["order_id"]) == [7, 8]

    def test_emits_events_in_the_canonical_time_order(self, tmp_path):
        # hftbacktest's own validate_event_order requires exch_ts and local_ts
        # to be non-decreasing; the canonical events are not always stored that
        # way (the Bitstamp path is ordered by order id).
        events = _events(
            [
                ("created", "bid", 99, 2.0, 7),
                ("created", "ask", 101, 3.0, 8),
                ("deleted", "bid", 99, 2.0, 7),
            ]
        )
        stamps = [
            pd.Timestamp("2026-01-05 10:00:02", tz="UTC"),
            pd.Timestamp("2026-01-05 10:00:00", tz="UTC"),
            pd.Timestamp("2026-01-05 10:00:01", tz="UTC"),
        ]
        events["timestamp"] = stamps
        events["exchange_timestamp"] = stamps
        dest = tmp_path / "session.npz"

        save_data({"events": events}, dest, fmt="hftbacktest", config=PipelineConfig())

        array = np.load(dest)[HFT_ARRAY_KEY]
        assert list(np.diff(array["exch_ts"]) >= 0) == [True, True]
        assert list(np.diff(array["local_ts"]) >= 0) == [True, True]
        # Sorted, not dropped: the order ids follow the timestamps.
        assert list(array["order_id"]) == [8, 7, 7]

    def test_carries_both_clocks_as_nanoseconds(self, tmp_path):
        events = _events([("created", "bid", 99, 2.0, 7)])
        events["exchange_timestamp"] = pd.Timestamp("2026-01-05 10:00:00", tz="UTC")
        events["timestamp"] = pd.Timestamp("2026-01-05 10:00:00.25", tz="UTC")
        dest = tmp_path / "session.npz"

        save_data({"events": events}, dest, fmt="hftbacktest", config=PipelineConfig())

        array = np.load(dest)[HFT_ARRAY_KEY]
        assert (
            array["exch_ts"][0] == pd.Timestamp("2026-01-05 10:00:00", tz="UTC").value
        )
        assert (
            array["local_ts"][0]
            == pd.Timestamp("2026-01-05 10:00:00.25", tz="UTC").value
        )


class TestNautilusWriter:
    """``save_data(fmt="nautilus")`` writes what Nautilus' wrangler reads.

    ``OrderBookDeltaDataWrangler.process`` (nautilus-trader 1.221.0) takes a
    pandas frame with a UTC ``DatetimeIndex`` and these seven columns.
    """

    def test_writes_the_seven_wrangler_columns_on_a_utc_index(self, tmp_path):
        events = _events(
            [
                ("created", "bid", 99, 2.0, 7),
                ("changed", "bid", 99, 1.0, 7),
                ("deleted", "ask", 101, 3.0, 8),
            ]
        )
        dest = tmp_path / "deltas.parquet"

        save_data(
            {"events": events},
            dest,
            fmt="nautilus",
            config=PipelineConfig(tick_size=0.5),
        )

        frame = pd.read_parquet(dest)
        assert list(frame.columns) == WRANGLER_COLUMNS
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert str(frame.index.tz) == "UTC"

    def test_uses_the_engines_own_action_and_side_words(self, tmp_path):
        events = _events(
            [
                ("created", "bid", 99, 2.0, 7),
                ("changed", "bid", 99, 1.0, 7),
                ("deleted", "ask", 101, 3.0, 8),
            ]
        )
        dest = tmp_path / "deltas.parquet"

        save_data(
            {"events": events},
            dest,
            fmt="nautilus",
            config=PipelineConfig(tick_size=0.5),
        )

        frame = pd.read_parquet(dest)
        assert list(frame["action"]) == ["ADD", "UPDATE", "DELETE"]
        assert list(frame["side"]) == ["BUY", "BUY", "SELL"]
        # Float prices, scaled off the integer ticks by the run's tick size.
        assert list(frame["price"]) == [49.5, 49.5, 50.5]
        assert list(frame["size"]) == [2.0, 1.0, 3.0]
        assert list(frame["order_id"]) == [7, 7, 8]

    def test_gives_a_filled_orders_delete_the_size_it_last_rested_at(self, tmp_path):
        # A fully-filled order leaves a `deleted` event carrying volume 0: the
        # canonical volume on a delete is the size *removed*, and an order that
        # was filled had nothing left to cancel.  Nautilus rejects a delta whose
        # size is zero, so the delete has to name the size the order last held.
        events = _events(
            [
                ("created", "bid", 99, 2.0, 7),
                ("changed", "bid", 99, 1.5, 7),
                ("deleted", "bid", 99, 0.0, 7),
            ]
        )
        dest = tmp_path / "deltas.parquet"

        save_data(
            {"events": events},
            dest,
            fmt="nautilus",
            config=PipelineConfig(tick_size=0.5),
        )

        frame = pd.read_parquet(dest)
        assert list(frame["action"]) == ["ADD", "UPDATE", "DELETE"]
        assert list(frame["size"]) == [2.0, 1.5, 1.5]

    def test_drops_a_delta_that_never_had_a_size(self, tmp_path):
        # Nothing sensible to tell the engine about an order that never rested.
        events = _events(
            [("created", "bid", 99, 0.0, 7), ("created", "ask", 101, 3.0, 8)]
        )
        dest = tmp_path / "deltas.parquet"

        save_data(
            {"events": events},
            dest,
            fmt="nautilus",
            config=PipelineConfig(tick_size=0.5),
        )

        frame = pd.read_parquet(dest)
        assert list(frame["order_id"]) == [8]

    def test_borrows_the_size_from_the_same_order_when_orders_interleave(
        self, tmp_path
    ):
        # The size a zero-size delete borrows must come from that order's own
        # history, not from whichever event happens to sit before it.
        events = _events(
            [
                ("created", "bid", 99, 2.0, 7),
                ("created", "ask", 101, 5.0, 8),
                ("deleted", "bid", 99, 0.0, 7),
                ("deleted", "ask", 101, 0.0, 8),
            ]
        )
        dest = tmp_path / "deltas.parquet"

        save_data(
            {"events": events},
            dest,
            fmt="nautilus",
            config=PipelineConfig(tick_size=0.5),
        )

        frame = pd.read_parquet(dest)
        assert list(frame["order_id"]) == [7, 8, 7, 8]
        assert list(frame["size"]) == [2.0, 5.0, 2.0, 5.0]
