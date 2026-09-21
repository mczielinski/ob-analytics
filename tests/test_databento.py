"""Tests for the Databento MBO source (issue #100) -- no network, no API key.

Most tests hand the loader a pandas frame of MBO records built by
:func:`mbo_frame`, which is the shape ``DBNStore.to_df(price_type="fixed")``
returns.  That path needs no ``databento`` install, so the action mapping is
covered everywhere.  The tests that read or write a real ``.dbn`` file build
one in memory with ``databento_dbn`` and skip without the extra.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from ob_analytics import Pipeline, PipelineConfig
from ob_analytics.databento import (
    DBN_PRICE_DIVISOR,
    F_MBP,
    F_TOB,
    MAX_ORDER_ID,
    UNDEF_PRICE,
    DatabentoLoader,
    DatabentoSettings,
    DatabentoSource,
    DatabentoTradeReader,
    DatabentoWriter,
)
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import FeedType, Level, OfflineSource, RunContext
from ob_analytics.schemas import validate_events_df, validate_trades_df
from ob_analytics.sources import get_source

_DATABENTO_INSTALLED = importlib.util.find_spec("databento") is not None

requires_databento = pytest.mark.skipif(
    not _DATABENTO_INSTALLED, reason="databento is an optional extra"
)

T0 = 1_700_000_000_000_000_000
INSTRUMENT_ID = 1108
PUBLISHER_ID = 2


def px(decimal: float) -> int:
    """Return *decimal* as Databento's fixed-point integer price."""
    return round(decimal * DBN_PRICE_DIVISOR)


def mbo_frame(records: list[tuple], **overrides) -> pd.DataFrame:
    """Build the MBO record frame from ``(order_id, action, side, price, size)``.

    One record per millisecond, in the order given.  ``overrides`` sets a
    whole column (``instrument_id``, ``publisher_id``, ``flags``, ``symbol``)
    for every row.
    """
    n = len(records)
    frame = pd.DataFrame(
        {
            "ts_recv": pd.to_datetime(
                [T0 + i * 1_000_000 + 500 for i in range(n)], utc=True
            ),
            "ts_event": pd.to_datetime(
                [T0 + i * 1_000_000 for i in range(n)], utc=True
            ),
            "action": [r[1] for r in records],
            "side": [r[2] for r in records],
            "price": [r[3] for r in records],
            "size": [r[4] for r in records],
            "order_id": [r[0] for r in records],
            "flags": np.zeros(n, dtype="uint8"),
            "sequence": np.arange(1, n + 1, dtype="uint32"),
            "instrument_id": np.full(n, INSTRUMENT_ID, dtype="uint32"),
            "publisher_id": np.full(n, PUBLISHER_ID, dtype="uint16"),
        }
    )
    for column, value in overrides.items():
        frame[column] = value
    return frame


#: An order added, partly cancelled, twice filled, then taken off the book.
#: Mirrors the life cycle in Databento's own "types of order book events" page.
LIFECYCLE = [
    (1, "A", "B", px(100.25), 100),  # bid 100 @ 100.25
    (2, "A", "A", px(100.35), 50),  # ask  50 @ 100.35
    (1, "C", "B", px(100.25), 40),  # partial cancel -> 60 left
    (0, "T", "B", px(100.35), 20),  # a buyer aggressed
    (2, "F", "A", px(100.35), 20),  # ... against the resting ask
    (2, "M", "A", px(100.35), 30),  # ... leaving 30
    (0, "T", "B", px(100.35), 30),
    (2, "F", "A", px(100.35), 30),
    (2, "C", "A", px(100.35), 30),  # fully executed -> off the book
]


@contextmanager
def warnings_logged() -> Iterator[list[str]]:
    """Collect the messages logged at WARNING or above inside the block."""
    messages: list[str] = []
    # The package disables its own logger on import, as a library should.
    logger.enable("ob_analytics")
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)
        logger.disable("ob_analytics")


def load(source, config=None, **loader_kwargs):
    """Load *source* and return ``(loader, events)``.

    A list of record tuples is turned into a frame first; anything else — a
    frame, a path, a ``DBNStore`` — goes to the loader as it is.
    """
    if isinstance(source, list):
        source = mbo_frame(source)
    loader = DatabentoLoader(config or _config(), **loader_kwargs)
    return loader, loader.load(source)


def _config(**overrides) -> PipelineConfig:
    """The config the pipeline would build for a Databento run."""
    defaults = DatabentoSource().config_defaults()
    return PipelineConfig(**{**defaults, **overrides})


# ── The action mapping ────────────────────────────────────────────────


class TestActionMapping:
    def test_add_cancel_modify_become_canonical_actions(self):
        _, events = load(LIFECYCLE)

        assert list(events["action"]) == [
            "created",  # 1 added
            "created",  # 2 added
            "changed",  # 1 partly cancelled
            "changed",  # 2 modified down
            "deleted",  # 2 cancelled away
        ]
        assert list(events["id"]) == [1, 2, 1, 2, 2]
        assert list(events["raw_event_type"]) == ["A", "A", "C", "M", "C"]

    def test_volume_is_the_outstanding_size_after_each_event(self):
        _, events = load(LIFECYCLE)

        # A/M state the new total; a partial cancel leaves 100 - 40 = 60; the
        # final cancel reports the size it removed.
        assert list(events["volume"]) == [100, 50, 60, 30, 30]
        # The record's own size field is kept as provenance.
        assert list(events["raw_size"]) == [100, 50, 40, 30, 30]

    def test_a_fill_is_charged_to_the_next_book_event_for_that_order(self):
        _, events = load(LIFECYCLE)

        by_event = dict(zip(events["event_id"], events["fill"]))
        modify = events.loc[events["raw_event_type"] == "M", "event_id"].iloc[0]
        cancel = events.loc[
            (events["raw_event_type"] == "C") & (events["id"] == 2), "event_id"
        ].iloc[0]
        assert by_event[modify] == 20
        assert by_event[cancel] == 30
        # A cancel the trader asked for is not a fill.
        partial = events.loc[events["id"] == 1, "event_id"].iloc[1]
        assert by_event[partial] == 0

    def test_several_fills_between_book_events_sum_onto_one_event(self):
        _, events = load(
            [
                (1, "A", "A", px(10.00), 100),
                (1, "F", "A", px(10.00), 30),
                (1, "F", "A", px(10.00), 20),
                (1, "M", "A", px(10.00), 50),
            ]
        )
        assert list(events["fill"]) == [0, 50]
        assert list(events["volume"]) == [100, 50]

    def test_modify_for_an_unseen_order_is_an_add(self):
        # Databento's own book builder treats a modify for an unknown order as
        # the record that puts it on the book.
        _, events = load([(7, "M", "B", px(9.99), 25)])
        assert list(events["action"]) == ["created"]
        assert list(events["volume"]) == [25]

    def test_cancel_for_an_unseen_order_deletes_the_size_removed(self):
        _, events = load([(7, "C", "B", px(9.99), 25)])
        assert list(events["action"]) == ["deleted"]
        assert list(events["volume"]) == [25]

    def test_cancel_that_empties_an_order_deletes_it(self):
        _, events = load(
            [
                (1, "A", "B", px(9.99), 30),
                (1, "C", "B", px(9.99), 10),
                (1, "C", "B", px(9.99), 20),
            ]
        )
        assert list(events["action"]) == ["created", "changed", "deleted"]
        assert list(events["volume"]) == [30, 20, 20]

    def test_trade_fill_and_none_records_are_not_book_events(self):
        _, events = load(
            [
                (1, "A", "B", px(9.99), 30),
                (0, "T", "A", px(9.99), 5),
                (0, "N", "N", px(9.99), 0),
            ]
        )
        assert len(events) == 1

    def test_a_modify_that_moves_the_price_is_a_changed_event(self):
        with warnings_logged() as logged:
            _, events = load(
                [
                    (1, "A", "B", px(9.99), 30),
                    (1, "M", "B", px(9.98), 30),
                ]
            )
        assert list(events["action"]) == ["created", "changed"]
        assert list(events["price"]) == [999, 998]
        # The depth follows a move now (issue #262), so there is nothing to
        # warn about.
        assert not logged

    def test_a_modify_that_grows_an_order_is_a_changed_event(self):
        with warnings_logged() as logged:
            _, events = load(
                [
                    (1, "A", "B", px(9.99), 30),
                    (1, "M", "B", px(9.99), 45),
                ]
            )
        assert list(events["volume"]) == [30, 45]
        assert not logged

    def test_a_modify_that_executes_and_moves_is_reported(self):
        # The depth reads a modify that carries a fill as an execution report,
        # so a move on the same record is the one case it still cannot follow.
        with warnings_logged() as logged:
            _, events = load(
                [
                    (1, "A", "B", px(9.99), 30),
                    (1, "F", "B", px(9.99), 10),
                    (1, "M", "B", px(9.98), 20),
                ]
            )
        assert list(events["fill"]) == [0, 10]
        assert any("1 modifies carry a fill" in m for m in logged)

    def test_a_modify_that_only_executes_is_not_reported(self):
        with warnings_logged() as logged:
            load(LIFECYCLE)
        assert not any("modifies carry a fill" in m for m in logged)

    def test_side_becomes_the_resting_direction(self):
        _, events = load([(1, "A", "B", px(9.99), 1), (2, "A", "A", px(10.01), 1)])
        assert list(events["direction"]) == ["bid", "ask"]


class TestBookClear:
    CLEARED: ClassVar[list[tuple]] = [
        (1, "A", "B", px(9.99), 30),
        (2, "A", "A", px(10.01), 40),
        (2, "C", "A", px(10.01), 40),  # gone before the clear
        (0, "R", "N", 0, 0),
        (1, "A", "B", px(9.50), 10),  # the id is reused after the clear
    ]

    def test_resting_orders_are_deleted_at_the_clear(self):
        _, events = load(self.CLEARED)

        cleared = events[events["raw_event_type"] == "R"]
        assert len(cleared) == 1
        assert cleared["id"].iloc[0] == 1
        assert cleared["action"].iloc[0] == "deleted"
        assert cleared["volume"].iloc[0] == 30
        # An order already off the book is not deleted twice.
        assert (events["id"] == 2).sum() == 2

    def test_the_synthetic_delete_sorts_before_the_clear(self):
        _, events = load(self.CLEARED)
        actions = list(events["action"])
        assert actions == ["created", "created", "deleted", "deleted", "created"]

    def test_a_reused_order_id_after_a_clear_is_a_new_order(self):
        _, events = load(self.CLEARED)
        reused = events[events["id"] == 1]
        assert list(reused["action"]) == ["created", "deleted", "created"]
        assert list(reused["price"]) == [999, 999, 950]


# ── Trades ────────────────────────────────────────────────────────────


class TestTrades:
    def test_fills_become_trades_with_the_aggressor_side_and_a_maker(self):
        loader, events = load(LIFECYCLE)
        trades = DatabentoTradeReader(_config(), loader=loader).load(events, None)

        assert len(trades) == 2
        # The resting order was a sell, so the aggressor bought.
        assert list(trades["direction"]) == ["buy", "buy"]
        assert list(trades["volume"]) == [20, 30]
        assert list(trades["price"]) == [10035, 10035]
        assert list(trades["maker"]) == [2, 2]
        # Each trade names the book event that took its size off.
        makers = dict(zip(events["event_id"], events["raw_event_type"]))
        assert [makers[e] for e in trades["maker_event_id"]] == ["M", "C"]

    def test_the_taker_is_not_identified(self):
        # A DBN trade record does not reliably carry the aggressing order's id.
        loader, events = load(LIFECYCLE)
        trades = DatabentoTradeReader(_config(), loader=loader).load(events, None)
        assert trades["taker"].isna().all()
        assert trades["taker_event_id"].isna().all()

    def test_trade_prints_are_used_when_the_file_has_no_fills(self):
        records = [
            (1, "A", "B", px(9.99), 30),
            (0, "T", "A", px(9.99), 12),  # a seller aggressed
            (1, "C", "B", px(9.99), 12),
        ]
        loader, events = load(records)
        trades = DatabentoTradeReader(_config(), loader=loader).load(events, None)

        assert len(trades) == 1
        assert trades["direction"].iloc[0] == "sell"
        assert trades["volume"].iloc[0] == 12
        assert pd.isna(trades["maker"].iloc[0])

    def test_a_trade_with_no_aggressor_side_is_classified(self):
        # Databento sends no side for auctions, non-displayed orders and
        # off-exchange prints. The pipeline labels those against the
        # reconstructed quotes rather than leaving them unset.
        recs = [
            (1, "A", "B", px(100.00), 10),
            (2, "A", "A", px(100.05), 10),
            (0, "T", "N", px(100.05), 4),
            (2, "C", "A", px(100.05), 4),
        ]
        result = Pipeline(source=DatabentoSource()).run(mbo_frame(recs))

        assert len(result.trades) == 1
        # A print at the ask is a buy.
        assert result.trades["direction"].iloc[0] == "buy"

    def test_a_venue_labelled_side_is_never_overwritten(self):
        result = Pipeline(source=DatabentoSource()).run(mbo_frame(LIFECYCLE))
        assert list(result.trades["direction"]) == ["buy", "buy"]

    def test_no_trades_at_all_gives_the_empty_frame(self):
        loader, events = load([(1, "A", "B", px(9.99), 30)])
        trades = DatabentoTradeReader(_config(), loader=loader).load(events, None)
        assert trades.empty
        validate_trades_df(trades)

    def test_the_reader_says_so_when_the_loader_has_not_run(self):
        reader = DatabentoTradeReader(_config(), loader=DatabentoLoader(_config()))
        with pytest.raises(ConfigError, match="has not read a file yet"):
            reader.load(pd.DataFrame(), None)

    def test_a_print_with_no_fill_behind_it_is_left_out(self):
        # An auction print with no passive side: the fills do not add up to
        # the printed volume, and only the fills make trades.  The loader logs
        # the gap so it is not silent.
        loader, events = load(LIFECYCLE + [(0, "T", "B", px(100.40), 500)])
        trades = DatabentoTradeReader(_config(), loader=loader).load(events, None)
        assert len(trades) == 2
        assert trades["volume"].sum() == 50


# ── Prices, sizes, clocks ─────────────────────────────────────────────


class TestEncoding:
    def test_fixed_point_prices_become_integer_ticks(self):
        _, events = load([(1, "A", "B", px(123.45), 1)])
        assert events["price"].iloc[0] == 12345
        assert events["price"].dtype == np.int64

    def test_a_coarser_tick_quantises_onto_its_own_grid(self):
        config = _config(tick_size=0.25)
        _, events = load([(1, "A", "B", px(4184.75), 1)], config=config)
        assert events["price"].iloc[0] == 16739  # 4184.75 / 0.25

    def test_sizes_become_integer_lots(self):
        _, events = load([(1, "A", "B", px(1.0), 137)])
        assert events["volume"].iloc[0] == 137
        assert events["volume"].dtype == np.int64

    def test_receive_time_is_the_pipeline_clock_and_event_time_the_venues(self):
        _, events = load([(1, "A", "B", px(1.0), 1)])
        assert str(events["timestamp"].dtype) == "datetime64[ns, UTC]"
        assert str(events["exchange_timestamp"].dtype) == "datetime64[ns, UTC]"
        # ts_recv is 500ns after ts_event in the fixture.
        delta = events["timestamp"].iloc[0] - events["exchange_timestamp"].iloc[0]
        assert delta == pd.Timedelta(500, "ns")

    def test_integer_nanosecond_clocks_are_accepted(self):
        frame = mbo_frame([(1, "A", "B", px(1.0), 1)])
        frame["ts_recv"] = frame["ts_recv"].astype("int64")
        frame["ts_event"] = frame["ts_event"].astype("int64")
        _, events = load(frame)
        assert str(events["timestamp"].dtype) == "datetime64[ns, UTC]"

    def test_the_ordering_keys_are_attached_when_asked_for(self):
        config = _config(track_sequence=True)
        _, events = load(LIFECYCLE, config=config)
        assert list(events["sequence"]) == [1, 2, 3, 6, 9]
        assert list(events["ingest_seq"]) == [0, 1, 2, 3, 4]

    def test_identity_columns_are_attached_when_a_symbol_is_given(self):
        _, events = load([(1, "A", "B", px(1.0), 1)], symbol="AAPL")
        assert list(events["venue"]) == ["databento"]
        assert list(events["symbol"]) == ["AAPL"]

    def test_identity_columns_are_absent_by_default(self):
        _, events = load([(1, "A", "B", px(1.0), 1)])
        assert "venue" not in events.columns


# ── What the loader refuses ───────────────────────────────────────────


class TestRefusals:
    def test_two_instruments_in_one_file_is_an_error(self):
        frame = mbo_frame([(1, "A", "B", px(1.0), 1), (2, "A", "A", px(2.0), 1)])
        frame.loc[1, "instrument_id"] = 999
        with pytest.raises(ConfigError, match="instrument_id"):
            load(frame)

    def test_two_publishers_of_one_instrument_is_an_error(self):
        # Two venues quoting the same symbol are two books.
        frame = mbo_frame([(1, "A", "B", px(1.0), 1), (2, "A", "A", px(2.0), 1)])
        frame.loc[1, "publisher_id"] = 41
        with pytest.raises(ConfigError, match="publisher_id"):
            load(frame)

    def test_naming_the_instrument_narrows_a_multi_instrument_file(self):
        frame = mbo_frame([(1, "A", "B", px(1.0), 1), (2, "A", "A", px(2.0), 1)])
        frame.loc[1, "instrument_id"] = 999
        _, events = load(frame, settings=DatabentoSettings(instrument_id=INSTRUMENT_ID))
        assert list(events["id"]) == [1]

    def test_naming_the_symbol_narrows_a_multi_instrument_file(self):
        frame = mbo_frame([(1, "A", "B", px(1.0), 1), (2, "A", "A", px(2.0), 1)])
        frame["symbol"] = ["AAPL", "MSFT"]
        frame.loc[1, "instrument_id"] = 999
        _, events = load(frame, settings=DatabentoSettings(raw_symbol="MSFT"))
        assert list(events["id"]) == [2]

    def test_a_filter_that_matches_nothing_is_an_error(self):
        with pytest.raises(ConfigError, match="no records left"):
            load(
                [(1, "A", "B", px(1.0), 1)],
                settings=DatabentoSettings(instrument_id=4242),
            )

    def test_a_filter_on_a_missing_column_is_an_error(self):
        # A file with no symbol mapping carries no ``symbol`` column.
        frame = mbo_frame([(1, "A", "B", px(1.0), 1)])
        assert "symbol" not in frame.columns
        with pytest.raises(ConfigError, match="no such column"):
            load(frame, settings=DatabentoSettings(raw_symbol="AAPL"))

    @pytest.mark.parametrize("flag", [F_TOB, F_MBP])
    def test_aggregated_records_are_refused(self, flag):
        # A top-of-book or price-level publisher's order ids mean nothing, so
        # per-order reconstruction would invent identity the feed never had.
        frame = mbo_frame([(1, "A", "B", px(1.0), 1)], flags=flag)
        with pytest.raises(ConfigError, match="aggregated, not per-order"):
            load(frame)

    def test_an_unknown_action_is_an_error(self):
        # A record whose meaning is unknown has no safe reading, and dropping
        # it silently would take its liquidity out of the book unannounced.
        with pytest.raises(ConfigError, match="action this loader does not know"):
            load([(1, "A", "B", px(1.0), 1), (2, "a", "A", px(2.0), 1)])

    def test_an_order_id_too_big_for_the_schema_is_an_error(self):
        # DBN order ids are unsigned 64-bit; the schema's is signed, so a cast
        # would wrap this to a negative id and could merge two orders.
        frame = mbo_frame([(1, "A", "B", px(1.0), 1)])
        frame["order_id"] = pd.array([MAX_ORDER_ID + 6], dtype="uint64")
        with pytest.raises(ConfigError, match="order id above"):
            load(frame)

    def test_an_order_id_at_the_ceiling_is_accepted(self):
        frame = mbo_frame([(1, "A", "B", px(1.0), 1)])
        frame["order_id"] = pd.array([MAX_ORDER_ID], dtype="uint64")
        _, events = load(frame)
        assert events["id"].iloc[0] == MAX_ORDER_ID

    def test_a_record_with_no_price_is_dropped(self):
        # UNDEF_PRICE is INT64_MAX. Kept, it would rest at ~9.2 billion and be
        # read back as the best bid.
        _, events = load(
            [
                (1, "A", "B", px(100.00), 10),
                (2, "A", "A", px(100.05), 10),
                (3, "A", "B", UNDEF_PRICE, 10),
            ]
        )
        assert list(events["id"]) == [1, 2]
        assert list(events["price"]) == [10000, 10005]

    def test_a_dropped_undefined_price_leaves_the_book_intact(self):
        recs = [
            (1, "A", "B", px(100.00), 10),
            (2, "A", "A", px(100.05), 10),
            (3, "A", "B", UNDEF_PRICE, 10),
        ]
        result = Pipeline(source=DatabentoSource()).run(mbo_frame(recs))
        last = result.depth_summary.iloc[-1]
        assert last["best_bid_price"] == 10000
        assert last["best_ask_price"] == 10005

    def test_a_book_record_with_no_side_is_dropped(self):
        # Databento states a side on every A/M/C. Without one there is no side
        # of the book to put the order on, and the depth rebuild would drop it
        # while the events frame kept it.
        _, events = load(
            [
                (1, "A", "B", px(100.00), 10),
                (2, "A", "N", px(100.05), 10),
                (2, "C", "N", px(100.05), 10),
            ]
        )
        assert list(events["id"]) == [1]
        assert events["direction"].notna().all()

    def test_a_frame_missing_a_required_column_is_an_error(self):
        frame = mbo_frame([(1, "A", "B", px(1.0), 1)]).drop(columns=["side"])
        with pytest.raises(ConfigError, match="missing required columns"):
            load(frame)


# ── The source descriptor ─────────────────────────────────────────────


class TestSource:
    def test_it_is_registered_and_offline_capable(self):
        assert get_source("databento") is DatabentoSource
        assert isinstance(DatabentoSource(), OfflineSource)

    def test_it_declares_its_coordinates(self):
        source = DatabentoSource()
        assert source.level is Level.L3
        assert source.feed_type is FeedType.MATCHED_BOOK
        assert source.name == "databento"

    def test_it_needs_no_run_context(self):
        assert DatabentoSource().required_context() == []

    def test_its_defaults_carry_the_fixed_point_scale(self):
        defaults = DatabentoSource().config_defaults()
        assert defaults["price_divisor"] == DBN_PRICE_DIVISOR
        assert defaults["lot_size"] == 1.0
        assert defaults["timestamp_unit"] == "ns"

    def test_the_pipeline_picks_up_the_defaults(self):
        pipeline = Pipeline(source=DatabentoSource())
        assert pipeline.config.price_divisor == DBN_PRICE_DIVISOR

    def test_the_trade_reader_shares_the_loaders_records(self):
        source = DatabentoSource()
        config = _config()
        ctx = RunContext()
        loader = source.create_loader(config, ctx)
        reader = source.create_trade_source(config, ctx)
        events = loader.load(mbo_frame(LIFECYCLE))
        assert len(reader.load(events, None)) == 2


class TestPipelineRun:
    def test_a_window_replays_end_to_end(self):
        result = Pipeline(source=DatabentoSource()).run(mbo_frame(LIFECYCLE))

        assert result.level is Level.L3
        validate_events_df(result.events)
        validate_trades_df(result.trades)
        assert len(result.trades) == 2
        assert not result.depth.empty
        assert not result.depth_summary.empty

    def test_the_reconstructed_book_matches_the_records(self):
        result = Pipeline(source=DatabentoSource()).run(mbo_frame(LIFECYCLE))

        # After the whole window: the bid has 60 left at 100.25 and the ask is
        # gone.
        final = result.depth.sort_values("timestamp").drop_duplicates(
            subset=["price", "direction"], keep="last"
        )
        resting = final[final["volume"] > 0]
        assert list(resting["price"]) == [10025]
        assert list(resting["volume"]) == [60]

    def test_the_touch_follows_orders_that_move_and_grow(self):
        # Issue #262: the reconstructed best bid and ask after every record
        # must be the ones the records themselves imply.
        records = [
            (1, "A", "B", px(100.00), 10),
            (2, "A", "A", px(100.10), 20),
            (3, "A", "B", px(99.90), 30),
            (1, "M", "B", px(100.05), 10),  # the best bid moves up
            (2, "M", "A", px(100.08), 25),  # the best ask moves down and grows
            (1, "M", "B", px(99.80), 10),  # the best bid moves away
            (3, "C", "B", px(99.90), 30),
        ]
        # (best bid, its size, best ask, its size) after each record.
        expected = [
            (10000, 10, None, 0),
            (10000, 10, 10010, 20),
            (10000, 10, 10010, 20),
            (10005, 10, 10010, 20),
            (10005, 10, 10008, 25),
            (9990, 30, 10008, 25),
            (9980, 10, 10008, 25),
        ]
        result = Pipeline(source=DatabentoSource()).run(mbo_frame(records))

        # A move writes two depth rows at one instant; the book after the
        # record is the last of them.
        after = result.depth_summary.groupby("timestamp", sort=True).last()
        got = [
            (bid if bid_vol else None, bid_vol, ask if ask_vol else None, ask_vol)
            for bid, bid_vol, ask, ask_vol in after[
                ["best_bid_price", "best_bid_vol", "best_ask_price", "best_ask_vol"]
            ]
            .to_numpy()
            .tolist()
        ]
        assert got == expected


# ── Reading and writing real DBN files ────────────────────────────────


def dbn_bytes(records: list[tuple], *, dataset="XNAS.ITCH", symbol="AAPL") -> bytes:
    """Encode *records* as a DBN MBO stream (metadata header + records)."""
    from databento_dbn import Action, MBOMsg, Metadata, Schema, Side, SType

    meta = Metadata(
        dataset=dataset,
        start=T0,
        stype_in=SType.from_str("raw_symbol"),
        stype_out=SType.from_str("instrument_id"),
        schema=Schema.from_str("mbo"),
        symbols=[symbol],
    )
    payload = bytearray(meta.encode())
    for i, (order_id, action, side, price, size) in enumerate(records):
        payload += bytes(
            MBOMsg(
                publisher_id=PUBLISHER_ID,
                instrument_id=INSTRUMENT_ID,
                ts_event=T0 + i * 1_000_000,
                order_id=order_id,
                price=price,
                size=size,
                action=Action.from_str(action),
                side=Side.from_str(side),
                ts_recv=T0 + i * 1_000_000 + 500,
                sequence=i + 1,
            )
        )
    return bytes(payload)


@requires_databento
class TestDbnFiles:
    def test_a_dbn_file_replays_through_the_pipeline(self, tmp_path):
        path = tmp_path / "lifecycle.mbo.dbn"
        path.write_bytes(dbn_bytes(LIFECYCLE))

        result = Pipeline(source=DatabentoSource()).run(path)

        assert list(result.events["action"]) == [
            "created",
            "created",
            "changed",
            "changed",
            "deleted",
        ]
        assert len(result.trades) == 2

    def test_an_open_store_is_accepted_too(self):
        import databento as db

        store = db.DBNStore.from_bytes(dbn_bytes(LIFECYCLE))
        _, events = load(store)
        assert len(events) == 5

    def test_a_price_level_schema_is_refused(self, tmp_path):
        from databento_dbn import Metadata, Schema, SType

        meta = Metadata(
            dataset="XNAS.ITCH",
            start=T0,
            stype_in=SType.from_str("raw_symbol"),
            stype_out=SType.from_str("instrument_id"),
            schema=Schema.from_str("mbp-10"),
            symbols=["AAPL"],
        )
        path = tmp_path / "depth.mbp10.dbn"
        path.write_bytes(bytes(meta.encode()))

        with pytest.raises(ConfigError, match="not per-order data"):
            load(path)

    def test_events_round_trip_through_the_writer(self, tmp_path):
        source = DatabentoSource()
        result = Pipeline(source=source).run(mbo_frame(LIFECYCLE))

        out = DatabentoWriter(result.config).write(
            {"events": result.events}, tmp_path / "out.dbn"
        )
        reloaded = Pipeline(source=DatabentoSource()).run(out)

        columns = ["id", "price", "volume", "action", "direction", "fill"]
        pd.testing.assert_frame_equal(
            result.events[columns].reset_index(drop=True),
            reloaded.events[columns].reset_index(drop=True),
        )

    def test_the_writer_fills_a_directory(self, tmp_path):
        result = Pipeline(source=DatabentoSource()).run(mbo_frame(LIFECYCLE))
        out = DatabentoWriter(result.config).write({"events": result.events}, tmp_path)
        assert out == tmp_path / "events.dbn"
        assert out.exists()

    def test_the_source_supplies_a_writer(self):
        writer = DatabentoSource().create_writer(_config(), RunContext())
        assert isinstance(writer, DatabentoWriter)
