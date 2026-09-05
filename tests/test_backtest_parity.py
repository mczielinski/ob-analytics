"""Cross-check our reconstruction against hftbacktest and Nautilus (issue #224).

Issue #143 proved the engine against golden files this project wrote itself,
which catches a change but not a shared misreading: if the rebuild and the
baseline are both wrong in the same way, both agree.  These tests replay the
same events through an engine written by someone else and compare the two
books.  A disagreement means one of the two is wrong.

The comparison is per event on the best bid and ask, which is the part both
engines model the same way.  Queue position is deliberately **not** compared —
see :class:`TestQueuePositionIsNotCompared` for why.

Both engines are a ``backtest-engines`` dependency group rather than a runtime
extra: the writers in :mod:`ob_analytics.interop` need neither installed, and
only these tests do.  Each half skips when its engine is absent, the way the
ccxt and cryptofeed tests already do.  Nautilus additionally publishes no build
that takes both Python 3.11 and pandas 3, so on 3.11 its half always skips.

This is what found the writer defect fixed alongside it: the export writers were
sending never-resting marketable orders as book liquidity, so the receiving
engine crossed its book and dropped the resting level they traded against.

It also found a second defect, in this library rather than in the writers, and
that one is **not** fixed here: a price level is a float running sum of adds,
cancels and fills, and it does not return to exactly zero when the last order
leaves.  The level stays live on residue such as ``5.55e-17`` and is reported as
the best bid or ask ahead of the real one.  The touch-parity tests below are
marked ``xfail`` for that reason, and issue #226 is what closes them; the rest
of the file passes today.  Leaving the check in and failing is the point — it is
the evidence for #226, and it turns green the moment #226 lands.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from ob_analytics.interop import to_hftbacktest_array, to_nautilus_deltas
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.synth import (
    SynthConfig,
    SyntheticLoader,
    SyntheticTradeSource,
    generate_session,
)

_HFTBACKTEST_INSTALLED = importlib.util.find_spec("hftbacktest") is not None
_NAUTILUS_INSTALLED = importlib.util.find_spec("nautilus_trader") is not None

needs_hftbacktest = pytest.mark.skipif(
    not _HFTBACKTEST_INSTALLED,
    reason="hftbacktest not installed (uv sync --group backtest-engines)",
)
needs_nautilus = pytest.mark.skipif(
    not _NAUTILUS_INSTALLED,
    reason=(
        "nautilus-trader not installed; it publishes no build taking both "
        "Python 3.11 and pandas 3 (uv sync --group backtest-engines)"
    ),
)

#: The tick grid the parity runs use.  Both engines read float prices, so the
#: writers scale the canonical integer ticks by this (issue #155).
TICK_SIZE = 0.01

#: The size grid handed to hftbacktest's own builder.  It is the engine's own
#: setting rather than ours: sizes still cross as base-asset floats until #226.
LOT_SIZE = 1e-8

#: Seeds whose sessions are replayed.  Several, because one seed's session
#: exercises one arrangement of crossings and cancellations; the writer defect
#: this file was written to catch showed on three of these five and not on the
#: other two.
PARITY_SEEDS: tuple[int, ...] = (224, 143, 7, 99, 1234)


def _session(seed: int, duration: float = 30.0) -> PipelineResult:
    """Run one seeded synthetic L3 session through the pipeline."""
    session = generate_session(SynthConfig(seed=seed, duration=duration))
    return Pipeline(
        loader=SyntheticLoader(session),
        trade_source=SyntheticTradeSource(session),
    ).run(source=None)


def _hftbacktest_touch(array: np.ndarray) -> pd.DataFrame:
    """Replay *array* through hftbacktest and return its best bid/ask per feed.

    Steps the engine one market feed at a time (``wait_next_feed`` returns 2 on
    a feed and 1 at the end of the data) and reads the book it has built.  The
    result is indexed by the engine's own clock and reduced to the last state
    at each instant, because several events can share one.
    """
    from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

    asset = (
        BacktestAsset()
        .data([array])
        .linear_asset(1.0)
        .constant_order_latency(0, 0)
        .l3_fifo_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(TICK_SIZE)
        .lot_size(LOT_SIZE)
    )
    backtest = HashMapMarketDepthBacktest([asset])
    rows: list[tuple[int, float, float]] = []
    try:
        outcome = backtest.wait_next_feed(False, 10**15)
        while outcome == 2:
            depth = backtest.depth(0)
            rows.append((backtest.current_timestamp, depth.best_bid, depth.best_ask))
            outcome = backtest.wait_next_feed(False, 10**15)
    finally:
        backtest.close()
    return _last_state_per_instant(pd.DataFrame(rows, columns=["ts", "bid", "ask"]))


def _last_state_per_instant(rows: pd.DataFrame) -> pd.DataFrame:
    """Reduce per-event book states to the final state at each instant.

    Deliberately not ``groupby("ts").last()``: that takes the last *non-null*
    value in each column independently, so a side emptied by the last event of
    an instant would keep reporting the price it held before — the book would
    look like it still had a side it does not.  Keeping the last whole row
    instead reports the emptied side as empty.
    """
    return (
        rows.set_index("ts")
        .pipe(lambda df: df[~df.index.duplicated(keep="last")])
        .sort_index()
    )


def _our_touch(result: PipelineResult) -> pd.DataFrame:
    """Our own best bid/ask per instant, in the engines' float units.

    An empty side is ``0`` ticks in ``depth_summary`` and ``NaN`` in
    hftbacktest, so it becomes ``NaN`` here and the two agree on "no book".
    """
    summary = result.depth_summary
    ours = _last_state_per_instant(
        pd.DataFrame(
            {
                "ts": summary["timestamp"].dt.as_unit("ns").astype("int64"),
                "bid": summary["best_bid_price"],
                "ask": summary["best_ask_price"],
            }
        )
    )
    return pd.DataFrame(
        {
            "bid": ours["bid"].replace(0, np.nan) * TICK_SIZE,
            "ask": ours["ask"].replace(0, np.nan) * TICK_SIZE,
        }
    )


def _disagreements(ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    """Rows where the two books name a different best bid or ask.

    Compared at half a tick, which is finer than the grid the two can differ
    on: a price is an exact integer on both sides before it is scaled, so a
    real disagreement is a whole tick or more.  Two empty sides agree.
    """
    joined = ours.join(theirs, how="inner", lsuffix="_ours", rsuffix="_theirs")
    disagrees = pd.Series(False, index=joined.index)
    for side in ("bid", "ask"):
        mine, yours = joined[f"{side}_ours"], joined[f"{side}_theirs"]
        both_empty = mine.isna() & yours.isna()
        close = (mine - yours).abs() <= TICK_SIZE / 2
        disagrees |= ~(close | both_empty)
    return joined[disagrees]


@needs_hftbacktest
class TestHftbacktestParity:
    """Our book and hftbacktest's must name the same touch, event for event."""

    @pytest.mark.xfail(
        reason=(
            "issue #226: a price level is a float running sum and does not "
            "empty to exactly zero, so a level left holding residue is still "
            "reported as the best bid or ask. Passes once sizes are integer "
            "lots. Not strict: the defect needs a crossing to show, so a seed "
            "whose session happens not to produce one already agrees."
        ),
        strict=False,
    )
    @pytest.mark.parametrize("seed", PARITY_SEEDS)
    def test_best_bid_and_ask_agree_on_every_event(self, seed: int):
        result = _session(seed)
        array = to_hftbacktest_array(result.events, tick_size=TICK_SIZE)
        theirs = _hftbacktest_touch(array)
        ours = _our_touch(result)

        differing = _disagreements(ours, theirs)
        assert len(theirs) > 100, "the replay produced too little to be a real check"
        assert differing.empty, (
            f"seed {seed}: {len(differing)} of {len(theirs)} instants disagree "
            f"with hftbacktest's own reconstruction\n{differing.head(10)}"
        )

    @pytest.mark.parametrize("seed", PARITY_SEEDS[:2])
    def test_written_feed_passes_the_engines_own_order_check(self, seed: int):
        # The cheap half of the cross-check: it needs no backtest, only the
        # engine's own validator, and it is what catches a feed written in the
        # wrong order for either of hftbacktest's two clocks.
        from hftbacktest.data import validate_event_order

        result = _session(seed)
        array = to_hftbacktest_array(result.events, tick_size=TICK_SIZE)
        validate_event_order(array)  # raises ValueError if out of order

    def test_the_writers_hardcoded_contract_matches_the_installed_engine(self):
        # ob_analytics.interop writes hftbacktest's dtype and event codes out
        # by hand, because the engine is not a dependency.  This is the only
        # place that can check the copy against the original.
        import hftbacktest as hft

        from ob_analytics.interop import HFT_EVENT_DTYPE

        assert HFT_EVENT_DTYPE == hft.event_dtype
        assert hft.ADD_ORDER_EVENT == 10
        assert hft.CANCEL_ORDER_EVENT == 11
        assert hft.MODIFY_ORDER_EVENT == 12
        assert hft.EXCH_EVENT == 1 << 31
        assert hft.LOCAL_EVENT == 1 << 30
        assert hft.BUY_EVENT == 1 << 29
        assert hft.SELL_EVENT == 1 << 28

    def test_never_resting_orders_are_left_out_of_the_feed(self):
        # The defect this file was written to catch.  A marketable order is
        # recorded as a transient add on its own side at the touch; exporting
        # it makes the engine cross its book and drop the resting level it
        # traded against, so its reconstruction drifts thinner than ours.
        result = _session(224)
        events = result.events
        taker_only = events.groupby("id")["type"].first() == "market"
        assert taker_only.any(), "this seed must contain some to be a real check"

        array = to_hftbacktest_array(events, tick_size=TICK_SIZE)
        excluded_ids = set(taker_only[taker_only].index)
        assert not (set(array["order_id"].tolist()) & excluded_ids)


@needs_nautilus
class TestNautilusParity:
    """The delta frame must be one Nautilus' own wrangler accepts and replays."""

    def test_wrangler_accepts_the_written_deltas(self):
        from nautilus_trader.persistence.wranglers import OrderBookDeltaDataWrangler
        from nautilus_trader.test_kit.providers import TestInstrumentProvider

        result = _session(224)
        deltas = to_nautilus_deltas(result.events, tick_size=TICK_SIZE)
        instrument = TestInstrumentProvider.btcusdt_binance()
        wrangled = OrderBookDeltaDataWrangler(instrument).process(deltas)

        assert len(wrangled) == len(deltas)

    @pytest.mark.xfail(
        reason="issue #226, as above: float price levels do not empty exactly.",
        strict=False,
    )
    def test_replaying_the_deltas_reproduces_our_touch(self):
        # Nautilus' own book, built from the deltas the writer emits, must name
        # the same best bid and ask as depth_summary.
        from nautilus_trader.model.book import OrderBook
        from nautilus_trader.model.enums import BookType
        from nautilus_trader.persistence.wranglers import OrderBookDeltaDataWrangler
        from nautilus_trader.test_kit.providers import TestInstrumentProvider

        result = _session(224)
        deltas = to_nautilus_deltas(result.events, tick_size=TICK_SIZE)
        instrument = TestInstrumentProvider.btcusdt_binance()
        wrangled = OrderBookDeltaDataWrangler(instrument).process(deltas)

        book = OrderBook(instrument.id, BookType.L3_MBO)
        rows: list[tuple[int, float, float]] = []
        for delta in wrangled:
            book.apply_delta(delta)
            best_bid = book.best_bid_price()
            best_ask = book.best_ask_price()
            rows.append(
                (
                    delta.ts_event,
                    float(best_bid) if best_bid is not None else np.nan,
                    float(best_ask) if best_ask is not None else np.nan,
                )
            )
        theirs = _last_state_per_instant(
            pd.DataFrame(rows, columns=["ts", "bid", "ask"])
        )

        differing = _disagreements(_our_touch(result), theirs)
        assert differing.empty, (
            f"{len(differing)} of {len(theirs)} instants disagree with "
            f"Nautilus' own book\n{differing.head(10)}"
        )


class TestQueuePositionIsNotCompared:
    """Say plainly which part of the reconstruction this file does not check.

    Issue #224 asks for queue position "where both model it", and the answer is
    that they do not model the same thing.  ``ob_analytics.queue.queue_positions``
    reconstructs, for every order in the market, how much size sits ahead of it
    at its level — a property of the observed book. hftbacktest's queue models
    (``l3_fifo_queue_model`` and the probabilistic ones) estimate where **the
    strategy's own** order would sit, which is a simulation input and is not
    defined for an order the strategy did not place.

    There is therefore no shared quantity to diff, and a test that invented one
    would be comparing our reconstruction against our own assumption dressed as
    an independent check. The touch parity above is the part that genuinely
    cross-checks, because the best bid and ask are the same quantity in both.
    """

    def test_our_queue_reconstruction_is_defined_for_market_orders(self):
        from ob_analytics.queue import queue_positions

        result = _session(224)
        queue = queue_positions(result.events, levels="all")
        # Every order in the queue is one the market placed, not one we did:
        # that is the difference that makes the two models incomparable.
        assert not queue.empty
        assert set(queue["id"]) <= set(result.events["id"])
