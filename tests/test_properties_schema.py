"""Property tests for the canonical schema rules (issue #143).

Issue #143 names three schema rules a correct engine must never break, and asks
for property tests that hold them over many generated inputs, not just the fixed
fixtures:

1. **Time goes forward** — events and trades are emitted in a non-decreasing
   time order, and the depth engine emits its rows in that same order.
2. **The book does not cross after an uncross** — every uncrossing path leaves
   ``best_bid < best_ask`` (strict), and the depth engine never leaves a
   strictly crossed touch (``best_bid <= best_ask``; equal-price locks are
   tolerated by design — see :class:`~ob_analytics.depth.DepthMetricsEngine`).
3. **Volume adds up** — outstanding size, executed size, and traded size
   reconcile, and price-level depth is never negative.

Rules 1 and 3 are driven by :mod:`ob_analytics.synth`, which emits a valid L3
stream with ground-truth trades. Rule 2 is driven by *hand-built, deliberately
crossable* books: the synthetic generator never crosses, so exercising the
crossed-level eviction needs inputs that do. This complements
:mod:`test_properties_lobster` (LOBSTER volume semantics) and the fixed-seed
structural checks in :mod:`test_synth`.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from ob_analytics.analytics import order_book, uncross_book_sides
from ob_analytics.depth import DepthMetricsEngine, depth_metrics, price_level_volume
from ob_analytics.synth import SynthConfig, generate_session

_T0 = pd.Timestamp("2020-01-01 00:00:00")


# ── Strategies ─────────────────────────────────────────────────────────────


@st.composite
def synth_configs(draw: st.DrawFn) -> SynthConfig:
    """A varied but always-populated :class:`SynthConfig`.

    Rates and duration are bounded so every draw produces a lively session
    quickly (hundreds of events, a mix of order types) without a runaway
    Hawkes path.
    """
    return SynthConfig(
        seed=draw(st.integers(min_value=0, max_value=10_000)),
        duration=draw(st.floats(min_value=20.0, max_value=45.0)),
        limit_rate=draw(st.floats(min_value=3.0, max_value=8.0)),
        cancel_rate=draw(st.floats(min_value=1.0, max_value=5.0)),
        market_rate=draw(st.floats(min_value=0.5, max_value=3.0)),
        mid_price=draw(st.sampled_from([50.0, 100.0, 250.0])),
        tick_size=draw(st.sampled_from([0.01, 0.05, 0.1])),
        half_spread_ticks=draw(st.integers(min_value=1, max_value=4)),
        depth_levels=draw(st.integers(min_value=3, max_value=12)),
        arrival_process=draw(st.sampled_from(["poisson", "hawkes"])),
        iceberg_fraction=draw(st.sampled_from([0.0, 0.2])),
        toxic_fraction=draw(st.sampled_from([0.0, 0.3])),
    )


def _price(draw: st.DrawFn) -> float:
    """A price in [90, 110] rounded to whole cents (matches the default tick)."""
    return round(
        draw(
            st.floats(
                min_value=90.0, max_value=110.0, allow_nan=False, allow_infinity=False
            )
        ),
        2,
    )


def _volume(draw: st.DrawFn, *, allow_zero: bool = False) -> float:
    low = 0.0 if allow_zero else 0.01
    return round(
        draw(
            st.floats(
                min_value=low, max_value=100.0, allow_nan=False, allow_infinity=False
            )
        ),
        4,
    )


@st.composite
def two_book_sides(draw: st.DrawFn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A pair of per-order (bids, asks) frames whose price ranges overlap.

    Both sides draw prices from the same [90, 110] window, so the two are
    frequently crossed or locked — exactly what the uncross paths must resolve.
    """

    def side(direction: str) -> pd.DataFrame:
        n = draw(st.integers(min_value=0, max_value=8))
        rows = [
            {
                "price": _price(draw),
                "volume": _volume(draw),
                "timestamp": _T0 + pd.Timedelta(seconds=draw(st.integers(0, 300))),
                "direction": direction,
            }
            for _ in range(n)
        ]
        return pd.DataFrame(rows, columns=["price", "volume", "timestamp", "direction"])

    bids, asks = side("bid"), side("ask")
    return bids, asks


@st.composite
def crossable_events(draw: st.DrawFn) -> pd.DataFrame:
    """Created-only resting orders on both sides, priced to cross frequently."""
    n = draw(st.integers(min_value=2, max_value=16))
    rows = []
    for k in range(n):
        rows.append(
            {
                "event_id": k + 1,
                "id": k + 1,
                "action": "created",
                "direction": draw(st.sampled_from(["bid", "ask"])),
                "type": "resting-limit",
                "timestamp": _T0 + pd.Timedelta(seconds=draw(st.integers(0, 300))),
                "price": _price(draw),
                "volume": _volume(draw),
            }
        )
    df = pd.DataFrame(rows)
    df["exchange_timestamp"] = df["timestamp"]
    df["action"] = pd.Categorical(
        df["action"], categories=["created", "changed", "deleted"], ordered=True
    )
    df["direction"] = pd.Categorical(
        df["direction"], categories=["bid", "ask"], ordered=True
    )
    return df


@st.composite
def crossable_depth(draw: st.DrawFn) -> pd.DataFrame:
    """A price-level depth stream (updates + deletes) that crosses freely."""
    n = draw(st.integers(min_value=1, max_value=30))
    rows = [
        {
            "timestamp": _T0 + pd.Timedelta(milliseconds=k),
            "price": _price(draw),
            "volume": _volume(draw, allow_zero=True),  # 0 deletes the level
            "direction": draw(st.sampled_from(["bid", "ask"])),
        }
        for k in range(n)
    ]
    df = pd.DataFrame(rows)
    df["direction"] = pd.Categorical(
        df["direction"], categories=["bid", "ask"], ordered=True
    )
    return df


# ── Rule 1: time goes forward ───────────────────────────────────────────────


@settings(max_examples=20, deadline=None)
@given(cfg=synth_configs())
def test_events_and_trades_move_forward_in_time(cfg: SynthConfig) -> None:
    session = generate_session(cfg)
    events = session.events.sort_values("event_id")

    # event_id is the dense 1..N ingest/replay key.
    assert events["event_id"].tolist() == list(range(1, len(events) + 1))
    # Time never runs backward along that key, on either clock.
    assert events["timestamp"].is_monotonic_increasing
    assert events["exchange_timestamp"].is_monotonic_increasing
    # Trades print in time order too.
    assert session.trades["timestamp"].is_monotonic_increasing


@settings(max_examples=20, deadline=None)
@given(cfg=synth_configs())
def test_depth_engine_emits_in_time_order(cfg: SynthConfig) -> None:
    events = generate_session(cfg).events
    summary = depth_metrics(price_level_volume(events))
    # The engine sorts by time, so consumers reading its output see a forward
    # clock regardless of the input row order.
    assert summary["timestamp"].is_monotonic_increasing


# ── Rule 2: the book does not cross after an uncross ─────────────────────────


@settings(max_examples=50, deadline=None)
@given(sides=two_book_sides())
def test_uncross_book_sides_leaves_no_cross(
    sides: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    bids, asks = sides
    ub, ua = uncross_book_sides(bids, asks)
    if not ub.empty and not ua.empty:
        # Returned best-first: best bid / best ask are the extreme prices.
        assert ub["price"].max() < ua["price"].min()


@settings(max_examples=50, deadline=None)
@given(events=crossable_events())
def test_order_book_uncross_leaves_no_cross(events: pd.DataFrame) -> None:
    book = order_book(events, uncross=True)
    bids, asks = book["bids"], book["asks"]
    if not bids.empty and not asks.empty:
        assert bids["price"].max() < asks["price"].min()


@settings(max_examples=50, deadline=None)
@given(depth=crossable_depth())
def test_depth_engine_never_strictly_crosses(depth: pd.DataFrame) -> None:
    summary = DepthMetricsEngine().compute(depth)
    both = (summary["best_bid_price"] > 0) & (summary["best_ask_price"] > 0)
    # Locks (equal best prices) are tolerated; a strict cross is not.
    assert (
        summary.loc[both, "best_bid_price"] <= summary.loc[both, "best_ask_price"]
    ).all()


def test_uncross_resolves_a_known_crossed_book() -> None:
    # A deterministic crossed book, so the eviction branch is covered even if
    # every random draw happened to be uncrossed.
    bids = pd.DataFrame(
        {
            "price": [103.0, 101.0, 99.0],
            "volume": [1.0, 1.0, 1.0],
            "timestamp": [_T0, _T0 + pd.Timedelta(seconds=2), _T0],
        }
    )
    asks = pd.DataFrame(
        {
            "price": [100.0, 102.0, 104.0],
            "volume": [1.0, 1.0, 1.0],
            "timestamp": [_T0 + pd.Timedelta(seconds=1), _T0, _T0],
        }
    )
    ub, ua = uncross_book_sides(bids, asks)
    assert ub["price"].max() < ua["price"].min()


# ── Rule 3: volume adds up ───────────────────────────────────────────────────


@settings(max_examples=20, deadline=None)
@given(cfg=synth_configs())
def test_volume_conservation(cfg: SynthConfig) -> None:
    session = generate_session(cfg)
    events, trades = session.events, session.trades

    # Non-negativity across the schema.
    assert (events["volume"] >= 0).all()
    assert (events["fill"] >= 0).all()
    assert (trades["volume"] > 0).all()

    # Global conservation: each fill is emitted twice — once on the maker event
    # and once on the taker event — so total event fill is exactly twice the
    # total traded volume.
    total_fill = float(events["fill"].sum())
    total_traded = float(trades["volume"].sum())
    assert total_fill == pytest.approx(2.0 * total_traded, rel=1e-9, abs=1e-9)

    # Per aggressor: a market order's created size equals the volume it swept.
    created_vol = events.loc[events["action"] == "created"].set_index("id")["volume"]
    swept = trades.groupby("taker")["volume"].sum()
    for taker_id, vol in swept.items():
        assert created_vol[taker_id] == pytest.approx(vol, rel=1e-9, abs=1e-6)

    # Price-level depth is a non-negative resting size everywhere.
    depth = price_level_volume(events)
    assert (depth["volume"] >= 0).all()


@settings(max_examples=20, deadline=None)
@given(cfg=synth_configs())
def test_outstanding_size_reconciles_with_fills(cfg: SynthConfig) -> None:
    """For every order, placed size minus cumulative fill equals outstanding.

    Walks each order's events in time order: outstanding stays non-negative and
    equals ``placed - Σfill`` after each event.
    """
    events = generate_session(cfg).events.sort_values("event_id")
    for _oid, group in events.groupby("id", sort=False):
        rows = group.sort_values("event_id")
        head = rows.iloc[0]
        assert head["action"] == "created"
        placed = float(head["volume"])
        cum_fill = 0.0
        for row in rows.iloc[1:].itertuples(index=False):
            cum_fill += float(row.fill)
            outstanding = placed - cum_fill
            assert outstanding >= -1e-9
            if row.action == "deleted" and row.fill > 0:
                assert abs(outstanding) < 1e-6  # a full execution empties it
            else:
                assert abs(float(row.volume) - outstanding) < 1e-6
        assert cum_fill <= placed + 1e-6
