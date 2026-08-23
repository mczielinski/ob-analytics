"""Tests for the synthetic L3 session generator (``ob_analytics.synth``).

Covers issue #114's acceptance: deterministic output for a given seed, a
generated session that satisfies the schema validators and flows through the
full pipeline, and the structural invariants of a valid L3 stream (no negative
outstanding volume; every ``changed`` / ``deleted`` references a previously
``created`` live order).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics import Pipeline
from ob_analytics.analytics import order_book, order_lifecycles
from ob_analytics.depth import price_level_volume
from ob_analytics.exceptions import ConfigError
from ob_analytics.schemas import (
    EVENT_COLUMNS,
    TRADE_COLUMNS,
    validate_events_df,
    validate_trades_df,
)
from ob_analytics.synth import (
    SynthConfig,
    SyntheticLoader,
    SyntheticTradeSource,
    SynthSession,
    generate_session,
)

# A short session keeps the tests fast while still producing a rich book
# (hundreds of orders, a mix of every produced order type).
_FAST = {"seed": 42, "duration": 90.0}


def _session(**overrides) -> SynthSession:
    params = {**_FAST, **overrides}
    return generate_session(SynthConfig(**params))


def _run_pipeline(session: SynthSession):
    return Pipeline(
        loader=SyntheticLoader(session),
        trade_source=SyntheticTradeSource(session),
    ).run(source=None)


# ── (a) Determinism ───────────────────────────────────────────────────


def test_same_seed_is_byte_identical():
    a = _session()
    b = _session()
    pd.testing.assert_frame_equal(a.events, b.events)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    # A byte-level check on the serialised frames, not just value equality.
    assert a.events.to_csv().encode() == b.events.to_csv().encode()
    assert a.trades.to_csv().encode() == b.trades.to_csv().encode()


def test_same_seed_identical_with_all_options():
    opts = {
        "arrival_process": "hawkes",
        "hawkes_excitation": 0.5,
        "iceberg_fraction": 0.2,
        "toxic_fraction": 0.25,
    }
    a = _session(seed=11, **opts)
    b = _session(seed=11, **opts)
    pd.testing.assert_frame_equal(a.events, b.events)
    pd.testing.assert_frame_equal(a.trades, b.trades)


def test_different_seed_differs():
    a = _session(seed=1)
    b = _session(seed=2)
    assert not a.events.equals(b.events)


def test_kwargs_override_config():
    session = generate_session(seed=5, duration=30.0)
    assert session.config.seed == 5
    assert session.config.duration == 30.0


# ── (b) Schema conformance + pipeline flow ────────────────────────────


def test_events_conform_to_schema():
    events = _session().events
    validate_events_df(events)
    for col in EVENT_COLUMNS:
        assert col in events.columns
    # event_id is a dense, sorted, unique 1..N key.
    assert events["event_id"].tolist() == list(range(1, len(events) + 1))
    assert events["event_id"].is_unique
    # Timestamps are tz-naive (schema policy) and datetime typed.
    assert events["timestamp"].dt.tz is None
    assert pd.api.types.is_datetime64_ns_dtype(events["timestamp"])
    assert pd.api.types.is_datetime64_ns_dtype(events["exchange_timestamp"])
    assert set(events["action"].unique()) <= {"created", "changed", "deleted"}
    assert set(events["direction"].unique()) <= {"bid", "ask"}


def test_trades_conform_to_schema():
    trades = _session().trades
    validate_trades_df(trades)
    for col in TRADE_COLUMNS:
        assert col in trades.columns
    assert set(trades["direction"].unique()) <= {"buy", "sell"}


def test_runs_through_full_pipeline():
    session = _session()
    result = _run_pipeline(session)
    # Every core table is populated and re-validates after the pipeline.
    validate_events_df(result.events)
    validate_trades_df(result.trades)
    assert len(result.events) > 0
    assert len(result.trades) > 0
    assert len(result.depth) > 0
    assert len(result.depth_summary) > 0
    assert "aggressiveness_bps" in result.events.columns
    # Depth (price-level resting volume) is never negative.
    assert (result.depth["volume"] >= 0).all()
    # The reconstructed touch is never crossed for this matched book.
    ds = result.depth_summary
    both = (ds["best_bid_price"] > 0) & (ds["best_ask_price"] > 0)
    assert (ds.loc[both, "best_bid_price"] < ds.loc[both, "best_ask_price"]).all()


def test_order_book_reconstruction_is_uncrossed():
    events = _session().events
    snapshot = order_book(events, uncross=False)
    bids, asks = snapshot["bids"], snapshot["asks"]
    if not bids.empty and not asks.empty:
        assert bids["price"].max() < asks["price"].min()
    # No duplicate active order ids (order_book raises otherwise).
    assert not bids["id"].duplicated().any()
    assert not asks["id"].duplicated().any()


def test_price_level_volume_non_negative():
    events = _session().events
    depth = price_level_volume(events)
    assert (depth["volume"] >= 0).all()


# ── (c) Structural invariants ─────────────────────────────────────────


def test_no_negative_volume_or_fill():
    events = _session().events
    assert (events["volume"] >= 0).all()
    assert (events["fill"] >= 0).all()


def test_outstanding_never_negative_and_consistent():
    """Reconstruct each order's outstanding size from its events.

    ``volume`` on a ``created`` / ``changed`` row is the outstanding size after
    the event; ``fill`` is the executed delta. Walking an order's events, the
    outstanding size must stay non-negative and equal ``placed - Σfill``.
    """
    events = _session().events.sort_values("event_id")
    for _oid, group in events.groupby("id", sort=False):
        rows = group.sort_values("event_id")
        head = rows.iloc[0]
        assert head["action"] == "created"
        assert head["fill"] == 0.0
        placed = head["volume"]
        cum_fill = 0.0
        for row in rows.iloc[1:].itertuples(index=False):
            cum_fill += row.fill
            outstanding = placed - cum_fill
            assert outstanding >= -1e-9
            if row.action == "changed":
                # A partial execution leaves a positive outstanding remainder.
                assert row.fill > 0
                assert abs(row.volume - outstanding) < 1e-6
                assert row.volume > 0
            else:  # deleted: full fill (fill>0, volume 0) or cancel (fill 0)
                if row.fill > 0:
                    assert abs(row.volume) < 1e-9
                    assert abs(outstanding) < 1e-6
                else:
                    assert abs(row.volume - outstanding) < 1e-6
        assert cum_fill <= placed + 1e-6


def test_changed_deleted_reference_a_prior_created():
    events = _session().events
    created_eid = events[events["action"] == "created"].groupby("id")["event_id"].min()
    non_created = events[events["action"] != "created"]
    # Every non-created event belongs to an order that was created.
    assert non_created["id"].isin(created_eid.index).all()
    # And the created event strictly precedes it.
    merged = non_created.merge(
        created_eid.rename("created_eid"), left_on="id", right_index=True
    )
    assert (merged["event_id"] > merged["created_eid"]).all()


def test_lifecycle_integrity():
    events = _session().events
    # Exactly one created per order.
    created_counts = events[events["action"] == "created"].groupby("id").size()
    assert (created_counts == 1).all()
    # At most one deleted, and it is the order's final event.
    deleted = events[events["action"] == "deleted"]
    assert (deleted.groupby("id").size() <= 1).all()
    last_eid = events.groupby("id")["event_id"].max()
    del_eid = deleted.groupby("id")["event_id"].max()
    assert (del_eid == last_eid.reindex(del_eid.index)).all()


def test_trades_reference_existing_events_and_orders():
    session = _session()
    events, trades = session.events, session.trades
    event_ids = set(events["event_id"])
    order_ids = set(events["id"])
    assert set(trades["maker_event_id"]).issubset(event_ids)
    assert set(trades["taker_event_id"]).issubset(event_ids)
    assert set(trades["maker"]).issubset(order_ids)
    assert set(trades["taker"]).issubset(order_ids)
    # A trade always has two distinct counterparties.
    assert (trades["maker"] != trades["taker"]).all()
    # Trade volume matches the fill recorded on the maker event it names.
    fill_by_eid = dict(zip(events["event_id"], events["fill"]))
    maker_fill = trades["maker_event_id"].map(fill_by_eid)
    assert np.allclose(maker_fill.to_numpy(), trades["volume"].to_numpy())


def test_no_unknown_order_types():
    events = _session().events
    assert (events["type"] != "unknown").all()
    # The default model produces exactly these three classes.
    assert set(events["type"].unique().astype(str)) <= {
        "flashed-limit",
        "resting-limit",
        "market",
    }


def test_passive_orders_do_not_cross():
    events = _session().events
    passive = events[events["type"].isin(["resting-limit", "flashed-limit"])]
    passive_bids = passive[passive["direction"] == "bid"]
    passive_asks = passive[passive["direction"] == "ask"]
    assert passive_bids["price"].max() < passive_asks["price"].min()


def test_lifecycle_outcomes_are_varied():
    """A healthy session exercises fills, partials, cancels, and resting."""
    events = _session().events
    outcomes = set(order_lifecycles(events)["outcome"].unique())
    assert {"filled", "cancelled"} <= outcomes


# ── Arrival processes ─────────────────────────────────────────────────


def test_hawkes_arrivals_valid_and_flow():
    session = _session(
        seed=3, arrival_process="hawkes", hawkes_excitation=0.6, hawkes_decay=1.5
    )
    validate_events_df(session.events)
    validate_trades_df(session.trades)
    result = _run_pipeline(session)
    assert len(result.depth) > 0


def test_hawkes_clusters_more_than_poisson():
    """Self-excitation packs more arrivals into the same span than Poisson."""
    poisson = _session(seed=8, arrival_process="poisson")
    hawkes = _session(
        seed=8, arrival_process="hawkes", hawkes_excitation=0.7, hawkes_decay=2.0
    )
    assert len(hawkes.events) > len(poisson.events)


# ── Optional injections (off by default) ──────────────────────────────


def test_injections_off_by_default():
    cfg = SynthConfig()
    assert cfg.iceberg_fraction == 0.0
    assert cfg.toxic_fraction == 0.0


def test_iceberg_injection_valid_and_flows():
    session = _session(seed=4, iceberg_fraction=0.25, iceberg_size_multiple=4.0)
    validate_events_df(session.events)
    result = _run_pipeline(session)
    assert len(result.depth) > 0
    # Invariants still hold with icebergs enabled.
    assert (session.events["volume"] >= 0).all()


def test_toxic_flow_injection_valid_and_flows():
    baseline = _session(seed=6, market_rate=2.0)
    toxic = _session(
        seed=6, market_rate=2.0, toxic_fraction=0.5, toxic_size_multiple=4.0
    )
    validate_trades_df(toxic.trades)
    _run_pipeline(toxic)
    # Toxic sweeps consume more, so total traded volume rises.
    assert toxic.trades["volume"].sum() > baseline.trades["volume"].sum()


# ── Config validation and edge cases ──────────────────────────────────


def test_config_is_frozen():
    cfg = SynthConfig()
    # A frozen dataclass raises FrozenInstanceError, a subclass of AttributeError.
    with pytest.raises(AttributeError):
        cfg.seed = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"duration": 0.0},
        {"level_decay": 0.0},
        {"level_decay": 1.5},
        {"depth_levels": 0},
        {"half_spread_ticks": 0},
        {"arrival_process": "hawkes", "hawkes_excitation": 1.0},
    ],
)
def test_invalid_config_raises(overrides):
    with pytest.raises(ConfigError):
        SynthConfig(**overrides)


def test_empty_session_raises():
    with pytest.raises(ConfigError, match="no events"):
        generate_session(
            SynthConfig(
                seed=0, duration=30.0, limit_rate=0.0, cancel_rate=0.0, market_rate=0.0
            )
        )


def test_loader_returns_independent_copy():
    session = _session(seed=2)
    loader = SyntheticLoader(session)
    frame = loader.load(source=None)
    frame.loc[frame.index[0], "price"] = -123.0
    # Mutating the loaded frame must not corrupt the stored session.
    assert session.events["price"].iloc[0] != -123.0
    trade_source = SyntheticTradeSource(session)
    tframe = trade_source.load(session.events, source=None)
    assert list(tframe.columns) == list(session.trades.columns)
