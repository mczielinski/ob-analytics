"""Property-based tests for the canonical per-order volume semantics.

Hypothesis generates random-but-consistent LOBSTER message streams and
asserts the loader's derived invariants (schemas.py):

* ``volume`` is the outstanding size after the event (created/changed) or
  the size removed (deleted) — non-negative, and non-increasing across an
  order's post-submission life;
* ``fill`` is the executed delta, present only on execution rows;
* writer → loader round-trips preserve the raw message columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from ob_analytics.config import PipelineConfig
from ob_analytics.lobster import LobsterLoader, LobsterWriter

_CFG = PipelineConfig(price_decimals=2, price_divisor=10_000, volume_decimals=0)


@st.composite
def lobster_streams(draw) -> pd.DataFrame:
    """A consistent LOBSTER message stream over a handful of orders.

    Every order is submitted (type 1) before any reduction; reductions
    (partial cancel 2, executions 4/5) never exceed the outstanding size;
    at most one terminal deletion (type 3) of whatever remains.
    """
    n_orders = draw(st.integers(min_value=1, max_value=6))
    rows: list[tuple[float, int, int, int, int, int]] = []
    t = 34_200.0
    for oid in range(1, n_orders + 1):
        t += draw(st.floats(min_value=0.001, max_value=1.0))
        size = draw(st.integers(min_value=1, max_value=500)) * 100
        price = 5_000_000 + draw(st.integers(min_value=-50, max_value=50)) * 100
        direction = draw(st.sampled_from([1, -1]))
        rows.append((t, 1, oid, size, price, direction))
        outstanding = size
        n_reductions = draw(st.integers(min_value=0, max_value=4))
        for _ in range(n_reductions):
            if outstanding == 0:
                break
            t += draw(st.floats(min_value=0.001, max_value=1.0))
            etype = draw(st.sampled_from([2, 4, 5]))
            qty = draw(st.integers(min_value=1, max_value=max(outstanding // 100, 1)))
            qty *= 100
            qty = min(qty, outstanding)
            rows.append((t, etype, oid, qty, price, direction))
            outstanding -= qty
        if outstanding > 0 and draw(st.booleans()):
            t += draw(st.floats(min_value=0.001, max_value=1.0))
            rows.append((t, 3, oid, outstanding, price, direction))
    # Interleave across orders in time order (stable for equal times).
    rows.sort(key=lambda r: r[0])
    return pd.DataFrame(
        rows, columns=["time", "event_type", "id", "volume", "price", "direction"]
    )


def _load(stream: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    p = tmp_path / "GEN_2012-06-21_x_y_message_5.csv"
    stream.to_csv(p, index=False, header=False)
    return LobsterLoader(_CFG, trading_date="2012-06-21").load(p)


@settings(max_examples=80, deadline=None)
@given(stream=lobster_streams())
def test_outstanding_size_invariants(stream: pd.DataFrame, tmp_path_factory) -> None:
    tmp = tmp_path_factory.mktemp("hyp")
    events = _load(stream, tmp)

    assert (events["volume"] >= 0).all()
    assert (events["fill"] >= 0).all()
    # fill only on executions
    non_exec = ~events["raw_event_type"].isin([4, 5])
    assert (events.loc[non_exec, "fill"] == 0).all()

    for _oid, grp in events.groupby("id"):
        vols = grp["volume"].to_numpy()
        raw = grp["raw_size"].to_numpy()
        etypes = grp["raw_event_type"].to_numpy()
        # Submission row carries the full size.
        assert etypes[0] == 1
        assert vols[0] == raw[0]
        outstanding = vols[0]
        for k in range(1, len(grp)):
            if etypes[k] == 3:
                # Deleted rows report the size removed = outstanding before.
                assert vols[k] == outstanding
                outstanding = 0.0
            else:
                # Reductions: outstanding drops by exactly the raw delta.
                assert vols[k] == outstanding - raw[k]
                outstanding = vols[k]
            assert vols[k] >= 0


@settings(max_examples=40, deadline=None)
@given(stream=lobster_streams())
def test_writer_loader_roundtrip_preserves_messages(
    stream: pd.DataFrame, tmp_path_factory
) -> None:
    tmp = tmp_path_factory.mktemp("hyp_rt")
    events = _load(stream, tmp)

    writer = LobsterWriter(_CFG, trading_date="2012-06-21")
    msg = writer._events_to_message(events)

    # The written message must reproduce the source stream's rows.
    got = msg[["event_type", "id", "volume", "price", "direction"]].reset_index(
        drop=True
    )
    want = stream[["event_type", "id", "volume", "price", "direction"]].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(
        got.astype(np.int64), want.astype(np.int64), check_dtype=False
    )
    # Times survive to float seconds-after-midnight precision.
    np.testing.assert_allclose(
        msg["time"].to_numpy(), stream["time"].to_numpy(), rtol=0, atol=1e-6
    )
