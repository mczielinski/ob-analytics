"""Golden-output gate for the full L3 path (issue #143).

This locks the exact numbers a fixed synthetic L3 session produces through the
pipeline and the two per-order reconstructions the issue names —
:func:`~ob_analytics.analytics.order_book` and
:func:`~ob_analytics.queue.queue_positions` — alongside price-level ``depth``.
A change that shifts any value fails here.

This is the correctness bar the breaking schema changes (#154 UTC time, #155
integer-tick prices) and the engine / dataframe rewrites (#136, #138, #104)
must clear: they may make the code faster or the dtypes cleaner, but a number
must not move unless the move is intended and this baseline is re-recorded in a
separate, labelled commit.

Unlike the Bitstamp regression snapshot (:mod:`test_regression_snapshot`), the
input here is a seeded, license-free synthetic session
(:mod:`ob_analytics.synth`). So the gate runs anywhere and exercises the
L3-only outputs — per-order queue positions — the price-level Bitstamp sample
cannot.

Determinism
-----------
The session is driven by one seeded numpy ``Generator`` and the frames hash
their raw value buffers, so the fingerprints are stable across the CI matrix:
the pinned numpy / pandas, Python 3.11-3.13, and both ubuntu and macos runners
(verified to agree byte-for-byte). Regenerate :data:`EXPECTED` only when an
output change is intended — run this file, read the printed ``FINGERPRINT``
lines, and paste them back with a rationale in the commit message.
"""

from __future__ import annotations

import pytest

from ob_analytics.analytics import order_book
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.queue import queue_positions
from ob_analytics.synth import (
    SynthConfig,
    SyntheticLoader,
    SyntheticTradeSource,
    SynthSession,
    generate_session,
)
from tests._golden import book_fingerprint, df_fingerprint

# A small, fixed session: enough events to exercise every pipeline stage and
# both per-order reconstructions, small enough to fingerprint fast. The pinned
# config + seed produce byte-identical output every run.
_GOLDEN_CONFIG = SynthConfig(seed=143, duration=60.0)

# Recorded baseline. Update ONLY when an intended output change lands, in a
# separate labelled commit with a before/after rationale. Verified identical on
# CPython 3.11 / 3.12 / 3.13 with the pinned numpy / pandas.
#
# 2026-08-25 (#154, tz-aware UTC nanosecond time model): both clocks are now
# ``datetime64[ns, UTC]`` instead of tz-naive. The synthetic anchor
# (2020-01-01) is now labelled UTC, which is the same int64 nanosecond instant,
# so every value — including the timestamps — is unchanged; only the timestamp
# DTYPE moved. The fingerprint hashes dtype + values (and a tz-aware datetime
# column hashes through its string form), so all six digests changed even
# though no number did.
EXPECTED: dict[str, str] = {
    "events": "ca3a477a2b6767ffd06e1535f75f8ffadd48745843cd232bd4b13b08b92d92f6",
    "trades": "d2343b9e9245de7684218379d8d63b8d3ebbfed4ef5f931af47db44514add13a",
    "depth": "244ed099147b29519c0131442968a0be64c141ff52926c7eed135af4f52d45aa",
    "depth_summary": "4d01b99c0675e4a0b2c75e918a5e1f23fcea0407ba0a177030f3e70a2fd988a9",
    "order_book": (
        "0d21fa4b99d2035163b6c2a0c8c554df10af066b43fae05bed63d3bac1b68c24:"
        "55471e7232e0e48e67588de731b485459c1e0fa00b0ada263d924c6866eed831"
    ),
    "queue_positions": (
        "150f254da8e74369b053fa37e2878763dc023d598b21726d68b7cd337778fb82"
    ),
}


@pytest.fixture(scope="module")
def golden_session() -> SynthSession:
    return generate_session(_GOLDEN_CONFIG)


@pytest.fixture(scope="module")
def golden_result(golden_session: SynthSession) -> PipelineResult:
    return Pipeline(
        loader=SyntheticLoader(golden_session),
        trade_source=SyntheticTradeSource(golden_session),
    ).run(source=None)


def _fingerprints(result: PipelineResult) -> dict[str, str]:
    """The six golden fingerprints for one run: pipeline frames + book + queue."""
    events = result.events
    return {
        "events": df_fingerprint(result.events),
        "trades": df_fingerprint(result.trades),
        "depth": df_fingerprint(result.depth),
        "depth_summary": df_fingerprint(result.depth_summary),
        "order_book": book_fingerprint(order_book(events)),
        "queue_positions": df_fingerprint(queue_positions(events, levels="all")),
    }


def test_golden_session_is_populated(
    golden_session: SynthSession, golden_result: PipelineResult
) -> None:
    # Guard the gate itself: empty or trivial frames would make the fingerprint
    # comparison pass vacuously, so assert the session is genuinely rich first.
    assert len(golden_session.events) > 100
    assert len(golden_session.trades) > 0
    for name in ("events", "trades", "depth", "depth_summary"):
        frame = getattr(golden_result, name)
        assert frame is not None and len(frame) > 0, f"{name} empty"


def test_golden_fingerprints(golden_result: PipelineResult) -> None:
    fps = _fingerprints(golden_result)
    # Printed so an intended re-baseline can copy the new values straight in.
    for name, fp in fps.items():
        print(f"FINGERPRINT {name} = {fp}")
    assert fps == EXPECTED


def test_golden_session_is_reproducible() -> None:
    # The gate is only meaningful if the session is deterministic: a second
    # generation from the same config must fingerprint identically.
    session = generate_session(_GOLDEN_CONFIG)
    result = Pipeline(
        loader=SyntheticLoader(session),
        trade_source=SyntheticTradeSource(session),
    ).run(source=None)
    assert _fingerprints(result) == EXPECTED
