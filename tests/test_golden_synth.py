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
# 2026-08-26 (#155, integer-tick prices): every ``price`` column is now
# ``int64`` ticks instead of a ``double`` in the quote currency. The simulator
# already carried prices as integer ticks, so the values are the same book,
# re-expressed: a former ``100.00`` float is now ``10000`` ticks (tick_size
# 0.01). The DTYPE moved (double -> int64) and the stored numbers moved (× 100),
# so every price-bearing frame's digest changed. Scale-free columns (bps depth,
# order-book ``bps`` / ``liquidity``, volumes) are unchanged in value; the
# reconstructed book is identical to the pre-tick one (verified element-wise
# against the float pipeline: ticks × tick_size reproduces it exactly).
EXPECTED: dict[str, str] = {
    "events": "6e88626676163bb92c67953720467f7ef31c63b18a13b6c9cb5ccc89d4b3c8b2",
    "trades": "aaba2cc755bfcf91f980b0f0f4d2141a36fbca1a446a755ff7161b661c74c0d8",
    "depth": "1c9df9cbf685c8e125706a1687d04a81384fdb6e45b557e93c9fb9561d70576e",
    "depth_summary": "6781a478fd59b6d3176a86306f60f3df5f5345f2ec7bbb4ac0759f1d40efdb16",
    "order_book": (
        "c21d053a23abe1e879972308de6398930912929a04c7efb47add18a99d24fe69:"
        "e2828c08dd91fc4236c96cbd5b22813d18cd497401c8de6828be62820d2652ad"
    ),
    "queue_positions": (
        "841eadc3982b32aed360f51fc6fd3d5096226f9c84ccf3c290c34b9f9d3a6cf0"
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
