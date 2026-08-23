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
EXPECTED: dict[str, str] = {
    "events": "4e9c94b6f4db79f0db62eba4907ef4387b7b594b9d78cfebd193ddd43a91c2d3",
    "trades": "c1d71ced8f7d833fb4e4aa08254534548639c3adf2a43a8ff0e7d809e0b608e0",
    "depth": "3d7051dd205fb9923f6b4f89e6ce576719445a92ad0e2581fc1ec2f1d3868629",
    "depth_summary": "9d34f0752fd3932611201bfe43a84ed78eae9415a74b3a158ef313a76c58895b",
    "order_book": (
        "14d7fbc508d5e74440608f277e57e611079954fd220731a8c231cef597cd6a9e:"
        "21566e282378953977fd4ee7e61c086e915dfae733a59a3e847eea1db4e7b1ff"
    ),
    "queue_positions": (
        "37914a3e88186a96e6735cee3e09209e8652c70f9261e7ae7db7f18ce298f1de"
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
