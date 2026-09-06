"""Golden-output gate for the full L3 path (issue #143).

This locks the exact numbers a fixed synthetic L3 session produces through the
pipeline and the two per-order reconstructions the issue names —
:func:`~ob_analytics.analytics.order_book`,
:func:`~ob_analytics.analytics.order_lifecycles`, and
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

from ob_analytics.analytics import order_book, order_lifecycles
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
# 2026-09-05 (#226, integer-lot sizes): every ``volume`` and ``fill`` column is
# now ``int64`` lots instead of a double in the base asset. The simulator now
# carries sizes the way it already carried prices — an exact integer on the
# instrument's grid — so its former ``_vol_eps`` tolerance is gone and an order
# is exhausted or it is not. Two things moved, and only one of them is a
# re-expression. The sizes are the same book scaled by 1/lot_size. The depth
# and the book are also *corrected*: a price level was a float running sum of
# adds, cancels and fills, and when the last order left, that sum landed on
# residue such as 5.55e-17 rather than 0, leaving the level live and reported
# as the best bid or ask ahead of the real one. Verified against an independent
# implementation rather than against ourselves: replaying the exported session
# through hftbacktest's own L3 reconstruction (#224) now agrees with
# depth_summary on the best bid and ask for every row across five seeds, where
# before it disagreed on up to 78 rows per seed. The derived size columns
# follow the same rule: every ``depth_summary`` volume column and
# ``filled_vol`` are int64 lot counts too, so their sums are exact — which is
# why ``filled_vol`` no longer carries the Kahan compensation it needed while
# sizes were floats. Sizes also cross the engine boundary as integers now, so
# the cumulative sums built from them — ``liquidity`` down a book side and
# ``ahead_volume`` along a queue — are exact as well. Those two are a dtype
# move only: summing the same integers in int64 rather than float64 gives the
# same numbers, and the fingerprint hashes dtype alongside values, so
# ``order_book`` and ``queue_positions`` changed digest without changing a
# number.
EXPECTED: dict[str, str] = {
    "events": "7be06840a399d9796b12a7e4144e44df9615a10804d316bd0bb6f39b9c752a01",
    "trades": "dfe87849f18a11294965564e23374563676aafadfb735df832737a33cdaba92e",
    "depth": "14259329e7614036e5aa8f642d2a903229c184f19cf665678dab2d71d417ea05",
    "depth_summary": "da5ff1da836e631cdf7fd985f388e5f70b35b443acf2d1ab4303f690910e39c6",
    "order_book": (
        "e4552ba2ff12c89af4ec608273d564364266c13446f361da53239581e6fdbb95:"
        "7902c4f9eabe3a88f35b989809ebf878cd092f9c24300e21a955f036142fb32e"
    ),
    "queue_positions": (
        "53ec1c0f45d83796a7ce18e7df0b64a13a682e5192b9da66d93e4a3f04000950"
    ),
    # 2026-08-31 (#136, engine separation): added, not re-baselined. Recorded
    # from the pre-#136 implementation and verified to still hold after the
    # move, so it pins the numbers the old code produced. The lifecycle table
    # had no fingerprint before, which is how a first pass at the engine lost
    # the compensated summation behind ``filled_vol`` without any gate noticing.
    "order_lifecycles": (
        "02dfb0836a90645d8303acc5c1d5c9eb4a714061d5d4959604f36b554b1b879d"
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
    """The golden fingerprints for one run: pipeline frames, book, lifecycles,
    and queue."""
    events = result.events
    return {
        "events": df_fingerprint(result.events),
        "trades": df_fingerprint(result.trades),
        "depth": df_fingerprint(result.depth),
        "depth_summary": df_fingerprint(result.depth_summary),
        "order_book": book_fingerprint(order_book(events)),
        "order_lifecycles": df_fingerprint(order_lifecycles(events)),
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
