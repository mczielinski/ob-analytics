"""Regression snapshots — guard behavior-preserving refactors (PR1).

These hash the Bitstamp demo's in-memory pipeline outputs and pin Kyle's
lambda to a recorded scalar. A refactor that changes any number fails
here. Update the recorded values ONLY when an output change is intended
and reviewed.

The pipeline runs exactly once (module-scoped fixture) and we fingerprint
the resulting DataFrames directly — no Parquet round-trip, no gallery, no
duplicate run — so this stays as cheap as a single pipeline pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics.bitstamp import BitstampFormat
from ob_analytics.flow_toxicity import compute_kyle_lambda
from ob_analytics.pipeline import Pipeline, PipelineResult
from tests._golden import df_fingerprint


def _frames(result: PipelineResult) -> dict[str, pd.DataFrame]:
    return {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
    }


@pytest.fixture(scope="module")
def demo_result(bitstamp_sample_dir) -> PipelineResult:
    return Pipeline(format=BitstampFormat()).run(
        str(bitstamp_sample_dir / "orders.csv.gz")
    )


def test_demo_outputs_present(demo_result):
    for name, df in _frames(demo_result).items():
        assert df is not None and len(df) > 0, f"{name} empty"


def test_demo_fingerprints(demo_result):
    # First run prints fingerprints; paste them into EXPECTED below, then
    # the equality assertion locks the numeric baseline.
    fps = {name: df_fingerprint(df) for name, df in _frames(demo_result).items()}
    for name, fp in fps.items():
        print(f"FINGERPRINT {name} = {fp}")
    # Update these ONLY when an intended output change lands (separate,
    # labeled commit with a before/after rationale).
    # 2026-06-12 (WS-1.1 PR 1/3): price_level_volume dropped the R-inherited
    # "changed & fill==0 adds full volume" branch.  On this sample it fired on
    # exactly two degenerate unknown-type rows (+0 and +1e-08 at one level),
    # so depth / depth_summary / events (via aggressiveness) shift by at most
    # 1e-08 at a single price level; trades are byte-identical (hash kept).
    # 2026-06-12 (correctness batch, WS-8.2): the 13 sample orders with no
    # created row are now typed "pre-existing" instead of "unknown" — only
    # the events type column moves; trades/depth/depth_summary hashes are
    # unchanged.
    # 2026-08-25 (#154, tz-aware UTC nanosecond time model): both clocks are
    # now datetime64[ns, UTC] instead of tz-naive datetime64[ms]. Bitstamp
    # epoch-milliseconds already count from the UTC epoch, so the reconstructed
    # instants are unchanged and depth / depth_summary values and row order are
    # byte-identical to before — only the timestamp DTYPE moved. The fingerprint
    # hashes dtype + values (tz-aware datetime hashes via its string form), so
    # all four digests changed even though no number did.
    EXPECTED: dict[str, str] = {
        "events": "d2d637dd45d4b61e3cd63d1321ceb5a1d9e5fe32998cdb79ae2a766d73262853",
        "trades": "24483443747e7cd1c21ef700733b90f9c17c9e54d1aa80cddf5fb1ede1df16d8",
        "depth": "15c2c34c57783af0ab3aeeb1c156e0a6dcd5a42ab456e9c25cece120be04cd0c",
        "depth_summary": "dc58f1c0838dcc6141910d5c9888cb0f1ce0b136b9d62d1daae7078f0ceb5fe2",
    }
    if EXPECTED:
        assert fps == EXPECTED


def test_kyle_lambda_baseline(demo_result):
    res = compute_kyle_lambda(demo_result.trades)
    # Record the baseline scalar; C4 (lstsq rewrite) must stay within rtol.
    print(f"KYLE_LAMBDA_BASELINE = {res.lambda_!r}")
    print(f"KYLE_RSQ_BASELINE = {res.r_squared!r}")
    BASELINE_LAMBDA: float | None = 8.651849748125398
    if BASELINE_LAMBDA is not None and not np.isnan(BASELINE_LAMBDA):
        assert np.isclose(res.lambda_, BASELINE_LAMBDA, rtol=1e-10, equal_nan=True)
