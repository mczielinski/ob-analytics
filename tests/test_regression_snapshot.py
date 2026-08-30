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

from ob_analytics.bitstamp import BitstampSource
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
    return Pipeline(source=BitstampSource()).run(
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
    # 2026-08-26 (#155, integer-tick prices): every price column is now int64
    # ticks instead of a double in the quote currency. The loader quantises the
    # raw cent-priced feed at tick_size 0.01, which is exactly the integer the
    # depth engine already computed internally, so the reconstruction is the
    # same book re-expressed: a former 133.30 float is now 13330 ticks. Verified
    # element-wise against the pre-tick float pipeline — ticks × 0.01 reproduces
    # the old events / depth / depth_summary prices to 0.0, and every volume /
    # direction / row is identical. DTYPE moved (double -> int64) and the numbers
    # moved (× 100), so all four digests changed.
    EXPECTED: dict[str, str] = {
        "events": "71f686a64b44fc215c092fcead1b960b80da6668a15919ff0be7f45c4e5c7d03",
        "trades": "54ec916a24c642cde2c6914561f7148057da30141b1974f422a8b63980a8cc7d",
        "depth": "b3fd876ed853b6f8df9d98c67c0ca374ddc34f335866e04762d4d7c4a3ebd764",
        "depth_summary": "6280117765ecedf6ed1a07399d75d609767a713f669788425c47beffdb24bf7c",
    }
    if EXPECTED:
        assert fps == EXPECTED


def test_kyle_lambda_baseline(demo_result):
    res = compute_kyle_lambda(demo_result.trades)
    # Record the baseline scalar; C4 (lstsq rewrite) must stay within rtol.
    print(f"KYLE_LAMBDA_BASELINE = {res.lambda_!r}")
    print(f"KYLE_RSQ_BASELINE = {res.r_squared!r}")
    # 2026-08-26 (#155, integer-tick prices): lambda is ΔPrice per unit signed
    # volume, and ΔPrice is now in ticks (int64) rather than dollars, so lambda
    # is exactly 100× the old 8.651849748125398 (tick_size 0.01). The regression
    # r_squared is scale-free and unchanged. Multiply by tick_size for the
    # quote-currency value.
    BASELINE_LAMBDA: float | None = 865.184974812539
    if BASELINE_LAMBDA is not None and not np.isnan(BASELINE_LAMBDA):
        assert np.isclose(res.lambda_, BASELINE_LAMBDA, rtol=1e-10, equal_nan=True)
