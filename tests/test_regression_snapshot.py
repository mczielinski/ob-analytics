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

import hashlib

import numpy as np
import pandas as pd
import pytest

from ob_analytics.bitstamp import BitstampFormat
from ob_analytics.flow_toxicity import compute_kyle_lambda
from ob_analytics.pipeline import Pipeline, PipelineResult


def _df_fingerprint(df: pd.DataFrame) -> str:
    """Stable content hash of a DataFrame (column names + dtypes + values).

    Numeric and datetime columns hash their raw value buffers (IEEE-754 /
    int64 bytes, native order — stable across pandas/pyarrow versions, unlike
    CSV float formatting, and ~200x faster than a to_csv dump on the sample);
    object-like columns (categoricals, mixed id columns) hash a string join.
    """
    if df is None:
        return "None"
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update("|".join(map(str, df.dtypes)).encode())
    for col in df.columns:
        arr = df[col].to_numpy()
        if arr.dtype == object:
            h.update("\x1f".join(map(str, arr.tolist())).encode())
        else:
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


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
        str(bitstamp_sample_dir / "orders.csv")
    )


def test_demo_outputs_present(demo_result):
    for name, df in _frames(demo_result).items():
        assert df is not None and len(df) > 0, f"{name} empty"


def test_demo_fingerprints(demo_result):
    # First run prints fingerprints; paste them into EXPECTED below, then
    # the equality assertion locks the numeric baseline.
    fps = {name: _df_fingerprint(df) for name, df in _frames(demo_result).items()}
    for name, fp in fps.items():
        print(f"FINGERPRINT {name} = {fp}")
    # Update these ONLY when an intended output change lands (separate,
    # labeled commit with a before/after rationale).
    EXPECTED: dict[str, str] = {
        "events": "8d7a5c6e30fc0761cf28121bc3b910470d76716b80dbccfa24f54f83cd89db46",
        "trades": "e724c05b6584bfaf111f054e5066da8aec9d59e3315bb2f0a8ca38ae9696dc1a",
        "depth": "763e6ea8f00d6f88ae9e27cb3c89c8eaf28181297d1f01c35b371aebc37c75b0",
        "depth_summary": "453474cd31de6b6717ef71200a85c4efff6ef1f26b067002682fa8bd3bcfdf4e",
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
