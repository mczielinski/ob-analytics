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
    """Stable content hash of a DataFrame (column names + dtypes + values)."""
    if df is None:
        return "None"
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update("|".join(map(str, df.dtypes)).encode())
    # to_parquet bytes are stable for identical content within a pyarrow version;
    # use a value-based CSV dump instead to be version-independent.
    h.update(df.to_csv(index=False).encode())
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
    EXPECTED: dict[str, str] = {
        "events": "0161445c1ea48ff76e4db98ecc1c0b7bc3bdf44700444d8b01c16da8f4011a65",
        "trades": "adcc4919350776d58988031395b929f5326853ff3d4b76389d2ca61a03d4384a",
        "depth": "f43bcdb57648fce327d91c94dac03648cd8a31e367852db84e1ccdd01101fb92",
        "depth_summary": "94d42be2c09571f82d50eb4751d341f3ef1c47b930f7119242a0f6123bf108bf",
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
