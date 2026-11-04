"""RunContext: per-run parameter container."""

from dataclasses import FrozenInstanceError

import pytest

from ob_analytics import LobsterSource, Pipeline
from ob_analytics.protocols import RunContext


def test_run_context_defaults_are_empty():
    ctx = RunContext()
    assert ctx.trading_date is None


def test_run_context_is_frozen():
    ctx = RunContext(trading_date="2012-06-21")
    with pytest.raises(FrozenInstanceError):
        ctx.trading_date = "2013-01-01"  # ty: ignore[invalid-assignment]


def test_lobster_source_requires_trading_date_via_ctx(tmp_path):
    # The loader is built when the run starts, from the run's context, so a
    # pipeline is built without one and the run with none fails.
    pipeline = Pipeline(source=LobsterSource())
    with pytest.raises(ValueError, match="trading_date is required"):
        pipeline.run(tmp_path)


def test_lobster_source_rejects_wrong_trading_date_type(tmp_path):
    ctx = RunContext(trading_date=12345)  # not str/Timestamp
    with pytest.raises(TypeError, match="trading_date must be str"):
        Pipeline(source=LobsterSource()).run(tmp_path, ctx=ctx)
