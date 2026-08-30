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
        ctx.trading_date = "2013-01-01"  # type: ignore[misc]


def test_lobster_source_requires_trading_date_via_ctx():
    source = LobsterSource()
    with pytest.raises(ValueError, match="trading_date is required"):
        # ctx defaults to RunContext() with no trading_date — should fail
        # in create_loader()
        Pipeline(source=source)


def test_lobster_source_constructs_with_ctx():
    source = LobsterSource()
    ctx = RunContext(trading_date="2012-06-21")
    # Should not raise; we never actually call .run() here.
    pipeline = Pipeline(source=source, ctx=ctx)
    assert pipeline is not None


def test_lobster_source_rejects_wrong_trading_date_type():
    source = LobsterSource()
    ctx = RunContext(trading_date=12345)  # not str/Timestamp
    with pytest.raises(TypeError, match="trading_date must be str"):
        Pipeline(source=source, ctx=ctx)
