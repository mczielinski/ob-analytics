"""RunContext: per-run parameter container."""

import pytest

from ob_analytics import LobsterFormat, Pipeline
from ob_analytics.protocols import RunContext


def test_run_context_defaults_are_empty():
    ctx = RunContext()
    assert ctx.trading_date is None


def test_run_context_is_frozen():
    ctx = RunContext(trading_date="2012-06-21")
    with pytest.raises(Exception):
        ctx.trading_date = "2013-01-01"  # type: ignore[misc]


def test_lobster_format_requires_trading_date_via_ctx():
    fmt = LobsterFormat()
    with pytest.raises(ValueError, match="trading_date is required"):
        # ctx defaults to RunContext() with no trading_date — should fail
        # in create_loader()
        Pipeline(format=fmt)


def test_lobster_format_constructs_with_ctx():
    fmt = LobsterFormat()
    ctx = RunContext(trading_date="2012-06-21")
    # Should not raise; we never actually call .run() here.
    pipeline = Pipeline(format=fmt, ctx=ctx)
    assert pipeline is not None


def test_lobster_format_rejects_wrong_trading_date_type():
    fmt = LobsterFormat()
    ctx = RunContext(trading_date=12345)  # not str/Timestamp
    with pytest.raises(TypeError, match="trading_date must be str"):
        Pipeline(format=fmt, ctx=ctx)
