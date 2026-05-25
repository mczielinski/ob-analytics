"""PipelineResult.extras: per-format auxiliary outputs."""

from pathlib import Path

from ob_analytics import LobsterFormat, Pipeline, sample_csv_path
from ob_analytics.protocols import RunContext


def test_bitstamp_run_produces_empty_extras():
    result = Pipeline().run(sample_csv_path())
    assert isinstance(result.extras, dict)
    assert result.extras == {}


def _write_lobster_message_with_halt_and_hidden(tmp_path: Path) -> Path:
    """LOBSTER message file with bid+ask depth, one halt (type=7), one
    cross (type=6) and one hidden execution (type=5) for extras testing.
    """
    p = tmp_path / "AAPL_2012-06-21_34200000_57600000_message_5.csv"
    p.write_text(
        # bid + ask opened together so depth_metrics has both sides
        "34200.000000000,1,11111,100,1000000,1\n"  # new bid @ $100
        "34200.000000000,1,22222,100,1010000,-1\n"  # new ask @ $101
        "34200.500000000,5,0,50,1005000,-1\n"  # hidden ask exec @ $100.5
        "34201.000000000,7,0,0,1000000,1\n"  # trading halt
        "34201.500000000,6,0,25,1000000,1\n"  # cross trade
        "34202.000000000,3,11111,100,1000000,1\n"  # delete the bid
        "34202.500000000,3,22222,100,1010000,-1\n"  # delete the ask
    )
    return p


def test_lobster_run_populates_extras(tmp_path: Path) -> None:
    msg = _write_lobster_message_with_halt_and_hidden(tmp_path)
    fmt = LobsterFormat()
    ctx = RunContext(trading_date="2012-06-21")
    result = Pipeline(format=fmt, ctx=ctx).run(msg)

    assert "trading_halts" in result.extras
    assert len(result.extras["trading_halts"]) == 1

    assert "cross_trades" in result.extras
    assert len(result.extras["cross_trades"]) == 1

    assert "hidden_executions" in result.extras
    assert len(result.extras["hidden_executions"]) == 1
    assert (result.extras["hidden_executions"]["raw_event_type"] == 5).all()


def test_lobster_run_omits_missing_extras(tmp_path: Path) -> None:
    """A LOBSTER file with no halts / crosses / hidden execs leaves those
    keys out of extras (rather than including empty frames)."""
    p = tmp_path / "AAPL_2012-06-21_34200000_57600000_message_5.csv"
    p.write_text(
        "34200.000000000,1,11111,100,1000000,1\n34201.000000000,3,11111,100,1000000,1\n"
    )
    fmt = LobsterFormat()
    ctx = RunContext(trading_date="2012-06-21")
    result = Pipeline(format=fmt, ctx=ctx).run(p)

    assert "trading_halts" not in result.extras
    assert "cross_trades" not in result.extras
    assert "hidden_executions" not in result.extras
