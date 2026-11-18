"""Golden numbers for a LOBSTER input and a price-level (L2) input.

:mod:`test_golden_synth` pins a synthetic L3 session and
:mod:`test_regression_snapshot` pins the Bitstamp sample.  Neither reads a
LOBSTER file or a price-level feed, so a change that moves their numbers
passes as long as their invariant tests still hold.  This module pins the
exact output of one input of each kind, both already in the test suite:

* the five-message LOBSTER day of ``tests/conftest.py`` (``write_lobster_day``);
* the toy L2 session of :mod:`ob_analytics.datasets`, written as a
  ``depth_csv`` capture.

Each input is small enough to check by hand, so beside the fingerprints a few
tests check the numbers against what the input says.  Those checks do not use
the package to compute the expected value.

Regenerate :data:`LOBSTER_EXPECTED` or :data:`L2_EXPECTED` only when an output
change is intended: run this file with ``-s``, copy the printed
``FINGERPRINT`` lines, and say in the commit message why the numbers moved.
"""

from __future__ import annotations

import pytest

from ob_analytics.book import order_book, order_lifecycles
from ob_analytics.datasets import toy_l2_depth, toy_l2_trades
from ob_analytics.depth_csv import DepthCsvWriter
from ob_analytics.lobster import LobsterSource
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.protocols import Level, RunContext
from tests._golden import book_fingerprint, df_fingerprint
from tests.conftest import LOBSTER_DAY_DATE, write_lobster_day

# Recorded 2026-11-17 on main at 28fac29 (issue #504).
LOBSTER_EXPECTED: dict[str, str] = {
    "events": "eb5bc08532f464b090bb153dec2ee220321fab58bc069e0efb2c07903e8b5309",
    "trades": "6240b7eac8b42805b355b1e9bb1f26ce7fe861abd759f80b0cf404fdfc5f424f",
    "depth": "a523d2d890de215a35020a02f56b37e82269f60715b439aa1efc4bcb4ced05f0",
    "depth_summary": "3066c7ad452c6e428d87754cddd4668d52403aba17e348379443c69ff5d45474",
    "order_book": (
        "31b78642cc1e2c96b663d1d726929b804b693e0a40bf77da853e61d16aaea447:"
        "1106cdee8d88dc0c126e91be51b6f8a2b12bc6569dfd5fe65c10db34728d4950"
    ),
    "order_lifecycles": (
        "3def2ae023f3fdb0ca1e6f1efffb034e22ed3d9c1392c264d3c6d8c4641e51dc"
    ),
}

# Recorded 2026-11-17 on main at 28fac29 (issue #504).
L2_EXPECTED: dict[str, str] = {
    "events": "6b0ef16c50f1db17bb393ffd26ce8458a94b0fc013e8c18db86686215e6e3b8e",
    "trades": "202a230d7d94f432ef86c54900a2e1112f27c436a90dc5c07324d1b52bb78d7f",
    "depth": "ea20ff833421cec039fa21a3f57595f8b8715c012567764db8f8798f5203c985",
    "depth_summary": "b03849f94067bf35f39bb2232208f54175a698d652bfab1a30487852d65ae515",
}


@pytest.fixture(scope="module")
def lobster_result(tmp_path_factory: pytest.TempPathFactory) -> PipelineResult:
    directory = write_lobster_day(tmp_path_factory.mktemp("lobster"))
    pipeline = Pipeline(
        source=LobsterSource(), ctx=RunContext(trading_date=LOBSTER_DAY_DATE)
    )
    return pipeline.run(directory)


@pytest.fixture(scope="module")
def l2_result(tmp_path_factory: pytest.TempPathFactory) -> PipelineResult:
    directory = tmp_path_factory.mktemp("toy_l2")
    DepthCsvWriter().write(
        {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, directory
    )
    return Pipeline.from_source("depth_csv").run(directory)


def _frames(result: PipelineResult) -> dict[str, str]:
    return {name: df_fingerprint(frame) for name, frame in result.tables().items()}


def _print(fps: dict[str, str]) -> None:
    # Printed so an intended re-baseline can copy the new values straight in.
    for name, fp in fps.items():
        print(f"FINGERPRINT {name} = {fp}")


# ── LOBSTER ───────────────────────────────────────────────────────────


def test_lobster_fingerprints(lobster_result: PipelineResult) -> None:
    events = lobster_result.events
    fps = {
        **_frames(lobster_result),
        "order_book": book_fingerprint(order_book(events)),
        "order_lifecycles": df_fingerprint(order_lifecycles(events)),
    }
    _print(fps)
    assert fps == LOBSTER_EXPECTED


def test_lobster_trades_are_the_two_executions(lobster_result: PipelineResult) -> None:
    # The day's two type-4 messages: 40 shares of the ask at 101.00, then 25
    # shares of the bid at 100.00.  An execution of the ask is a buy.
    trades = lobster_result.trades
    tick = lobster_result.config.tick_size
    lot = lobster_result.config.lot_size
    assert lobster_result.level is Level.L3
    assert [round(p * tick, 2) for p in trades["price"]] == [101.0, 100.0]
    assert [round(v * lot) for v in trades["volume"]] == [40, 25]
    assert trades["direction"].astype(str).tolist() == ["buy", "sell"]


def test_lobster_touch_follows_the_orderbook_file(
    lobster_result: PipelineResult,
) -> None:
    # The orderbook file's best ask and bid after each message, in dollars.
    # After the last message the bid side is empty, so the touch is the ask.
    summary = lobster_result.depth_summary
    tick = lobster_result.config.tick_size
    last = summary.groupby("timestamp", sort=True).tail(1)
    asks = [round(p * tick, 2) for p in last["best_ask_price"]]
    assert asks[1:] == [101.0, 101.0, 101.0, 101.0]
    bids = [round(p * tick, 2) for p in last["best_bid_price"]]
    assert bids[:4] == [100.0, 100.0, 100.0, 100.0]


# ── Price-level (L2) ──────────────────────────────────────────────────


def test_l2_fingerprints(l2_result: PipelineResult) -> None:
    fps = _frames(l2_result)
    _print(fps)
    assert fps == L2_EXPECTED


def test_l2_run_keeps_every_update_and_trade(l2_result: PipelineResult) -> None:
    depth, trades = toy_l2_depth(), toy_l2_trades()
    assert l2_result.level is Level.L2
    assert l2_result.events.empty
    assert len(l2_result.depth) == len(depth)
    assert len(l2_result.trades) == len(trades)
    # The toy frames are in ticks and lots, and so is the run.
    for column in ("price", "volume"):
        assert l2_result.trades[column].tolist() == trades[column].tolist()
    assert l2_result.depth["volume"].sum() == depth["volume"].sum()


def test_l2_final_touch(l2_result: PipelineResult) -> None:
    # The toy session ends with a best bid of 99 ticks and a best ask of 100.
    last = l2_result.depth_summary.iloc[-1]
    assert last["best_bid_price"] == 99
    assert last["best_ask_price"] == 100
