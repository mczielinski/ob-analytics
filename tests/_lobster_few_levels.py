"""LOBSTER files with one requested level, for the few-levels tests.

A message file holds only the events inside the requested levels.  An order
cancelled after it moved outside them keeps resting in the book rebuilt from
the messages, so that book can cross while the orderbook file does not.
"""

from __future__ import annotations

from pathlib import Path

STEM = "TEST_2024-01-02_34200000_57600000"

# Ask A @ 101.00, bid B @ 100.00, ask C @ 100.50 pushes A out of the level, C
# is deleted: A was cancelled out of sight and an ask at 102.00 the messages
# never named is now the best.  Bid E @ 101.50 then sits above A in the
# rebuilt book, from 34204 until E goes at 34210: 60% of the file's time.
CROSSING_MESSAGES = (
    "34200.0,1,1,100,1010000,-1\n"
    "34201.0,1,2,100,1000000,1\n"
    "34202.0,1,3,100,1005000,-1\n"
    "34203.0,3,3,100,1005000,-1\n"
    "34204.0,1,5,100,1015000,1\n"
    "34210.0,3,5,100,1015000,1\n"
)
CROSSING_ORDERBOOK = (
    "1010000,100,-9999999999,0\n"
    "1010000,100,1000000,100\n"
    "1005000,100,1000000,100\n"
    "1020000,100,1000000,100\n"
    "1020000,100,1015000,100\n"
    "1020000,100,1000000,100\n"
)
# The same orders, every one inside the level until it goes.
CLEAN_MESSAGES = (
    "34200.0,1,1,100,1010000,-1\n"
    "34201.0,1,2,100,1000000,1\n"
    "34202.0,3,1,100,1010000,-1\n"
    "34210.0,3,2,100,1000000,1\n"
)
CLEAN_ORDERBOOK = (
    "1010000,100,-9999999999,0\n"
    "1010000,100,1000000,100\n"
    "9999999999,0,1000000,100\n"
    "9999999999,0,-9999999999,0\n"
)


def write_files(directory: Path, messages: str, orderbook: str | None) -> Path:
    """Write a one-level message file, and the orderbook file unless ``None``."""
    (directory / f"{STEM}_message_1.csv").write_text(messages)
    if orderbook is not None:
        (directory / f"{STEM}_orderbook_1.csv").write_text(orderbook)
    return directory


def run(directory: Path):
    """Run the LOBSTER pipeline on the files in *directory*."""
    from ob_analytics.lobster import LobsterSource
    from ob_analytics.pipeline import Pipeline
    from ob_analytics.protocols import RunContext

    return Pipeline(
        source=LobsterSource(), ctx=RunContext(trading_date="2024-01-02")
    ).run(directory)
