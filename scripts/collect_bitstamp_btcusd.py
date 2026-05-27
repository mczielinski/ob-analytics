#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ob-analytics[live]",
# ]
# ///
"""Thin compatibility wrapper for the historical bitstamp capture script.

The actual implementation now lives in :mod:`ob_analytics.live.bitstamp`.
Prefer the new CLI verb:

    ob-analytics capture bitstamp --pair btcusd --minutes 30 --out DIR

This wrapper preserves the original ``--pair``, ``--minutes``, and ``--out``
flags so existing automation continues to work. The output directory layout
(orders.csv / trades.csv / raw.jsonl / meta.json) is unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from ob_analytics.live import get_capturer
from ob_analytics.live._base import CaptureConfig
from ob_analytics.live._runner import run_capturer


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    p = argparse.ArgumentParser(description="Bitstamp BTC/USD live capture")
    p.add_argument(
        "--minutes",
        type=float,
        default=10.0,
        help="Capture duration in minutes (default: 10).",
    )
    p.add_argument(
        "--pair",
        default="btcusd",
        help="Bitstamp currency pair, lowercase (default: btcusd).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "Desktop",
        help="Parent directory for the run folder (default: ~/Desktop).",
    )
    p.add_argument(
        "--no-raw",
        action="store_true",
        default=False,
        help="Don't write raw.jsonl.",
    )
    args = p.parse_args()

    out_dir = args.out / f"bitstamp_{args.pair}_{_utc_stamp()}"

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    capturer = get_capturer("bitstamp")()
    config = CaptureConfig(
        pair=args.pair,
        out_dir=out_dir,
        minutes=args.minutes,
        keep_raw=not args.no_raw,
    )

    try:
        asyncio.run(run_capturer(capturer, config))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
