#!/usr/bin/env python3
"""LOBSTER data demo for ob-analytics.

Thin wrapper around :func:`ob_analytics._demos.run_lobster_demo`.

    uv run python scripts/lobster_demo.py path/to/lobster_dir --trading-date 2012-06-21
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
from loguru import logger

from ob_analytics._demos import run_lobster_demo


def main() -> None:
    matplotlib.use("Agg")
    logger.enable("ob_analytics")
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(description="LOBSTER data demo")
    parser.add_argument("source", help="LOBSTER data directory")
    parser.add_argument("--trading-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--output", default=None, help="Output directory (default: ./lobster_output)"
    )
    args = parser.parse_args()

    run_lobster_demo(args.source, args.trading_date, args.output)


if __name__ == "__main__":
    main()
