#!/usr/bin/env python3
"""Bitstamp data demo for ob-analytics.

Thin wrapper around :func:`ob_analytics._demos.run_bitstamp_demo` for use
outside the installed CLI. Run via:

    uv run python scripts/bitstamp_demo.py
    uv run python scripts/bitstamp_demo.py --input path/to/orders.csv
    uv run python scripts/bitstamp_demo.py --output ~/Desktop/bitstamp_gallery
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
from loguru import logger

from ob_analytics._demos import run_bitstamp_demo


def main() -> None:
    matplotlib.use("Agg")
    logger.enable("ob_analytics")
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(description="Bitstamp data demo")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to orders.csv or a directory containing it (default: bundled sample)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: ./bitstamp_output)",
    )
    args = parser.parse_args()

    run_bitstamp_demo(args.input, args.output)


if __name__ == "__main__":
    main()
