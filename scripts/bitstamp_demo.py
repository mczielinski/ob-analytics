#!/usr/bin/env python3
"""Bitstamp data demo for ob-analytics.

Runs the full pipeline on a Bitstamp-format CSV, saves results as
Parquet, performs a round-trip write/re-read verification, and
generates a plot gallery.

Usage::

    uv run python scripts/bitstamp_demo.py
    uv run python scripts/bitstamp_demo.py --input path/to/orders.csv
    uv run python scripts/bitstamp_demo.py --output ~/Desktop/bitstamp_gallery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from loguru import logger

logger.enable("ob_analytics")
logger.remove()
logger.add(sys.stderr, level="INFO")

from ob_analytics.bitstamp import BitstampFormat, BitstampWriter
from ob_analytics.data import save_data
from ob_analytics.gallery import generate_gallery
from ob_analytics.pipeline import Pipeline

_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "inst" / "extdata" / "orders.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitstamp data demo")
    parser.add_argument(
        "--input",
        default=None,
        help=f"Path to Bitstamp-format CSV (default: {_DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: ./bitstamp_output)",
    )
    args = parser.parse_args()

    csv_path = Path(args.input) if args.input else _DEFAULT_CSV
    if not csv_path.exists():
        logger.error("CSV file not found: {}", csv_path)
        logger.error(
            "Provide a Bitstamp-format CSV with --input or place one at {}",
            _DEFAULT_CSV,
        )
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path("bitstamp_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Bitstamp Demo: {}", csv_path.name)
    logger.info("=" * 60)
    logger.info("Input: {}", csv_path)

    # 1. Run pipeline
    logger.info("Running pipeline...")
    pipeline = Pipeline(format=BitstampFormat())
    result = pipeline.run(str(csv_path))

    logger.info("Events:  {:,}", len(result.events))
    logger.info("Trades:  {:,}", len(result.trades))
    logger.info("Depth:   {:,}", len(result.depth))
    logger.info("Metadata: {}", result.metadata)

    # 2. Save as Parquet
    parquet_dir = output_dir / "parquet"
    result_dict = {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
    }
    save_data(result_dict, parquet_dir)
    logger.info("Parquet saved to: {}", parquet_dir)

    # 3. Round-trip: write as Bitstamp CSV, re-read, verify
    logger.info("Round-trip verification...")
    roundtrip_dir = output_dir / "roundtrip"
    roundtrip_csv = roundtrip_dir / "orders.csv"
    save_data(result_dict, roundtrip_csv, writer=BitstampWriter())
    logger.info("Bitstamp CSV written to: {}", roundtrip_csv)

    rt_result = Pipeline(format=BitstampFormat()).run(str(roundtrip_csv))
    logger.info("Re-read events:  {:,}", len(rt_result.events))
    logger.info("Re-read trades:  {:,}", len(rt_result.trades))

    events_match = len(result.events) == len(rt_result.events)
    logger.info("Event count match: {}", "YES" if events_match else "NO")

    # 4. Generate gallery
    logger.info("Generating plot gallery...")
    gallery_dir = output_dir / "gallery"
    gallery_path = generate_gallery(
        result,
        gallery_dir,
        volume_scale=1e-8,
        title=f"Bitstamp ({csv_path.name}) -- ob-analytics",
    )
    logger.info("Gallery: {}", gallery_path)
    logger.info("Open in browser: file://{}", gallery_path.resolve())

    logger.info("=" * 60)
    logger.info("Done! All output in: {}", output_dir.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
