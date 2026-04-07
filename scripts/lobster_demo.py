#!/usr/bin/env python3
"""LOBSTER data demo for ob-analytics.

Runs the full pipeline on locally available LOBSTER data, performs a
round-trip write/re-read verification, and generates a plot gallery.

Usage::

    uv run python scripts/lobster_demo.py /path/to/lobster_data --trading-date 2012-06-21
    uv run python scripts/lobster_demo.py /path/to/lobster_data --output ~/Desktop/lobster_gallery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
from loguru import logger

from ob_analytics.data import save_data
from ob_analytics.gallery import generate_gallery
from ob_analytics.lobster import LobsterFormat
from ob_analytics.pipeline import Pipeline


def main() -> None:
    matplotlib.use("Agg")
    logger.enable("ob_analytics")
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(description="LOBSTER data demo")
    parser.add_argument(
        "source",
        help="Path to LOBSTER data directory (containing message + orderbook CSVs)",
    )
    parser.add_argument(
        "--trading-date",
        required=True,
        help="Trading date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for gallery (default: ./lobster_output)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=10,
        help="Number of orderbook levels for round-trip write (default: 10)",
    )
    args = parser.parse_args()

    source = Path(args.source)
    trading_date: str = args.trading_date
    levels: int = args.levels

    if not source.exists():
        logger.error("Data path does not exist: {}", source)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path("lobster_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run pipeline
    logger.info("=" * 60)
    logger.info("LOBSTER Demo: {} ({})", source, trading_date)
    logger.info("=" * 60)

    fmt = LobsterFormat(trading_date=trading_date)
    pipeline = Pipeline(format=fmt)
    result = pipeline.run(source)

    logger.info("Events:  {:,}", len(result.events))
    logger.info("Trades:  {:,}", len(result.trades))
    logger.info("Depth:   {:,}", len(result.depth))
    logger.info("Metadata: {}", result.metadata)

    # 2. Save as parquet
    parquet_dir = output_dir / "parquet"
    result_dict = {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
    }
    save_data(result_dict, parquet_dir)
    logger.info("Parquet saved to: {}", parquet_dir)

    # 3. Round-trip: write as LOBSTER format, re-read, verify
    logger.info("Round-trip verification...")
    roundtrip_dir = output_dir / "roundtrip"
    save_data(
        result_dict,
        roundtrip_dir,
        writer=pipeline.writer,
        ticker="DATA",
        num_levels=levels,
    )
    logger.info("LOBSTER files written to: {}", roundtrip_dir)

    rt_pipeline = Pipeline(format=LobsterFormat(trading_date=trading_date))
    rt_result = rt_pipeline.run(roundtrip_dir)
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
        volume_scale=1.0,
        title=f"LOBSTER ({trading_date}) -- ob-analytics",
    )
    logger.info("Gallery: {}", gallery_path)
    logger.info("Open in browser: file://{}", gallery_path.resolve())

    logger.info("=" * 60)
    logger.info("Done! All output in: {}", output_dir.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
