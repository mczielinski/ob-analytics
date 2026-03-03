#!/usr/bin/env python3
"""LOBSTER data demo for ob-analytics.

Downloads free LOBSTER sample data (AAPL, 2012-06-21), runs the full
pipeline, performs a round-trip write/re-read verification, and
generates a plot gallery.

Usage::

    uv run python scripts/lobster_demo.py
    uv run python scripts/lobster_demo.py --ticker MSFT
    uv run python scripts/lobster_demo.py --output ~/Desktop/lobster_gallery
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

from ob_analytics.data import save_data
from ob_analytics.gallery import generate_gallery
from ob_analytics.lobster import LobsterFormat, download_sample
from ob_analytics.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="LOBSTER data demo")
    parser.add_argument(
        "--ticker",
        default="AAPL",
        choices=["AAPL", "AMZN", "GOOG", "INTC", "MSFT"],
        help="Ticker to download (default: AAPL)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for gallery (default: ./lobster_output/<ticker>)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=10,
        help="Number of orderbook levels (default: 10)",
    )
    args = parser.parse_args()

    ticker: str = args.ticker
    levels: int = args.levels
    trading_date = "2012-06-21"

    output_dir = Path(args.output) if args.output else Path("lobster_output") / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download sample data
    print(f"\n{'='*60}")
    print(f"  LOBSTER Demo: {ticker} ({trading_date})")
    print(f"{'='*60}")

    data_dir = download_sample(ticker=ticker, levels=levels)
    print(f"  Data directory: {data_dir}")

    # 2. Run pipeline
    print(f"\n  Running pipeline...")
    fmt = LobsterFormat(trading_date=trading_date)
    pipeline = Pipeline(format=fmt)
    result = pipeline.run(data_dir)

    print(f"  Events:  {len(result.events):,}")
    print(f"  Trades:  {len(result.trades):,}")
    print(f"  Depth:   {len(result.depth):,}")
    print(f"  Metadata: {result.metadata}")

    # 3. Save as parquet
    parquet_dir = output_dir / "parquet"
    result_dict = {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
    }
    save_data(result_dict, parquet_dir)
    print(f"\n  Parquet saved to: {parquet_dir}")

    # 4. Round-trip: write as LOBSTER format, re-read, verify
    print(f"\n  Round-trip verification...")
    roundtrip_dir = output_dir / "roundtrip"
    save_data(
        result_dict,
        roundtrip_dir,
        writer=pipeline.writer,
        ticker=ticker,
        num_levels=levels,
    )
    print(f"  LOBSTER files written to: {roundtrip_dir}")

    rt_pipeline = Pipeline(format=LobsterFormat(trading_date=trading_date))
    rt_result = rt_pipeline.run(roundtrip_dir)
    print(f"  Re-read events:  {len(rt_result.events):,}")
    print(f"  Re-read trades:  {len(rt_result.trades):,}")

    events_match = len(result.events) == len(rt_result.events)
    print(f"  Event count match: {'YES' if events_match else 'NO'}")

    # 5. Generate gallery
    print(f"\n  Generating plot gallery...")
    gallery_dir = output_dir / "gallery"
    gallery_path = generate_gallery(
        result,
        gallery_dir,
        volume_scale=1.0,
        title=f"LOBSTER {ticker} ({trading_date}) -- ob-analytics",
    )
    print(f"\n  Gallery: {gallery_path}")
    print(f"  Open in browser: file://{gallery_path.resolve()}")

    print(f"\n{'='*60}")
    print(f"  Done! All output in: {output_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
