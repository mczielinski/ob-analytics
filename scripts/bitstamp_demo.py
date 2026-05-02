#!/usr/bin/env python3
"""Bitstamp data demo for ob-analytics.

Runs the full pipeline on Bitstamp-format `orders.csv` with a sibling
`trades.csv`, saves results as Parquet, performs a round-trip write/re-read
verification, and generates a plot gallery.

Usage::

    uv run python scripts/bitstamp_demo.py
    uv run python scripts/bitstamp_demo.py --input path/to/orders.csv
    uv run python scripts/bitstamp_demo.py --input path/to/run_dir
    uv run python scripts/bitstamp_demo.py --output ~/Desktop/bitstamp_gallery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from loguru import logger

from ob_analytics.bitstamp import BitstampFormat, BitstampWriter
from ob_analytics.data import save_data
from ob_analytics.visualization.gallery import generate_gallery
from ob_analytics.pipeline import Pipeline

_DEFAULT_CSV = (
    Path(__file__).resolve().parent.parent
    / "ob_analytics"
    / "_sample_data"
    / "orders.csv"
)


def _resolve_orders_path(path: Path) -> Path:
    """Return path to orders.csv (directory containing orders.csv is ok)."""
    if path.is_dir():
        return path / "orders.csv"
    return path


def _write_trades_csv_for_reader(trades: pd.DataFrame, dest: Path) -> None:
    """Write capture-style trades.csv so BitstampTradeReader can load after round-trip."""
    cols = [
        "trade_id",
        "timestamp",
        "exchange_timestamp",
        "price",
        "amount",
        "buy_order_id",
        "sell_order_id",
        "side",
    ]
    if trades.empty:
        pd.DataFrame(columns=cols).to_csv(dest, index=False)
        return

    side = trades["direction"].astype(str)
    buy_order_id = np.where(side == "buy", trades["taker"], trades["maker"])
    sell_order_id = np.where(side == "buy", trades["maker"], trades["taker"])
    ts_ns = trades["timestamp"].astype("datetime64[ns]").astype(np.int64)
    ts_ms = ts_ns // 1_000_000
    out = pd.DataFrame(
        {
            "trade_id": np.arange(1, len(trades) + 1, dtype=np.int64),
            "timestamp": ts_ms,
            "exchange_timestamp": ts_ms,
            "price": trades["price"].to_numpy(),
            "amount": trades["volume"].to_numpy(),
            "buy_order_id": buy_order_id,
            "sell_order_id": sell_order_id,
            "side": side.to_numpy(),
        }
    )
    out.to_csv(dest, index=False)


def main() -> None:
    matplotlib.use("Agg")
    logger.enable("ob_analytics")
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(description="Bitstamp data demo")
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to orders.csv, or directory containing orders.csv + trades.csv "
            f"(default: {_DEFAULT_CSV})"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: ./bitstamp_output)",
    )
    args = parser.parse_args()

    raw_input = Path(args.input) if args.input else _DEFAULT_CSV
    orders_path = _resolve_orders_path(raw_input)
    if not orders_path.exists():
        logger.error("orders.csv not found: {}", orders_path)
        logger.error(
            "Provide --input pointing to orders.csv or a directory that contains it "
            "(see {})",
            _DEFAULT_CSV.parent,
        )
        sys.exit(1)

    trades_path = orders_path.parent / "trades.csv"
    if not trades_path.exists():
        logger.error(
            "Companion trades.csv missing next to {}. "
            "Capture both with scripts/collect_bitstamp_btcusd.py.",
            orders_path.name,
        )
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path("bitstamp_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Bitstamp Demo: {}", orders_path.name)
    logger.info("=" * 60)
    logger.info("Orders:  {}", orders_path)
    logger.info("Trades:  {}", trades_path)

    # 1. Run pipeline
    logger.info("Running pipeline...")
    pipeline = Pipeline(format=BitstampFormat())
    result = pipeline.run(str(orders_path))

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

    # 3. Round-trip: write as Bitstamp CSV + trades, re-read, verify
    logger.info("Round-trip verification...")
    roundtrip_dir = output_dir / "roundtrip"
    roundtrip_dir.mkdir(parents=True, exist_ok=True)
    roundtrip_csv = roundtrip_dir / "orders.csv"
    save_data(result_dict, roundtrip_csv, writer=BitstampWriter())
    _write_trades_csv_for_reader(result.trades, roundtrip_dir / "trades.csv")
    logger.info("Bitstamp CSV written to: {}", roundtrip_csv)
    logger.info("Synthetic trades.csv written next to round-trip orders.")

    rt_result = Pipeline(format=BitstampFormat()).run(str(roundtrip_csv))
    logger.info("Re-read events:  {:,}", len(rt_result.events))
    logger.info("Re-read trades:  {:,}", len(rt_result.trades))

    events_match = len(result.events) == len(rt_result.events)
    trades_match = len(result.trades) == len(rt_result.trades)
    logger.info("Event count match: {}", "YES" if events_match else "NO")
    logger.info("Trade count match: {}", "YES" if trades_match else "NO")

    # 4. Generate gallery
    logger.info("Generating plot gallery...")
    gallery_dir = output_dir / "gallery"
    gallery_path = generate_gallery(
        result,
        gallery_dir,
        title=f"Bitstamp ({orders_path.name}) -- ob-analytics",
    )
    logger.info("Gallery: {}", gallery_path)
    logger.info("Open in browser: file://{}", gallery_path.resolve())

    logger.info("=" * 60)
    logger.info("Done! All output in: {}", output_dir.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
