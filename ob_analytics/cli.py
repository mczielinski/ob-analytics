"""Command-line interface for ob-analytics.

Entry point registered as ``ob-analytics`` in pyproject.toml.

Usage::

    ob-analytics process orders.csv --output results/
    ob-analytics process data/ --format lobster --trading-date 2012-06-21
    ob-analytics gallery results/parquet/ --output my_gallery/
    ob-analytics bitstamp-demo --input orders.csv --output demo_out/
    ob-analytics lobster-demo --ticker AAPL --output demo_out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _setup_logging(verbose: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from loguru import logger

    logger.enable("ob_analytics")
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")


def _cmd_process(args: argparse.Namespace) -> None:
    """Run the pipeline on a data source and save results."""
    _setup_logging(args.verbose)
    from loguru import logger

    from ob_analytics.config import PipelineConfig
    from ob_analytics.data import save_data
    from ob_analytics.event_processing import BitstampFormat
    from ob_analytics.lobster import LobsterFormat
    from ob_analytics.pipeline import Pipeline

    source = args.source
    fmt_name = args.format

    if fmt_name == "lobster":
        trading_date = args.trading_date
        if trading_date is None:
            logger.error("--trading-date is required for LOBSTER format")
            sys.exit(1)
        fmt = LobsterFormat(trading_date=trading_date)
    elif fmt_name == "bitstamp":
        fmt = BitstampFormat()
    else:
        logger.error("Unknown format {!r}", fmt_name)
        sys.exit(1)

    config_overrides: dict = {}
    if args.vpin_bucket_volume is not None:
        config_overrides["vpin_bucket_volume"] = args.vpin_bucket_volume

    config = PipelineConfig(**{**fmt.config_defaults(), **config_overrides})
    pipeline = Pipeline(config=config, format=fmt)

    logger.info("Processing {} (format={})...", source, fmt_name)
    result = pipeline.run(source)

    logger.info("Events: {:,}", len(result.events))
    logger.info("Trades: {:,}", len(result.trades))
    logger.info("Depth:  {:,}", len(result.depth))

    output = Path(args.output)
    result_dict = {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
    }
    save_data(result_dict, output)
    logger.info("Saved to: {}", output.resolve())

    if args.gallery:
        _generate_gallery_from_result(result, output, fmt_name, source)


def _cmd_gallery(args: argparse.Namespace) -> None:
    """Generate an HTML plot gallery from saved Parquet results."""
    _setup_logging(args.verbose)
    from loguru import logger

    from ob_analytics.data import load_data
    from ob_analytics.gallery import generate_gallery
    from ob_analytics.pipeline import PipelineResult

    data_path = Path(args.data)
    output = Path(args.output)

    logger.info("Loading data from {}...", data_path)
    data = load_data(data_path)

    result = PipelineResult(
        events=data["events"],
        trades=data["trades"],
        depth=data["depth"],
        depth_summary=data["depth_summary"],
        vpin=data.get("vpin"),
        ofi=data.get("ofi"),
    )

    gallery_path = generate_gallery(
        result,
        output,
        volume_scale=args.volume_scale,
        title=args.title or f"ob-analytics gallery -- {data_path.name}",
    )

    logger.info("Gallery: {}", gallery_path.resolve())
    logger.info("Open in browser: file://{}", gallery_path.resolve())


def _cmd_bitstamp_demo(args: argparse.Namespace) -> None:
    """Run the Bitstamp demo pipeline with gallery generation."""
    _setup_logging(args.verbose)
    from loguru import logger

    from ob_analytics.data import save_data
    from ob_analytics.gallery import generate_gallery
    from ob_analytics.pipeline import Pipeline

    default_csv = (
        Path(__file__).resolve().parent.parent / "inst" / "extdata" / "orders.csv"
    )

    csv_path = Path(args.input) if args.input else default_csv
    if not csv_path.exists():
        logger.error("CSV file not found: {}", csv_path)
        logger.error(
            "Provide a Bitstamp-format CSV with --input or place one at {}",
            default_csv,
        )
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path("bitstamp_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Bitstamp Demo: {}", csv_path.name)
    logger.info("=" * 60)

    pipeline = Pipeline()
    result = pipeline.run(str(csv_path))

    logger.info("Events: {:,}", len(result.events))
    logger.info("Trades: {:,}", len(result.trades))
    logger.info("Depth:  {:,}", len(result.depth))

    result_dict = {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
    }
    parquet_dir = output_dir / "parquet"
    save_data(result_dict, parquet_dir)
    logger.info("Parquet saved to: {}", parquet_dir)

    gallery_dir = output_dir / "gallery"
    gallery_path = generate_gallery(
        result,
        gallery_dir,
        volume_scale=1e-8,
        title=f"Bitstamp ({csv_path.name}) -- ob-analytics",
    )
    logger.info("Gallery: {}", gallery_path.resolve())
    logger.info("Open in browser: file://{}", gallery_path.resolve())
    logger.info("Done! All output in: {}", output_dir.resolve())


def _cmd_lobster_demo(args: argparse.Namespace) -> None:
    """Run the LOBSTER demo pipeline with gallery generation."""
    _setup_logging(args.verbose)
    from loguru import logger

    from ob_analytics.data import save_data
    from ob_analytics.gallery import generate_gallery
    from ob_analytics.lobster import LobsterFormat, download_sample
    from ob_analytics.pipeline import Pipeline

    ticker: str = args.ticker
    levels: int = args.levels
    trading_date = "2012-06-21"

    output_dir = Path(args.output) if args.output else Path("lobster_output") / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LOBSTER Demo: {} ({})", ticker, trading_date)
    logger.info("=" * 60)

    data_dir = download_sample(ticker=ticker, levels=levels)
    logger.info("Data directory: {}", data_dir)

    fmt = LobsterFormat(trading_date=trading_date)
    pipeline = Pipeline(format=fmt)
    result = pipeline.run(data_dir)

    logger.info("Events: {:,}", len(result.events))
    logger.info("Trades: {:,}", len(result.trades))
    logger.info("Depth:  {:,}", len(result.depth))

    result_dict = {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
    }
    parquet_dir = output_dir / "parquet"
    save_data(result_dict, parquet_dir)
    logger.info("Parquet saved to: {}", parquet_dir)

    gallery_dir = output_dir / "gallery"
    gallery_path = generate_gallery(
        result,
        gallery_dir,
        volume_scale=1.0,
        title=f"LOBSTER {ticker} ({trading_date}) -- ob-analytics",
    )
    logger.info("Gallery: {}", gallery_path.resolve())
    logger.info("Open in browser: file://{}", gallery_path.resolve())
    logger.info("Done! All output in: {}", output_dir.resolve())


def _generate_gallery_from_result(
    result: Any, output: Path, fmt_name: str, source: str
) -> None:
    """Helper to generate a gallery alongside process output."""
    from loguru import logger

    from ob_analytics.gallery import generate_gallery

    volume_scale = 1e-8 if fmt_name == "bitstamp" else 1.0
    gallery_dir = output / "gallery"
    gallery_path = generate_gallery(
        result,
        gallery_dir,
        volume_scale=volume_scale,
        title=f"{fmt_name} ({Path(source).name}) -- ob-analytics",
    )
    logger.info("Gallery: {}", gallery_path.resolve())
    logger.info("Open in browser: file://{}", gallery_path.resolve())


def main() -> None:
    """Entry point for the ``ob-analytics`` CLI."""
    parser = argparse.ArgumentParser(
        prog="ob-analytics",
        description="Limit order book analytics and visualization",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- process --
    p_process = subparsers.add_parser(
        "process",
        help="Run the pipeline on a data source and save results",
    )
    p_process.add_argument("source", help="Path to data file or directory")
    p_process.add_argument(
        "-f",
        "--format",
        default="bitstamp",
        choices=["bitstamp", "lobster"],
        help="Data format (default: bitstamp)",
    )
    p_process.add_argument(
        "-o",
        "--output",
        default="output",
        help="Output directory for Parquet results (default: ./output)",
    )
    p_process.add_argument(
        "--trading-date",
        default=None,
        help="Trading date for LOBSTER format (YYYY-MM-DD)",
    )
    p_process.add_argument(
        "--vpin-bucket-volume",
        type=float,
        default=None,
        help="Bucket volume for VPIN computation (omit to skip)",
    )
    p_process.add_argument(
        "--gallery",
        action="store_true",
        default=False,
        help="Also generate an HTML plot gallery",
    )
    p_process.set_defaults(func=_cmd_process)

    # -- gallery --
    p_gallery = subparsers.add_parser(
        "gallery",
        help="Generate an HTML plot gallery from saved Parquet data",
    )
    p_gallery.add_argument(
        "data",
        help="Path to Parquet directory (output of 'process')",
    )
    p_gallery.add_argument(
        "-o",
        "--output",
        default="gallery_output",
        help="Output directory for the gallery (default: ./gallery_output)",
    )
    p_gallery.add_argument(
        "--volume-scale",
        type=float,
        default=1.0,
        help="Volume display scale factor (default: 1.0)",
    )
    p_gallery.add_argument(
        "--title",
        default=None,
        help="Gallery page title",
    )
    p_gallery.set_defaults(func=_cmd_gallery)

    # -- bitstamp-demo --
    p_bs = subparsers.add_parser(
        "bitstamp-demo",
        help="Run the Bitstamp demo (pipeline + gallery)",
    )
    p_bs.add_argument(
        "--input",
        default=None,
        help="Path to Bitstamp CSV (default: inst/extdata/orders.csv)",
    )
    p_bs.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default: ./bitstamp_output)",
    )
    p_bs.set_defaults(func=_cmd_bitstamp_demo)

    # -- lobster-demo --
    p_lob = subparsers.add_parser(
        "lobster-demo",
        help="Download LOBSTER sample data and run the demo (pipeline + gallery)",
    )
    p_lob.add_argument(
        "--ticker",
        default="AAPL",
        choices=["AAPL", "AMZN", "GOOG", "INTC", "MSFT"],
        help="Ticker to download (default: AAPL)",
    )
    p_lob.add_argument(
        "--levels",
        type=int,
        default=10,
        help="Number of orderbook levels (default: 10)",
    )
    p_lob.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default: ./lobster_output/<ticker>)",
    )
    p_lob.set_defaults(func=_cmd_lobster_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
