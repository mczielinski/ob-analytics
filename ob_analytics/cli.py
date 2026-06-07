"""Command-line interface for ob-analytics.

Entry point registered as ``ob-analytics`` in pyproject.toml.

Usage::

    ob-analytics process orders.csv --output results/
    ob-analytics process data/ --format lobster --trading-date 2012-06-21
    ob-analytics gallery results/parquet/ --output my_gallery/
    ob-analytics bitstamp-demo --input /path/to/dir_with_orders_and_trades/ --output demo_out/
    ob-analytics bitstamp-demo --view comparison   # L2-vs-L3 counterparts side by side
    ob-analytics lobster-demo /path/to/lobster_data --trading-date 2012-06-21 --output demo_out/
    ob-analytics capture bitstamp --pair btcusd --minutes 30 --out /tmp/capture
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
    from ob_analytics.pipeline import FORMATS, Pipeline

    from ob_analytics.protocols import RunContext

    source = args.source
    fmt_name = args.format

    try:
        fmt = FORMATS.get(fmt_name)()
    except KeyError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if fmt_name == "lobster":
        if args.trading_date is None:
            logger.error("--trading-date is required for LOBSTER format")
            sys.exit(1)
        ctx = RunContext(trading_date=args.trading_date)
    else:
        ctx = RunContext()

    config = PipelineConfig(**fmt.config_defaults())
    pipeline = Pipeline(config=config, format=fmt, ctx=ctx)

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
        _generate_gallery_from_result(result, output, fmt_name, source, view=args.view)


def _cmd_gallery(args: argparse.Namespace) -> None:
    """Generate an HTML plot gallery from saved Parquet results."""
    _setup_logging(args.verbose)
    from loguru import logger

    from ob_analytics.config import PipelineConfig
    from ob_analytics.data import load_data
    from ob_analytics.visualization.gallery import generate_gallery
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
        config=PipelineConfig(),
    )

    gallery_path = generate_gallery(
        result,
        output,
        view=args.view,
        volume_scale=args.volume_scale,
        title=args.title or f"ob-analytics gallery -- {data_path.name}",
    )

    logger.info("Gallery: {}", gallery_path.resolve())
    logger.info("Open in browser: file://{}", gallery_path.resolve())


def _cmd_bitstamp_demo(args: argparse.Namespace) -> None:
    """Run the Bitstamp demo pipeline with gallery generation."""
    _setup_logging(args.verbose)
    from ob_analytics._demos import run_bitstamp_demo

    run_bitstamp_demo(args.input, args.output, view=args.view)


def _cmd_lobster_demo(args: argparse.Namespace) -> None:
    """Run the LOBSTER demo pipeline with gallery generation."""
    _setup_logging(args.verbose)
    from ob_analytics._demos import run_lobster_demo

    run_lobster_demo(args.source, args.trading_date, args.output, view=args.view)


def _cmd_capture(args: argparse.Namespace) -> None:
    """Run a live market-data capture."""
    _setup_logging(args.verbose)
    import asyncio

    from loguru import logger

    from ob_analytics.live import get_capturer, list_capturers
    from ob_analytics.live._base import CaptureConfig
    from ob_analytics.live._runner import run_capturer

    if getattr(args, "list", False):
        registered = list_capturers()
        if not registered:
            logger.error(
                "No capturers registered. Install with: "
                'pip install "ob-analytics[live]"'
            )
            sys.exit(1)
        for name in registered:
            print(name)
        return

    if not args.venue:
        logger.error(
            "venue is required (e.g. 'bitstamp'). Use --list to see registered capturers."
        )
        sys.exit(2)
    if not args.out:
        logger.error("--out is required")
        sys.exit(2)

    try:
        capturer_cls = get_capturer(args.venue)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    capturer = capturer_cls()
    config = CaptureConfig(
        pair=args.pair,
        out_dir=Path(args.out),
        minutes=args.minutes,
        keep_raw=not args.no_raw,
    )
    result = asyncio.run(run_capturer(capturer, config))
    logger.info("Capture complete: {}", result.out_dir)


def _generate_gallery_from_result(
    result: Any, output: Path, fmt_name: str, source: str, *, view: str = "both"
) -> None:
    """Helper to generate a gallery alongside process output.

    ``volume_scale`` is intentionally left to the gallery's auto-inference;
    the previous ``1e-8`` / ``1.0`` hard-codes leaked Bitstamp/LOBSTER
    conventions into the CLI.
    """
    from loguru import logger

    from ob_analytics.visualization.gallery import generate_gallery

    gallery_dir = output / "gallery"
    gallery_path = generate_gallery(
        result,
        gallery_dir,
        view=view,
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
        "--gallery",
        action="store_true",
        default=False,
        help="Also generate an HTML plot gallery",
    )
    p_process.add_argument(
        "--view",
        default="both",
        choices=["l2", "l3", "both", "comparison"],
        help=(
            "Gallery view: resolution level(s) to render "
            "(l2|l3|both|comparison; default: both)"
        ),
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
        default=None,
        help=(
            "Volume display scale factor. Omit to auto-infer a "
            "power-of-10 scale from the data."
        ),
    )
    p_gallery.add_argument(
        "--title",
        default=None,
        help="Gallery page title",
    )
    p_gallery.add_argument(
        "--view",
        default="both",
        choices=["l2", "l3", "both", "comparison"],
        help=(
            "Gallery view: resolution level(s) to render "
            "(l2|l3|both|comparison; default: both)"
        ),
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
        help="Path to Bitstamp CSV (default: bundled sample data)",
    )
    p_bs.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default: ./bitstamp_output)",
    )
    p_bs.add_argument(
        "--view",
        default="both",
        choices=["l2", "l3", "both", "comparison"],
        help=(
            "Gallery view: resolution level(s) to render "
            "(l2|l3|both|comparison; default: both)"
        ),
    )
    p_bs.set_defaults(func=_cmd_bitstamp_demo)

    # -- lobster-demo --
    p_lob = subparsers.add_parser(
        "lobster-demo",
        help="Run the LOBSTER demo on local data (pipeline + gallery)",
    )
    p_lob.add_argument(
        "source",
        help="Path to LOBSTER data directory (containing message + orderbook CSVs)",
    )
    p_lob.add_argument(
        "--trading-date",
        required=True,
        help="Trading date for LOBSTER format (YYYY-MM-DD)",
    )
    p_lob.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default: ./lobster_output)",
    )
    p_lob.add_argument(
        "--view",
        default="both",
        choices=["l2", "l3", "both", "comparison"],
        help=(
            "Gallery view: resolution level(s) to render "
            "(l2|l3|both|comparison; default: both)"
        ),
    )
    p_lob.set_defaults(func=_cmd_lobster_demo)

    # -- capture --
    p_cap = subparsers.add_parser(
        "capture",
        help="Live-capture market data from a registered venue",
    )
    p_cap.add_argument(
        "venue",
        nargs="?",
        help=("Venue name (e.g. 'bitstamp'). Use --list to see registered capturers."),
    )
    p_cap.add_argument(
        "--pair",
        default="btcusd",
        help="Venue-specific pair symbol (default: btcusd)",
    )
    p_cap.add_argument(
        "--minutes",
        type=float,
        default=10.0,
        help="Capture duration in minutes (default: 10.0)",
    )
    p_cap.add_argument(
        "--out",
        default=None,
        help=(
            "Output directory (will contain orders.csv, trades.csv, "
            "raw.jsonl, meta.json). Required unless --list."
        ),
    )
    p_cap.add_argument(
        "--no-raw",
        action="store_true",
        default=False,
        help="Don't write raw.jsonl (saves disk for long runs)",
    )
    p_cap.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List registered capturers and exit (ignores other flags)",
    )
    p_cap.set_defaults(func=_cmd_capture)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
