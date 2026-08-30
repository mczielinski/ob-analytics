"""Command-line interface for ob-analytics.

Entry point registered as ``ob-analytics`` in pyproject.toml.

Usage::

    ob-analytics process orders.csv --output results/
    ob-analytics process data/ --source lobster --trading-date 2012-06-21
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

    from ob_analytics.data import save_data
    from ob_analytics.pipeline import Pipeline
    from ob_analytics.protocols import RunContext
    from ob_analytics.sources import get_source

    data_path = args.path
    source_name = args.source

    try:
        source = get_source(source_name)()
    except KeyError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # Ask the source what RunContext it needs rather than hard-coding "lobster".
    required = getattr(source, "required_context", list)()
    if "trading_date" in required and args.trading_date is None:
        logger.error("--trading-date is required for the %s source", source_name)
        sys.exit(1)
    ctx = (
        RunContext(trading_date=args.trading_date)
        if args.trading_date is not None
        else RunContext()
    )

    try:
        pipeline = Pipeline(source=source, ctx=ctx)
    except TypeError as exc:  # e.g. a live-only source cannot replay files
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Processing {} (source={})...", data_path, source_name)
    result = pipeline.run(data_path)

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
    # Pass the config so each Parquet file is tagged with its tick size
    # (issue #155), letting a reader recover the quote-currency price.
    save_data(result_dict, output, config=result.config)
    logger.info("Saved to: {}", output.resolve())

    if args.gallery:
        _generate_gallery_from_result(
            result, output, source_name, data_path, view=args.view
        )


def _cmd_validate(args: argparse.Namespace) -> None:
    """Run the pipeline and print a per-run data-quality summary."""
    _setup_logging(args.verbose)
    from loguru import logger

    from ob_analytics.analytics import data_quality_summary
    from ob_analytics.pipeline import Pipeline
    from ob_analytics.protocols import FeedType, RunContext
    from ob_analytics.sources import get_source

    try:
        source = get_source(args.source)()
    except KeyError as exc:
        logger.error(str(exc))
        sys.exit(1)

    required = getattr(source, "required_context", list)()
    if "trading_date" in required and args.trading_date is None:
        logger.error("--trading-date is required for the {} source", args.source)
        sys.exit(1)
    ctx = (
        RunContext(trading_date=args.trading_date)
        if args.trading_date is not None
        else RunContext()
    )

    try:
        pipeline = Pipeline(source=source, ctx=ctx)
    except TypeError as exc:  # e.g. a live-only source cannot replay files
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Validating {} (source={})...", args.path, args.source)
    result = pipeline.run(args.path)

    summary = data_quality_summary(
        result.events,
        result.trades,
        feed_type=getattr(source, "feed_type", FeedType.UNKNOWN),
        depth=result.depth,
    )

    if args.json:
        import json

        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(summary.render())


def _cmd_gallery(args: argparse.Namespace) -> None:
    """Generate an HTML plot gallery from saved Parquet results."""
    _setup_logging(args.verbose)
    from loguru import logger

    from ob_analytics.config import PipelineConfig
    from ob_analytics.data import load_data
    from ob_analytics.pipeline import PipelineResult
    from ob_analytics.visualization.gallery import generate_gallery

    data_path = Path(args.data)
    output = Path(args.output)

    logger.info("Loading data from {}...", data_path)
    data = load_data(data_path)

    # Recover the tick size the data was written with (issue #155), surfaced by
    # load_data on each frame's ``attrs``, so the gallery renders quote-currency
    # prices.  A legacy (pre-#155) file has no tick size and already stores float
    # prices, so fall back to 1.0 (prices shown as-is).
    tick_size = next(
        (df.attrs["tick_size"] for df in data.values() if "tick_size" in df.attrs),
        1.0,
    )

    result = PipelineResult(
        events=data["events"],
        trades=data["trades"],
        depth=data["depth"],
        depth_summary=data["depth_summary"],
        config=PipelineConfig(tick_size=tick_size),
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

    run_bitstamp_demo(args.input, args.output, view=args.view, roundtrip=args.roundtrip)


def _cmd_sources(args: argparse.Namespace) -> None:
    """List the registered sources: capability (offline/live) and required context."""
    from ob_analytics.sources import SOURCES, list_sources

    for name in list_sources():
        source = SOURCES.get(name)()
        caps = []
        if hasattr(source, "create_loader"):
            caps.append("offline")
        if hasattr(source, "stream"):
            caps.append("live")
        cap_str = "/".join(caps) if caps else "?"
        required = getattr(source, "required_context", list)()
        req_str = f", requires: {', '.join(required)}" if required else ""
        print(f"{name}  [{cap_str}{req_str}]")


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

    from ob_analytics.live import CaptureConfig, LiveSource
    from ob_analytics.live._runner import run_capturer
    from ob_analytics.live.ccxt_source import CcxtSettings
    from ob_analytics.sources import SOURCES, get_source, list_sources

    def _is_live(name: str) -> bool:
        return hasattr(SOURCES.get(name), "stream")

    if getattr(args, "list", False):
        for name in list_sources():
            if _is_live(name):
                print(name)
        return

    if not args.venue:
        logger.error(
            "venue is required (e.g. 'bitstamp'). Use --list to see live sources."
        )
        sys.exit(2)
    if not args.out:
        logger.error("--out is required")
        sys.exit(2)

    try:
        source_cls = get_source(args.venue)
    except KeyError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # Build typed per-source settings. Only ccxt takes venue knobs today; a
    # source with no knobs is constructed with its empty default settings.
    if args.venue.lower() == "ccxt":
        ccxt_kwargs: dict[str, Any] = {}
        if getattr(args, "exchange", None):
            ccxt_kwargs["exchange"] = args.exchange
        if getattr(args, "depth_limit", None) is not None:
            ccxt_kwargs["depth_limit"] = args.depth_limit
        if getattr(args, "poll_interval", None) is not None:
            ccxt_kwargs["poll_interval"] = args.poll_interval
        # The registry is typed as `type[Source]`; the core protocol declares
        # no constructor, so passing settings is a checked-at-runtime dynamic
        # call (every built-in source accepts an optional `settings`).
        source = source_cls(settings=CcxtSettings(**ccxt_kwargs))  # ty: ignore[unknown-argument]
    else:
        source = source_cls()

    if not isinstance(source, LiveSource):
        logger.error("Source %r has no live capture; it is offline-only.", args.venue)
        sys.exit(1)

    config = CaptureConfig(
        pair=args.pair,
        out_dir=Path(args.out),
        minutes=args.minutes,
        keep_raw=not args.no_raw,
    )
    result = asyncio.run(run_capturer(source, config))
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


def _add_view_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--view`` gallery-resolution argument."""
    parser.add_argument(
        "--view",
        default="both",
        choices=["l2", "l3", "both", "comparison"],
        help=(
            "Gallery view: resolution level(s) to render "
            "(l2|l3|both|comparison; default: both)"
        ),
    )


def main() -> None:
    """Entry point for the ``ob-analytics`` CLI."""
    from ob_analytics.sources import list_sources

    sources = list_sources()

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
    p_process.add_argument("path", help="Path to data file or directory")
    p_process.add_argument(
        "-s",
        "--source",
        default="bitstamp",
        choices=sources,
        help=f"Data source (default: bitstamp; registered: {', '.join(sources)})",
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
        help="Trading date for the LOBSTER source (YYYY-MM-DD)",
    )
    p_process.add_argument(
        "--gallery",
        action="store_true",
        default=False,
        help="Also generate an HTML plot gallery",
    )
    _add_view_arg(p_process)
    p_process.set_defaults(func=_cmd_process)

    # -- validate --
    p_validate = subparsers.add_parser(
        "validate",
        help="Report per-run data-quality metrics for a data source",
    )
    p_validate.add_argument("path", help="Path to data file or directory")
    p_validate.add_argument(
        "-s",
        "--source",
        default="bitstamp",
        choices=sources,
        help=f"Data source (default: bitstamp; registered: {', '.join(sources)})",
    )
    p_validate.add_argument(
        "--trading-date",
        default=None,
        help="Trading date for the LOBSTER source (YYYY-MM-DD)",
    )
    p_validate.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit the summary as JSON instead of text",
    )
    p_validate.set_defaults(func=_cmd_validate)

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
    _add_view_arg(p_gallery)
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
        "--roundtrip",
        action="store_true",
        default=False,
        help="Write then re-read the result to verify the round-trip (slower)",
    )
    _add_view_arg(p_bs)
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
    _add_view_arg(p_lob)
    p_lob.set_defaults(func=_cmd_lobster_demo)

    # -- capture --
    p_cap = subparsers.add_parser(
        "capture",
        help="Live-capture market data from a registered venue",
    )
    p_cap.add_argument(
        "venue",
        nargs="?",
        help=("Venue name (e.g. 'bitstamp'). Use --list to see live sources."),
    )
    p_cap.add_argument(
        "--pair",
        default="btcusd",
        help="Venue-specific pair symbol (default: btcusd)",
    )
    p_cap.add_argument(
        "--exchange",
        default=None,
        help=(
            "For the 'ccxt' venue: the CCXT exchange id (e.g. 'binance', "
            "'kraken', 'coinbase'). Use CCXT pair notation, e.g. 'BTC/USDT'."
        ),
    )
    p_cap.add_argument(
        "--depth-limit",
        type=int,
        default=None,
        help="ccxt: order-book depth (levels per side) to request",
    )
    p_cap.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="ccxt: seconds between REST polls (REST-only venues)",
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
            "Output directory (orders.csv for L3 or depth.csv for L2, plus "
            "trades.csv, raw.jsonl, meta.json). Required unless --list."
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
        help="List live-capable sources and exit (ignores other flags)",
    )
    p_cap.set_defaults(func=_cmd_capture)

    # -- sources --
    p_src = subparsers.add_parser(
        "sources",
        help="List registered sources, their capability (offline/live), and required context",
    )
    p_src.set_defaults(func=_cmd_sources)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
