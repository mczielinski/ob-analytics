"""Command-line interface for ob-analytics.

Entry point registered as ``ob-analytics`` in pyproject.toml.

Usage::

    ob-analytics process orders.csv --output results/
    ob-analytics process data/ --source lobster --trading-date 2012-06-21
    ob-analytics process orders.csv --tick-size 0.001 --lot-size 0.0001
    ob-analytics process orders.csv --cut 2015-05-01T02:00 --output windows/
    ob-analytics audit orders.csv --strict
    ob-analytics gallery results/ --output my_gallery/ --backend plotly
    ob-analytics bitstamp-demo /path/to/dir_with_orders_and_trades/ --output demo_out/
    ob-analytics bitstamp-demo --view comparison   # L2-vs-L3 counterparts side by side
    ob-analytics lobster-demo /path/to/lobster_data --trading-date 2012-06-21 --output demo_out/
    ob-analytics capture bitstamp --pair btcusd --minutes 30 --output /tmp/capture
    ob-analytics capture ccxt --venue kraken --pair BTC/USD --output /tmp/kraken

``ob-analytics capture <source> --help`` lists the flags of that source's
settings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ob_analytics.config import PipelineConfig

__all__ = [
    "main",
]


def _setup_logging(verbose: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from loguru import logger

    from ob_analytics._secrets import redact_log_record

    logger.enable("ob_analytics")
    logger.remove()
    # No API key a source holds is logged, even inside an error message.
    logger.configure(patcher=redact_log_record)
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")


def _cmd_process(args: argparse.Namespace) -> None:
    """Run the pipeline on a data source and save results.

    A segmented live capture (a directory with ``manifest.json``) is
    processed one segment at a time, each into the folder of the same name
    under the output, and the manifest is copied alongside.
    """
    _setup_logging(args.verbose)
    from loguru import logger

    if args.cut and args.format != "parquet":
        logger.error("--cut writes Parquet; leave out --format {}", args.format)
        sys.exit(2)
    if args.cut and args.gallery:
        logger.error(
            "--cut does not build a gallery; run 'ob-analytics gallery' on its "
            "output instead"
        )
        sys.exit(2)
    if args.backend and not args.gallery:
        logger.error("--backend applies only with --gallery")
        sys.exit(2)
    if args.no_carry and not args.cut:
        logger.error("--no-carry applies only with --cut")
        sys.exit(2)
    root = Path(args.path)
    manifest = _read_capture_manifest(root)
    if manifest is None:
        _process_one(args, args.path, Path(args.output))
        return
    if _writes_a_file(args.format, _run_context(args)):
        logger.error(
            "A segmented capture is saved as one folder per segment, and "
            "--format {} writes one file; process one segment at a time to "
            "write it",
            args.format,
        )
        sys.exit(2)
    output = Path(args.output)
    seg_dirs = manifest.segment_dirs(root)
    _note_open_segments(manifest, "processed")
    if not seg_dirs:
        _no_segment_data(root, manifest)
    for seg_dir in seg_dirs:
        _process_one(args, str(_segment_input(seg_dir)), output / seg_dir.name)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_bytes((root / "manifest.json").read_bytes())


def _writes_a_file(fmt: str, ctx: Any) -> bool:
    """Whether the ``save_data`` format *fmt* writes one file, not a folder."""
    from ob_analytics.data import _named_writer
    from ob_analytics.protocols import WriteTarget

    try:
        writer = _named_writer(fmt, None, ctx)
    except Exception:  # noqa: BLE001 -- save_data reports it, with the run's config
        return False
    return getattr(writer, "target", None) == WriteTarget.FILE


def _note_open_segments(manifest: Any, verb: str) -> None:
    """Say which segments are left out because they are not closed yet."""
    from loguru import logger

    for segment in manifest.open_segments:
        logger.warning(
            "{} was not {}: it is still being captured (or its capture process "
            "died and has not been restarted), so it has no closing rows yet",
            segment.name,
            verb,
        )


def _no_segment_data(root: Path, manifest: Any) -> None:
    """Stop: a segmented capture in which no closed segment holds book rows."""
    from loguru import logger

    if manifest.open_segments:
        logger.error(
            "No closed segment of the capture in {} holds data yet. Check it "
            "once a segment has rolled or the capture has stopped.",
            root,
        )
    else:
        logger.error(
            "No segment of the capture in {} holds data: see its manifest.json "
            "for why each one ended.",
            root,
        )
    sys.exit(1)


def _segment_input(seg_dir: Path) -> Path:
    """What to read for one segment: its book file, or a saved result's folder.

    Some loaders take a directory and some only the file, so a capture
    segment is read through its ``orders.csv`` (L3) or ``depth.csv`` (L2), as
    a single capture is.  A ``process`` output has neither and is read whole.
    """
    from ob_analytics.capture_files import book_file

    return book_file(seg_dir) or seg_dir


def _read_capture_manifest(path: Path) -> Any:
    """The manifest of a segmented capture at *path*, or ``None``."""
    from loguru import logger

    from ob_analytics.exceptions import ConfigError
    from ob_analytics.live import read_manifest

    try:
        return read_manifest(path) if path.is_dir() else None
    except ConfigError as exc:
        logger.error(str(exc))
        sys.exit(1)


def _process_one(args: argparse.Namespace, data_path: str, output: Path) -> None:
    """Run the pipeline on one capture (or file) and save the results."""
    from loguru import logger

    from ob_analytics.data import save_data
    from ob_analytics.exceptions import ConfigError

    source_name = args.source
    pipeline = _build_pipeline(args, _build_source(source_name), source_name)

    result = None
    try:
        if args.cut:
            logger.info(
                "Processing {} (source={}) in {} windows...",
                data_path,
                source_name,
                len(args.cut) + 1,
            )
            pipeline.run_windows(data_path, args.cut, output, carry=not args.no_carry)
        else:
            logger.info("Processing {} (source={})...", data_path, source_name)
            result = pipeline.run(data_path)
            # A Parquet folder records the run's config, so `gallery` and
            # `audit --from-parquet` read the run back with its own units.
            save_data(result, output, fmt=args.format, ctx=_run_context(args))
    except ConfigError as exc:
        logger.error(str(exc))
        sys.exit(1)
    if output.is_dir():
        _keep_capture_record(Path(data_path), output)
    logger.info("Saved to: {}", output.resolve())

    if result is not None and args.gallery:
        gallery_dir = (
            output / "gallery"
            if output.is_dir()
            else output.with_name(f"{output.stem}_gallery")
        )
        _generate_gallery_from_result(
            result,
            gallery_dir,
            source_name,
            data_path,
            view=args.view,
            backends=args.backend,
        )


def _cmd_audit(args: argparse.Namespace) -> None:
    """Audit a run's data quality and exit non-zero when a check fails.

    A segmented live capture (a directory with ``manifest.json``, or the
    ``process`` output made from one) is audited one segment at a time, and
    the manifest adds the capture's own checks: gaps between segments,
    segments a dead process left open, dropped messages, errors that ended
    capturing early, and resyncs.  A single live capture (or its ``process``
    output) adds the dropped-message and error checks from its
    ``meta.json``.
    """
    _setup_logging(args.verbose)
    import json

    from loguru import logger

    from ob_analytics.live._manifest import record_checks

    if args.from_parquet:
        given = [flag for dest, flag in _RUN_FLAGS.items() if getattr(args, dest)]
        if given:
            logger.error(
                "{} set how the pipeline runs; a saved result keeps the "
                "settings of its own run, so leave them out with --from-parquet",
                ", ".join(given),
            )
            sys.exit(2)
    root = Path(args.path)
    manifest = _read_capture_manifest(root)
    capture_checks: tuple[Any, ...]
    if manifest is None:
        summary = _audit_one(args, root)
        capture_checks = record_checks(root)
        if args.json:
            report = summary.to_dict()
            if capture_checks:
                report["capture_checks"] = [c.to_dict() for c in capture_checks]
            print(json.dumps(report, indent=2))
        else:
            print(summary.render())
            failed_capture = [c for c in capture_checks if not c.passed]
            if failed_capture:
                print(f"Capture checks: {len(failed_capture)} warning(s)")
                for c in failed_capture:
                    print(f"  {c.severity.value.upper():<7} {c.name}: {c.detail}")
        named = [("", summary)]
    else:
        seg_dirs = manifest.segment_dirs(root)
        _note_open_segments(manifest, "checked")
        named = [(d.name, _audit_one(args, _segment_input(d))) for d in seg_dirs]
        capture_checks = manifest.checks()
        if args.json:
            report = {
                "capture": {
                    **manifest.to_dict(),
                    "checks": [c.to_dict() for c in capture_checks],
                },
                "segments": {name: summary.to_dict() for name, summary in named},
            }
            print(json.dumps(report, indent=2))
        else:
            for name, summary in named:
                print(f"== {name} ==\n{summary.render()}\n")
            print(manifest.render())
        if not seg_dirs:
            # Nothing to check is not a pass.
            _no_segment_data(root, manifest)

    failed = [
        f"{name + ': ' if name else ''}{c.name} — {c.detail}"
        for name, summary in named
        for c in summary.errors + (summary.warnings if args.strict else ())
    ]
    if args.strict:
        failed += [f"{c.name} — {c.detail}" for c in capture_checks if not c.passed]
    if failed:
        logger.error("Audit failed: {}", "; ".join(failed))
        sys.exit(1)


def _audit_one(args: argparse.Namespace, path: Path) -> Any:
    """Audit one capture, file or saved result; return its quality summary.

    The result carries the declarations it is checked against: its source's,
    with what a live capture recorded in their place.
    """
    from loguru import logger

    from ob_analytics.exceptions import ConfigError

    # Running the pipeline needs a source, so it falls back to the default one.
    # Reading a saved result does not: it records its own declarations, and
    # one that records none (an older output) stays undeclared without
    # --source rather than guessed, which keeps the crossing check honest
    # (see FeedType).
    source_name = args.source
    if source_name is None and not args.from_parquet:
        source_name = "bitstamp"
    source = _build_source(source_name) if source_name is not None else None

    if args.from_parquet:
        return _load_saved_result(path, source=source).audit()

    # Dropped-message detection reads the venue sequence, which is off by
    # default elsewhere.  Only the fields given are set, so the source's own
    # config defaults still apply to the rest.
    pipeline = _build_pipeline(args, source, source_name, track_sequence=True)
    logger.info("Auditing {} (source={})...", path, source_name)
    try:
        result = pipeline.run(path)
    except ConfigError as exc:
        logger.error(str(exc))
        sys.exit(1)
    # A source that supplies its own book (LOBSTER's orderbook file) lets the
    # rebuilt book be compared with it.
    read_venue_book = getattr(source, "venue_book", None)
    venue_book = None
    if read_venue_book is not None:
        venue_book = read_venue_book(path, result.config)
    return result.audit(venue_book=venue_book)


def _build_source(source_name: str) -> Any:
    """The registered source called *source_name*; exit when there is none."""
    from loguru import logger

    from ob_analytics.exceptions import UnknownNameError
    from ob_analytics.sources import get_source

    try:
        return get_source(source_name)()
    except UnknownNameError as exc:
        logger.error(str(exc))
        sys.exit(1)


#: The ``process`` and ``audit`` flags that set the run's config or context,
#: by ``argparse`` destination.  A saved result keeps its own, so ``audit
#: --from-parquet`` refuses them.
_RUN_FLAGS = {
    "trading_date": "--trading-date",
    "session_tz": "--session-tz",
    "tick_size": "--tick-size",
    "lot_size": "--lot-size",
    "depth_bins": "--depth-bins",
    "depth_bps": "--depth-bps",
}


def _run_context(args: argparse.Namespace) -> Any:
    """The run context the ``--trading-date`` and ``--session-tz`` flags give."""
    from ob_analytics.protocols import RunContext

    return RunContext(trading_date=args.trading_date, session_tz=args.session_tz)


def _run_config(args: argparse.Namespace, **fields: Any) -> PipelineConfig | None:
    """The config the run flags give, laid over *fields*; ``None`` when neither.

    Only the fields given are set, so the source's own defaults, and a
    capture's tick and lot size, still apply to the rest.
    """
    from ob_analytics.config import PipelineConfig, instrument_fields

    fields.update(instrument_fields(tick_size=args.tick_size, lot_size=args.lot_size))
    for name in ("depth_bins", "depth_bps"):
        if getattr(args, name) is not None:
            fields[name] = getattr(args, name)
    return PipelineConfig(**fields) if fields else None


def _build_pipeline(
    args: argparse.Namespace,
    source: Any,
    source_name: str,
    **fields: Any,
) -> Any:
    """The pipeline that ``process`` and ``audit`` run on *source*.

    The source says which run context it needs, so a missing
    ``--trading-date`` stops here with a message naming the source.  The
    config sets *fields* and the fields the run flags give; the source's
    defaults apply to the rest, and the pipeline takes a capture's tick and
    lot size from its ``meta.json`` unless ``--tick-size`` or ``--lot-size``
    is given.
    """
    from loguru import logger

    from ob_analytics.pipeline import Pipeline
    from ob_analytics.protocols import required_context_of

    required = required_context_of(source)
    if "trading_date" in required and args.trading_date is None:
        logger.error("--trading-date is required for the {} source", source_name)
        sys.exit(1)
    try:
        return Pipeline(
            _run_config(args, **fields), source=source, ctx=_run_context(args)
        )
    except TypeError as exc:  # e.g. a live-only source cannot replay files
        logger.error(str(exc))
        sys.exit(1)


def _keep_capture_record(data_path: Path, output: Path) -> None:
    """Copy a live capture's ``meta.json`` into the processed output.

    The record says which source made the capture, what that source
    declares about its feed, and the capture's counters.  The output's
    Parquet files already record the run's declarations; the copy keeps the
    rest of the record with the output.

    Input without a record removes any ``meta.json`` an earlier run left in
    *output*, so a reused output directory never carries another capture's
    record.
    """
    from ob_analytics.capture_record import RECORD_NAME, record_path

    meta = record_path(data_path)
    dest = output / RECORD_NAME
    if meta.is_file() and meta.resolve() == dest.resolve():
        return
    if meta.is_file():
        dest.write_bytes(meta.read_bytes())
    else:
        dest.unlink(missing_ok=True)


def _load_saved_result(data_path: Path, *, source: Any = None) -> Any:
    """Load a saved run (a ``process`` output) for ``gallery`` and ``audit``.

    *source* declares the feed of a run that records no declarations.
    """
    from loguru import logger

    from ob_analytics.data import load_result
    from ob_analytics.exceptions import ConfigError

    logger.info("Loading data from {}...", data_path)
    try:
        return load_result(data_path, source=source)
    except (ConfigError, FileNotFoundError) as exc:
        logger.error(str(exc))
        sys.exit(1)


def _cmd_gallery(args: argparse.Namespace) -> None:
    """Generate an HTML plot gallery from saved Parquet results."""
    _setup_logging(args.verbose)
    from loguru import logger

    from ob_analytics.visualization.gallery import generate_gallery

    data_path = Path(args.path)
    output = Path(args.output)

    result = _load_saved_result(data_path)

    gallery_path = generate_gallery(
        result,
        output,
        view=args.view,
        volume_scale=args.volume_scale,
        backends=args.backend,
        title=args.title or f"ob-analytics gallery -- {data_path.name}",
    )

    logger.info("Gallery: {}", gallery_path.resolve())
    logger.info("Open in browser: file://{}", gallery_path.resolve())


def _cmd_bitstamp_demo(args: argparse.Namespace) -> None:
    """Run the Bitstamp demo pipeline with gallery generation."""
    _setup_logging(args.verbose)
    from ob_analytics._demos import run_bitstamp_demo

    run_bitstamp_demo(args.path, args.output, view=args.view, roundtrip=args.roundtrip)


def _cmd_sources(args: argparse.Namespace) -> None:
    """List the registered sources: capability, required context and API keys."""
    from ob_analytics.config import SourceSettings
    from ob_analytics.exceptions import ConfigError
    from ob_analytics.live import LiveSource
    from ob_analytics.protocols import OfflineSource, required_context_of
    from ob_analytics.sources import SOURCES, list_sources

    for name in list_sources():
        try:
            source = SOURCES.get(name)()
        except ConfigError as exc:  # a key file that cannot be read, say
            print(f"{name}  [cannot be built: {exc}]")
            continue
        caps = []
        if isinstance(source, OfflineSource):
            caps.append("offline")
        if isinstance(source, LiveSource):
            caps.append("live")
        cap_str = "/".join(caps) if caps else "?"
        required = required_context_of(source)
        req_str = f", requires: {', '.join(required)}" if required else ""
        settings = getattr(source, "settings", None)
        keys = (
            [c.env for c in type(settings).credentials().values()]
            if isinstance(settings, SourceSettings)
            else []
        )
        key_str = f", key: {', '.join(keys)}" if keys else ""
        print(f"{name}  [{cap_str}{req_str}{key_str}]")


def _cmd_lobster_demo(args: argparse.Namespace) -> None:
    """Run the LOBSTER demo pipeline with gallery generation."""
    _setup_logging(args.verbose)
    from ob_analytics._demos import run_lobster_demo

    run_lobster_demo(args.path, args.trading_date, args.output, view=args.view)


def _cmd_capture(args: argparse.Namespace) -> None:
    """Run a live market-data capture.

    The flags after the source's name are read once the source is known: the
    capture's own, and one for each field of the source's settings class.
    """
    _setup_logging(args.verbose)
    import asyncio

    from loguru import logger

    from ob_analytics.exceptions import ConfigError, ObAnalyticsError, UnknownNameError
    from ob_analytics.live import (
        CaptureConfig,
        LiveSource,
        SupportsPreflight,
        run_capture,
    )
    from ob_analytics.sources import SOURCES, get_source, list_sources

    if args.list:
        for name in list_sources():
            if hasattr(SOURCES.get(name), "stream"):
                print(name)
        return

    if not args.source:
        logger.error(
            "source is required (e.g. 'bitstamp'). Use --list to see live sources."
        )
        sys.exit(2)

    try:
        source_cls = get_source(args.source)
    except UnknownNameError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # Building a source reads any API key from the environment; a key file
    # that cannot be read is user error, not a crash.  A pydantic
    # ValidationError is a ValueError.
    try:
        settings_cls = _settings_class(source_cls)
    except (ConfigError, ValueError) as exc:
        logger.error("Source {!r} cannot be built: {}", args.source, exc)
        sys.exit(1)
    try:
        parser = _capture_parser(args.source, settings_cls)
    except ConfigError as exc:  # a plug-in whose field clashes with a flag
        logger.error(str(exc))
        sys.exit(1)
    options = parser.parse_args(args.flags)

    given = {
        name: getattr(options, dest)
        for dest, name in _settings_dests(settings_cls).items()
        if hasattr(options, dest)
    }
    # A source holds one run's connection and book, so each segment of the
    # capture gets a fresh one.  Every segment reuses these settings, so a key
    # file moved or rotated during a long capture does not end it at the next
    # roll.  The registry is typed as `type[Source]`; the core protocol
    # declares no constructor, so passing settings is a checked-at-runtime
    # dynamic call (every built-in source accepts an optional `settings`).
    settings: Any = None
    if settings_cls is not None:
        try:
            settings = settings_cls(**given)
        except (ConfigError, ValueError) as exc:
            # A pydantic ValidationError is a ValueError.
            logger.error("Bad settings for source {!r}: {}", args.source, exc)
            sys.exit(2)

    def make_source() -> Any:
        if settings is None:
            return source_cls()
        return source_cls(settings=settings)  # ty: ignore[unknown-argument]

    try:
        source = make_source()
    except (ConfigError, ValueError) as exc:
        logger.error("Source {!r} cannot be built: {}", args.source, exc)
        sys.exit(1)

    # A venue that cannot supply the requested resolution is user error, not a
    # crash: report the explanation and stop.  (Reading ``level`` is what
    # resolves it, and the isinstance check below is the first read.)
    try:
        source_level = getattr(source, "level", None)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    logger.debug("Source {!r} captures at {}", args.source, source_level)

    if not isinstance(source, LiveSource):
        logger.error(
            "Source {!r} has no live capture; it is offline-only.", args.source
        )
        sys.exit(1)

    # A missing optional extra or an unusable venue stops the run here, before
    # the output directory exists (run_capture checks again for library use).
    if isinstance(source, SupportsPreflight):
        try:
            source.preflight()
        except (ImportError, ObAnalyticsError) as exc:
            logger.error(str(exc))
            sys.exit(1)

    config = CaptureConfig(
        pair=options.pair,
        out_dir=Path(options.output),
        minutes=options.minutes,
        keep_raw=not options.no_raw,
        roll_minutes=options.roll_minutes,
        roll_mb=options.roll_mb,
    )
    try:
        run = asyncio.run(run_capture(make_source, config))
    except ConfigError as exc:  # an --output that holds something else
        logger.error(str(exc))
        sys.exit(1)
    if run.error is not None:
        # The capture never streamed. The rows written before the error are
        # kept and the error is in the manifest; the run is still a failure.
        # This includes the venue or its settings being wrong for this run (a
        # location the venue refuses, a mirror it does not have).
        logger.error("Capture failed: {}; partial output in {}", run.error, run.out_dir)
        sys.exit(1)
    manifest = run.manifest
    if manifest.gaps:
        logger.warning(
            "Capture has {} gap(s), {:.1f} s in total: see {}",
            len(manifest.gaps),
            manifest.gap_seconds,
            run.out_dir / "manifest.json",
        )
    logger.info("Capture complete: {}", run.out_dir)


def _settings_class(source_cls: Any) -> Any:
    """The settings class the command line can set for *source_cls*, or ``None``.

    It is the class the constructor's ``settings`` parameter names, when that
    is a subclass of :class:`~ob_analytics.config.SourceSettings`.  Otherwise
    it is the type of a default source's settings, since the protocol
    requires the value, not a declaration of its class; a source whose
    settings have a field with no default should name its class, as it
    cannot build a default.  ``None`` when the source takes no ``settings``
    argument, or its settings are not a ``SourceSettings``.
    """
    import inspect
    import types
    from typing import Union, get_args, get_origin, get_type_hints

    from ob_analytics.config import SourceSettings

    try:
        parameters = inspect.signature(source_cls).parameters
    except (TypeError, ValueError):  # a constructor with no signature to read
        return None
    if "settings" not in parameters:
        return None
    try:
        hint = get_type_hints(source_cls.__init__).get("settings")
    except Exception:  # noqa: BLE001 -- a hint that cannot be resolved
        hint = None
    if get_origin(hint) in (Union, types.UnionType):
        named = [a for a in get_args(hint) if a is not type(None)]
        hint = named[0] if len(named) == 1 else None
    if (
        isinstance(hint, type)
        and issubclass(hint, SourceSettings)
        and hint is not SourceSettings
    ):
        return hint
    settings = getattr(source_cls(), "settings", None)
    return type(settings) if isinstance(settings, SourceSettings) else None


#: The prefix of a settings flag's ``argparse`` destination, so a field never
#: shares a destination with one of the capture's own flags.
_SETTING_DEST = "setting__"


def _settings_dests(settings_cls: Any) -> dict[str, str]:
    """The field each settings flag sets, by its ``argparse`` destination."""
    if settings_cls is None:
        return {}
    return {_SETTING_DEST + name: name for name in settings_cls.model_fields}


def _capture_parser(source_name: str, settings_cls: Any) -> argparse.ArgumentParser:
    """The parser for the flags after ``capture <source_name>``."""
    from ob_analytics.capture_files import (
        DEPTH_NAME,
        ORDERS_NAME,
        RAW_NAME,
        TRADES_NAME,
    )
    from ob_analytics.capture_record import RECORD_NAME
    from ob_analytics.live._manifest import MANIFEST_NAME

    parser = argparse.ArgumentParser(
        prog=f"ob-analytics capture {source_name}",
        description=f"Live-capture market data from the {source_name!r} source.",
    )
    parser.add_argument(
        "--pair",
        default="btcusd",
        help="Venue-specific pair symbol (default: btcusd)",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=10.0,
        help="Capture duration in minutes (default: 10.0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help=(
            f"Capture directory: {MANIFEST_NAME} plus one folder per segment "
            f"({ORDERS_NAME} for L3 or {DEPTH_NAME} for L2, plus {TRADES_NAME}, "
            f"{RAW_NAME}, {RECORD_NAME}). A directory that already holds a "
            "capture of the same source and pair is continued."
        ),
    )
    parser.add_argument(
        "--roll-minutes",
        type=float,
        default=None,
        help="Start a new segment every this many minutes (default: never)",
    )
    parser.add_argument(
        "--roll-mb",
        type=float,
        default=None,
        help=(
            "Start a new segment once one has written this many MB since its "
            "first live event (default: never)"
        ),
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        default=False,
        help=f"Don't write {RAW_NAME} (saves disk for long runs)",
    )
    if settings_cls is not None:
        _add_settings_flags(parser, settings_cls)
    return parser


def _add_settings_flags(parser: argparse.ArgumentParser, settings_cls: Any) -> None:
    """Add one flag to *parser* for each field of *settings_cls* it can read.

    A flag given sets its field; a flag left out leaves the field at the
    class's default.  A :class:`~ob_analytics.config.Credential` field gets
    no flag: a key on a command line stays in the shell history.
    """
    from pydantic_core import PydanticUndefined

    from ob_analytics.exceptions import ConfigError
    from ob_analytics.sources import CliOption

    group = parser.add_argument_group(f"source settings ({settings_cls.__name__})")
    credentials = settings_cls.credentials()
    for name, field in settings_cls.model_fields.items():
        if name in credentials:
            continue
        option = next((m for m in field.metadata if isinstance(m, CliOption)), None)
        kind = _flag_kind(field.annotation, option)
        if kind is None:  # a type the command line cannot read
            continue
        flag = "--" + ((option.flag if option else None) or name).replace("_", "-")
        help_text = (option.help if option else None) or field.description or name
        required = field.default is PydanticUndefined and field.default_factory is None
        if kind == "bool" and not required:
            help_text = f"{help_text} (default: {'on' if field.default else 'off'})"
        elif not required and field.default not in (None, ""):
            shown = getattr(field.default, "value", field.default)
            help_text = f"{help_text} (default: {shown})"
        kwargs: dict[str, Any] = {
            "dest": _SETTING_DEST + name,
            "help": help_text,
            "default": argparse.SUPPRESS,
            "required": required,
        }
        if kind == "bool":
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            convert, choices = kind
            kwargs["type"] = convert
            kwargs["metavar"] = (
                "{" + ",".join(choices) + "}"
                if choices is not None
                else flag.removeprefix("--").replace("-", "_").upper()
            )
        try:
            group.add_argument(flag, **kwargs)
        except argparse.ArgumentError:
            raise ConfigError(
                f"{settings_cls.__name__}.{name} would be the flag {flag}, which "
                "the capture command has already; give the field "
                "CliOption(flag=...) with another name"
            ) from None


def _flag_kind(annotation: Any, option: Any) -> Any:
    """How a flag reads a value of *annotation*.

    ``"bool"`` for an on/off flag, ``(convert, choices)`` for a flag that
    takes a value, where *choices* is ``None`` or the names to show, and
    ``None`` for a type the command line cannot read.  The marker's *type*
    wins over the annotation.
    """
    import types
    from enum import Enum
    from typing import Literal, Union, get_args, get_origin

    if option is not None and option.type is not None:
        return option.type, None
    if get_origin(annotation) in (Union, types.UnionType):
        members = [a for a in get_args(annotation) if a is not type(None)]
        if len(members) != 1:
            return None
        annotation = members[0]
    if annotation is bool:
        return "bool"
    if annotation in (str, int, float, Path):
        return annotation, None
    if get_origin(annotation) is Literal:
        values = get_args(annotation)
        return _choice_reader({str(v): v for v in values}), [str(v) for v in values]
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        members = {str(m.value): m for m in annotation}
        return _choice_reader(members), list(members)
    return None


def _choice_reader(values: dict[str, Any]) -> Any:
    """Read one of *values* by its name, ignoring case."""
    by_folded = {name.casefold(): value for name, value in values.items()}

    def read(text: str) -> Any:
        try:
            return by_folded[text.casefold()]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"{text!r} is not one of {', '.join(values)}"
            ) from None

    read.__name__ = "choice"
    return read


def _generate_gallery_from_result(
    result: Any,
    gallery_dir: Path,
    fmt_name: str,
    path: str,
    *,
    view: str = "both",
    backends: list[str] | None = None,
) -> None:
    """Helper to generate a gallery alongside process output.

    ``volume_scale`` is intentionally left to the gallery's auto-inference;
    the previous ``1e-8`` / ``1.0`` hard-codes leaked Bitstamp/LOBSTER
    conventions into the CLI.
    """
    from loguru import logger

    from ob_analytics.visualization.gallery import generate_gallery

    gallery_path = generate_gallery(
        result,
        gallery_dir,
        view=view,
        backends=backends,
        title=f"{fmt_name} ({Path(path).name}) -- ob-analytics",
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


def _add_backend_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--backend`` gallery argument."""
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        help=(
            "Plot backend to draw the gallery with (matplotlib, plotly, bokeh, "
            "or a registered plug-in); give it more than once for more columns. "
            "Default: plotly and matplotlib when plotly is installed, else "
            "matplotlib."
        ),
    )


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not value > 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not more than 0")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not more than 0")
    return value


def _time_zone(text: str) -> str:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a time zone name, such as America/New_York"
        ) from None
    return text


def _cut_time(text: str) -> Any:
    import pandas as pd

    try:
        return pd.Timestamp(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a time, such as 2015-05-01T02:00"
        ) from None


def _add_run_args(
    parser: argparse.ArgumentParser, sources: list[str], *, source_default: str | None
) -> None:
    """Add the flags ``process`` and ``audit`` share: the source and the run's settings."""
    parser.add_argument("path", help="Path to data file or directory")
    default_text = (
        f"default: {source_default}"
        if source_default
        else "default: bitstamp when running the pipeline"
    )
    parser.add_argument(
        "-s",
        "--source",
        default=source_default,
        choices=sources,
        help=f"Data source ({default_text}; registered: {', '.join(sources)})",
    )
    parser.add_argument(
        "--trading-date",
        default=None,
        help="Trading date, for a source that needs one (LOBSTER), as YYYY-MM-DD",
    )
    parser.add_argument(
        "--session-tz",
        type=_time_zone,
        default=None,
        help=(
            "The venue's time zone, for a feed whose times count from the "
            "session's midnight (LOBSTER; default: America/New_York)"
        ),
    )
    parser.add_argument(
        "--tick-size",
        type=_positive_float,
        default=None,
        help=(
            "The instrument's price step, in the quote currency (e.g. 0.001). "
            "Default: the source's, or a capture's own from its meta.json"
        ),
    )
    parser.add_argument(
        "--lot-size",
        type=_positive_float,
        default=None,
        help=(
            "The instrument's size step, in the base asset (e.g. 0.0001). "
            "Default: the source's, or a capture's own from its meta.json"
        ),
    )
    parser.add_argument(
        "--depth-bins",
        type=_positive_int,
        default=None,
        help="Number of depth bins on each side of the book (default: 20)",
    )
    parser.add_argument(
        "--depth-bps",
        type=_positive_int,
        default=None,
        help="Width of each depth bin, in basis points (default: 25)",
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``ob-analytics`` CLI.

    *argv* is the arguments after the command name; ``None`` reads
    ``sys.argv``.
    """
    from ob_analytics.data import list_writers
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
    _add_run_args(p_process, sources, source_default="bitstamp")
    p_process.add_argument(
        "-o",
        "--output",
        default="output",
        help=(
            "Where to save the results (default: ./output): a folder, or a "
            "file for a --format that writes one"
        ),
    )
    p_process.add_argument(
        "--format",
        default="parquet",
        choices=list_writers(),
        help="Format to save the results in (default: parquet)",
    )
    p_process.add_argument(
        "--cut",
        action="append",
        type=_cut_time,
        default=None,
        metavar="TIME",
        help=(
            "Run one time window at a time, cut at TIME (e.g. 2015-05-01T02:00; "
            "a time with no zone is UTC). Give it more than once for more "
            "windows. Peak memory is set by the largest window. Writes Parquet"
        ),
    )
    p_process.add_argument(
        "--no-carry",
        action="store_true",
        default=False,
        help=(
            "With --cut: start each window from an empty book, as if it were a "
            "separate input, instead of from the book the window before ended with"
        ),
    )
    p_process.add_argument(
        "--gallery",
        action="store_true",
        default=False,
        help="Also generate an HTML plot gallery",
    )
    _add_view_arg(p_process)
    _add_backend_arg(p_process)
    p_process.set_defaults(func=_cmd_process)

    # -- audit --
    p_audit = subparsers.add_parser(
        "audit",
        aliases=["validate"],
        help="Check a run's data quality; exit non-zero when a check fails",
    )
    _add_run_args(p_audit, sources, source_default=None)
    p_audit.add_argument(
        "--from-parquet",
        action="store_true",
        default=False,
        help=(
            "Read a saved 'process' output directory instead of running the "
            "pipeline. --source then only declares the feed of an output that "
            "records no declarations (one written by an older version); omit "
            "it to leave those undeclared"
        ),
    )
    p_audit.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail on warnings too, not only on errors",
    )
    p_audit.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit the summary as JSON instead of text",
    )
    p_audit.set_defaults(func=_cmd_audit)

    # -- gallery --
    p_gallery = subparsers.add_parser(
        "gallery",
        help="Generate an HTML plot gallery from saved Parquet data",
    )
    p_gallery.add_argument(
        "path",
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
    _add_backend_arg(p_gallery)
    p_gallery.set_defaults(func=_cmd_gallery)

    # -- bitstamp-demo --
    p_bs = subparsers.add_parser(
        "bitstamp-demo",
        help="Run the Bitstamp demo (pipeline + gallery)",
    )
    p_bs.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "Path to a Bitstamp orders.csv, or the folder holding it and its "
            "trades.csv (default: bundled sample data)"
        ),
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
        "path",
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
    # The flags after the source's name depend on the source, so they are
    # read by _cmd_capture once the source is known.
    p_cap = subparsers.add_parser(
        "capture",
        help="Live-capture market data from a registered source",
        description=(
            "Live-capture market data from a registered source. Run "
            "'ob-analytics capture <source> --help' for that source's flags: "
            "the capture's own (--pair, --minutes, -o/--output, ...) and one "
            "for each of its settings (for ccxt: --venue, --depth-limit, ...)."
        ),
    )
    p_cap.add_argument(
        "source",
        nargs="?",
        help="Source name (e.g. 'bitstamp', 'ccxt'). Use --list to see live sources.",
    )
    p_cap.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List live-capable sources and exit",
    )
    p_cap.add_argument("flags", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    p_cap.set_defaults(func=_cmd_capture)

    # -- sources --
    p_src = subparsers.add_parser(
        "sources",
        help="List registered sources, their capability (offline/live), and required context",
    )
    p_src.set_defaults(func=_cmd_sources)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
