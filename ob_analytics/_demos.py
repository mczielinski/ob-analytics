"""Internal demo runners shared between the ob-analytics CLI and scripts/.

Both ``scripts/bitstamp_demo.py`` and ``cli.py:_cmd_bitstamp_demo`` (and the
LOBSTER equivalents) are thin wrappers around these functions, so the
"what a full pipeline run looks like" logic lives in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from ob_analytics.bitstamp import BitstampFormat, BitstampWriter
from ob_analytics.data import save_data
from ob_analytics.lobster import LobsterFormat
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.protocols import RunContext
from ob_analytics.visualization.gallery import (
    PlotSpec,
    generate_gallery,
    trading_halts_panel,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _result_dict(result: PipelineResult) -> dict[str, pd.DataFrame]:
    return {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
    }


def _save_and_gallery(
    result: PipelineResult,
    output_dir: Path,
    title: str,
    extra_panels: list[PlotSpec] | None = None,
) -> Path:
    """Save Parquet + generate gallery; return the gallery HTML path."""
    parquet_dir = output_dir / "parquet"
    save_data(_result_dict(result), parquet_dir)
    logger.info("Parquet saved to: {}", parquet_dir)

    gallery_dir = output_dir / "gallery"
    gallery_path = generate_gallery(
        result, gallery_dir, title=title, extra_panels=extra_panels
    )
    logger.info("Gallery: {}", gallery_path.resolve())
    logger.info("Open in browser: file://{}", gallery_path.resolve())
    return gallery_path


# ---------------------------------------------------------------------------
# Bitstamp demo
# ---------------------------------------------------------------------------


def _resolve_orders_path(path: Path) -> Path:
    """Return path to orders.csv (directory containing orders.csv is ok)."""
    return path / "orders.csv" if path.is_dir() else path


def run_bitstamp_demo(
    input_path: Path | str | None,
    output_dir: Path | str | None,
    *,
    roundtrip: bool = True,
) -> Path:
    """Run the Bitstamp demo pipeline.

    Parameters
    ----------
    input_path
        Path to orders.csv, a directory containing it, or None to use the
        bundled sample data.
    output_dir
        Where to write parquet + gallery. Defaults to ./bitstamp_output.
    roundtrip
        If True, perform a write/re-read verification step.

    Returns
    -------
    Path to the generated gallery HTML.
    """
    default_csv = Path(__file__).resolve().parent / "_sample_data" / "orders.csv"
    raw_input = Path(input_path) if input_path else default_csv
    orders_path = _resolve_orders_path(raw_input)
    if not orders_path.exists():
        raise FileNotFoundError(
            f"orders.csv not found: {orders_path}. "
            f"Provide --input or place one at {default_csv}"
        )
    trades_path = orders_path.parent / "trades.csv"
    if not trades_path.exists():
        raise FileNotFoundError(
            f"Companion trades.csv missing next to {orders_path.name}. "
            f"Capture both with scripts/collect_bitstamp_btcusd.py."
        )

    out = Path(output_dir) if output_dir else Path("bitstamp_output")
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Bitstamp Demo: {}", orders_path.name)
    logger.info("=" * 60)

    pipeline = Pipeline(format=BitstampFormat())
    result = pipeline.run(str(orders_path))

    logger.info("Events: {:,}", len(result.events))
    logger.info("Trades: {:,}", len(result.trades))
    logger.info("Depth:  {:,}", len(result.depth))

    if roundtrip:
        rt_dir = out / "roundtrip"
        rt_dir.mkdir(parents=True, exist_ok=True)
        rt_csv = rt_dir / "orders.csv"
        # BitstampWriter emits the companion trades.csv from the "trades"
        # frame in _result_dict, so no separate shim is needed here.
        save_data(_result_dict(result), rt_csv, writer=BitstampWriter())
        rt = Pipeline(format=BitstampFormat()).run(str(rt_csv))
        logger.info(
            "Round-trip events: {} (match: {})",
            len(rt.events),
            len(rt.events) == len(result.events),
        )

    return _save_and_gallery(
        result,
        out,
        title=f"Bitstamp ({orders_path.name}) -- ob-analytics",
    )


# ---------------------------------------------------------------------------
# LOBSTER demo
# ---------------------------------------------------------------------------


def run_lobster_demo(
    source: Path | str,
    trading_date: str,
    output_dir: Path | str | None,
) -> Path:
    """Run the LOBSTER demo pipeline.

    Returns the path to the generated gallery HTML.
    """
    src = Path(source)
    out = Path(output_dir) if output_dir else Path("lobster_output")
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LOBSTER Demo: {} ({})", src, trading_date)
    logger.info("=" * 60)

    fmt = LobsterFormat()
    ctx = RunContext(trading_date=trading_date)
    pipeline = Pipeline(format=fmt, ctx=ctx)
    result = pipeline.run(str(src))

    logger.info("Events: {:,}", len(result.events))
    logger.info("Trades: {:,}", len(result.trades))
    logger.info("Depth:  {:,}", len(result.depth))

    # LOBSTER halts are not part of the slim PipelineResult; read them off the
    # loader and overlay them on the gallery via extra_panels.
    halts = getattr(pipeline.loader, "trading_halts", None)
    extra_panels: list[PlotSpec] = []
    if halts is not None and not halts.empty:
        extra_panels.append(trading_halts_panel(result.trades, halts))

    return _save_and_gallery(
        result,
        out,
        title=f"LOBSTER {src.name} ({trading_date}) -- ob-analytics",
        extra_panels=extra_panels,
    )
