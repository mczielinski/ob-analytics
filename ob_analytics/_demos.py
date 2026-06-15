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
    build_gallery_model,
    generate_gallery,
    ofi_horizon_panel,
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
    *,
    analytics: list[PlotSpec] | None = None,
    view: str = "both",
) -> Path:
    """Save Parquet + generate gallery; return the gallery HTML path.

    *analytics* are level-less panels (built with the ``*_panel`` helpers)
    appended to the model's :attr:`~...gallery.GalleryModel.analytics`.
    """
    parquet_dir = output_dir / "parquet"
    save_data(_result_dict(result), parquet_dir)
    logger.info("Parquet saved to: {}", parquet_dir)

    model = None
    if analytics:
        model = build_gallery_model(result)
        model.analytics.extend(analytics)

    gallery_dir = output_dir / "gallery"
    gallery_path = generate_gallery(
        result, gallery_dir, model=model, view=view, title=title
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
    roundtrip: bool = False,
    view: str = "both",
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
        If True, perform a write/re-read verification step (runs the pipeline a
        second time).  Off by default -- it only logs a ``match: True`` line.
    view
        Gallery view: ``l2``, ``l3``, ``both``, or ``comparison``. The
        ``comparison`` view renders L2-vs-L3 counterpart plots side by side.

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
        view=view,
    )


# ---------------------------------------------------------------------------
# LOBSTER demo
# ---------------------------------------------------------------------------


def run_lobster_demo(
    source: Path | str,
    trading_date: str,
    output_dir: Path | str | None,
    *,
    view: str = "both",
) -> Path:
    """Run the LOBSTER demo pipeline.

    *view* is the gallery view (``l2``, ``l3``, ``both``, or ``comparison``);
    ``comparison`` renders L2-vs-L3 counterpart plots side by side.

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
    # loader and append them to the gallery model's analytics.
    halts = getattr(pipeline.loader, "trading_halts", None)
    analytics: list[PlotSpec] = []
    if halts is not None and not halts.empty:
        analytics.append(trading_halts_panel(result.trades, halts))
    if not result.trades.empty:
        analytics.append(ofi_horizon_panel(result.trades))

    return _save_and_gallery(
        result,
        out,
        title=f"LOBSTER {src.name} ({trading_date}) -- ob-analytics",
        analytics=analytics,
        view=view,
    )
