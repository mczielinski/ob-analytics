"""Demo runners behind the ``bitstamp-demo`` and ``lobster-demo`` CLI commands.

``cli.py:_cmd_bitstamp_demo`` and ``cli.py:_cmd_lobster_demo`` are thin
wrappers around these functions, so the "what a full pipeline run looks like"
logic lives in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from ob_analytics.bars import bars
from ob_analytics.bitstamp import BitstampSource, BitstampWriter
from ob_analytics.capture_files import ORDERS_NAME, TRADES_NAME
from ob_analytics.data import save_data
from ob_analytics.lobster import LobsterLoader, LobsterSource
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.visualization.gallery import (
    PlotSpec,
    bars_panel,
    build_gallery_model,
    display_result,
    generate_gallery,
    trading_halts_panel,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
    appended to the model's :attr:`~...gallery.GalleryModel.analytics`, on top
    of the bars panel every demo shows.
    """
    parquet_dir = output_dir / "parquet"
    # Each Parquet file records the run's config, tick and lot size included.
    save_data(result, parquet_dir)
    logger.info("Parquet saved to: {}", parquet_dir)

    panels = list(analytics or [])
    panels.extend(_bars_panels(display_result(result).trades))

    model = None
    if panels:
        model = build_gallery_model(result)
        model.analytics.extend(panels)

    gallery_dir = output_dir / "gallery"
    gallery_path = generate_gallery(
        result, gallery_dir, model=model, view=view, title=title
    )
    logger.info("Gallery: {}", gallery_path.resolve())
    logger.info("Open in browser: file://{}", gallery_path.resolve())
    return gallery_path


def _bars_panels(trades: pd.DataFrame) -> list[PlotSpec]:
    """Build the demo's bar faces: one cut by the clock, one by traded volume.

    The pair is the point of the card — the same trades cut two ways. *trades*
    must already be in display units (see :func:`display_result`) so bar
    prices read in the quote currency, like every other face in the gallery.
    """
    if trades.empty:
        return []
    panels: list[PlotSpec] = []
    for rule in ("time", "volume"):
        try:
            panels.append(bars_panel(bars(trades, rule)))
        except Exception as e:  # noqa: BLE001 -- a demo face must not sink the run
            logger.warning("Demo: {} bars failed: {}", rule, e)
    return panels


def _lobster_analytics_panels(
    result: PipelineResult, halts: pd.DataFrame | None
) -> list[PlotSpec]:
    """Build the LOBSTER-only analytic panel: trading halts.

    Trading halts draws trade price as a line with halt bands overlaid, so it
    needs the same display-unit trades as every other panel in the gallery
    (see :func:`display_result`).
    """
    panels: list[PlotSpec] = []
    if halts is not None and not halts.empty:
        display_trades = display_result(result).trades
        panels.append(trading_halts_panel(display_trades, halts))
    return panels


# ---------------------------------------------------------------------------
# Bitstamp demo
# ---------------------------------------------------------------------------


def _resolve_orders_path(path: Path) -> Path:
    """Return path to orders.csv (directory containing orders.csv is ok)."""
    return path / ORDERS_NAME if path.is_dir() else path


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
    if input_path:
        raw_input = Path(input_path)
    else:
        # The bundled sample (orders.csv.gz; pandas reads the gzip transparently).
        from ob_analytics import sample_csv_path

        raw_input = sample_csv_path()
    orders_path = _resolve_orders_path(raw_input)
    if not orders_path.exists():
        raise FileNotFoundError(
            f"orders.csv not found: {orders_path}. "
            "Give the path of a Bitstamp orders.csv, or of the folder holding it."
        )
    trades_path = orders_path.parent / TRADES_NAME
    if not trades_path.exists():
        raise FileNotFoundError(
            f"Companion trades.csv missing next to {orders_path.name}. "
            f"Capture both with `ob-analytics capture bitstamp`."
        )

    out = Path(output_dir) if output_dir else Path("bitstamp_output")
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Bitstamp Demo: {}", orders_path.name)
    logger.info("=" * 60)

    pipeline = Pipeline(source=BitstampSource())
    result = pipeline.run(str(orders_path))

    if roundtrip:
        rt_dir = out / "roundtrip"
        # BitstampWriter writes orders.csv and the companion trades.csv into
        # the folder.  Pass the run's config so ticks convert back on the
        # run's grid (issue #155).
        save_data(result, rt_dir, writer=BitstampWriter(result.config))
        rt = Pipeline(source=BitstampSource()).run(str(rt_dir / ORDERS_NAME))
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
    path: Path | str,
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
    src = Path(path)
    out = Path(output_dir) if output_dir else Path("lobster_output")
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LOBSTER Demo: {} ({})", src, trading_date)
    logger.info("=" * 60)

    # The run's own loader, so its trading halts can be read after the run.
    loader = LobsterLoader(trading_date=trading_date)
    result = Pipeline(source=LobsterSource(), loader=loader).run(str(src))

    # LOBSTER halts are not part of the slim PipelineResult; read them off the
    # loader and append them to the gallery model's analytics.
    halts = loader.trading_halts
    analytics = _lobster_analytics_panels(result, halts)

    return _save_and_gallery(
        result,
        out,
        title=f"LOBSTER {src.name} ({trading_date}) -- ob-analytics",
        analytics=analytics,
        view=view,
    )
