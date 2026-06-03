"""Limit order book analytics and visualization.

Load order events, attach trades (from ``trades.csv`` or embedded
executions), classify order types, compute depth metrics, and visualize
market microstructure.

Quick start::

    from ob_analytics import Pipeline, sample_csv_path

    result = Pipeline().run(sample_csv_path())

The package exposes two layers:

* **High-level**: :class:`Pipeline` runs the full processing
  sequence (load → trades → classify → depth → metrics)
  with sensible defaults.  When called without arguments it defaults
  to the Bitstamp format (orders + companion ``trades.csv``).
* **Low-level**: Individual classes and functions for step-by-step control.
  Two symmetric format implementations are provided:

  - Bitstamp: :class:`BitstampLoader`, :class:`BitstampTradeReader`,
    :class:`BitstampWriter`, :class:`BitstampFormat`
  - LOBSTER: :class:`LobsterLoader`, :class:`LobsterTradeReader`,
    :class:`LobsterWriter`, :class:`LobsterFormat`

All processing stages are pluggable via :mod:`~ob_analytics.protocols`.
"""

from pathlib import Path

import pandas as pd
from loguru import logger
from ob_analytics.config import PipelineConfig
from ob_analytics.data import (
    list_writers,
    load_data,
    register_writer,
    save_data,
)
from ob_analytics.depth import (
    DepthMetricsEngine,
    depth_metrics,
    filter_depth,
    get_spread,
    price_level_volume,
)
from ob_analytics.analytics import (
    order_aggressiveness,
    order_book,
    set_order_types,
    trade_impacts,
)
from ob_analytics.bitstamp import (
    BitstampFormat,
    BitstampLoader,
    BitstampTradeReader,
    BitstampWriter,
)
from ob_analytics.exceptions import (
    ConfigurationError,
    InsufficientDataError,
    InvalidDataError,
    MatchingError,
    ObAnalyticsError,
)
from ob_analytics.flow_toxicity import (
    compute_kyle_lambda,
    compute_vpin,
    order_flow_imbalance,
)
from ob_analytics.metrics import (
    KyleLambda,
    Ofi,
    ToxicityMetric,
    Vpin,
    list_metrics,
    register_metric,
)
from ob_analytics.lobster import (
    LobsterFormat,
    LobsterLoader,
    LobsterTradeReader,
    LobsterWriter,
    lobster_depth_from_orderbook,
)
from ob_analytics.models import (
    DepthLevel,
    KyleLambdaResult,
    OrderBookSnapshot,
    OrderEvent,
    Trade,
)
from ob_analytics.pipeline import (
    Pipeline,
    PipelineResult,
    list_formats,
    register_format,
)
from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    Format,
    RunContext,
    TradeSource,
)
from ob_analytics.visualization import (
    PlotTheme,
    get_plot_theme,
    infer_volume_scale,
    plot_current_depth,
    plot_event_map,
    plot_events_histogram,
    plot_hidden_executions,
    plot_kyle_lambda,
    plot_order_flow_imbalance,
    plot_price_levels,
    plot_time_series,
    plot_trades,
    plot_trading_halts,
    plot_volume_map,
    plot_volume_percentiles,
    plot_vpin,
    register_plot_backend,
    save_figure,
    set_plot_theme,
)


# ── Register built-in formats and writers ─────────────────────────────
def _make_lobster_writer(config, ctx):
    td = ctx.trading_date
    if td is None:
        raise ValueError(
            "LOBSTER writer requires ctx.trading_date. "
            "Pass ctx=RunContext(trading_date=...) to save_data()."
        )
    if not isinstance(td, (str, pd.Timestamp)):
        raise TypeError(
            f"ctx.trading_date must be str or pandas.Timestamp, got {type(td).__name__}"
        )
    return LobsterWriter(config, trading_date=td)


# Formats self-register at import time (see ob_analytics/bitstamp.py and
# ob_analytics/lobster.py). Writers are still registered here until S3.3.
register_writer("bitstamp", lambda config, ctx: BitstampWriter(config))
register_writer("lobster", _make_lobster_writer)

logger.disable("ob_analytics")


def sample_data_dir() -> Path:
    """Return the directory holding the bundled Bitstamp sample.

    The directory contains ``orders.csv``, ``trades.csv``, and a
    ``meta.json`` describing the live capture.  Pass this directory to
    :class:`Pipeline` (or to :class:`BitstampTradeReader.load`) so the
    companion ``trades.csv`` is auto-located.
    """
    return Path(__file__).parent / "_sample_data"


def sample_csv_path() -> Path:
    """Return the path to the bundled Bitstamp sample ``orders.csv``.

    The companion ``trades.csv`` in the same directory is produced by
    the live capture script and is required for :class:`Pipeline` runs.
    """
    return sample_data_dir() / "orders.csv"


__all__ = [
    # ── Sample data ──────────────────────────────────────────────────
    "sample_csv_path",
    "sample_data_dir",
    # ── Symmetric format pairs (Bitstamp ↔ LOBSTER) ──────────────────
    "BitstampFormat",
    "LobsterFormat",
    "BitstampLoader",
    "LobsterLoader",
    "BitstampTradeReader",
    "LobsterTradeReader",
    "BitstampWriter",
    "LobsterWriter",
    # ── Pipeline orchestration ───────────────────────────────────────
    "Pipeline",
    "PipelineResult",
    "register_format",
    "list_formats",
    # ── Protocols / extension points ─────────────────────────────────
    "EventLoader",
    "TradeSource",
    "DataWriter",
    "Format",
    "RunContext",
    # ── Format-agnostic analytics ───────────────────────────────────
    "order_aggressiveness",
    "trade_impacts",
    # ── Order book processing ────────────────────────────────────────
    "set_order_types",
    "order_book",
    # ── Depth computation ───────────────────────────────────────────
    "DepthMetricsEngine",
    "price_level_volume",
    "depth_metrics",
    "filter_depth",
    "get_spread",
    # ── Data I/O + writer registry ───────────────────────────────────
    "save_data",
    "load_data",
    "register_writer",
    "list_writers",
    # ── LOBSTER-specific utilities ───────────────────────────────────
    "lobster_depth_from_orderbook",
    # ── Flow toxicity ───────────────────────────────────────────────
    "compute_vpin",
    "compute_kyle_lambda",
    "order_flow_imbalance",
    # ── Pluggable metrics ───────────────────────────────────────────
    "ToxicityMetric",
    "Vpin",
    "KyleLambda",
    "Ofi",
    "register_metric",
    "list_metrics",
    # ── Domain models ────────────────────────────────────────────────
    "OrderEvent",
    "Trade",
    "DepthLevel",
    "OrderBookSnapshot",
    "KyleLambdaResult",
    # ── Configuration ────────────────────────────────────────────────
    "PipelineConfig",
    # ── Exceptions ───────────────────────────────────────────────────
    "ObAnalyticsError",
    "InvalidDataError",
    "MatchingError",
    "InsufficientDataError",
    "ConfigurationError",
    # ── Visualization ────────────────────────────────────────────────
    "PlotTheme",
    "set_plot_theme",
    "get_plot_theme",
    "save_figure",
    "infer_volume_scale",
    "register_plot_backend",
    "plot_time_series",
    "plot_trades",
    "plot_price_levels",
    "plot_event_map",
    "plot_volume_map",
    "plot_current_depth",
    "plot_volume_percentiles",
    "plot_events_histogram",
    "plot_vpin",
    "plot_order_flow_imbalance",
    "plot_kyle_lambda",
    "plot_hidden_executions",
    "plot_trading_halts",
]
