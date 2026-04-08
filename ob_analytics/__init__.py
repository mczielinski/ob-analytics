"""Limit order book analytics and visualization.

Reconstruct trades from raw exchange events, classify order types,
compute depth metrics, and visualize market microstructure.

Quick start::

    from ob_analytics import Pipeline, sample_csv_path

    result = Pipeline().run(sample_csv_path())

The package exposes two layers:

* **High-level**: :class:`Pipeline` runs the full processing
  sequence (load → match → trades → classify → depth → metrics)
  with sensible defaults.  When called without arguments it defaults
  to the Bitstamp format.
* **Low-level**: Individual classes and functions for step-by-step control.
  Two symmetric format implementations are provided:

  - Bitstamp: :class:`BitstampLoader`, :class:`BitstampMatcher`,
    :class:`BitstampTradeInferrer`, :class:`BitstampWriter`,
    :class:`BitstampFormat`
  - LOBSTER: :class:`LobsterLoader`, :class:`LobsterMatcher`,
    :class:`LobsterTradeInferrer`, :class:`LobsterWriter`,
    :class:`LobsterFormat`

All processing stages are pluggable via :mod:`~ob_analytics.protocols`.
"""

from pathlib import Path

from loguru import logger
from ob_analytics.config import PipelineConfig
from ob_analytics.data import (
    get_zombie_ids,
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
    BitstampMatcher,
    BitstampTradeInferrer,
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
from ob_analytics.lobster import (
    LobsterFormat,
    LobsterLoader,
    LobsterMatcher,
    LobsterTradeInferrer,
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
    MatchingEngine,
    TradeInferrer,
)
from ob_analytics.visualization import (
    PlotTheme,
    get_plot_theme,
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
register_format("bitstamp", BitstampFormat)
register_format("lobster", LobsterFormat)
register_writer("bitstamp", BitstampWriter)
# Note: LobsterWriter is NOT registered here because it requires a trading_date
# argument that cannot be auto-inferred. Use:
#   LobsterFormat(trading_date=...).create_writer(config)
# or pass writer= directly to save_data().

logger.disable("ob_analytics")


def sample_csv_path() -> Path:
    """Return the path to the bundled Bitstamp sample CSV.

    The file contains ~5 hours of Bitstamp BTC/USD limit order events
    (2015-05-01 00:00--05:00 UTC).
    """
    return Path(__file__).parent / "_sample_data" / "orders.csv"


__all__ = [
    # ── Sample data ──────────────────────────────────────────────────
    "sample_csv_path",
    # ── Symmetric format pairs (Bitstamp ↔ LOBSTER) ──────────────────
    "BitstampFormat",
    "LobsterFormat",
    "BitstampLoader",
    "LobsterLoader",
    "BitstampMatcher",
    "LobsterMatcher",
    "BitstampTradeInferrer",
    "LobsterTradeInferrer",
    "BitstampWriter",
    "LobsterWriter",
    # ── Pipeline orchestration ───────────────────────────────────────
    "Pipeline",
    "PipelineResult",
    "register_format",
    "list_formats",
    # ── Protocols / extension points ─────────────────────────────────
    "EventLoader",
    "MatchingEngine",
    "TradeInferrer",
    "DataWriter",
    "Format",
    # ── Format-agnostic analytics ────────────────────────────────────
    "order_aggressiveness",
    "trade_impacts",
    # ── Order book processing ────────────────────────────────────────
    "set_order_types",
    "order_book",
    # ── Depth computation ────────────────────────────────────────────
    "DepthMetricsEngine",
    "price_level_volume",
    "depth_metrics",
    "filter_depth",
    "get_spread",
    # ── Data I/O + writer registry ───────────────────────────────────
    "save_data",
    "load_data",
    "get_zombie_ids",
    "register_writer",
    "list_writers",
    # ── LOBSTER-specific utilities ───────────────────────────────────
    "lobster_depth_from_orderbook",
    # ── Flow toxicity ────────────────────────────────────────────────
    "compute_vpin",
    "compute_kyle_lambda",
    "order_flow_imbalance",
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
