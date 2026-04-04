"""Limit order book analytics and visualization.

Reconstruct trades from raw exchange events, classify order types,
compute depth metrics, and visualize market microstructure.

Quick start::

    from ob_analytics import Pipeline

    result = Pipeline().run("orders.csv")

The package exposes two layers:

* **High-level**: :class:`Pipeline` runs the full processing
  sequence (load → match → trades → classify → depth → metrics)
  with sensible defaults.
* **Low-level**: Individual functions (``load_event_data``,
  ``event_match``, ``match_trades``, etc.) for step-by-step control.

All processing stages are pluggable via :mod:`~ob_analytics.protocols`.
"""

from loguru import logger
from ob_analytics.config import PipelineConfig
from ob_analytics.data import (
    get_zombie_ids,
    list_writers,
    load_data,
    process_data,
    register_writer,
    save_data,
)
from ob_analytics.depth import (
    depth_metrics,
    filter_depth,
    get_spread,
    price_level_volume,
)
from ob_analytics.analytics import order_aggressiveness, trade_impacts
from ob_analytics.event_processing import (
    BitstampFormat,
    BitstampLoader,
    BitstampWriter,
    load_event_data,
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
    download_sample,
)
from ob_analytics.matching_engine import NeedlemanWunschMatcher, event_match
from ob_analytics.models import (
    DepthLevel,
    KyleLambdaResult,
    OrderBookSnapshot,
    OrderEvent,
    Trade,
)
from ob_analytics.order_book_reconstruction import order_book
from ob_analytics.order_types import set_order_types
from ob_analytics.pipeline import Pipeline, PipelineResult, list_formats, register_format
from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    Format,
    MatchingEngine,
    TradeInferrer,
)
from ob_analytics.trades import DefaultTradeInferrer, match_trades
from ob_analytics.visualisation import (
    PlotTheme,
    get_plot_theme,
    plot_current_depth,
    plot_event_map,
    plot_events_histogram,
    plot_hidden_executions,
    plot_kyle_lambda,
    plot_order_flow_imbalance,
    plot_price_levels,
    plot_price_levels_faster,
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

__all__ = [
    # Pipeline class
    "Pipeline",
    "PipelineResult",
    # Pipeline functions (backward-compatible)
    "get_zombie_ids",
    "load_data",
    "process_data",
    "save_data",
    "depth_metrics",
    "filter_depth",
    "get_spread",
    "price_level_volume",
    "load_event_data",
    "order_aggressiveness",
    "event_match",
    "order_book",
    "set_order_types",
    "match_trades",
    "trade_impacts",
    # Flow toxicity
    "compute_vpin",
    "compute_kyle_lambda",
    "order_flow_imbalance",
    "KyleLambdaResult",
    # Configuration
    "PipelineConfig",
    # Protocols and base classes
    "EventLoader",
    "MatchingEngine",
    "TradeInferrer",
    "DataWriter",
    "Format",
    # Format registration
    "register_format",
    "list_formats",
    "register_writer",
    "list_writers",
    # Bitstamp implementations
    "BitstampLoader",
    "BitstampWriter",
    "BitstampFormat",
    # LOBSTER implementations
    "LobsterLoader",
    "LobsterMatcher",
    "LobsterTradeInferrer",
    "LobsterWriter",
    "LobsterFormat",
    "download_sample",
    # Other implementations
    "NeedlemanWunschMatcher",
    "DefaultTradeInferrer",
    # Domain models
    "OrderEvent",
    "Trade",
    "DepthLevel",
    "OrderBookSnapshot",
    # Visualization
    "PlotTheme",
    "set_plot_theme",
    "get_plot_theme",
    "save_figure",
    "register_plot_backend",
    "plot_time_series",
    "plot_trades",
    "plot_price_levels",
    "plot_price_levels_faster",
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
    # Exceptions
    "ObAnalyticsError",
    "InvalidDataError",
    "MatchingError",
    "InsufficientDataError",
    "ConfigurationError",
]
