"""Limit order book analytics and visualization.

Reconstruct trades from raw exchange events, classify order types,
compute depth metrics, and visualize market microstructure.

Quick start::

    from ob_analytics import Pipeline

    result = Pipeline().run("orders.csv")

The package exposes two layers:

* **High-level**: :class:`Pipeline` runs the full 8-step processing
  sequence with sensible defaults.
* **Low-level**: Individual functions (``load_event_data``,
  ``event_match``, ``match_trades``, etc.) for step-by-step control.

All processing stages are pluggable via :mod:`~ob_analytics.protocols`.
"""

from loguru import logger
from ob_analytics.config import PipelineConfig
from ob_analytics.data import get_zombie_ids, load_data, process_data, save_data
from ob_analytics.depth import (
    depth_metrics,
    filter_depth,
    get_spread,
    price_level_volume,
)
from ob_analytics.event_processing import (
    BitstampLoader,
    load_event_data,
    order_aggressiveness,
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
from ob_analytics.matching_engine import NeedlemanWunschMatcher, event_match
from ob_analytics.models import DepthLevel, KyleLambdaResult, OrderBookSnapshot, OrderEvent, Trade
from ob_analytics.order_book_reconstruction import order_book
from ob_analytics.order_types import set_order_types
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.protocols import EventLoader, MatchingEngine, TradeInferrer
from ob_analytics.trades import DefaultTradeInferrer, match_trades, trade_impacts
from ob_analytics.visualisation import (
    PlotTheme,
    get_plot_theme,
    plot_current_depth,
    plot_event_map,
    plot_events_histogram,
    plot_kyle_lambda,
    plot_order_flow_imbalance,
    plot_price_levels,
    plot_time_series,
    plot_trades,
    plot_volume_map,
    plot_volume_percentiles,
    plot_vpin,
    register_plot_backend,
    save_figure,
    set_plot_theme,
)

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
    # Protocols
    "EventLoader",
    "MatchingEngine",
    "TradeInferrer",
    # Default implementations
    "BitstampLoader",
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
    "plot_event_map",
    "plot_volume_map",
    "plot_current_depth",
    "plot_volume_percentiles",
    "plot_events_histogram",
    "plot_vpin",
    "plot_order_flow_imbalance",
    "plot_kyle_lambda",
    # Exceptions
    "ObAnalyticsError",
    "InvalidDataError",
    "MatchingError",
    "InsufficientDataError",
    "ConfigurationError",
]
