"""ob-analytics: Limit order book analytics and visualization."""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from ob_analytics.config import PipelineConfig
from ob_analytics.data import get_zombie_ids, load_data, process_data, save_data
from ob_analytics.depth import depth_metrics, filter_depth, get_spread, price_level_volume
from ob_analytics.event_processing import BitstampLoader, load_event_data, order_aggressiveness
from ob_analytics.exceptions import (
    ConfigurationError,
    InsufficientDataError,
    InvalidDataError,
    MatchingError,
    ObAnalyticsError,
)
from ob_analytics.matching_engine import NeedlemanWunschMatcher, event_match
from ob_analytics.models import DepthLevel, OrderBookSnapshot, OrderEvent, Trade
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
    plot_price_levels,
    plot_time_series,
    plot_trades,
    plot_volume_map,
    plot_volume_percentiles,
    save_figure,
    set_plot_theme,
)

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
    "plot_time_series",
    "plot_trades",
    "plot_price_levels",
    "plot_event_map",
    "plot_volume_map",
    "plot_current_depth",
    "plot_volume_percentiles",
    "plot_events_histogram",
    # Exceptions
    "ObAnalyticsError",
    "InvalidDataError",
    "MatchingError",
    "InsufficientDataError",
    "ConfigurationError",
]
