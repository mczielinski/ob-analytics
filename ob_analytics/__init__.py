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
  Two symmetric source implementations are provided:

  - Bitstamp: :class:`BitstampLoader`, :class:`BitstampTradeReader`,
    :class:`BitstampWriter`, :class:`BitstampSource`
  - LOBSTER: :class:`LobsterLoader`, :class:`LobsterTradeReader`,
    :class:`LobsterWriter`, :class:`LobsterSource`

All processing stages are pluggable via :mod:`~ob_analytics.protocols`; a whole
new data source registers via :func:`~ob_analytics.sources.register_source`.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from loguru import logger

from ob_analytics.analytics import (
    DataQualitySummary,
    SequenceGapReport,
    data_quality_summary,
    detect_sequence_gaps,
)

# Importing the source modules fires their register_source(...) self-registration
# at import time; the Source classes are also the public per-venue entry points.
from ob_analytics.bitstamp import BitstampSource
from ob_analytics.config import PipelineConfig, SourceSettings
from ob_analytics.data import load_data, save_data
from ob_analytics.datasets import toy_events, toy_l2_depth, toy_l2_trades, toy_trades
from ob_analytics.depth_l2 import DepthCsvSource
from ob_analytics.exceptions import ConfigError, ObAnalyticsError
from ob_analytics.flow_toxicity import (
    KyleLambdaResult,
    compute_kyle_lambda,
    compute_vpin,
    order_flow_imbalance,
)

# Importing the live package registers the ccxt and cryptofeed live sources
# (the bitstamp live capability rides on BitstampSource, already registered
# above); then discover any third-party sources advertised through the
# entry-point group.
from ob_analytics.live import LiveSource
from ob_analytics.lobster import LobsterSource
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.protocols import (
    DataWriter,
    DepthSource,
    EventLoader,
    FeedType,
    Level,
    OfflineSource,
    RunContext,
    Source,
    TradeSource,
)
from ob_analytics.schemas import (
    SYMBOL_COLUMN,
    VENUE_COLUMN,
    group_by_instrument,
)
from ob_analytics.sources import (
    get_source,
    list_sources,
    load_source_plugins,
    register_source,
)
from ob_analytics.trade_sign import (
    bulk_volume_classification,
    classify_trade_sign,
    lee_ready,
    tick_rule,
)

load_source_plugins()

logger.disable("ob_analytics")

try:
    __version__ = version("ob-analytics")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"


def sample_data_dir() -> Path:
    """Return the directory holding the bundled Bitstamp sample.

    The directory contains ``orders.csv.gz`` (gzip-compressed; pandas reads it
    transparently), ``trades.csv``, and a ``meta.json`` describing the live
    capture.  Pass it to :class:`Pipeline` (or to
    :class:`BitstampTradeReader.load`) so the companion ``trades.csv`` is
    auto-located.
    """
    return Path(__file__).parent / "_sample_data"


def sample_csv_path() -> Path:
    """Return the path to the bundled Bitstamp sample ``orders.csv.gz``.

    The orders capture ships gzip-compressed (~23 MB -> ~2.9 MB) so it does not
    bloat installs; :func:`pandas.read_csv` decompresses it transparently.  The
    companion ``trades.csv`` in the same directory is required for
    :class:`Pipeline` runs.
    """
    return sample_data_dir() / "orders.csv.gz"


__all__ = [
    # ── Instrument identity (issue #147) ─────────────────────────────
    "SYMBOL_COLUMN",
    "VENUE_COLUMN",
    # ── Sources (per-venue entry points) ─────────────────────────────
    "BitstampSource",
    "ConfigError",
    "DataQualitySummary",
    "DataWriter",
    "DepthCsvSource",
    "DepthSource",
    "EventLoader",
    "FeedType",
    "KyleLambdaResult",
    "Level",
    "LiveSource",
    "LobsterSource",
    # ── Exceptions ───────────────────────────────────────────────────
    "ObAnalyticsError",
    # ── Protocols / extension points ─────────────────────────────────
    "OfflineSource",
    # ── Pipeline orchestration ───────────────────────────────────────
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "RunContext",
    "SequenceGapReport",
    "Source",
    "SourceSettings",
    "TradeSource",
    "__version__",
    # ── Trade-sign classification ────────────────────────────────────
    "bulk_volume_classification",
    "classify_trade_sign",
    "compute_kyle_lambda",
    # ── Flow toxicity ────────────────────────────────────────────────
    "compute_vpin",
    # ── Data quality ─────────────────────────────────────────────────
    "data_quality_summary",
    "detect_sequence_gaps",
    "get_source",
    "group_by_instrument",
    "lee_ready",
    "list_sources",
    "load_data",
    "load_source_plugins",
    "order_flow_imbalance",
    "register_source",
    # ── Sample data ──────────────────────────────────────────────────
    "sample_csv_path",
    "sample_data_dir",
    # ── Data I/O ─────────────────────────────────────────────────────
    "save_data",
    "tick_rule",
    "toy_events",
    "toy_l2_depth",
    "toy_l2_trades",
    "toy_trades",
]
