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

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from loguru import logger

# Importing the bitstamp and lobster modules fires their format/writer
# self-registration at import time; the Format classes are also the public
# symmetric-pair entry points.
from ob_analytics.bitstamp import BitstampFormat
from ob_analytics.config import PipelineConfig
from ob_analytics.data import load_data, save_data
from ob_analytics.datasets import toy_events, toy_trades
from ob_analytics.exceptions import ConfigError, ObAnalyticsError
from ob_analytics.flow_toxicity import (
    KyleLambdaResult,
    compute_kyle_lambda,
    compute_vpin,
    order_flow_imbalance,
)
from ob_analytics.lobster import LobsterFormat
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
    "__version__",
    # ── Sample data ──────────────────────────────────────────────────
    "sample_csv_path",
    "sample_data_dir",
    "toy_events",
    "toy_trades",
    # ── Pipeline orchestration ───────────────────────────────────────
    "Pipeline",
    "PipelineResult",
    "PipelineConfig",
    "register_format",
    "list_formats",
    # ── Formats (symmetric-pair entry points) ────────────────────────
    "BitstampFormat",
    "LobsterFormat",
    # ── Protocols / extension points ─────────────────────────────────
    "Format",
    "EventLoader",
    "TradeSource",
    "DataWriter",
    "RunContext",
    # ── Data I/O ─────────────────────────────────────────────────────
    "save_data",
    "load_data",
    # ── Flow toxicity ────────────────────────────────────────────────
    "compute_vpin",
    "compute_kyle_lambda",
    "order_flow_imbalance",
    "KyleLambdaResult",
    # ── Exceptions ───────────────────────────────────────────────────
    "ObAnalyticsError",
    "ConfigError",
]
