"""Protocol interfaces for ob-analytics.

These define the contracts that pluggable components must satisfy.
Implementations are discovered by structural (duck) typing -- there is
no need to inherit from these classes.

Built-in implementations ship with the package (one symmetric set per format):

* Bitstamp: :class:`~ob_analytics.bitstamp.BitstampLoader`,
  :class:`~ob_analytics.bitstamp.BitstampMatcher`,
  :class:`~ob_analytics.bitstamp.BitstampTradeInferrer`,
  :class:`~ob_analytics.bitstamp.BitstampWriter`
* LOBSTER: :class:`~ob_analytics.lobster.LobsterLoader`,
  :class:`~ob_analytics.lobster.LobsterMatcher`,
  :class:`~ob_analytics.lobster.LobsterTradeInferrer`,
  :class:`~ob_analytics.lobster.LobsterWriter`

Users can substitute their own by passing any object that satisfies the
protocol to :class:`~ob_analytics.pipeline.Pipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class EventLoader(Protocol):
    """Loads raw order-book events from a data source.

    The returned DataFrame must contain at least the columns documented
    in :class:`~ob_analytics.models.OrderEvent`.
    """

    def load(self, source: Any) -> pd.DataFrame:
        """Load events from *source* and return a DataFrame.

        Parameters
        ----------
        source
            Data source identifier.  The canonical type is ``str | Path``
            (a file path), but loaders may accept richer descriptors
            such as dicts, dataclasses, or connection strings.

        Returns
        -------
        pandas.DataFrame
            Events with at least the columns described by
            :class:`~ob_analytics.models.OrderEvent`.
        """
        ...


@runtime_checkable
class MatchingEngine(Protocol):
    """Pairs bid and ask fills to identify which events are part of the same trade.

    Receives the events DataFrame (with ``fill`` column populated) and
    returns the same DataFrame with a ``matching_event`` column added.
    """

    def match(self, events: pd.DataFrame) -> pd.DataFrame:
        """Pair bid/ask fills and return events with a ``matching_event`` column.

        Parameters
        ----------
        events : pandas.DataFrame
            Events DataFrame with a ``fill`` column.

        Returns
        -------
        pandas.DataFrame
            Same DataFrame with ``matching_event`` added.
        """
        ...


@runtime_checkable
class TradeInferrer(Protocol):
    """Constructs trade records from matched event pairs.

    Receives the events DataFrame (with ``matching_event`` populated)
    and returns a trades DataFrame.
    """

    def infer_trades(self, events: pd.DataFrame) -> pd.DataFrame:
        """Build a trades DataFrame from matched events.

        Parameters
        ----------
        events : pandas.DataFrame
            Events with ``matching_event`` column populated.

        Returns
        -------
        pandas.DataFrame
            Trades with ``timestamp``, ``price``, ``volume``,
            ``direction``, ``maker_event_id``, ``taker_event_id``.
        """
        ...


@runtime_checkable
class DataWriter(Protocol):
    """Writes pipeline results to a format-specific output."""

    def write(
        self,
        data: dict[str, pd.DataFrame],
        dest: str | Path,
        **kwargs: Any,
    ) -> Path | tuple[Path, ...]:
        """Write pipeline DataFrames to *dest*.

        Parameters
        ----------
        data : dict of str to DataFrame
            Pipeline output keyed by name (e.g. ``"events"``, ``"trades"``,
            ``"depth"``, ``"depth_summary"``).
        dest : str or Path
            Output path (file or directory, format-dependent).
        """
        ...


class Format:
    """Base class for data format descriptors.

    Subclass to define how a specific data format's events should be
    loaded, matched, trade-inferred, and optionally written back.  Pass
    instances to ``Pipeline(format=...)`` for one-line setup.

    Individual components can still be overridden::

        Pipeline(format=LobsterFormat(...), matcher=MyMatcher())

    Subclasses should set the ``name`` class variable to a short lowercase
    string identifying the exchange format (e.g. ``name = "bitstamp"``).
    This value appears in ``PipelineResult.metadata["format"]``.
    """

    name: str = ""

    def create_loader(self, config: Any) -> EventLoader:
        """Return a loader for this format."""
        raise NotImplementedError

    def create_matcher(self, config: Any) -> MatchingEngine:
        """Return a matching engine for this format.

        Subclasses must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_matcher()"
        )

    def create_trade_inferrer(self, config: Any) -> TradeInferrer:
        """Return a trade inferrer for this format.

        Subclasses must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_trade_inferrer()"
        )

    def create_writer(self, config: Any) -> DataWriter | None:
        """Return a writer for this format, or ``None`` if not supported."""
        return None

    def compute_depth(
        self,
        events: pd.DataFrame,
        config: Any,
        source: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Optionally compute depth and depth_summary directly.

        Return ``None`` (default) to use the standard
        ``price_level_volume`` → ``depth_metrics`` pipeline.  Return
        ``(depth, depth_summary)`` to override with format-specific
        ground-truth data (e.g. LOBSTER orderbook files).
        """
        return None

    def config_defaults(self) -> dict[str, Any]:
        """Return default :class:`PipelineConfig` overrides for this format."""
        return {}
