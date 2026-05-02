"""Protocol interfaces for ob-analytics.

These define the contracts that pluggable components must satisfy.
Implementations are discovered by structural (duck) typing -- there is
no need to inherit from these classes.

Built-in implementations ship with the package (one symmetric set per format):

* Bitstamp: :class:`~ob_analytics.bitstamp.BitstampLoader`,
  :class:`~ob_analytics.bitstamp.BitstampTradeReader`,
  :class:`~ob_analytics.bitstamp.BitstampWriter`
* LOBSTER: :class:`~ob_analytics.lobster.LobsterLoader`,
  :class:`~ob_analytics.lobster.LobsterTradeReader`,
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
class TradeSource(Protocol):
    """Builds the trades DataFrame for a given run.

    Implementations read explicit trade records (a separate
    ``trades.csv``, LOBSTER execution rows embedded in the events
    frame, etc.) and project them into the canonical trades schema.

    Returned DataFrame columns:

    * ``timestamp``        — pandas datetime64[ns]
    * ``price``            — float
    * ``volume``           — float
    * ``direction``        — categorical ``buy``/``sell`` (taker side)
    * ``maker_event_id``   — integer event id of the resting order
    * ``taker_event_id``   — integer event id of the aggressing order
    * ``maker``            — order id of the resting order
    * ``taker``            — order id of the aggressing order
    * ``maker_og``         — original_number of the maker event
    * ``taker_og``         — original_number of the taker event
    """

    def load(self, events: pd.DataFrame, source: Any) -> pd.DataFrame:
        """Build the trades DataFrame.

        Parameters
        ----------
        events : pandas.DataFrame
            The processed events frame (post-loader).
        source
            The same ``source`` value passed to :meth:`EventLoader.load`.
            Used by file-based readers to locate companion files.

        Returns
        -------
        pandas.DataFrame
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
    loaded, how trades are acquired, and optionally written back.  Pass
    instances to ``Pipeline(format=...)`` for one-line setup.

    Individual components can still be overridden::

        Pipeline(format=LobsterFormat(...), loader=MyLoader())

    Subclasses should set the ``name`` class variable to a short lowercase
    string identifying the exchange format (e.g. ``name = "bitstamp"``).
    This value appears in ``PipelineResult.metadata["format"]``.
    """

    name: str = ""

    def create_loader(self, config: Any) -> EventLoader:
        """Return a loader for this format."""
        raise NotImplementedError

    def create_trade_source(self, config: Any) -> TradeSource:
        """Return a trade source for this format."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_trade_source()"
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
