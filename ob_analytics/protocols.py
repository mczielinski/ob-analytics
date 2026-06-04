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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class RunContext:
    """Per-run parameters that don't belong on the Format constructor.

    Passed to ``Pipeline.run(source, ctx=...)`` and forwarded to
    ``Format.create_loader/create_trade_source/create_writer``.

    Attributes
    ----------
    trading_date : str or pd.Timestamp, optional
        Calendar date anchor (LOBSTER needs this; venues with continuous
        trading do not).
    """

    trading_date: object | None = None


@runtime_checkable
class EventLoader(Protocol):
    """Loads raw order-book events from a data source.

    The returned DataFrame must contain at least the columns required by
    ``ob_analytics.schemas.validate_events_df``.
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
            Events with at least the columns required by
            ``ob_analytics.schemas.validate_events_df``.
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


@runtime_checkable
class Format(Protocol):
    """Structural contract for a data-format descriptor.

    A Format bundles the per-format factories the pipeline needs: how to
    load events, how to acquire trades, and (optionally) how to write
    results or compute depth directly. Pass instances to
    ``Pipeline(format=...)``.

    There is **no base class to inherit** — any object providing these
    members satisfies the contract (structural typing). ``name`` is a
    short lowercase identifier (e.g. ``"bitstamp"``).
    """

    name: str

    def create_loader(self, config: Any, ctx: RunContext) -> EventLoader:
        """Return a loader for this format."""
        ...

    def create_trade_source(self, config: Any, ctx: RunContext) -> TradeSource:
        """Return a trade source for this format."""
        ...

    def create_writer(self, config: Any, ctx: RunContext) -> DataWriter | None:
        """Return a writer for this format, or ``None`` if unsupported."""
        ...

    def compute_depth(
        self,
        events: pd.DataFrame,
        config: Any,
        source: Any,
        ctx: RunContext,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Return ``(depth, depth_summary)`` to override the standard
        depth pipeline, or ``None`` to use it."""
        ...

    def config_defaults(self) -> dict[str, Any]:
        """Return default :class:`PipelineConfig` overrides for this format."""
        ...
