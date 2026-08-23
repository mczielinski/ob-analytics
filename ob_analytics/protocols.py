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
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd


class Level(str, Enum):
    """Order-book resolution a feed (or a plot) works at — MBP vs MBO.

    The granularity axis, orthogonal to :class:`FeedType`'s crossing
    invariant.  A format declares its :attr:`Format.resolution` so the
    pipeline knows which stages apply:

    * :attr:`L2` — Market-By-Price (MBP): aggregate volume per price level,
      with **no persistent order identity**.  Price-level feeds (Binance,
      Kalshi, Polymarket, most CCXT sources) are L2.  The per-order stages
      (:func:`~ob_analytics.analytics.set_order_types`,
      :func:`~ob_analytics.analytics.order_aggressiveness`, queue
      reconstruction) have nothing to key on and are skipped; depth / spread
      / trade analytics run directly on the price-level book.
    * :attr:`L3` — Market-By-Order (MBO): one primitive per resting order,
      with stable identity (queue position recoverable).  The reconstruction
      model ob-analytics was built for (Bitstamp, LOBSTER, Databento).

    The ``str`` mixin lets members slot directly into the visualization
    renderer-registry tuple keys and, via the :meth:`__str__` override, render
    as the bare token (``"L2"``) in file stems and f-strings rather than
    ``"Level.L2"``.
    """

    L2 = "L2"
    L3 = "L3"

    def __str__(self) -> str:
        return self.value


class FeedType(str, Enum):
    """How a data feed represents the order book — its crossing invariant.

    A format declares its feed type so downstream code can reason about
    crossed books *by coordinate, not by format name*.  The distinction is
    a property of the source, not of the reconstruction:

    * :attr:`MATCHED_BOOK` — an L3 feed emitted by the venue's own matching
      engine (LOBSTER, exchange MBO such as Databento).  Bids can never rest
      above asks, so an uncrossed book is a guaranteed invariant of the data.
    * :attr:`DIFF_FEED` — an L3 feed reconstructed from a public
      placement/cancellation *diff stream* (the Bitstamp public feed).  It
      can contain genuinely crossed *resting* orders (a bid resting above an
      ask, neither filling); :func:`~ob_analytics.analytics.order_book`
      replays this faithfully — a crossed book in the output is a property of
      the feed, not a reconstruction bug.
    * :attr:`UNKNOWN` — a format that does not declare its feed type (the
      structural default for third-party formats predating this attribute).

    Mixes in ``str`` so members compare and serialise as their value
    (``FeedType.DIFF_FEED == "diff_feed"``), which keeps CLI/JSON output and
    equality checks ergonomic.
    """

    MATCHED_BOOK = "matched_book"
    DIFF_FEED = "diff_feed"
    UNKNOWN = "unknown"


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
    symbol : str, optional
        The instrument this run covers (e.g. ``"BTC/USD"``).  When supplied,
        loaders tag each row with an optional ``symbol`` column so cross-venue
        frames can be told apart (issue #147).  ``None`` leaves it untagged.
    venue : str, optional
        The source venue this run covers (e.g. ``"bitstamp"``).  When supplied,
        it overrides the loader's own source name in the optional ``venue``
        column; ``None`` falls back to that source name.  Supplying either
        *symbol* or *venue* is what turns identity tagging on.
    """

    trading_date: object | None = None
    symbol: str | None = None
    venue: str | None = None


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
class DepthSource(Protocol):
    """Loads a **price-level (L2) depth stream** into the canonical depth frame.

    The L2 counterpart to :class:`EventLoader`.  Where an ``EventLoader``
    returns per-order events that the pipeline later folds into depth via
    :func:`~ob_analytics.depth.price_level_volume`, a ``DepthSource`` returns
    the depth frame **directly** — a price-level feed *is* a depth stream, so
    there is nothing to reconstruct.

    An :attr:`Format.resolution` of :attr:`Level.L2` format's
    :meth:`Format.create_loader` returns a ``DepthSource``; the pipeline
    validates its output with
    :func:`~ob_analytics.schemas.validate_depth_df` and feeds it straight to
    :class:`~ob_analytics.depth.DepthMetricsEngine`.

    Returned DataFrame columns (see
    :data:`~ob_analytics.schemas.DEPTH_COLUMNS`):

    * ``timestamp``  — pandas datetime64[ns]
    * ``price``      — float, the price level
    * ``volume``     — float, the level's **new absolute** resting size after
      the update (``0`` removes the level); *not* a signed delta
    * ``direction``  — categorical ``bid``/``ask``
    """

    def load(self, source: Any) -> pd.DataFrame:
        """Load a price-level depth stream from *source* and return the frame.

        Parameters
        ----------
        source
            Data source identifier — canonically a ``str | Path`` (a snapshot
            + price-level-delta file), but implementations may accept richer
            descriptors.

        Returns
        -------
        pandas.DataFrame
            Depth with at least the columns required by
            :func:`~ob_analytics.schemas.validate_depth_df`.
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
    short lowercase identifier (e.g. ``"bitstamp"``).  ``feed_type``
    declares the source's crossing invariant (:class:`FeedType`); callers
    should treat a missing attribute as :attr:`FeedType.UNKNOWN` (structural
    default) rather than special-casing format names.  ``resolution``
    declares the granularity (:class:`Level`): :attr:`Level.L3` (default) for
    per-order feeds, :attr:`Level.L2` for price-level feeds — callers should
    treat a missing attribute as :attr:`Level.L3` (structural default).
    """

    name: str
    feed_type: FeedType
    resolution: Level

    def create_loader(self, config: Any, ctx: RunContext) -> EventLoader | DepthSource:
        """Return the loader for this format.

        An :attr:`Level.L3` format returns an :class:`EventLoader` (per-order
        events); an :attr:`Level.L2` format returns a :class:`DepthSource`
        (the price-level depth frame directly).
        """
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

    def required_context(self) -> list[str]:
        """:class:`RunContext` field names this format requires.

        E.g. LOBSTER returns ``["trading_date"]`` because its filenames carry
        no date; Bitstamp returns ``[]``.  Lets the CLI/pipeline validate
        required context generically instead of special-casing format names.
        Callers should treat a missing method as ``[]`` (structural default).
        """
        ...
