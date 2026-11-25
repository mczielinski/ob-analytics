"""Composable pipeline for limit order book analytics.

:class:`Pipeline` orchestrates the full processing sequence using
pluggable components that satisfy the protocols defined in
:mod:`ob_analytics.protocols`.

Usage with defaults (Bitstamp CSV + companion ``trades.csv`)::

    from ob_analytics import Pipeline, sample_csv_path

    result = Pipeline().run(sample_csv_path())
    print(result.events.shape, result.trades.shape)

Usage with custom configuration::

    from ob_analytics import Pipeline, PipelineConfig, sample_csv_path

    config = PipelineConfig(depth_bps=50)
    result = Pipeline(config=config).run(sample_csv_path())

Usage with a custom loader (any object satisfying EventLoader)::

    Pipeline(loader=my_custom_loader, trade_reader=my_trade_reader).run("data/")

Usage with a Source descriptor::

    from ob_analytics import Pipeline, BitstampSource

    result = Pipeline(source=BitstampSource()).run("my_data/orders.csv")
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from ob_analytics._utils import _UNSET, empty_events, require_per_order_events
from ob_analytics._windows import (
    OrderTypeFiles,
    ParquetAppender,
    resting_orders,
    seed_rows,
    window_bounds,
    window_positions,
)
from ob_analytics.analytics import (
    _OrderFacts,
    order_aggressiveness,
    set_order_types,
)
from ob_analytics.capture_record import read_instrument, run_declarations
from ob_analytics.config import PipelineConfig, instrument_fields
from ob_analytics.depth import (
    BookLevels,
    DepthSummaryEngine,
    price_level_volume,
    readable_quotes,
    summarize_depth,
)
from ob_analytics.protocols import (
    EventLoader,
    FeedDeclarations,
    Level,
    LoadedWindow,
    OfflineSource,
    RunContext,
    Source,
    SupportsComputeDepth,
    TradeReader,
    WindowedLoader,
    grid_of,
)
from ob_analytics.quality import DataQualitySummary, data_quality_summary
from ob_analytics.schemas import (
    loader_counts,
    lot_sizes_from_config,
    put_config_units,
    tick_sizes_from_config,
    validate_depth_df,
    validate_events_df,
    validate_trades_df,
)
from ob_analytics.sources import get_source
from ob_analytics.trade_sign import (
    _unlabelled_trades,
    classify_trade_sign,
    quote_before,
    resolve_direction,
)

if TYPE_CHECKING:
    from datetime import datetime

    from matplotlib.axes import Axes

    from ob_analytics.book import BookSnapshot

__all__ = [
    "Pipeline",
    "PipelineResult",
]

#: The quote columns a windowed run keeps for labelling trades after the
#: last window: what Lee-Ready reads, and nothing else.
_ARRIVAL_COLUMNS = ["timestamp", "best_bid_price", "best_ask_price"]

#: The config fields that describe the instrument's grid, each step with its
#: display precision.  A capture's record sets a pair only when the caller set
#: neither field of it.
_GRID_FIELDS = (
    frozenset(instrument_fields(tick_size=1.0)),
    frozenset(instrument_fields(lot_size=1.0)),
)


#: The per-order tables ``save_data`` can save beside a result's four tables,
#: each with the key :meth:`PipelineResult.lifecycles` and
#: :meth:`PipelineResult.queue` keep it under by default, so ``load_result``
#: hands a saved table back to the accessor.
def _queue_key(levels: str) -> Hashable:
    """The key :meth:`PipelineResult.queue` keeps its table under."""
    return ("queue", levels)


_SAVABLE_TABLES: dict[str, Hashable] = {
    "lifecycles": "lifecycles",
    "queue": _queue_key("touch"),
}


@dataclass(frozen=True)
class PipelineResult:
    """Immutable container for the core outputs of a pipeline run.

    Analytic outputs (VPIN, trade imbalance, Kyle's λ) are intentionally **not** stored
    here — compute them post-pipeline from ``trades`` and append them to the
    gallery model's ``analytics`` (build cards with the ``*_card`` helpers).

    Attributes
    ----------
    events, trades, depth, depth_summary : pandas.DataFrame
        Core pipeline tables.  For an :attr:`~ob_analytics.protocols.Level.L2`
        run ``events`` is **empty** (a schema-valid zero-row frame): a
        price-level feed has no per-order identity, so the per-order stages do
        not run — read ``depth`` / ``depth_summary`` / ``trades`` instead.
    config : PipelineConfig
        The configuration used for the run.
    level : Level
        The order-book resolution the run was produced at
        (:attr:`~ob_analytics.protocols.Level.L3` by default,
        :attr:`~ob_analytics.protocols.Level.L2` for price-level feeds).
        Downstream code (the gallery, data-quality) reads it to decide which
        per-order faces / metrics apply.
    declarations : FeedDeclarations
        What the run's data is declared to show: its source's declarations,
        with what a live capture recorded in their place (see
        :func:`~ob_analytics.capture_record.run_declarations`).
        :meth:`audit` holds the run to them.  The defaults declare nothing.

    Notes
    -----
    The tables built from the run after it ends are kept on the result:
    :meth:`lifecycles`, :meth:`queue`, :meth:`book`, :meth:`bars` and
    :meth:`features` build their table on the first call, and a later call
    with the same settings returns the same table.  The display-unit tables
    and the gallery's view of the run, which :meth:`plot` and the gallery
    build, are kept the same way.  So the kept tables are shared: copy one
    before you change it.  A result made with :func:`dataclasses.replace`
    starts with nothing kept.
    """

    events: pd.DataFrame
    trades: pd.DataFrame
    depth: pd.DataFrame
    depth_summary: pd.DataFrame
    config: PipelineConfig
    level: Level = Level.L3
    declarations: FeedDeclarations = field(default_factory=FeedDeclarations)
    # What the accessors built, keyed by what built it.  Not a field of the
    # run: it is left out of the constructor, of equality and of the repr.
    _kept: dict[Hashable, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def _keep(self, key: Hashable, build: Callable[[], Any]) -> Any:
        """Return what *build* returns, built once per *key* and kept.

        A key that cannot be hashed (a setting such as a list) is built each
        time and not kept.
        """
        try:
            return self._kept[key]
        except KeyError:
            pass
        except TypeError:
            return build()
        value = self._kept[key] = build()
        return value

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return the run's four core tables keyed by name.

        This is what :func:`~ob_analytics.data.save_data` saves for a result,
        and what :meth:`to_arrow` converts.

        Returns
        -------
        dict of str to pandas.DataFrame
            ``events``, ``trades``, ``depth`` and ``depth_summary``.  The keys
            are the same for every run: on an
            :attr:`~ob_analytics.protocols.Level.L2` run ``events`` is a
            schema-valid zero-row frame rather than a missing key.
        """
        return {
            "events": self.events,
            "trades": self.trades,
            "depth": self.depth,
            "depth_summary": self.depth_summary,
        }

    def to_arrow(self) -> dict[str, pa.Table]:
        """Return the run's four core tables as Arrow tables.

        Each table carries the same key-value metadata a canonical Parquet file
        carries — the schema version, the run's tick and lot size, its config,
        its declarations and the loader's counts (see
        :mod:`ob_analytics.schemas`) — so a reader
        handed these tables in memory is no worse off than one reading the
        files.

        Returns
        -------
        dict of str to pyarrow.Table
            ``events``, ``trades``, ``depth`` and ``depth_summary``.

        Examples
        --------
        >>> from ob_analytics import Pipeline, sample_csv_path
        >>> tables = Pipeline().run(sample_csv_path()).to_arrow()  # doctest: +SKIP
        >>> sorted(tables)  # doctest: +SKIP
        ['depth', 'depth_summary', 'events', 'trades']
        """
        from ob_analytics.data import _to_arrow_table

        tick_sizes = tick_sizes_from_config(self.config)
        lot_sizes = lot_sizes_from_config(self.config)
        return {
            name: _to_arrow_table(
                df,
                tick_sizes=tick_sizes,
                lot_sizes=lot_sizes,
                config=self.config,
                declarations=self.declarations,
                loader_counts=loader_counts(df.attrs),
            )
            for name, df in self.tables().items()
        }

    def to_polars(self) -> dict[str, Any]:
        """Return the run's four core tables as Polars DataFrames.

        Polars is **not** a dependency of ob-analytics (see
        ``adr/0002-dataframe-library.md``): the public API takes and returns
        pandas, and this accessor is a convenience for users who already have
        Polars installed.  Install it yourself with ``pip install polars``.

        Returns
        -------
        dict of str to polars.DataFrame
            The same keys as :meth:`to_arrow`.

        Raises
        ------
        ImportError
            When polars is not installed.

        Notes
        -----
        Polars keeps no schema-level key-value metadata, so the schema version
        and tick size that :meth:`to_arrow` attaches do not survive this
        conversion.  Read the tick size from ``result.config.tick_size``, or use
        :meth:`to_arrow` when the metadata has to travel with the tables.
        """
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "PipelineResult.to_polars() requires polars, which is not a "
                "dependency of ob-analytics: pip install polars. "
                "Use to_arrow() for a pyarrow table instead."
            ) from exc

        return {name: pl.from_arrow(table) for name, table in self.to_arrow().items()}

    def audit(self, *, venue_book: BookLevels | None = None) -> DataQualitySummary:
        """Check this run's data quality against its :attr:`declarations`.

        The same checks as ``ob-analytics audit``: this calls
        :func:`~ob_analytics.quality.data_quality_summary` with the run's
        tables, its faithful ``depth``, its tick size and its declarations.

        Parameters
        ----------
        venue_book : BookLevels, optional
            The venue's own book after every message, when the source
            supplies one (LOBSTER's orderbook file, read by
            :meth:`~ob_analytics.lobster.LobsterSource.venue_book`).  The
            book rebuilt from the events is then compared with it.

        Returns
        -------
        DataQualitySummary
        """
        declared = self.declarations
        return data_quality_summary(
            self.events,
            self.trades,
            feed_type=declared.feed_type,
            depth=self.depth,
            tick_size=self.config.tick_size,
            sequence_kind=declared.sequence_kind,
            sequence_restarts=declared.sequence_restarts,
            trade_attribution=declared.trade_attribution,
            clocks=declared.clocks,
            trade_order=declared.trade_order,
            venue_book=venue_book,
        )

    def metric(self, name: str, **settings: Any) -> pd.DataFrame:
        """Compute the registered metric *name* over this result.

        Metrics are computed on demand, not stored: a run pays for a metric
        only when it is asked for, and a third-party metric that raises cannot
        break :meth:`Pipeline.run`.

        Parameters
        ----------
        name : str
            Registered metric name (case-insensitive), e.g. ``"amihud"``.
            See :func:`~ob_analytics.metrics.list_metrics`.
        **settings
            Keyword arguments for the metric's
            :meth:`~ob_analytics.protocols.Metric.compute`, e.g.
            ``result.metric("vpin", n_buckets=20)``.  Leave them out to
            use the metric's defaults.

        Returns
        -------
        pandas.DataFrame
            The metric's own table, as its
            :meth:`~ob_analytics.protocols.Metric.compute` returns it.

        Raises
        ------
        UnknownNameError
            If no metric is registered under *name*; the message names
            the registered names closest to it.
        """
        from ob_analytics.metrics import get_metric

        return get_metric(name).compute(self, **settings)

    def metrics(self) -> dict[str, pd.DataFrame]:
        """Compute every registered metric that applies to this run.

        A metric declares the resolutions it applies to (
        :attr:`~ob_analytics.protocols.Metric.levels`) and the tables it needs
        (:attr:`~ob_analytics.protocols.Metric.needs`), so an L3-only metric
        is skipped on an L2 run, and a trade metric on a run with no trades,
        rather than failing on an empty table.

        Returns
        -------
        dict of str to pandas.DataFrame
            Metric name → its table, for the metrics that apply to this run
            (see :func:`~ob_analytics.metrics.skip_reason`).
        """
        from ob_analytics.metrics import applicable_metrics

        return {
            metric.name: metric.compute(self) for metric in applicable_metrics(self)
        }

    def lifecycles(self) -> pd.DataFrame:
        """Return one row per order, from placement to outcome.

        The table :func:`~ob_analytics.book.order_lifecycles` builds from
        :attr:`events`.  It is built on the first call and kept.

        Returns
        -------
        pandas.DataFrame
            One row per order id, with the columns
            :func:`~ob_analytics.book.order_lifecycles` describes.

        Raises
        ------
        ConfigError
            On a run with no per-order events (an
            :attr:`~ob_analytics.protocols.Level.L2` run).
        """
        from ob_analytics.book import order_lifecycles

        return self._keep(
            _SAVABLE_TABLES["lifecycles"], lambda: order_lifecycles(self.events)
        )

    def queue(self, *, levels: str = "touch") -> pd.DataFrame:
        """Return each visible limit order's place in its price level's queue.

        The table :func:`~ob_analytics.queue.queue_positions` builds from
        :attr:`events`.  It is built on the first call for each *levels* and
        kept.

        Parameters
        ----------
        levels : {"touch", "all"}
            ``"touch"`` (the default) keeps the rows where the order rests at
            the best bid or ask.  ``"all"`` keeps every visible level.

        Returns
        -------
        pandas.DataFrame
            One row per order event, with the columns
            :func:`~ob_analytics.queue.queue_positions` describes.

        Raises
        ------
        ConfigError
            On a run with no per-order events, or for an unknown *levels*.
        """
        from ob_analytics.queue import queue_positions

        return self._keep(
            _queue_key(levels), lambda: queue_positions(self.events, levels=levels)
        )

    def book(
        self,
        at: datetime | pd.Timestamp | None = None,
        *,
        max_levels: int | None = None,
        bps_range: int = 0,
        uncross: bool = False,
    ) -> BookSnapshot:
        """Return the order book at *at*, rebuilt from :attr:`events`.

        The snapshot :func:`~ob_analytics.book.order_book` returns for the
        same arguments.  One snapshot is kept for each set of arguments, so a
        second call for the same instant costs nothing.  A call for a new
        instant rebuilds the book.

        Parameters
        ----------
        at : datetime.datetime or pandas.Timestamp, optional
            The instant to show.  ``None`` (the default) shows the book after
            the last event.
        max_levels, bps_range, uncross
            As :func:`~ob_analytics.book.order_book` takes them.

        Returns
        -------
        BookSnapshot

        Raises
        ------
        ConfigError
            On a run with no per-order events, or events without the order
            ``type`` column.
        """
        from ob_analytics.book import order_book

        return self._keep(
            ("book", at, max_levels, bps_range, uncross),
            lambda: order_book(
                self.events,
                at=at,
                max_levels=max_levels,
                bps_range=bps_range,
                uncross=uncross,
            ),
        )

    def bars(
        self,
        rule: str = "time",
        threshold: Any = None,
        *,
        target_bars: int = 50,
        sign_method: str | None = None,
    ) -> pd.DataFrame:
        """Return :attr:`trades` resampled into bars.

        The table :func:`~ob_analytics.bars.bars` builds from :attr:`trades`,
        with :attr:`depth_summary` as the quotes the Lee-Ready classifier
        reads.  It is built on the first call for each set of arguments and
        kept.

        Parameters
        ----------
        rule, threshold, target_bars, sign_method
            As :func:`~ob_analytics.bars.bars` takes them.

        Returns
        -------
        pandas.DataFrame
            One row per bar, with the columns
            :data:`~ob_analytics.bars.BAR_COLUMNS`.
        """
        from ob_analytics.bars import bars

        return self._keep(
            ("bars", rule, threshold, target_bars, sign_method),
            lambda: bars(
                self.trades,
                rule,
                threshold,
                target_bars=target_bars,
                sign_method=sign_method,
                quotes=self.depth_summary,
            ),
        )

    def features(
        self,
        rule: str = "time",
        threshold: Any = None,
        *,
        target_bars: int = 50,
        include: Sequence[str] | None = None,
        sign_method: str | None = None,
    ) -> pd.DataFrame:
        """Return the feature table of this run: one row per bar.

        The table :func:`~ob_analytics.features.features` builds from
        :attr:`trades`, with :attr:`depth_summary` as the book.  It is built
        on the first call for each set of arguments and kept.

        Parameters
        ----------
        rule, threshold, target_bars, include, sign_method
            As :func:`~ob_analytics.features.features` takes them.

        Returns
        -------
        pandas.DataFrame
            One row per bar, with the columns
            :func:`~ob_analytics.features.features` describes.
        """
        from ob_analytics.features import features

        # A bare string stays a string, so features() refuses it as before.
        names = (
            include if include is None or isinstance(include, str) else tuple(include)
        )
        return self._keep(
            ("features", rule, threshold, target_bars, names, sign_method),
            lambda: features(
                self.trades,
                self.depth_summary,
                rule,
                threshold,
                target_bars=target_bars,
                include=include,
                sign_method=sign_method,
            ),
        )

    def plot(
        self,
        concept: str,
        level: Level | str | None = _UNSET,
        *,
        backend: str = "matplotlib",
        ax: Axes | None = None,
        **overrides: Any,
    ) -> Any:
        """Render one plot *concept* from this result in a single call.

        The method form of :func:`ob_analytics.visualization.plot_result`,
        which takes the same arguments, e.g.
        ``result.plot("depth_heatmap", col_bias=0.1)``.  Pass *level* when
        this result has the concept at both L2 and L3.  See
        :func:`~ob_analytics.visualization.available_concepts` for what a given
        result can plot (it varies by format).
        """
        from ob_analytics.visualization import plot_result

        return plot_result(
            self,
            concept,
            level,
            backend=backend,
            ax=ax,
            **overrides,
        )


class Pipeline:
    """Configurable, composable order book analytics pipeline.

    Each processing stage is handled by a pluggable component that
    satisfies the corresponding protocol.  Pass your own implementations
    to override any stage.

    Parameters
    ----------
    config : PipelineConfig, optional
        Central configuration.  The fields it sets are laid over the source's
        instrument grid (``source.grid``), and the result is what each run's
        loader and trade reader are built from.
    source : OfflineSource, optional
        A source descriptor that builds each run's loader and trade reader, and
        gives the instrument grid the config starts from (its ``grid``).
        Defaults to :class:`~ob_analytics.bitstamp.BitstampSource`.  Explicit
        component arguments take precedence over the source's factories.
    loader : EventLoader, optional
        Loads raw events from a data source.  Overrides the source's loader,
        and is used for every run.
    trade_reader : TradeReader, optional
        Builds the trades DataFrame.  Overrides the source's trade reader, and
        is used for every run.
    ctx : RunContext, optional
        The per-run parameters (such as LOBSTER's ``trading_date``) every run
        uses unless :meth:`run` is given its own.

    Notes
    -----
    A live capture records the instrument's tick size and lot size in its
    ``meta.json`` (see :mod:`ob_analytics.capture_record`).  :meth:`run` and
    :meth:`run_windows` use them for that capture when *config* does not set
    them, so a capture replays on its own price and size grid, as
    ``ob-analytics process`` reads it.  A tick size you set (or a
    ``price_decimals``) keeps the recorded tick size out, and a lot size (or a
    ``volume_decimals``) the recorded lot size.  When you pass your own
    *loader* or *trade_reader*, the record is not used: those components hold
    their own config.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        source: Source | None = None,
        loader: EventLoader | None = None,
        trade_reader: TradeReader | None = None,
        ctx: RunContext | None = None,
    ) -> None:
        self._ctx = ctx or RunContext()
        if source is None:
            # Deferred import (not at module top) so the bitstamp source module
            # can import from pipeline without a cycle; this is the default
            # offline source when none is supplied.
            from ob_analytics.bitstamp import BitstampSource

            source = BitstampSource()
        if not isinstance(source, OfflineSource):
            raise TypeError(
                f"Pipeline needs an offline-capable source; {source.name!r} "
                "cannot replay stored files (it lacks create_loader or "
                "create_trade_reader)."
            )

        # What the caller set, so a capture's record fills only the rest.
        self._caller_fields = frozenset(config.model_fields_set if config else ())
        # The source's grid underlies the fields the caller set, so a LOBSTER
        # run is on whole shares unless the caller says otherwise.
        self.config = grid_of(source).config(config)
        self._loader = loader
        self._trade_reader = trade_reader
        self._source = source

    def _for_run(self, path: Any, ctx: RunContext) -> tuple[PipelineConfig, Any, Any]:
        """Return the config, loader and trade reader for one run on *path*.

        The source builds the loader and trade reader for each run, from that
        run's config and *ctx*; a loader or trade reader the caller passed is
        used as it is.  The config is the pipeline's own, unless *path* is a
        capture whose record gives a tick or lot size the caller did not set.
        """
        config = self.config
        if self._loader is None and self._trade_reader is None:
            config = self._recorded_config(path)
        loader = self._loader or self._source.create_loader(config, ctx)
        trade_reader = self._trade_reader or self._source.create_trade_reader(
            config, ctx, loader
        )
        return config, loader, trade_reader

    def _recorded_config(self, path: Any) -> PipelineConfig:
        """Return the config with the grid a capture recorded, where unset."""
        recorded = read_instrument(path)
        fields = {
            name: value
            for name, value in recorded.items()
            for pair in _GRID_FIELDS
            if name in pair and not pair & self._caller_fields
        }
        if not fields:
            return self.config
        config = PipelineConfig(**{**self.config.model_dump(), **fields})
        if config != self.config:
            logger.info(
                "Pipeline: using the instrument the capture recorded: {}", fields
            )
        return config

    @classmethod
    def from_source(
        cls, name: str, *, ctx: RunContext | None = None, **kwargs: Any
    ) -> Pipeline:
        """Create a pipeline from a registered source name.

        Parameters
        ----------
        name : str
            Registered source name (case-insensitive), e.g. ``"bitstamp"``
            or ``"lobster"``.
        ctx : RunContext, optional
            Per-run parameters (e.g. ``trading_date``) forwarded to
            ``OfflineSource.create_*`` factories.
        **kwargs
            Passed to the :class:`~ob_analytics.protocols.Source` constructor.

        Raises
        ------
        UnknownNameError
            If no source is registered under *name*; the message names
            the registered names closest to it.
        """
        source = get_source(name)(**kwargs)
        return cls(source=source, ctx=ctx)

    def run(self, path: Any, *, ctx: RunContext | None = None) -> PipelineResult:
        """Execute the full pipeline on *path* and return results.

        Parameters
        ----------
        path
            What the loader reads, typically a file or directory path.
        ctx : RunContext, optional
            Override the pipeline's default :class:`RunContext` for this
            single call.  When ``None``, the ``ctx`` provided at
            construction (or the default empty context) is used.

        Returns
        -------
        PipelineResult
            Frozen dataclass with ``events``, ``trades``, ``depth``,
            ``depth_summary``, ``config``, ``level`` and ``declarations``.

        Steps (L3 / per-order feeds)
        ----------------------------
        1. Load events (``EventLoader.load``)
        2. Build trades (``TradeReader.load``)
        3. Classify order types
        4. Compute price-level depth
        5. Compute depth metrics, and sign the trades the venue left
           unlabelled
        6. Compute order aggressiveness

        Steps 3 to 6 are ``STAGES`` in this module.  For an
        :attr:`~ob_analytics.protocols.Level.L2` source the loader yields the
        depth frame directly, depth metrics and trade signs are computed on
        it, and the per-order stages (3, 6) are skipped.
        """
        run_ctx = ctx if ctx is not None else self._ctx
        config, loader, trade_reader = self._for_run(path, run_ctx)
        declarations = run_declarations(path, self._source)
        rows, trades, loaded_attrs = self._load_inputs(path, loader, trade_reader)

        if self._source.level is Level.L2:
            # A price-level feed has no per-order identity, so the per-order
            # stages (set_order_types, order_aggressiveness) are skipped
            # and ``events`` comes back empty but schema-valid.
            tables = {"events": empty_events(), "trades": trades, "depth": rows}
            _depth_metrics(tables, config)
            return _result(tables, config, Level.L2, declarations)

        tables = {"events": rows, "trades": trades}
        for name, stage in STAGES:
            if name == "price_level_volume":
                depth_override = (
                    self._source.compute_depth(
                        tables["events"], config, path, run_ctx, loader
                    )
                    if isinstance(self._source, SupportsComputeDepth)
                    else None
                )
                if depth_override is not None:
                    tables["depth"], tables["depth_summary"] = depth_override
                    logger.info(
                        "Pipeline: using source-provided depth ({} rows, {} "
                        "summary rows)",
                        len(tables["depth"]),
                        len(tables["depth_summary"]),
                    )
            stage(tables, config)
        tables["events"].attrs.update(loaded_attrs)
        return _result(tables, config, Level.L3, declarations)

    def _load_inputs(
        self, path: Any, loader: Any, trade_reader: Any
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Load the rows the depth stages read, and the trades.

        For a per-order source the rows are the events, and for an
        :attr:`~ob_analytics.protocols.Level.L2` source the price-level depth
        frame.  The trades are as the trade reader builds them: the ones the
        venue left unlabelled are signed later, against the run's quotes.

        Returns
        -------
        tuple
            The rows, the trades, and what the loader recorded on the rows'
            ``attrs`` (such as the records the venue flagged after a gap).  The
            later stages build new frames, and pandas does not carry ``attrs``
            over, so a caller that keeps them puts them back at the end.
        """
        if self._source.level is Level.L2:
            logger.info(
                "Pipeline: L2 resolution — loading price-level depth from {}", path
            )
            depth = loader.load(path)
            validate_depth_df(depth)  # data contract (schemas.py)
            logger.info("Pipeline: building trades")
            # The trade reader ignores the (empty) events frame for L2: trades
            # come from the venue's own prints, not from order lifecycles.
            trades = trade_reader.load(empty_events(), path)
            validate_trades_df(trades)  # data contract (schemas.py)
            return depth, trades, dict(depth.attrs)

        logger.info("Pipeline: loading events from {}", path)
        events = loader.load(path)
        loaded_attrs = dict(events.attrs)
        logger.info("Pipeline: building trades")
        trades = trade_reader.load(events, path)
        validate_trades_df(trades)  # data contract (schemas.py)
        return events, trades, loaded_attrs

    def run_windows(
        self,
        path: Any,
        boundaries: Iterable[Any],
        output: str | Path,
        *,
        carry: bool = True,
        ctx: RunContext | None = None,
    ) -> Path:
        """Run the pipeline one time window at a time and write the result.

        Cuts *path* at each of the *boundaries*, runs the depth stages on one
        window at a time, and writes each window's rows to *output* as soon as
        the window is done.  The depth stages are what makes a run large, so
        their peak memory is set by the largest window rather than by the whole
        input.  The output is one Parquet file per table, the same folder a
        single run saved with :func:`~ob_analytics.data.save_data` gives, and
        :func:`~ob_analytics.data.load_data` reads it back.

        With *carry* on, each window starts from the book the previous window
        ended with, so the cuts do not show in the output: every table matches
        a single run row for row, though ``events`` comes out in window order.
        With *carry* off, each window starts from an empty book, as if it were
        a separate input: the orders resting at a cut are missing from the next
        window's depth.

        Parameters
        ----------
        path
            What the loader reads, typically a file or directory path, as for
            :meth:`run`.
        boundaries : time or iterable of times
            Where to cut, anything :class:`pandas.Timestamp` accepts; one time
            on its own is one cut.  A time with no zone is read as UTC.  *n* cuts make *n + 1* windows that
            cover the whole input: a window holds the rows at or after its start
            and before its end.
        output : str or Path
            The folder to write ``events``, ``trades``, ``depth`` and
            ``depth_summary`` to, created when missing.  Files already there
            under those names are replaced, but only once every window is done:
            a run that fails part-way leaves the folder as it was.
        carry : bool, optional
            Start each window from the previous window's book.  Default
            ``True``.
        ctx : RunContext, optional
            The context for this run, as for :meth:`run`.

        Returns
        -------
        Path
            The output folder.

        Raises
        ------
        ConfigError
            If no boundary is given, if the boundaries do not strictly
            increase, or if no window produced rows for one of the tables.

        Notes
        -----
        A loader that can read one window at a time does so when the source
        built both the loader and the trade reader.  The loaders of the
        Databento, Bitstamp and L2 CSV sources can; so can every capture that
        replays through the Bitstamp or the L2 CSV source.  The run then never
        holds more than one window's events or depth rows.

        For per-order events the run goes over the windows twice.  The first
        time it reads each window and saves its events, and the few values per
        order that the order types are read from, to a working folder inside
        *output*.  The trade reader then builds the trades from what the
        loader kept: the Databento loader keeps the trade records, and the
        Bitstamp loader matches the trades to their orders as it reads.  Order
        types and trade signs are decided over the whole run, as in a single
        run: the order types one group of orders at a time, from the saved
        values.  The second time, the run reads each saved window back and
        runs the depth stages on it.  While it does, it holds only the time of
        each trade; the trades wait in the working folder until the last
        window is done.  *carry* does not change how such a loader reads: an
        order's size and price always carry from one window to the next, as
        they do when the input is read whole.

        A loader that reads a CSV file (Bitstamp, L2) first sorts its rows on
        disk, in the working folder, so the file need not be in time order.
        The Bitstamp loader sorts them by order, puts each order's rows in
        order, and then sorts them by window.  The Databento loader reads the
        file in order, so its records must be in receive-time order.

        Any other loader, and a loader the caller passes, reads the whole
        input, and the ``events`` table (or, for a price-level source, the
        depth rows) is held for the whole run.

        A source's own depth is not used.  LOBSTER's order book file states the
        book after each message of the whole session, so it cannot be cut; a
        windowed LOBSTER run rebuilds its depth from the messages instead.
        """
        windows = window_bounds(boundaries)
        level = self._source.level
        config, loader, trade_reader = self._for_run(
            path, ctx if ctx is not None else self._ctx
        )
        declarations = run_declarations(path, self._source)

        with ExitStack() as stack:
            # The trade reader builds the trades from what the loader kept, so
            # both must be the pair the source built.
            windowed = (
                self._loader is None
                and self._trade_reader is None
                and isinstance(loader, WindowedLoader)
            )
            Path(output).mkdir(parents=True, exist_ok=True)
            working = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix=".run_windows-", dir=output)
                )
            )
            if level is Level.L3 and windowed:
                trades, typed_windows, like, counts = _read_windows(
                    path, loader, trade_reader, windows, working
                )
            elif windowed:
                trades, typed_windows, like, counts = _read_depth_windows(
                    path, loader, trade_reader, windows, working
                )
            else:
                trades, typed_windows, like, counts = self._cut_inputs(
                    path, loader, trade_reader, windows, config
                )
            # A loader can keep what it read for its trade reader, as the
            # Databento loader keeps the trade records.  The trades are built,
            # so let both go before the depth stages.
            del loader, trade_reader
            # The trades are signed and written once the last window is done.
            # Until then only their times are needed, so the rest waits on disk.
            saved_trades = working / "trades.pkl"
            trades.to_pickle(saved_trades)
            del trades
            self._write_windows(
                windows,
                typed_windows,
                like,
                saved_trades,
                config,
                ParquetAppender(output, config, declarations, counts, level=level),
                carry=carry,
            )

        logger.info("Pipeline: wrote {} windows to {}", len(windows), output)
        return Path(output)

    def _cut_inputs(
        self,
        path: Any,
        loader: Any,
        trade_reader: Any,
        windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
        config: PipelineConfig,
    ) -> tuple[pd.DataFrame, Iterator[pd.DataFrame], pd.DataFrame, dict[str, int]]:
        """Load the whole input and cut its rows into *windows*.

        Returns the trades, each window's rows (order types set, for a
        per-order source), a frame holding a value of each column, and the
        loader's counts (none for a price-level source, as in :meth:`run`).
        """
        rows, trades, loaded_attrs = self._load_inputs(path, loader, trade_reader)
        counts: dict[str, int] = {}
        if self._source.level is Level.L3:
            counts = loader_counts(loaded_attrs)
            tables = {"events": rows, "trades": trades}
            # The stage builds a new events frame; let the loaded one go.
            del rows
            _set_order_types(tables, config)
            rows = tables.pop("events")
        positions = window_positions(rows["timestamp"], windows)
        return trades, (rows.iloc[p] for p in positions), rows, counts

    def _write_windows(
        self,
        windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
        typed_windows: Iterator[pd.DataFrame],
        like: pd.DataFrame,
        saved_trades: Path,
        config: PipelineConfig,
        appender: ParquetAppender,
        *,
        carry: bool,
    ) -> None:
        """Run the depth stages on each window in turn and write the tables.

        *saved_trades* is the run's trades, pickled.  Only their times are
        held while the windows run; the trades are read back at the end.
        """
        level = self._source.level
        trades = pd.read_pickle(saved_trades)
        trade_times = trades[["timestamp"]].copy()
        # The quotes are kept only for trades the venue left unlabelled.
        unlabelled = _unlabelled_trades(trades)
        del trades
        trade_windows = window_positions(trade_times["timestamp"], windows)
        arrived_into: list[pd.DataFrame] = []
        # Trades whose own window held no quote before them.
        no_quote = np.zeros(len(trade_times), dtype=bool)
        resting: pd.DataFrame | None = None
        engine = DepthSummaryEngine(config)
        last_quote = last_readable = pd.DataFrame()
        had_quotes = False
        with appender as out:
            for number, ((start, _), window, at_trades) in enumerate(
                zip(windows, typed_windows, trade_windows, strict=True), 1
            ):
                if resting is None:
                    resting = window.iloc[:0]
                logger.info(
                    "Pipeline: window {} of {} from {}: {} rows, {} orders carried in",
                    number,
                    len(windows),
                    start,
                    len(window),
                    len(resting),
                )
                if not carry:
                    engine = DepthSummaryEngine(config)
                    last_quote, last_readable = last_quote[:0], last_readable[:0]

                if level is Level.L2:
                    # A price-level row states its level's whole size, so the
                    # summary engine, which holds the book, is the whole carry.
                    depth = window
                else:
                    depth, resting = self._window_price_levels(
                        window, resting, start, carry=carry
                    )

                depth_summary = engine.compute(depth) if len(depth) else None
                quotes = _led_by(last_quote, depth_summary)
                if unlabelled.any():
                    readable = _led_by(
                        last_readable,
                        None
                        if depth_summary is None
                        else readable_quotes(depth_summary[_ARRIVAL_COLUMNS]),
                    )
                    _note_quotes(
                        arrived_into, no_quote, trade_times, at_trades, readable
                    )
                    last_readable = readable.tail(1)

                if level is Level.L3 and len(window):
                    if quotes.empty:
                        window = window.assign(aggressiveness_bps=np.nan)
                    else:
                        window = order_aggressiveness(window, quotes)
                    out.write("events", window, like=like)
                if depth_summary is not None:
                    had_quotes = had_quotes or not depth_summary.empty
                    out.write("depth", depth)
                    out.write("depth_summary", depth_summary)
                    last_quote = depth_summary.tail(1)

            if level is Level.L2:
                out.write("events", empty_events())
            # The rows the trades arrived into.  When the run had quotes but
            # none stood before a trade, Lee-Ready signs every trade by the
            # tick rule, as it would in a single run; a run with no quotes at
            # all leaves them empty, as a single run does.
            standing = (
                pd.concat(arrived_into, ignore_index=True) if arrived_into else None
            )
            trades = pd.read_pickle(saved_trades)
            trades = _sign_unlabelled(
                trades, standing, fallback="tick" if had_quotes else None
            )
            if standing is not None and not carry:
                # Without carry a window is a separate input: a trade with no
                # quote in its own window must not read one kept from an
                # earlier window.  Lee-Ready signs it by the tick rule.
                trades = _tick_signs(trades, unlabelled & no_quote)
            validate_trades_df(trades)
            out.write("trades", trades)
            out.finish(("events", "trades", "depth", "depth_summary"))

    @staticmethod
    def _window_price_levels(
        window: pd.DataFrame,
        resting: pd.DataFrame,
        start: pd.Timestamp | None,
        *,
        carry: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Rebuild one window's price levels from its events and the carried orders.

        The carried orders go back on their price levels as seed rows just
        before *start*, so the window's own rows change those levels from the
        right size.  The depth rows the seeds produce restate the previous
        window's book and are dropped.

        Returns
        -------
        tuple of pandas.DataFrame
            The window's depth rows, and the orders still resting at its end
            (none when *carry* is off).
        """
        seeds = seed_rows(resting, start) if start is not None else resting
        frame = pd.concat([seeds, window]) if len(seeds) else window
        if frame.empty:
            return frame[:0], frame
        depth = price_level_volume(frame)
        if len(seeds):
            depth = depth[depth["timestamp"] >= start]
        return depth, resting_orders(frame) if carry else frame.iloc[:0]


# ── Stages ─────────────────────────────────────────────────────────────
#
# Each stage reads the run's tables and adds or replaces some, in place.  A
# depth stage leaves a table a source already provided alone.


def _set_order_types(tables: dict[str, pd.DataFrame], config: PipelineConfig) -> None:
    """Classify each order's type from the events and trades."""
    logger.info("Pipeline: classifying order types")
    tables["events"] = set_order_types(tables["events"], tables["trades"])
    validate_events_df(tables["events"])  # data contract (schemas.py)


def _price_level_volume(
    tables: dict[str, pd.DataFrame], config: PipelineConfig
) -> None:
    """Rebuild the volume at each price level from the events."""
    if "depth" not in tables:
        logger.info("Pipeline: computing price-level volume")
        tables["depth"] = price_level_volume(tables["events"])


def _depth_metrics(tables: dict[str, pd.DataFrame], config: PipelineConfig) -> None:
    """Summarise the depth, then sign the trades the venue left unlabelled.

    Signing needs the quotes, which exist only from here on.  It is a no-op
    for a feed that states the aggressor on every trade, which is every L3
    crypto source; Databento leaves auction and off-exchange prints unset.
    """
    if "depth_summary" not in tables:
        logger.info(
            "Pipeline: building the depth summary ({} depth rows)", len(tables["depth"])
        )
        tables["depth_summary"] = summarize_depth(
            tables["depth"], depth_bps=config.depth_bps, depth_bins=config.depth_bins
        )
    tables["trades"] = _sign_unlabelled(tables["trades"], tables["depth_summary"])
    validate_trades_df(tables["trades"])  # data contract (schemas.py)


def _order_aggressiveness(
    tables: dict[str, pd.DataFrame], config: PipelineConfig
) -> None:
    """Measure how far into the book each order was placed."""
    logger.info("Pipeline: computing order aggressiveness")
    tables["events"] = order_aggressiveness(tables["events"], tables["depth_summary"])


#: The stages of a per-order run after loading, in the order :meth:`Pipeline.run`
#: runs them.  ``scripts/bench_scale.py`` times the same stages one by one.
STAGES: tuple[
    tuple[str, Callable[[dict[str, pd.DataFrame], PipelineConfig], None]], ...
] = (
    ("set_order_types", _set_order_types),
    ("price_level_volume", _price_level_volume),
    ("depth_metrics", _depth_metrics),
    ("order_aggressiveness", _order_aggressiveness),
)


def _result(
    tables: dict[str, pd.DataFrame],
    config: PipelineConfig,
    level: Level,
    declarations: FeedDeclarations,
) -> PipelineResult:
    """Return the run's tables as a result, with the run's units on each."""
    events, trades = tables["events"], tables["trades"]
    depth, depth_summary = tables["depth"], tables["depth_summary"]
    put_config_units(config, events, trades, depth, depth_summary)
    logger.info(
        "Pipeline: complete — {:,} events, {:,} trades, {:,} depth rows",
        len(events),
        len(trades),
        len(depth),
    )
    return PipelineResult(
        events=events,
        trades=trades,
        depth=depth,
        depth_summary=depth_summary,
        config=config,
        level=level,
        declarations=declarations,
    )


def _sign_unlabelled(
    trades: pd.DataFrame,
    quotes: pd.DataFrame | None,
    *,
    fallback: str | None = None,
) -> pd.DataFrame:
    """Label the trades the venue left unlabelled, by Lee-Ready against *quotes*.

    *quotes* is ``None`` or empty when the run produced no quotes at all,
    and the trades then get *fallback* (see
    :func:`~ob_analytics.trade_sign.resolve_direction`).

    L3 crypto states the taker side on every trade; many price-level venues
    (and CCXT sources) do not, and Databento sends none for an auction, a
    trade against a non-displayed order, or an off-exchange print.  Those rows
    are labelled by :func:`~ob_analytics.trade_sign.resolve_direction`, the
    same rule the signed-flow analytics use, against the quote each trade
    arrived into.  Rows the venue *did* label keep the venue's answer: a
    classifier is an estimate and the venue's is not.  A run with no quotes
    at all leaves the unlabelled trades empty, with a warning, rather than
    guessing.
    """
    if trades.empty:
        return trades
    if quotes is not None and quotes.empty:
        quotes = None
    return resolve_direction(trades, None, quotes, "Pipeline", fallback=fallback)


def _tick_signs(trades: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Return *trades* with the *rows* signed by the tick rule."""
    if not rows.any():
        return trades
    direction = trades["direction"].copy()
    direction[rows] = classify_trade_sign(trades, sign_method="tick")[rows]
    return trades.assign(direction=direction)


def _note_quotes(
    arrived_into: list[pd.DataFrame],
    no_quote: np.ndarray,
    trade_times: pd.DataFrame,
    positions: np.ndarray,
    quotes: pd.DataFrame,
) -> None:
    """Keep the readable quote each of one window's trades arrived into.

    Trade signs are decided once, over every trade, after the last window,
    because the tick rule they fall back on reads the trades before and after.
    The quotes they need are only in memory one window at a time, so each
    window leaves behind the rows its trades will read: at most one per trade.
    Each trade still finds its own row among them, because no readable quote
    falls between a trade and the row it arrived into.  *quotes* are the
    window's readable quotes, led by the last one before the window;
    *positions* are the window's trades, as positions in *trade_times*, the
    ``timestamp`` of every trade of the run.  The trades with no quote before
    them in the window are marked in *no_quote*.
    """
    if not len(positions):
        return
    if quotes.empty:
        no_quote[positions] = True
        return
    standing = quote_before(trade_times.iloc[positions], quotes)
    no_quote[positions[standing < 0]] = True
    rows = np.unique(standing[standing >= 0])
    if len(rows):
        arrived_into.append(quotes.iloc[rows])


def _read_windows(
    path: Any,
    loader: WindowedLoader,
    trade_reader: Any,
    windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
    working: Path,
) -> tuple[pd.DataFrame, Iterator[pd.DataFrame], pd.DataFrame, dict[str, int]]:
    """Read *path* one window at a time, saving each window's events.

    An order's type is decided over the whole run, so the first pass saves
    the events to *working*, with what the rule reads of them
    (:class:`~ob_analytics._windows.OrderTypeFiles`).  The trade reader then
    builds the whole run's trades from what the loader kept, and the order
    types are decided from the saved values, a group of orders at a time.

    Returns
    -------
    tuple
        The trades; each window's events with their order types, read back
        from *working* one at a time; a frame holding a value of each text
        column, for the output's column types; and the loader's counts over
        the whole input, the sum of each window's.
    """
    logger.info("Pipeline: reading {} one window at a time", path)
    paths: list[Path] = []
    types = OrderTypeFiles(working)
    texts: dict[str, Any] = {}
    counts: dict[str, int] = {}
    events = empty_events()
    loaded = _loaded_windows(loader, path, windows, working / "loader")
    for number, (events, named) in enumerate(loaded):
        types.add(_OrderFacts.of(events, named))
        for key, count in loader_counts(events.attrs).items():
            counts[key] = counts.get(key, 0) + count
        for column in events.columns:
            if column in texts or events[column].dtype != object:
                continue
            first = events[column].first_valid_index()
            if first is not None:
                texts[column] = events[column].loc[first]
        paths.append(working / f"window-{number}.pkl")
        events.to_pickle(paths[-1])
    if types.empty:
        require_per_order_events(events, "set_order_types")
    del events

    logger.info("Pipeline: building trades")
    # The loader kept what the trade reader reads (see WindowedLoader).
    trades = trade_reader.load(empty_events(), path)
    validate_trades_df(trades)  # data contract (schemas.py)
    logger.info("Pipeline: classifying order types")
    types.decide(trades)

    def typed() -> Iterator[pd.DataFrame]:
        # Row labels go on across the windows, as they do in a frame read
        # whole: the next window's seed rows come from this one's.
        start = 0
        for saved in paths:
            window = types.with_types(pd.read_pickle(saved))
            saved.unlink()
            window.index = pd.RangeIndex(start, start + len(window))
            start += len(window)
            validate_events_df(window)  # data contract (schemas.py)
            yield window

    like = pd.DataFrame({c: pd.Series([v], dtype=object) for c, v in texts.items()})
    return trades, typed(), like, counts


def _loaded_windows(
    loader: WindowedLoader,
    path: Any,
    windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
    working: Path,
) -> Iterator[LoadedWindow]:
    """Return what *loader* yields for *windows*.

    The loader gets *working*, inside the run's own working folder, for the
    files it may keep while it reads: the Bitstamp and L2 CSV loaders sort
    their rows there.
    """
    return loader.load_windows(path, windows, working=working)


def _read_depth_windows(
    path: Any,
    loader: WindowedLoader,
    trade_reader: Any,
    windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
    working: Path,
) -> tuple[pd.DataFrame, Iterator[pd.DataFrame], pd.DataFrame, dict[str, int]]:
    """Read the price-level depth of *path* one window at a time.

    The trades of a price-level source come from the venue's own prints, not
    from the depth, so the trade reader reads them first, given an empty
    events frame, as in a single run.

    Returns
    -------
    tuple
        The trades; each window's depth rows, read as they are needed; an
        empty events frame, for the output's column types; and no counts, as
        a single run of a price-level source records none.
    """
    logger.info("Pipeline: L2 resolution — reading {} one window at a time", path)
    logger.info("Pipeline: building trades")
    trades = trade_reader.load(empty_events(), path)
    validate_trades_df(trades)  # data contract (schemas.py)

    def checked() -> Iterator[pd.DataFrame]:
        for depth, _ in _loaded_windows(loader, path, windows, working / "loader"):
            validate_depth_df(depth)  # data contract (schemas.py)
            yield depth

    return trades, checked(), empty_events(), {}


def _led_by(
    last_quote: pd.DataFrame, depth_summary: pd.DataFrame | None
) -> pd.DataFrame:
    """Return *depth_summary* with the previous window's last row in front.

    The quote standing when a window opens is the last one of the window
    before.  A lookup of the quote standing at an order or a trade early in the
    window needs that row; it is not written again.
    """
    if depth_summary is None or depth_summary.empty:
        return last_quote
    if last_quote.empty:
        return depth_summary
    return pd.concat([last_quote, depth_summary], ignore_index=True)
