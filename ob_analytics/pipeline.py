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

    Pipeline(loader=my_custom_loader, trade_source=my_trade_source).run("data/")

Usage with a Source descriptor::

    from ob_analytics import Pipeline, BitstampSource

    result = Pipeline(source=BitstampSource()).run("my_data/orders.csv")
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from ob_analytics._utils import empty_events, validate_non_empty
from ob_analytics._windows import (
    ParquetAppender,
    WindowedLoader,
    resting_orders,
    seed_rows,
    window_bounds,
    window_positions,
)
from ob_analytics.analytics import (
    DataQualitySummary,
    _order_types,
    _OrderFacts,
    _with_order_types,
    data_quality_summary,
    order_aggressiveness,
    set_order_types,
)
from ob_analytics.capture_record import read_instrument, run_declarations
from ob_analytics.config import PipelineConfig, instrument_fields
from ob_analytics.depth import (
    BookLevels,
    DepthMetricsEngine,
    depth_metrics,
    price_level_volume,
    readable_quotes,
)
from ob_analytics.protocols import (
    EventLoader,
    FeedDeclarations,
    Level,
    OfflineSource,
    RunContext,
    Source,
    SupportsComputeDepth,
    TradeSource,
    grid_of,
)
from ob_analytics.schemas import (
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


@dataclass(frozen=True)
class PipelineResult:
    """Immutable container for the core outputs of a pipeline run.

    Analytic outputs (VPIN, OFI, Kyle's λ) are intentionally **not** stored
    here — compute them post-pipeline from ``trades`` and append them to the
    gallery model's ``analytics`` (build panels with the ``*_panel`` helpers).

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
    """

    events: pd.DataFrame
    trades: pd.DataFrame
    depth: pd.DataFrame
    depth_summary: pd.DataFrame
    config: PipelineConfig
    level: Level = Level.L3
    declarations: FeedDeclarations = field(default_factory=FeedDeclarations)

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
        carries — the schema version, the run's tick and lot size, its config
        and its declarations (see :mod:`ob_analytics.schemas`) — so a reader
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
        :func:`~ob_analytics.analytics.data_quality_summary` with the run's
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
        KeyError
            If no metric is registered under *name*; the message lists the
            registered names.
        """
        from ob_analytics.metrics import get_metric

        return get_metric(name).compute(self, **settings)

    def metrics(self) -> dict[str, pd.DataFrame]:
        """Compute every registered metric that applies to this run.

        A metric declares the resolutions it applies to (
        :attr:`~ob_analytics.protocols.Metric.levels`), so an L3-only metric is
        skipped on an L2 run rather than failing on its empty ``events`` table.

        Returns
        -------
        dict of str to pandas.DataFrame
            Metric name → its table, for the metrics whose ``levels`` include
            this run's :attr:`level`.
        """
        from ob_analytics.metrics import METRICS

        applicable = (METRICS.get(name) for name in METRICS.list())
        return {
            metric.name: metric.compute(self)
            for metric in applicable
            if self.level in metric.levels
        }

    def plot(
        self,
        concept: str,
        level: Any = None,
        *,
        backend: str = "matplotlib",
        volume_scale: float | None = None,
        **overrides: Any,
    ) -> Any:
        """Render one plot *concept* from this result in a single call.

        Thin convenience wrapper over
        :func:`ob_analytics.visualization.plot_result`, e.g.
        ``result.plot("depth_heatmap", col_bias=0.1)``.  See
        :func:`~ob_analytics.visualization.available_concepts` for what a given
        result can plot (it varies by format).
        """
        from ob_analytics.visualization import plot_result

        return plot_result(
            self,
            concept,
            level,
            backend=backend,
            volume_scale=volume_scale,
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
        loader and trade source are built from.
    source : OfflineSource, optional
        A source descriptor that builds each run's loader and trade source, and
        gives the instrument grid the config starts from (its ``grid``).
        Defaults to :class:`~ob_analytics.bitstamp.BitstampSource`.  Explicit
        component arguments take precedence over the source's factories.
    loader : EventLoader, optional
        Loads raw events from a data source.  Overrides the source's loader,
        and is used for every run.
    trade_source : TradeSource, optional
        Builds the trades DataFrame.  Overrides the source's trade source, and
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
    *loader* or *trade_source*, the record is not used: those components hold
    their own config.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        source: Source | None = None,
        loader: EventLoader | None = None,
        trade_source: TradeSource | None = None,
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
                "create_trade_source)."
            )

        # What the caller set, so a capture's record fills only the rest.
        self._caller_fields = frozenset(config.model_fields_set if config else ())
        # The source's grid underlies the fields the caller set, so a LOBSTER
        # run is on whole shares unless the caller says otherwise.
        self.config = grid_of(source).config(config)
        self._loader = loader
        self._trade_source = trade_source
        self._source = source

    def _for_run(self, source: Any, ctx: RunContext) -> tuple[PipelineConfig, Any, Any]:
        """Return the config, loader and trade source for one run on *source*.

        The source builds the loader and trade source for each run, from that
        run's config and *ctx*; a loader or trade source the caller passed is
        used as it is.  The config is the pipeline's own, unless *source* is a
        capture whose record gives a tick or lot size the caller did not set.
        """
        config = self.config
        if self._loader is None and self._trade_source is None:
            config = self._recorded_config(source)
        loader = self._loader or self._source.create_loader(config, ctx)
        trade_source = self._trade_source or self._source.create_trade_source(
            config, ctx, loader
        )
        return config, loader, trade_source

    def _recorded_config(self, source: Any) -> PipelineConfig:
        """Return the config with the grid a capture recorded, where unset."""
        recorded = read_instrument(source)
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
        """
        try:
            source_cls = get_source(name)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        source = source_cls(**kwargs)
        return cls(source=source, ctx=ctx)

    def run(self, source: Any, *, ctx: RunContext | None = None) -> PipelineResult:
        """Execute the full pipeline on *source* and return results.

        Parameters
        ----------
        source
            Data source for the loader (typically a file path).
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
        2. Build trades (``TradeSource.load``)
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
        config, loader, trade_source = self._for_run(source, run_ctx)
        declarations = run_declarations(source, self._source)
        rows, trades, loaded_attrs = self._load_inputs(source, loader, trade_source)

        if self._source.level is Level.L2:
            # A price-level feed has no per-order identity, so the per-order
            # stages (set_order_types, order_aggressiveness, queue) are skipped
            # and ``events`` comes back empty but schema-valid.
            tables = {"events": empty_events(), "trades": trades, "depth": rows}
            _depth_metrics(tables, config)
            return _result(tables, config, Level.L2, declarations)

        tables = {"events": rows, "trades": trades}
        for name, stage in STAGES:
            if name == "price_level_volume":
                depth_override = (
                    self._source.compute_depth(
                        tables["events"], config, source, run_ctx, loader
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
        self, source: Any, loader: Any, trade_source: Any
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Load the rows the depth stages read, and the trades.

        For a per-order source the rows are the events, and for an
        :attr:`~ob_analytics.protocols.Level.L2` source the price-level depth
        frame.  The trades are as the trade source builds them: the ones the
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
                "Pipeline: L2 resolution — loading price-level depth from {}", source
            )
            depth = loader.load(source)
            validate_depth_df(depth)  # data contract (schemas.py)
            logger.info("Pipeline: building trades")
            # The trade source ignores the (empty) events frame for L2: trades
            # come from the venue's own prints, not from order lifecycles.
            trades = trade_source.load(empty_events(), source)
            validate_trades_df(trades)  # data contract (schemas.py)
            return depth, trades, dict(depth.attrs)

        logger.info("Pipeline: loading events from {}", source)
        events = loader.load(source)
        loaded_attrs = dict(events.attrs)
        logger.info("Pipeline: building trades")
        trades = trade_source.load(events, source)
        validate_trades_df(trades)  # data contract (schemas.py)
        return events, trades, loaded_attrs

    def run_windows(
        self,
        source: Any,
        boundaries: Iterable[Any],
        output: str | Path,
        *,
        carry: bool = True,
        ctx: RunContext | None = None,
    ) -> Path:
        """Run the pipeline one time window at a time and write the result.

        Cuts *source* at each of the *boundaries*, runs the depth stages on one
        window at a time, and writes each window's rows to *output* as soon as
        the window is done.  The depth stages are what makes a run large, so
        their peak memory is set by the largest window rather than by the whole
        input.  The output is one Parquet file per table, the same folder a
        single run saved with :func:`~ob_analytics.data.save_data` gives, and
        :func:`~ob_analytics.data.load_data` reads it back.

        With *carry* on, each window starts from the book the previous window
        ended with, so the cuts do not show in the output: every table matches
        a single run row for row, though ``events`` comes out in window order.
        One column can differ.  ``aggressiveness_bps`` reads the quote standing
        before each order by ``event_id``, so on a source whose ``event_id`` is
        not in time order (Bitstamp numbers its events by order) it can read a
        quote from another window, and the value differs from a single run's.
        With *carry* off, each window starts from an empty book, as if it were
        a separate input: the orders resting at a cut are missing from the next
        window's depth.

        Parameters
        ----------
        source
            Data source for the loader (typically a file path), as for
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
        A loader that can read one window at a time (a
        :class:`~ob_analytics.databento.DatabentoLoader` can) does so when the
        source built both the loader and the trade source, which then builds
        the trades from what the loader kept.  The run then never holds more
        than one window's events, and goes over the windows twice.
        The first time it reads each window, keeps a few values per order and
        the trade records, and saves the window's events to a working folder
        inside *output*.  Order types and trade signs are decided over the
        whole run, as in a single run, so the second time it reads each saved
        window back and runs the depth stages on it.  *carry* does not change
        how such a loader reads: an order's size and price always carry from
        one window to the next, as they do when the input is read whole.

        Any other loader reads the whole input, and the ``events`` and
        ``trades`` tables are held for the whole run.  Those two tables are a
        small part of a run's memory; ``depth_summary`` alone is more than
        twice the size of ``events``.

        A source's own depth is not used.  LOBSTER's order book file states the
        book after each message of the whole session, so it cannot be cut; a
        windowed LOBSTER run rebuilds its depth from the messages instead.
        """
        windows = window_bounds(boundaries)
        level = self._source.level
        config, loader, trade_source = self._for_run(
            source, ctx if ctx is not None else self._ctx
        )
        declarations = run_declarations(source, self._source)

        with ExitStack() as stack:
            # The trade source builds the trades from what the loader kept, so
            # both must be the pair the source built.
            windowed = (
                self._loader is None
                and self._trade_source is None
                and isinstance(loader, WindowedLoader)
            )
            if level is Level.L3 and windowed:
                Path(output).mkdir(parents=True, exist_ok=True)
                working = stack.enter_context(
                    tempfile.TemporaryDirectory(prefix=".run_windows-", dir=output)
                )
                trades, typed_windows, like = _read_windows(
                    source, loader, trade_source, windows, Path(working)
                )
            else:
                trades, typed_windows, like = self._cut_inputs(
                    source, loader, trade_source, windows, config
                )
            self._write_windows(
                windows,
                typed_windows,
                like,
                trades,
                config,
                declarations,
                output,
                carry=carry,
            )

        logger.info("Pipeline: wrote {} windows to {}", len(windows), output)
        return Path(output)

    def _cut_inputs(
        self,
        source: Any,
        loader: Any,
        trade_source: Any,
        windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
        config: PipelineConfig,
    ) -> tuple[pd.DataFrame, Iterator[pd.DataFrame], pd.DataFrame]:
        """Load the whole input and cut its rows into *windows*.

        Returns the trades, each window's rows (order types set, for a
        per-order source), and a frame holding a value of each column.
        """
        rows, trades, _ = self._load_inputs(source, loader, trade_source)
        if self._source.level is Level.L3:
            tables = {"events": rows, "trades": trades}
            # The stage builds a new events frame; let the loaded one go.
            del rows
            _set_order_types(tables, config)
            rows = tables.pop("events")
        positions = window_positions(rows["timestamp"], windows)
        return trades, (rows.iloc[p] for p in positions), rows

    def _write_windows(
        self,
        windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
        typed_windows: Iterator[pd.DataFrame],
        like: pd.DataFrame,
        trades: pd.DataFrame,
        config: PipelineConfig,
        declarations: FeedDeclarations,
        output: str | Path,
        *,
        carry: bool,
    ) -> None:
        """Run the depth stages on each window in turn and write the tables."""
        level = self._source.level
        trade_windows = window_positions(trades["timestamp"], windows)
        arrived_into: list[pd.DataFrame] = []
        # Trades whose own window held no quote before them.
        no_quote = np.zeros(len(trades), dtype=bool)
        # The quotes are kept only for trades the venue left unlabelled.
        unlabelled = _unlabelled_trades(trades)
        resting: pd.DataFrame | None = None
        engine = DepthMetricsEngine(config)
        last_quote = last_readable = pd.DataFrame()
        had_quotes = False
        with ParquetAppender(output, config, declarations) as out:
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
                    engine = DepthMetricsEngine(config)
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
                    _note_quotes(arrived_into, no_quote, trades, at_trades, readable)
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
            "Pipeline: computing depth metrics ({} depth rows)", len(tables["depth"])
        )
        tables["depth_summary"] = depth_metrics(
            tables["depth"], bps=config.depth_bps, bins=config.depth_bins
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
    direction[rows] = classify_trade_sign(trades, method="tick")[rows]
    return trades.assign(direction=direction)


def _note_quotes(
    arrived_into: list[pd.DataFrame],
    no_quote: np.ndarray,
    trades: pd.DataFrame,
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
    *positions* are the window's trades, as positions in *trades*.  The trades
    with no quote before them in the window are marked in *no_quote*.
    """
    if not len(positions):
        return
    if quotes.empty:
        no_quote[positions] = True
        return
    standing = quote_before(trades["timestamp"].iloc[positions].to_frame(), quotes)
    no_quote[positions[standing < 0]] = True
    rows = np.unique(standing[standing >= 0])
    if len(rows):
        arrived_into.append(quotes.iloc[rows])


def _read_windows(
    source: Any,
    loader: WindowedLoader,
    trade_source: Any,
    windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
    working: Path,
) -> tuple[pd.DataFrame, Iterator[pd.DataFrame], pd.DataFrame]:
    """Read *source* one window at a time, saving each window's events.

    An order's type is decided over the whole run, so the first pass keeps
    only what the rule reads (:class:`~ob_analytics.analytics._OrderFacts`),
    and saves the events to *working*.  The trade source then builds the
    whole run's trades from what the loader kept.

    Returns
    -------
    tuple
        The trades; each window's events with their order types, read back
        from *working* one at a time; and a frame holding a value of each
        text column, for the output's column types.
    """
    logger.info("Pipeline: reading {} one window at a time", source)
    paths: list[Path] = []
    facts: list[_OrderFacts] = []
    texts: dict[str, Any] = {}
    events = empty_events()
    for number, (events, named) in enumerate(loader.load_windows(source, windows)):
        facts.append(_OrderFacts.of(events, named))
        for column in events.columns:
            if column in texts or events[column].dtype != object:
                continue
            first = events[column].first_valid_index()
            if first is not None:
                texts[column] = events[column].loc[first]
        paths.append(working / f"window-{number}.pkl")
        events.to_pickle(paths[-1])
    if not any(len(part.ids) for part in facts):
        validate_non_empty(events, "set_order_types")
    del events

    logger.info("Pipeline: building trades")
    # The loader kept what the trade source reads (see WindowedLoader).
    trades = trade_source.load(empty_events(), source)
    validate_trades_df(trades)  # data contract (schemas.py)
    logger.info("Pipeline: classifying order types")
    types = _order_types(_OrderFacts.combine(facts), trades)
    del facts

    def typed() -> Iterator[pd.DataFrame]:
        # Row labels go on across the windows, as they do in a frame read
        # whole: the next window's seed rows come from this one's.
        start = 0
        for path in paths:
            window = _with_order_types(pd.read_pickle(path), types)
            path.unlink()
            window.index = pd.RangeIndex(start, start + len(window))
            start += len(window)
            validate_events_df(window)  # data contract (schemas.py)
            yield window

    like = pd.DataFrame({c: pd.Series([v], dtype=object) for c, v in texts.items()})
    return trades, typed(), like


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
