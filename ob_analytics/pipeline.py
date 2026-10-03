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

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from ob_analytics._utils import empty_events
from ob_analytics._windows import (
    ParquetAppender,
    resting_orders,
    seed_rows,
    window_bounds,
    window_positions,
)
from ob_analytics.analytics import order_aggressiveness, set_order_types
from ob_analytics.config import PipelineConfig
from ob_analytics.depth import DepthMetricsEngine, depth_metrics, price_level_volume
from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    Level,
    OfflineSource,
    RunContext,
    Source,
    TradeSource,
)
from ob_analytics.schemas import (
    validate_depth_df,
    validate_events_df,
    validate_trades_df,
)
from ob_analytics.sources import get_source
from ob_analytics.trade_sign import classify_trade_sign, prevailing_mid


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
    """

    events: pd.DataFrame
    trades: pd.DataFrame
    depth: pd.DataFrame
    depth_summary: pd.DataFrame
    config: PipelineConfig
    level: Level = Level.L3

    def _frames(self) -> dict[str, pd.DataFrame]:
        """Return the run's four core tables keyed by name.

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
        carries — the schema version and the run's tick size (see
        :mod:`ob_analytics.schemas`) — so a reader handed these tables in memory
        is no worse off than one reading the files.

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
        from ob_analytics.data import (
            _lot_sizes_from_config,
            _tick_sizes_from_config,
            _to_arrow_table,
        )

        tick_sizes = _tick_sizes_from_config(self.config)
        lot_sizes = _lot_sizes_from_config(self.config)
        return {
            name: _to_arrow_table(df, tick_sizes=tick_sizes, lot_sizes=lot_sizes)
            for name, df in self._frames().items()
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
            ``result.metric("vpin", bucket_volume=5.0)``.  Leave them out to
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
        Central configuration.  Passed to default components when they
        are not explicitly provided.
    source : OfflineSource, optional
        A source descriptor that provides the default loader, trade source,
        writer, and config overrides.  Defaults to
        :class:`~ob_analytics.bitstamp.BitstampSource`.  Explicit component
        arguments take precedence over the source's factories.
    loader : EventLoader, optional
        Loads raw events from a data source.  Overrides the source's loader.
    trade_source : TradeSource, optional
        Builds the trades DataFrame.  Overrides the source's trade source.
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
                "cannot replay stored files (it has no create_loader)."
            )

        # The source's config defaults underlie the fields the caller
        # explicitly set.  Without this, Pipeline(config=..., source=
        # LobsterSource()) silently dropped price_divisor=10_000 and produced
        # prices wrong by four orders of magnitude.
        defaults = source.config_defaults()
        if config is None:
            config = PipelineConfig(**defaults)
        else:
            explicit = {k: getattr(config, k) for k in config.model_fields_set}
            config = PipelineConfig(**{**defaults, **explicit})
        self.config = config
        self.loader = loader or source.create_loader(config, self._ctx)
        self.trade_source = trade_source or source.create_trade_source(
            config, self._ctx
        )
        self._writer: DataWriter | None = source.create_writer(config, self._ctx)
        self._source = source

    @property
    def writer(self) -> DataWriter | None:
        """The source-provided writer, if any."""
        return self._writer

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
            ``depth_summary``, ``config``, and ``level``.

        Steps (L3 / per-order feeds)
        ----------------------------
        1. Load events (``EventLoader.load``)
        2. Build trades (``TradeSource.load``)
        3. Classify order types
        4. Compute price-level depth
        5. Compute depth metrics
        6. Compute order aggressiveness

        For an :attr:`~ob_analytics.protocols.Level.L2` source the run takes
        the price-level path instead (see :meth:`_run_l2`): the loader yields
        the depth frame directly, depth metrics and trade signs are computed
        on it, and the per-order stages (3, 6) are skipped.
        """
        run_ctx = ctx if ctx is not None else self._ctx

        if self._source.level is Level.L2:
            return self._run_l2(source, run_ctx)

        logger.info("Pipeline: loading events from {}", source)
        events = self.loader.load(source)

        logger.info("Pipeline: building trades")
        trades = self.trade_source.load(events, source)
        validate_trades_df(trades)  # data contract (schemas.py)

        logger.info("Pipeline: classifying order types")
        events = set_order_types(events, trades)
        validate_events_df(events)  # data contract (schemas.py)

        depth_override = self._source.compute_depth(
            events, self.config, source, run_ctx
        )

        if depth_override is not None:
            depth, depth_summary = depth_override
            logger.info(
                "Pipeline: using source-provided depth ({} rows, {} summary rows)",
                len(depth),
                len(depth_summary),
            )
        else:
            logger.info("Pipeline: computing price-level volume")
            depth = price_level_volume(events)

            logger.info("Pipeline: computing depth metrics")
            depth_summary = depth_metrics(
                depth,
                bps=self.config.depth_bps,
                bins=self.config.depth_bins,
            )

        # Now that the quotes exist, label any trade the venue left unlabelled.
        # A no-op for a feed that states the aggressor on every trade, which is
        # every L3 crypto source; Databento leaves auction and off-exchange
        # prints unset.
        trades = self._ensure_trade_signs(trades, depth_summary)
        validate_trades_df(trades)  # data contract (schemas.py)

        logger.info("Pipeline: computing order aggressiveness")
        events = order_aggressiveness(events, depth_summary)

        logger.info("Pipeline: complete")
        return PipelineResult(
            events=events,
            trades=trades,
            depth=depth,
            depth_summary=depth_summary,
            config=self.config,
            level=Level.L3,
        )

    def _run_l2(self, source: Any, run_ctx: RunContext) -> PipelineResult:
        """Run the price-level (L2) path: depth in, per-order stages skipped.

        A price-level feed carries ``(price, side, new absolute size)``
        updates and no order IDs, so the reconstruction stages have nothing
        to key on.  The loader (a :class:`~ob_analytics.protocols.DepthSource`)
        yields the canonical depth frame directly; from there depth metrics
        and — for feeds whose trades don't label the aggressor — trade signs
        are computed, while ``set_order_types`` / ``order_aggressiveness`` /
        queue reconstruction are **skipped by construction** (no per-order
        identity to classify).  ``events`` comes back empty but schema-valid.
        """
        logger.info(
            "Pipeline: L2 resolution — loading price-level depth from {}", source
        )
        depth = self.loader.load(source)
        validate_depth_df(depth)  # data contract (schemas.py)

        logger.info("Pipeline: computing depth metrics ({} depth rows)", len(depth))
        depth_summary = depth_metrics(
            depth,
            bps=self.config.depth_bps,
            bins=self.config.depth_bins,
        )

        logger.info("Pipeline: building trades")
        # The trade source ignores the (empty) events frame for L2 — trades come
        # from the venue's own prints, not from reconstructed order lifecycles.
        events = empty_events()
        trades = self.trade_source.load(events, source)
        trades = self._ensure_trade_signs(trades, depth_summary)
        validate_trades_df(trades)  # data contract (schemas.py)

        logger.info(
            "Pipeline: L2 complete — per-order stages (set_order_types, "
            "order_aggressiveness, queue) skipped: price-level feed has no "
            "order identity"
        )
        return PipelineResult(
            events=events,
            trades=trades,
            depth=depth,
            depth_summary=depth_summary,
            config=self.config,
            level=Level.L2,
        )

    def run_windows(
        self,
        source: Any,
        boundaries: Iterable[Any],
        output: str | Path,
        *,
        carry: bool = True,
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
        The loader still reads the whole input, and the ``events`` and
        ``trades`` tables are held for the whole run: order types and trade
        signs are decided over every row, as in a single run.  Those two tables
        are a small part of a run's memory; ``depth_summary`` alone is more
        than twice the size of ``events``.

        Where the loader already warns that the depth is off (a Databento
        modify that carries a fill and also moves the order), the carried order
        goes onto its new price level, so the windowed depth can differ from a
        single run's by the same amount.

        A source's own depth is not used.  LOBSTER's order book file states the
        book after each message of the whole session, so it cannot be cut; a
        windowed LOBSTER run rebuilds its depth from the messages instead.
        """
        windows = window_bounds(boundaries)
        level = self._source.level

        if level is Level.L2:
            logger.info(
                "Pipeline: L2 resolution — loading price-level depth from {}", source
            )
            rows = self.loader.load(source)
            validate_depth_df(rows)
            trades = self.trade_source.load(empty_events(), source)
            validate_trades_df(trades)
        else:
            logger.info("Pipeline: loading events from {}", source)
            events = self.loader.load(source)

            logger.info("Pipeline: building trades")
            trades = self.trade_source.load(events, source)
            validate_trades_df(trades)

            logger.info("Pipeline: classifying order types")
            rows = set_order_types(events, trades)
            validate_events_df(rows)
            del events

        trade_windows = window_positions(trades["timestamp"], windows)
        mids = np.full(len(trades), np.nan)
        resting = rows.iloc[:0]
        engine = DepthMetricsEngine(self.config)
        last_quote = pd.DataFrame()
        with ParquetAppender(output, self.config) as out:
            for number, ((start, _), positions, at_trades) in enumerate(
                zip(
                    windows,
                    window_positions(rows["timestamp"], windows),
                    trade_windows,
                    strict=True,
                ),
                1,
            ):
                window = rows.iloc[positions]
                logger.info(
                    "Pipeline: window {} of {} from {}: {} rows, {} orders carried in",
                    number,
                    len(windows),
                    start,
                    len(window),
                    len(resting),
                )
                if not carry:
                    engine, last_quote = DepthMetricsEngine(self.config), last_quote[:0]

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
                _note_mids(mids, trades, at_trades, quotes)

                if level is Level.L3 and len(window):
                    if quotes.empty:
                        window = window.assign(aggressiveness_bps=np.nan)
                    else:
                        window = order_aggressiveness(window, quotes)
                    out.write("events", window, like=rows)
                if depth_summary is not None:
                    out.write("depth", depth)
                    out.write("depth_summary", depth_summary)
                    last_quote = depth_summary.tail(1)

            if level is Level.L2:
                out.write("events", empty_events())
            out.write("trades", self._signs_from_mids(trades, mids))
            out.finish(("events", "trades", "depth", "depth_summary"))

        logger.info("Pipeline: wrote {} windows to {}", len(windows), output)
        return Path(output)

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

    def _signs_from_mids(self, trades: pd.DataFrame, mids: np.ndarray) -> pd.DataFrame:
        """Label the unlabelled trades from the mids the windows recorded.

        The same Lee–Ready pass :meth:`run` makes, given each trade's own mid
        as its quote.  Trades that share an instant share a mid, so the lookup
        finds the same value the windows did.
        """
        if trades.empty:
            return trades
        quotes = pd.DataFrame({"timestamp": trades["timestamp"], "mid": mids})
        quotes = quotes.sort_values("timestamp", kind="stable").drop_duplicates(
            "timestamp", keep="last"
        )
        signed = self._ensure_trade_signs(trades, quotes)
        validate_trades_df(signed)
        return signed

    @staticmethod
    def _ensure_trade_signs(
        trades: pd.DataFrame, depth_summary: pd.DataFrame
    ) -> pd.DataFrame:
        """Fill any unlabelled trade ``direction`` via Lee–Ready.

        L3 crypto ships the taker side for free; many price-level venues (and
        CCXT sources) don't.  Where the trade reader left ``direction`` unset,
        classify the aggressor with Lee–Ready against the reconstructed BBO
        (``depth_summary``), falling back to the tick rule at the mid — the
        trade-sign classifiers added for exactly this case (see
        :mod:`ob_analytics.trade_sign`).  Rows the venue *did* label keep the
        venue's answer: a classifier is an estimate and the venue's is not.

        Part of a feed can be unlabelled while the rest is not.  Databento
        states the aggressor on most trades but sends none for an auction, a
        trade against a non-displayed order, or an off-exchange print, so the
        unset rows are filled one subset at a time rather than all or nothing.
        """
        if trades.empty or "direction" not in trades.columns:
            return trades
        unlabelled = trades["direction"].isna()
        if not unlabelled.any():
            return trades  # venue labelled every aggressor side
        if depth_summary is None or depth_summary.empty:
            logger.warning(
                "Pipeline: {} trades carry no aggressor side and there are no "
                "quotes to classify them against; leaving them unset",
                int(unlabelled.sum()),
            )
            return trades
        logger.info(
            "Pipeline: classifying {} of {} trade signs (Lee–Ready)",
            int(unlabelled.sum()),
            len(trades),
        )
        direction = classify_trade_sign(
            trades, method="lee_ready", quotes=depth_summary
        )
        # Through ``object`` and back: assigning one categorical into another
        # widens the column, and the trades schema wants the ordered
        # buy/sell categorical.
        filled = trades["direction"].astype(object)
        filled[unlabelled] = direction.astype(object)[unlabelled]
        return trades.assign(
            direction=pd.Categorical(filled, categories=["buy", "sell"], ordered=True)
        )


def _note_mids(
    mids: np.ndarray,
    trades: pd.DataFrame,
    positions: np.ndarray,
    quotes: pd.DataFrame,
) -> None:
    """Record the mid standing at each of one window's trades into *mids*.

    Trade signs are decided once, over every trade, after the last window
    (see :meth:`Pipeline._signs_from_mids`), because the tick rule they fall
    back on reads the trades before and after.  The quotes they need are only
    in memory one window at a time, so each window leaves its mids behind.
    *positions* are the window's trades, as positions in *trades*.
    """
    if quotes.empty or not len(positions):
        return
    times = trades["timestamp"].to_numpy()[positions]
    order = np.argsort(times, kind="stable")
    mids[positions[order]] = prevailing_mid(times[order], quotes)


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
