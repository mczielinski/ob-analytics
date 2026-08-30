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

from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from ob_analytics._utils import empty_events
from ob_analytics.analytics import order_aggressiveness, set_order_types
from ob_analytics.config import PipelineConfig
from ob_analytics.depth import depth_metrics, price_level_volume
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
from ob_analytics.trade_sign import classify_trade_sign


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

    @staticmethod
    def _ensure_trade_signs(
        trades: pd.DataFrame, depth_summary: pd.DataFrame
    ) -> pd.DataFrame:
        """Fill an unlabelled L2 trades ``direction`` via Lee–Ready.

        L3 crypto ships the taker side for free; many price-level venues (and
        CCXT sources) don't.  When the trade reader leaves ``direction``
        entirely unset, classify the aggressor with Lee–Ready against the
        reconstructed BBO (``depth_summary``), falling back to the tick rule
        at the mid — the trade-sign classifiers added for exactly this case
        (see :mod:`ob_analytics.trade_sign`).  A reader that *does* label the
        side (native ``side`` column) is left untouched.
        """
        if trades.empty or "direction" not in trades.columns:
            return trades
        if not trades["direction"].isna().all():
            return trades  # venue already labelled the aggressor side
        logger.info("Pipeline: classifying {} trade signs (Lee–Ready)", len(trades))
        direction = classify_trade_sign(
            trades, method="lee_ready", quotes=depth_summary
        )
        return trades.assign(direction=direction)
