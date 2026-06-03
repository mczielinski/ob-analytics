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

Usage with a Format descriptor::

    from ob_analytics import Pipeline, BitstampFormat

    result = Pipeline(format=BitstampFormat()).run("my_data/orders.csv")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ob_analytics.config import PipelineConfig
from ob_analytics.depth import depth_metrics, price_level_volume
from ob_analytics.analytics import order_aggressiveness, set_order_types
from ob_analytics.schemas import validate_events_df, validate_trades_df
from ob_analytics._registry import Registry
from loguru import logger

from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    Format,
    RunContext,
    TradeSource,
)


# ── Format registry ───────────────────────────────────────────────────
#
# Defined *before* the ``bitstamp`` import below so format modules can
# self-register at import time. ``bitstamp`` and ``lobster`` end with
# ``from ob_analytics.pipeline import register_format``; that runs while this
# module is still partially initialised, so ``register_format`` must already
# exist at that point.

FORMATS: Registry[str, type[Format]] = Registry("format")


def register_format(name: str, format_cls: type[Format]) -> None:
    """Register a :class:`Format` implementation under *name* for lookup via
    :meth:`Pipeline.from_format`.
    """
    FORMATS.register(name, format_cls)


def list_formats() -> list[str]:
    """Return a sorted list of registered format names."""
    return FORMATS.list()


# Imported here (not with the other top-of-module imports) so the registry
# above already exists when the format modules self-register. See the note
# above the registry definition.
from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader  # noqa: E402


@dataclass(frozen=True)
class PipelineResult:
    """Immutable container for the core outputs of a pipeline run.

    Analytic outputs (VPIN, OFI, Kyle's λ) are intentionally **not** stored
    here — compute them post-pipeline from ``trades`` and pass them to the
    gallery via ``extra_panels=`` (see the migration guide).

    Attributes
    ----------
    events, trades, depth, depth_summary : pandas.DataFrame
        Core pipeline tables.
    config : PipelineConfig
        The configuration used for the run.
    """

    events: pd.DataFrame
    trades: pd.DataFrame
    depth: pd.DataFrame
    depth_summary: pd.DataFrame
    config: PipelineConfig


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
    format : Format, optional
        A format descriptor that provides default loader, trade source,
        writer, and config overrides.  Explicit component arguments take
        precedence over format defaults.
    loader : EventLoader, optional
        Loads raw events from a data source.  Defaults to
        :class:`BitstampLoader`.
    trade_source : TradeSource, optional
        Builds the trades DataFrame.  Defaults to
        :class:`BitstampTradeReader`.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        format: Format | None = None,
        loader: EventLoader | None = None,
        trade_source: TradeSource | None = None,
        ctx: RunContext | None = None,
    ) -> None:
        self._ctx = ctx or RunContext()
        if format is not None:
            defaults = format.config_defaults()
            if config is None:
                config = PipelineConfig(**defaults)
            self.config = config
            self.loader = loader or format.create_loader(config, self._ctx)
            self.trade_source = trade_source or format.create_trade_source(
                config, self._ctx
            )
            self._writer: DataWriter | None = format.create_writer(config, self._ctx)
        else:
            self.config = config or PipelineConfig()
            self.loader = loader or BitstampLoader(self.config)
            if trade_source is not None:
                self.trade_source = trade_source
            else:
                self.trade_source = BitstampTradeReader(self.config)
            self._writer = None

        self._format = format

    @property
    def writer(self) -> DataWriter | None:
        """The format-provided writer, if any."""
        return self._writer

    @classmethod
    def from_format(
        cls, name: str, *, ctx: RunContext | None = None, **kwargs: Any
    ) -> Pipeline:
        """Create a pipeline from a registered format name.

        Parameters
        ----------
        name : str
            Registered format name (case-insensitive), e.g. ``"bitstamp"``
            or ``"lobster"``.
        ctx : RunContext, optional
            Per-run parameters (e.g. ``trading_date``) forwarded to
            ``Format.create_*`` factories.
        **kwargs
            Passed to the :class:`Format` constructor.
        """
        try:
            fmt_cls = FORMATS.get(name)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        fmt = fmt_cls(**kwargs)
        return cls(format=fmt, ctx=ctx)

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
            ``depth_summary``, and ``config``.

        Steps
        -----
        1. Load events (``EventLoader.load``)
        2. Build trades (``TradeSource.load``)
        3. Classify order types
        4. Compute price-level depth
        5. Compute depth metrics
        6. Compute order aggressiveness
        """
        run_ctx = ctx if ctx is not None else self._ctx

        logger.info("Pipeline: loading events from {}", source)
        events = self.loader.load(source)

        logger.info("Pipeline: building trades")
        trades = self.trade_source.load(events, source)
        validate_trades_df(trades)  # data contract (schemas.py)

        logger.info("Pipeline: classifying order types")
        events = set_order_types(events, trades)
        validate_events_df(events)  # data contract (schemas.py)

        depth_override = None
        if self._format is not None:
            depth_override = self._format.compute_depth(
                events, self.config, source, run_ctx
            )

        if depth_override is not None:
            depth, depth_summary = depth_override
            logger.info(
                "Pipeline: using format-provided depth ({} rows, {} summary rows)",
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
        )
