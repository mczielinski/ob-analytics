"""Composable pipeline for limit order book analytics.

:class:`Pipeline` orchestrates the full processing sequence using
pluggable components that satisfy the protocols defined in
:mod:`ob_analytics.protocols`.

Usage with defaults (Bitstamp CSV, Needleman-Wunsch matching)::

    from ob_analytics import Pipeline, sample_csv_path

    result = Pipeline().run(sample_csv_path())
    print(result.events.shape, result.trades.shape)

Usage with custom configuration::

    from ob_analytics import Pipeline, PipelineConfig, sample_csv_path

    config = PipelineConfig(match_cutoff_ms=1000, price_jump_threshold=50.0)
    result = Pipeline(config=config).run(sample_csv_path())

Usage with a custom loader (any object satisfying EventLoader)::

    Pipeline(loader=my_custom_loader).run("data/feed.csv")

Usage with a Format descriptor::

    from ob_analytics import Pipeline, BitstampFormat

    result = Pipeline(format=BitstampFormat()).run("my_data.csv")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ob_analytics.config import PipelineConfig
from ob_analytics.data import get_zombie_ids
from ob_analytics.depth import depth_metrics, price_level_volume
from ob_analytics.analytics import order_aggressiveness
from ob_analytics.bitstamp import BitstampLoader, BitstampMatcher, BitstampTradeInferrer
from ob_analytics.flow_toxicity import compute_vpin, order_flow_imbalance
from ob_analytics.analytics import set_order_types
from loguru import logger

from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    Format,
    MatchingEngine,
    TradeInferrer,
)


# ── Format registry ───────────────────────────────────────────────────

_FORMATS: dict[str, type[Format]] = {}


def register_format(name: str, format_cls: type[Format]) -> None:
    """Register a :class:`Format` subclass under *name* for lookup via
    :meth:`Pipeline.from_format`.
    """
    _FORMATS[name.lower()] = format_cls


def list_formats() -> list[str]:
    """Return a sorted list of registered format names."""
    return sorted(_FORMATS)


@dataclass(frozen=True)
class PipelineResult:
    """Immutable container for the outputs of a pipeline run.

    Attributes
    ----------
    events : pandas.DataFrame
        Processed events with order types and aggressiveness.
    trades : pandas.DataFrame
        Inferred trades with maker/taker attribution.
    depth : pandas.DataFrame
        Price-level volume time series.
    depth_summary : pandas.DataFrame
        Depth metrics (best bid/ask, BPS bins, spread).
    vpin : pandas.DataFrame or None
        VPIN buckets (only when ``config.vpin_bucket_volume`` is set).
    ofi : pandas.DataFrame or None
        Order flow imbalance per minute (only when VPIN is computed).
    metadata : dict
        Provenance and format-specific metadata populated during the run.
    """

    events: pd.DataFrame
    trades: pd.DataFrame
    depth: pd.DataFrame
    depth_summary: pd.DataFrame
    vpin: pd.DataFrame | None = None
    ofi: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
        A format descriptor that provides default loader, matcher,
        trade inferrer, writer, and config overrides.  Explicit
        component arguments take precedence over format defaults.
    loader : EventLoader, optional
        Loads raw events from a data source.  Defaults to
        :class:`BitstampLoader`.
    matcher : MatchingEngine, optional
        Pairs bid/ask fills.  Defaults to
        :class:`BitstampMatcher`.
    trade_inferrer : TradeInferrer, optional
        Builds trade records from matched events.  Defaults to
        :class:`BitstampTradeInferrer`.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        format: Format | None = None,
        loader: EventLoader | None = None,
        matcher: MatchingEngine | None = None,
        trade_inferrer: TradeInferrer | None = None,
    ) -> None:
        if format is not None:
            defaults = format.config_defaults()
            if config is None:
                config = PipelineConfig(**defaults)
            self.config = config
            self.loader = loader or format.create_loader(config)
            self.matcher = matcher or format.create_matcher(config)
            self.trade_inferrer = trade_inferrer or format.create_trade_inferrer(config)
            self._writer: DataWriter | None = format.create_writer(config)
        else:
            self.config = config or PipelineConfig()
            self.loader = loader or BitstampLoader(self.config)
            self.matcher = matcher or BitstampMatcher(self.config)
            self.trade_inferrer = trade_inferrer or BitstampTradeInferrer(self.config)
            self._writer = None

        self._format = format

    @property
    def writer(self) -> DataWriter | None:
        """The format-provided writer, if any."""
        return self._writer

    @classmethod
    def from_format(cls, name: str, **kwargs: Any) -> Pipeline:
        """Create a pipeline from a registered format name.

        Parameters
        ----------
        name : str
            Registered format name (case-insensitive), e.g. ``"bitstamp"``
            or ``"lobster"``.
        **kwargs
            Passed to the :class:`Format` constructor.
        """
        key = name.lower()
        if key not in _FORMATS:
            available = ", ".join(sorted(_FORMATS)) or "(none)"
            raise ValueError(
                f"Unknown format {name!r}. Registered formats: {available}"
            )
        fmt = _FORMATS[key](**kwargs)
        return cls(format=fmt)

    def run(self, source: Any) -> PipelineResult:
        """Execute the full pipeline on *source* and return results.

        Parameters
        ----------
        source
            Data source for the loader (typically a file path).

        Returns
        -------
        PipelineResult
            Frozen dataclass with ``events``, ``trades``, ``depth``,
            ``depth_summary``, and optionally ``vpin``, ``ofi``, and
            ``metadata`` fields.

        Steps
        -----
        1. Load events (``EventLoader.load``)
        2. Match bid/ask fills (``MatchingEngine.match``)
        3. Infer trades (``TradeInferrer.infer_trades``)
        4. Classify order types
        5. Remove zombie orders
        6. Compute price-level depth
        7. Compute depth metrics
        8. Compute order aggressiveness
        9. (Optional) Compute VPIN and order flow imbalance
        """
        logger.info("Pipeline: loading events from {}", source)
        events = self.loader.load(source)

        logger.info("Pipeline: matching {} events", len(events))
        events = self.matcher.match(events)

        logger.info("Pipeline: inferring trades")
        trades = self.trade_inferrer.infer_trades(events)

        logger.info("Pipeline: classifying order types")
        events = set_order_types(events, trades)

        if self.config.skip_zombie_detection:
            logger.info("Pipeline: zombie detection skipped (config)")
        else:
            logger.info("Pipeline: detecting zombie orders")
            zombie_ids = get_zombie_ids(events, trades)
            if zombie_ids:
                logger.info("Pipeline: removing {} zombie orders", len(zombie_ids))
            events = events[~events["id"].isin(zombie_ids)]

        depth_override = None
        if self._format is not None:
            depth_override = self._format.compute_depth(events, self.config, source)

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

        offset = pd.Timedelta(seconds=self.config.zombie_offset_seconds)
        depth_summary = depth_summary[
            depth_summary["timestamp"] >= events["timestamp"].min() + offset
        ]

        # ── Optional flow toxicity metrics ─────────────────────────────
        vpin_df = None
        ofi_df = None
        if self.config.vpin_bucket_volume is not None:
            logger.info(
                "Pipeline: computing VPIN (bucket_volume={})",
                self.config.vpin_bucket_volume,
            )
            vpin_df = compute_vpin(trades, self.config.vpin_bucket_volume)
            logger.info("Pipeline: computing order flow imbalance")
            ofi_df = order_flow_imbalance(trades)

        # ── Provenance metadata ────────────────────────────────────────
        metadata: dict[str, Any] = {
            "source": str(source),
            "format": self._format.name if self._format else None,
            "config": self.config.model_dump(),
            "n_events": len(events),
            "n_trades": len(trades),
        }

        logger.info("Pipeline: complete")
        return PipelineResult(
            events=events,
            trades=trades,
            depth=depth,
            depth_summary=depth_summary,
            vpin=vpin_df,
            ofi=ofi_df,
            metadata=metadata,
        )
