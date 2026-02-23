"""Composable pipeline for limit order book analytics.

:class:`Pipeline` orchestrates the full processing sequence using
pluggable components that satisfy the protocols defined in
:mod:`ob_analytics.protocols`.

Usage with defaults (Bitstamp CSV, Needleman-Wunsch matching)::

    from ob_analytics.pipeline import Pipeline

    result = Pipeline().run("orders.csv")
    print(result.events.shape, result.trades.shape)

Usage with custom configuration::

    from ob_analytics.pipeline import Pipeline
    from ob_analytics.config import PipelineConfig

    config = PipelineConfig(match_cutoff_ms=1000, price_jump_threshold=50.0)
    result = Pipeline(config=config).run("orders.csv")

Usage with a custom loader (any object satisfying EventLoader)::

    Pipeline(loader=my_custom_loader).run("data/feed.csv")
"""


from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ob_analytics.config import PipelineConfig
from ob_analytics.data import get_zombie_ids
from ob_analytics.depth import depth_metrics, price_level_volume
from ob_analytics.event_processing import BitstampLoader, order_aggressiveness
from ob_analytics.flow_toxicity import compute_vpin, order_flow_imbalance
from ob_analytics.matching_engine import NeedlemanWunschMatcher
from ob_analytics.order_types import set_order_types
from loguru import logger

from ob_analytics.protocols import EventLoader, MatchingEngine, TradeInferrer
from ob_analytics.trades import DefaultTradeInferrer


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
    """

    events: pd.DataFrame
    trades: pd.DataFrame
    depth: pd.DataFrame
    depth_summary: pd.DataFrame
    vpin: pd.DataFrame | None = None
    ofi: pd.DataFrame | None = None


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
    loader : EventLoader, optional
        Loads raw events from a data source.  Defaults to
        :class:`BitstampLoader`.
    matcher : MatchingEngine, optional
        Pairs bid/ask fills.  Defaults to
        :class:`NeedlemanWunschMatcher`.
    trade_inferrer : TradeInferrer, optional
        Builds trade records from matched events.  Defaults to
        :class:`DefaultTradeInferrer`.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        loader: EventLoader | None = None,
        matcher: MatchingEngine | None = None,
        trade_inferrer: TradeInferrer | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.loader = loader or BitstampLoader(self.config)
        self.matcher = matcher or NeedlemanWunschMatcher(self.config)
        self.trade_inferrer = trade_inferrer or DefaultTradeInferrer(self.config)

    def run(self, source: str | Path) -> PipelineResult:
        """Execute the full pipeline on *source* and return results.

        Parameters
        ----------
        source : str or Path
            Path to the raw events file (e.g. a Bitstamp CSV).

        Returns
        -------
        PipelineResult
            Frozen dataclass with ``events``, ``trades``, ``depth``,
            ``depth_summary``, and optionally ``vpin`` and ``ofi``
            DataFrames.

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

        logger.info("Pipeline: detecting zombie orders")
        zombie_ids = get_zombie_ids(events, trades)
        if zombie_ids:
            logger.info("Pipeline: removing {} zombie orders", len(zombie_ids))
        events = events[~events["id"].isin(zombie_ids)]

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

        logger.info("Pipeline: complete")
        return PipelineResult(
            events=events,
            trades=trades,
            depth=depth,
            depth_summary=depth_summary,
            vpin=vpin_df,
            ofi=ofi_df,
        )
