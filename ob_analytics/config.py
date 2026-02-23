"""Pipeline configuration for ob-analytics.

Centralises the numeric thresholds and parameters that were previously
scattered as literals across multiple modules.
"""


from typing import Literal

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Validated, immutable configuration for the ob-analytics pipeline.

    Every parameter that was previously a hard-coded literal now lives here
    with a sensible default matching the original R package behaviour (Bitstamp
    BTC/USD, 2015).  Override individual values for different instruments,
    exchanges, or precision requirements.
    """

    model_config = {"frozen": True}

    # ── Price / volume precision ──────────────────────────────────────────
    price_decimals: int = Field(
        default=2,
        ge=0,
        le=18,
        description=(
            "Number of decimal places in price.  2 for USD equities / "
            "BTC-USD; 8 for satoshi-denominated pairs; 4-5 for FX."
        ),
    )
    volume_decimals: int = Field(
        default=8,
        ge=0,
        le=18,
        description="Number of decimal places in volume.",
    )
    timestamp_unit: Literal["ms", "us", "ns"] = Field(
        default="ms",
        description=(
            "Unit of raw integer timestamps in the source data.  "
            "'ms' (milliseconds, default) matches Bitstamp CSV format; "
            "'us' for microseconds; 'ns' for nanosecond-precision feeds."
        ),
    )

    # ── Matching engine ───────────────────────────────────────────────────
    match_cutoff_ms: int = Field(
        default=5000,
        gt=0,
        description=(
            "Maximum elapsed time (ms) between a bid fill and an ask fill "
            "for the Needleman-Wunsch matcher to consider them part of the "
            "same trade.  5 000 ms suits 2015 Bitstamp; modern HFT venues "
            "may need < 100 ms."
        ),
    )

    # ── Trade inference ───────────────────────────────────────────────────
    price_jump_threshold: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Absolute price difference between consecutive trades that "
            "triggers the maker/taker swap heuristic in matchTrades.  "
            "The original value ($10) targets 2015 BTC prices."
        ),
    )

    # ── Depth metrics ─────────────────────────────────────────────────────
    depth_bps: int = Field(
        default=25,
        gt=0,
        description="Width of each depth bin in basis points.",
    )
    depth_bins: int = Field(
        default=20,
        gt=0,
        description="Number of depth bins on each side of the book.",
    )
    # ── Data processing ───────────────────────────────────────────────────
    zombie_offset_seconds: int = Field(
        default=60,
        ge=0,
        description=(
            "Seconds to skip at the start of the depth summary to allow "
            "the order book to populate before computing metrics."
        ),
    )

    # ── Flow toxicity ─────────────────────────────────────────────────
    vpin_bucket_volume: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Volume per VPIN bucket.  When set, the pipeline computes "
            "VPIN and Order Flow Imbalance as part of the standard run.  "
            "When None (default), these metrics are skipped — call "
            "compute_vpin() directly with an instrument-appropriate "
            "bucket size."
        ),
    )

    # ── Derived helpers ───────────────────────────────────────────────────
    @property
    def price_multiplier(self) -> int:
        """Multiplier to convert float prices to integer price units."""
        return 10**self.price_decimals

    @property
    def bps_labels(self) -> list[str]:
        """Column suffixes for depth-metric BPS bins (e.g. '25bps', '50bps' …)."""
        return [
            f"{i * self.depth_bps}bps"
            for i in range(1, self.depth_bins + 1)
        ]
