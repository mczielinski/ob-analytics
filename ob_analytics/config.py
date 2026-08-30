"""Pipeline configuration for ob-analytics.

Centralises the numeric thresholds and parameters that were previously
scattered as literals across multiple modules.
"""

from typing import Literal

from pydantic import BaseModel, Field


class SourceSettings(BaseModel):
    """Base class for a data source's typed, immutable settings.

    The typed replacement for the untyped per-source settings dict that live
    capturers used to carry (``CaptureConfig.extras``).  Every
    :class:`~ob_analytics.protocols.Source` declares a ``settings`` value of
    this type; a source that needs no configuration uses the empty base, and a
    source with venue knobs subclasses it with typed, validated fields — e.g.
    :class:`~ob_analytics.live.ccxt_source.CcxtSettings` (``exchange`` /
    ``depth_limit`` / ``poll_interval``).

    Frozen so a source's settings are fixed for the run, matching
    :class:`PipelineConfig`.
    """

    model_config = {"frozen": True}


class PipelineConfig(BaseModel):
    """Validated, immutable configuration for the ob-analytics pipeline.

    Every parameter that was previously a hard-coded literal now lives here
    with a sensible default matching the original R package behaviour (Bitstamp
    BTC/USD, 2015).  Override individual values for different instruments,
    exchanges, or precision requirements.
    """

    model_config = {"frozen": True}

    # ── Price / volume precision ──────────────────────────────────────────
    tick_size: float = Field(
        default=0.01,
        gt=0,
        description=(
            "The instrument's minimum price increment, in the quote currency "
            "(issue #155).  Prices are stored as a whole number of ticks "
            "(``int64``); the quote-currency price is ``ticks * tick_size``.  "
            "0.01 (default) is a cent grid (USD equities, BTC-USD); use the "
            "venue's real tick for small-tick crypto or 0-1 prediction markets. "
            "By default it matches ``price_decimals`` (``10 ** -price_decimals``)."
        ),
    )
    price_decimals: int = Field(
        default=2,
        ge=0,
        le=18,
        description=(
            "Display precision: decimal places to show when a tick price is "
            "rendered back to the quote currency for a plot or CSV.  2 for USD "
            "equities / BTC-USD; 8 for satoshi-denominated pairs; 4-5 for FX.  "
            "The stored price grid is ``tick_size``, not this — see issue #155."
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

    price_divisor: int = Field(
        default=1,
        ge=1,
        description=(
            "Raw-feed encoding scale: the divisor that turns a source's raw "
            "integer price into the quote currency, before it is converted to "
            "ticks.  1 (default) means the raw price is already in the quote "
            "currency (Bitstamp).  LOBSTER uses 10 000 (prices are in "
            "ten-thousandths of a dollar).  This is the feed's encoding, "
            "separate from the instrument's ``tick_size``."
        ),
    )

    # ── Sequence / ordering keys ──────────────────────────────────────────
    track_sequence: bool = Field(
        default=False,
        description=(
            "Attach the ordering-key columns to loaded frames (see "
            "ob_analytics.schemas): the local monotonic 'ingest_seq' counter, "
            "and the venue 'sequence' number when the source carries one.  Off "
            "by default so the standard pipeline output is unchanged; turn it "
            "on to detect dropped or reordered messages via "
            "ob_analytics.analytics.detect_sequence_gaps."
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

    # ── Derived helpers ───────────────────────────────────────────────────
    @property
    def price_multiplier(self) -> int:
        """Integer inverse of :attr:`tick_size` (``round(1 / tick_size)``).

        The multiplier that turns a quote-currency price into an integer tick
        count, defined only when the tick is a reciprocal integer (the usual
        case: a cent, nickel, or quarter grid).  With the default ``tick_size``
        of ``0.01`` this is ``100``, matching the former ``10 ** price_decimals``.

        Raises
        ------
        ValueError
            If :attr:`tick_size` has no integer inverse; convert prices with
            ``ob_analytics._utils.price_to_ticks`` (which divides) instead.
        """
        from ob_analytics._utils import tick_multiplier

        multiplier = tick_multiplier(self.tick_size)
        if multiplier is None:
            raise ValueError(
                f"tick_size={self.tick_size!r} has no integer inverse; use "
                "ob_analytics._utils.price_to_ticks to convert prices to ticks."
            )
        return multiplier

    @property
    def bps_labels(self) -> list[str]:
        """Column suffixes for depth-metric BPS bins (e.g. '25bps', '50bps' …)."""
        return [f"{i * self.depth_bps}bps" for i in range(1, self.depth_bins + 1)]
