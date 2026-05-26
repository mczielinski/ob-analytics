"""VPIN metric wrapper around :func:`compute_vpin`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from ob_analytics.flow_toxicity import compute_vpin

if TYPE_CHECKING:
    from ob_analytics.pipeline import PipelineResult


@dataclass(frozen=True)
class Vpin:
    """Volume-Synchronised Probability of Informed Trading.

    Parameters
    ----------
    bucket_volume : float
        Volume per bucket (instrument-specific; ~ ADV/50 is a starting point).
    n_buckets : int
        Trailing window length for ``vpin_avg``. Default 50.
    """

    bucket_volume: float
    n_buckets: int = 50
    name: str = "vpin"
    requires: tuple[str, ...] = ("trades",)
    primary_column: str = "vpin_avg"

    def compute(self, result: "PipelineResult", config: Any) -> pd.DataFrame:
        if result.trades is None or result.trades.empty:
            return pd.DataFrame(columns=["timestamp", self.primary_column])
        df = compute_vpin(result.trades, self.bucket_volume, n_buckets=self.n_buckets)
        if "timestamp_end" in df.columns:
            df = df.assign(timestamp=df["timestamp_end"])
        return df
