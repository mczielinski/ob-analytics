"""Order-flow imbalance metric wrapper around :func:`order_flow_imbalance`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from ob_analytics.flow_toxicity import order_flow_imbalance

if TYPE_CHECKING:
    from ob_analytics.pipeline import PipelineResult


@dataclass(frozen=True)
class Ofi:
    """Normalised buy/sell volume imbalance per time window.

    Parameters
    ----------
    window : str
        Pandas offset alias for resampling (default ``"1min"``).
    """

    window: str = "1min"
    name: str = "ofi"
    requires: tuple[str, ...] = ("trades",)
    primary_column: str = "ofi"

    def compute(self, result: "PipelineResult", config: Any) -> pd.DataFrame:
        if result.trades is None or result.trades.empty:
            return pd.DataFrame(columns=["timestamp", self.primary_column])
        return order_flow_imbalance(result.trades, window=self.window)
