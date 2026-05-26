"""Kyle's Lambda metric wrapper around :func:`compute_kyle_lambda`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from ob_analytics.flow_toxicity import compute_kyle_lambda

if TYPE_CHECKING:
    from ob_analytics.pipeline import PipelineResult


@dataclass(frozen=True)
class KyleLambda:
    """Kyle (1985) λ price-impact coefficient.

    Parameters
    ----------
    window : str
        Pandas offset alias for the regression window (default ``"5min"``).
    """

    window: str = "5min"
    name: str = "kyle_lambda"
    requires: tuple[str, ...] = ("trades",)
    primary_column: str = "lambda"

    def compute(self, result: "PipelineResult", config: Any) -> pd.DataFrame:
        if result.trades is None or result.trades.empty:
            return pd.DataFrame(columns=["timestamp", self.primary_column])
        out = compute_kyle_lambda(result.trades, window=self.window)
        # KyleLambdaResult → single-row summary DataFrame for protocol parity.
        return pd.DataFrame(
            [
                {
                    "timestamp": result.trades["timestamp"].iloc[-1],
                    "lambda": out.lambda_,
                    "t_stat": out.t_stat,
                    "r_squared": out.r_squared,
                    "n_windows": out.n_windows,
                }
            ]
        )
