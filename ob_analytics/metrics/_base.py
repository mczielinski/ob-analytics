"""Protocol and shared types for pluggable order-flow / liquidity metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from ob_analytics.pipeline import PipelineResult


@runtime_checkable
class ToxicityMetric(Protocol):
    """A computation over a :class:`PipelineResult` that returns a tidy DataFrame.

    The contract is intentionally minimal: implementations decide their own
    inputs (consuming via ``requires``) and parameters (passed to ``__init__``).
    The output must be a DataFrame indexed or columned with a ``timestamp``
    field and at least one ``value`` column suitable for plotting.

    Attributes
    ----------
    name : str
        Stable lowercase identifier; used as the key in
        :attr:`PipelineResult.metrics` and as the gallery panel id.
    requires : tuple[str, ...]
        Names of :class:`PipelineResult` tables this metric reads, e.g.
        ``("trades",)`` or ``("trades", "depth_summary")``. Used by the
        runner to skip metrics whose inputs are absent.
    primary_column : str, optional
        Column name in the returned DataFrame that the gallery should
        plot by default. Defaults to ``"value"``.
    """

    name: str
    requires: tuple[str, ...]
    primary_column: str

    def compute(self, result: "PipelineResult", config: Any) -> pd.DataFrame:
        """Compute the metric.

        Must return a DataFrame with ``timestamp`` and at least one numeric
        column (``primary_column`` by default).
        """
        ...
