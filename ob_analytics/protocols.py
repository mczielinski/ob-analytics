"""Protocol interfaces for ob-analytics.

These define the contracts that pluggable components must satisfy.
Implementations are discovered by structural (duck) typing -- there is
no need to inherit from these classes.

Default implementations ship with the package:

* :class:`BitstampLoader` in ``event_processing.py``
* :class:`NeedlemanWunschMatcher` in ``matching_engine.py``
* :class:`DefaultTradeInferrer` in ``trades.py``
* :class:`DepthMetricsEngine` in ``depth.py``

Users can substitute their own by passing any object that satisfies the
protocol to :class:`Pipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class EventLoader(Protocol):
    """Loads raw order-book events from a data source.

    The returned DataFrame must contain at least the columns documented
    in :class:`~ob_analytics.models.OrderEvent`.
    """

    def load(self, source: str | Path) -> pd.DataFrame: ...


@runtime_checkable
class MatchingEngine(Protocol):
    """Pairs bid and ask fills to identify which events are part of the same trade.

    Receives the events DataFrame (with ``fill`` column populated) and
    returns the same DataFrame with a ``matching.event`` column added.
    """

    def match(self, events: pd.DataFrame) -> pd.DataFrame: ...


@runtime_checkable
class TradeInferrer(Protocol):
    """Constructs trade records from matched event pairs.

    Receives the events DataFrame (with ``matching.event`` populated)
    and returns a trades DataFrame.
    """

    def infer_trades(self, events: pd.DataFrame) -> pd.DataFrame: ...
