"""Pydantic domain models for ob-analytics.

These models serve as **data contracts** at package boundaries:

* **Input validation**: raw data → validated model → DataFrame
* **Output serialisation**: DataFrame → validated model → export
* **Documentation**: each model defines exactly which fields the pipeline
  expects and produces, replacing implicit column-name conventions.

The pipeline continues to use DataFrames internally for performance.
"""



from datetime import datetime
from decimal import Decimal
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field


class OrderEvent(BaseModel):
    """A single order-book event as received from the exchange.

    Corresponds to one row of the events DataFrame that flows through
    ``loadEventData → eventMatch → setOrderTypes → orderAggressiveness``.
    """

    model_config = {"frozen": True}

    event_id: int = Field(description="Sequential event identifier assigned during loading.")
    order_id: int = Field(description="Exchange-assigned order identifier.")
    timestamp: datetime
    exchange_timestamp: datetime
    price: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)
    action: Literal["created", "changed", "deleted"]
    direction: Literal["bid", "ask"]
    fill: Decimal = Field(default=Decimal(0), ge=0)
    raw_event_type: str | int | None = Field(
        default=None,
        description=(
            "Preserved source-specific event type for loaders that expose "
            "richer semantics (e.g. LOBSTER event codes 1-7)."
        ),
    )


class Trade(BaseModel):
    """An inferred trade execution.

    Corresponds to one row of the trades DataFrame produced by
    ``matchTrades``.
    """

    model_config = {"frozen": True}

    timestamp: datetime
    price: Decimal = Field(gt=0)
    volume: Decimal = Field(gt=0)
    direction: Literal["buy", "sell"]
    maker_event_id: int
    taker_event_id: int


class DepthLevel(BaseModel):
    """Volume at a single price level at a point in time.

    Corresponds to one row of the depth DataFrame produced by
    ``priceLevelVolume``.
    """

    model_config = {"frozen": True}

    timestamp: datetime
    price: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)
    side: Literal["bid", "ask"]


class OrderBookSnapshot(BaseModel):
    """Full order-book state at a specific point in time.

    Returned by ``order_book()`` and useful for serialisation or display.
    """

    model_config = {"frozen": True}

    timestamp: datetime
    bids: list[DepthLevel] = Field(default_factory=list)
    asks: list[DepthLevel] = Field(default_factory=list)


class KyleLambdaResult(BaseModel):
    """Result of a Kyle's Lambda OLS regression.

    Attributes
    ----------
    lambda_ : float
        Slope coefficient — price change per unit of signed order flow.
        Higher values indicate a less liquid (more adverse-selection-
        prone) market.
    t_stat : float
        *t*-statistic for ``lambda_``.  ``|t_stat| > 2`` is a common
        threshold for statistical significance.
    r_squared : float
        Coefficient of determination (fraction of ΔPrice variance
        explained by signed order flow).
    n_windows : int
        Number of time windows used in the regression.
    regression_df : pandas.DataFrame
        Per-window data with columns ``timestamp``, ``delta_price``,
        ``signed_volume`` — useful for scatter-plot visualisation.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    lambda_: float
    t_stat: float
    r_squared: float
    n_windows: int
    regression_df: pd.DataFrame
