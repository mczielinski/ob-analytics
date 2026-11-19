"""Convert a run's tables from integer ticks and lots to display units.

A pipeline result stores each price as a whole number of ticks and each size
as a whole number of lots (see :mod:`ob_analytics.schemas`).  The plots, and
anyone who wants to read the numbers in the quote currency and the
instrument's own units, need floats instead.  :func:`display_result` makes
that conversion.  Which columns are prices and which are sizes comes from the
table contracts in :data:`ob_analytics.schemas.TABLES`, so this module keeps
no column list of its own.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from ob_analytics._utils import lots_to_size, ticks_to_price
from ob_analytics.schemas import PRICE, SIZE, TABLES, Kind

__all__ = ["display_result"]

_ResultT = TypeVar("_ResultT")


def display_result(result: _ResultT) -> _ResultT:
    """Return *result* with prices and sizes converted to display units.

    Canonical prices are integer ticks and canonical sizes integer lots
    (:mod:`ob_analytics.schemas`).  This converts each price column to the
    ``ticks * tick_size`` float in the quote currency, and each size column to
    the ``lots * lot_size`` float in the instrument's units, reading both steps
    and the display precision (``price_decimals``, ``volume_decimals``) from
    ``result.config``.  The table contracts in
    :data:`~ob_analytics.schemas.TABLES` say which columns are prices and which
    are sizes.

    The gallery and :func:`~ob_analytics.visualization.plot_result` call it
    once, so every face works in display units.  Call it yourself before the
    low-level ``prepare.*`` builders when you plot straight from a result, or
    to read a run's tables in the quote currency.

    Idempotent and legacy-safe: only integer columns are scaled, so a result
    whose prices or sizes are already floats (a file written before prices and
    sizes were integers, or a result already passed through here) is returned
    unchanged.

    Parameters
    ----------
    result : PipelineResult
        A pipeline result, or any dataclass with a ``config`` and table fields
        named as in :data:`~ob_analytics.schemas.TABLES`.

    Returns
    -------
    PipelineResult
        A new result of the same type, with each table field converted.
    """
    config = getattr(result, "config", None)
    tick_size = getattr(config, "tick_size", 1.0)
    decimals = getattr(config, "price_decimals", None)
    lot_size = getattr(config, "lot_size", 1.0)
    size_decimals = getattr(config, "volume_decimals", None)
    converters: dict[Kind, Callable[[np.ndarray], np.ndarray]] = {
        PRICE: lambda a: ticks_to_price(a, tick_size, decimals=decimals),
        SIZE: lambda a: lots_to_size(a, lot_size, decimals=size_decimals),
    }

    changes: dict[str, pd.DataFrame] = {}
    for field in dataclasses.fields(_dataclass(result)):
        table = TABLES.get(field.name)
        frame = getattr(result, field.name)
        if table is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        converted = frame
        for kind, convert in converters.items():
            present = [
                c
                for c in table.columns_of(frame.columns, kind)
                if pd.api.types.is_integer_dtype(frame[c])
            ]
            if not present:
                continue
            if converted is frame:
                converted = frame.copy()
            for column in present:
                converted[column] = convert(converted[column].to_numpy())
        if converted is not frame:
            changes[field.name] = converted
    return dataclasses.replace(_dataclass(result), **changes)


def _dataclass(result: Any) -> Any:
    # ``dataclasses.fields`` and ``replace`` need a dataclass instance; say
    # what is wrong rather than letting them fail on a plain object.
    if not dataclasses.is_dataclass(result) or isinstance(result, type):
        raise TypeError(
            f"display_result needs a pipeline result, got {type(result).__name__}."
        )
    return result
