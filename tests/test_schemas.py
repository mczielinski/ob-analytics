"""Tests for the schemas.py column contract."""

import pandas as pd
import pytest

from ob_analytics import schemas
from ob_analytics.exceptions import InvalidDataError


def test_validate_events_df_accepts_valid(tiny_events):
    # tiny_events lacks 'type'/'id' in some fixtures; add the required set.
    df = tiny_events.assign(type="limit")
    if "id" not in df.columns:
        df = df.assign(id=df["event_id"])
    schemas.validate_events_df(df)  # must not raise


def test_validate_events_df_rejects_missing():
    with pytest.raises(InvalidDataError, match="missing required columns"):
        schemas.validate_events_df(pd.DataFrame({"price": [1.0]}))


def test_validate_depth_df_uses_direction(tiny_depth):
    schemas.validate_depth_df(tiny_depth)  # tiny_depth has 'direction'


def test_validate_trades_df_rejects_missing():
    with pytest.raises(InvalidDataError, match="missing required columns"):
        schemas.validate_trades_df(pd.DataFrame({"price": [1.0]}))
