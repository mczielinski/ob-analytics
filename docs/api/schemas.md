---
title: Data Contracts
---

# Data Contracts

Column-list constants and validators defining the DataFrame contracts at
package boundaries (replaces the old Pydantic `models.py`). Validators check
that the required columns are present and that the events and trades columns
have the dtypes in `EVENT_DTYPES` and `TRADE_DTYPES`; extra columns are
allowed.

::: ob_analytics.schemas.validate_events_df

::: ob_analytics.schemas.validate_trades_df

::: ob_analytics.schemas.validate_depth_df

::: ob_analytics.schemas.EVENT_DTYPES

::: ob_analytics.schemas.TRADE_DTYPES

::: ob_analytics.schemas.cast_events

::: ob_analytics.schemas.MAYBE_BAD_BOOK_ATTR
