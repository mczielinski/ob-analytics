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

::: ob_analytics.schemas.HIDDEN_ORDER_ID

## Instrument columns

A frame that holds more than one instrument names each row's venue and symbol
in two optional columns.

::: ob_analytics.schemas.VENUE_COLUMN

::: ob_analytics.schemas.SYMBOL_COLUMN

::: ob_analytics.schemas.group_by_instrument
