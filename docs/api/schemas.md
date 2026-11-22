---
title: Data Contracts
---

# Data Contracts

The table contracts, the converter to display units, and the validators that
check a frame at a package boundary. Validators check
that the required columns are present and that the events and trades columns
have the dtypes in `EVENT_DTYPES` and `TRADE_DTYPES`; extra columns are
allowed.

## Table contracts

One contract per table the package returns. Each column has its dtype and says
whether it is a price, a size or neither. The [schema page](../schema.md)
describes the same tables.

::: ob_analytics.schemas.TABLES

::: ob_analytics.schemas.Table

::: ob_analytics.schemas.Column

::: ob_analytics.schemas.PRICE

::: ob_analytics.schemas.SIZE

::: ob_analytics.schemas.depth_bins

::: ob_analytics.schemas.DepthBin

::: ob_analytics.schemas.depth_bin_name

## Display units

::: ob_analytics.display.display_result

## Validators and dtypes

::: ob_analytics.schemas.validate_events_df

::: ob_analytics.schemas.validate_trades_df

::: ob_analytics.schemas.validate_depth_df

::: ob_analytics.schemas.EVENT_DTYPES

::: ob_analytics.schemas.TRADE_DTYPES

::: ob_analytics.schemas.cast_events

::: ob_analytics.schemas.MAYBE_BAD_BOOK_ATTR

::: ob_analytics.schemas.SET_ASIDE_ORDERS_ATTR

::: ob_analytics.schemas.HIDDEN_ORDER_ID

## Instrument columns

A frame that holds more than one instrument names each row's venue and symbol
in two optional columns.

::: ob_analytics.schemas.VENUE_COLUMN

::: ob_analytics.schemas.SYMBOL_COLUMN

::: ob_analytics.schemas.group_by_instrument
