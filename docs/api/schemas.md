---
title: Data Contracts
---

# Data Contracts

Column-list constants and validators defining the DataFrame contracts at
package boundaries (replaces the old Pydantic `models.py`). Validators check
for required columns only; extra columns are allowed.

::: ob_analytics.schemas.validate_events_df

::: ob_analytics.schemas.validate_trades_df

::: ob_analytics.schemas.validate_depth_df
