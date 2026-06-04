---
title: Exceptions
---

# Exceptions

All exceptions inherit from `ObAnalyticsError`, allowing callers to catch
the full hierarchy with a single `except` clause. Configuration and
data-contract problems are raised as the `ConfigError` subclass.

::: ob_analytics.exceptions.ObAnalyticsError

::: ob_analytics.exceptions.ConfigError
