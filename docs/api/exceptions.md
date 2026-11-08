---
title: Exceptions
---

# Exceptions

Every error the package raises for a mistake in what it was given inherits
from `ObAnalyticsError`, so one `except ObAnalyticsError` clause catches them
all. Each kind of mistake has one type:

- `ConfigError` for a bad value: a setting or argument out of range, or input
  data that is empty or missing columns. It is also a `ValueError`.
- `UnknownNameError` for a name the package does not know: a source, format,
  metric, feature, bar rule, plot, backend or style. It is also a `KeyError`.
  Its message names the known names closest to the one given.

A path that does not exist raises `FileNotFoundError`, a missing optional
dependency raises `ImportError`, and an argument of the wrong type raises
`TypeError`, as in the standard library.

::: ob_analytics.exceptions.ObAnalyticsError

::: ob_analytics.exceptions.ConfigError

::: ob_analytics.exceptions.UnknownNameError
