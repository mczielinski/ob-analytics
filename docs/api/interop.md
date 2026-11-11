---
title: Backtester export
---

# Backtester export

Writers for two backtesting engines: hftbacktest and Nautilus Trader. Neither
engine is a dependency. `save_data(..., fmt="hftbacktest")` and
`save_data(..., fmt="nautilus")` call these writers; the two `to_*` functions
return the same data without writing a file. See
[Export to a backtester](../howto/backtester.md).

::: ob_analytics.interop.HftbacktestWriter

::: ob_analytics.interop.to_hftbacktest_array

::: ob_analytics.interop.NautilusWriter

::: ob_analytics.interop.to_nautilus_deltas
