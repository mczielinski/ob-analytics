---
title: LOBSTER
---

# LOBSTER

Support for [LOBSTER](https://lobsterdata.com/) message files (event types
1–7), orderbook-backed depth, and round-trip writers.

Use via `Pipeline(source=LobsterSource(...))` or
`Pipeline.from_source("lobster", ...)`.

Key differences from Bitstamp:

- **Executions in the message file** — `LobsterTradeReader` builds trades
  directly from type 4/5 rows in the events DataFrame (no separate trades file).
- **Orderbook-backed depth** — `LobsterSource.compute_depth` reads the official
  orderbook file for ground-truth depth instead of reconstructing from events
- **Integer prices** — raw prices are in ten-thousandths of a dollar
  (`LOBSTER_PRICE_DIVISOR`). The loader and writer know this encoding, so a
  `LobsterLoader` built by hand reads the same prices as a pipeline run.

::: ob_analytics.lobster.LobsterLoader

::: ob_analytics.lobster.LobsterTradeReader

::: ob_analytics.lobster.LobsterWriter

::: ob_analytics.lobster.LobsterSource

::: ob_analytics.lobster.lobster_depth_from_orderbook
