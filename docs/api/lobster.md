---
title: LOBSTER
---

# LOBSTER Format

Support for [LOBSTER](https://lobsterdata.com/) message files (event types
1–7), orderbook-backed depth, and round-trip writers.

Use via `Pipeline(format=LobsterFormat(...))` or
`Pipeline.from_format("lobster", ...)`.

Key differences from Bitstamp:

- **Single-sided executions** — `LobsterMatcher` is a pass-through; trades are
  inferred directly from type 4/5 events by `LobsterTradeInferrer`
- **Orderbook-backed depth** — `LobsterFormat.compute_depth` reads the official
  orderbook file for ground-truth depth instead of reconstructing from events
- **Integer prices** — raw prices are in ten-thousandths of a dollar
  (`price_divisor=10000`)

::: ob_analytics.lobster.LobsterLoader

::: ob_analytics.lobster.LobsterMatcher

::: ob_analytics.lobster.LobsterTradeInferrer

::: ob_analytics.lobster.LobsterWriter

::: ob_analytics.lobster.LobsterFormat
