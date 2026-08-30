---
title: Protocols
---

# Protocol Interfaces

Contracts that pluggable pipeline components must satisfy. Uses structural
(duck) typing — implement the right method signature and it works, no
inheritance required.

| Protocol | Method | Purpose |
|----------|--------|---------|
| `EventLoader` | `load(source) → DataFrame` | Parse raw L3 data into events |
| `DepthSource` | `load(source) → DataFrame` | Load an L2 price-level feed into the depth frame |
| `TradeSource` | `load(events, source) → DataFrame` | Build the canonical trades DataFrame |
| `DataWriter` | `write(data, dest)` | Serialize pipeline outputs |
| `Source` | `level` · `feed_type` · `settings` | The shape shared by every data source, file or live |
| `OfflineSource` | factory methods | A `Source` that replays stored files (loader, trade source, writer) |
| `LiveSource` | `snapshot` · `stream` · `shutdown_synthetic_events` | A `Source` that captures a live venue feed |

A `Source` declares two coordinates: a `FeedType` (`matched_book` vs
`diff_feed`; see [Data quality](../data-quality.md)) and a `Level` (`L2` vs
`L3`; see [Process L2 feeds](../howto/l2-depth.md)). It carries typed `settings`
and registers in the source registry (see [Sources](sources.md)).

::: ob_analytics.protocols.Level

::: ob_analytics.protocols.FeedType

::: ob_analytics.protocols.EventLoader

::: ob_analytics.protocols.DepthSource

::: ob_analytics.protocols.TradeSource

::: ob_analytics.protocols.DataWriter

::: ob_analytics.protocols.Source

::: ob_analytics.protocols.OfflineSource

::: ob_analytics.live.LiveSource
