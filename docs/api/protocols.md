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
| `Format` | factory methods | Bundle loader, trade source, and writer for a venue |

A `Format` declares two axes: a `FeedType` (`matched_book` vs `diff_feed`; see
[Data quality](../data-quality.md)) and a `Level` resolution (`L2` vs `L3`; see
[Process L2 feeds](../howto/l2-depth.md)).

::: ob_analytics.protocols.Level

::: ob_analytics.protocols.FeedType

::: ob_analytics.protocols.EventLoader

::: ob_analytics.protocols.DepthSource

::: ob_analytics.protocols.TradeSource

::: ob_analytics.protocols.DataWriter

::: ob_analytics.protocols.Format
