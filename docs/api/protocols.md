---
title: Protocols
---

# Protocol Interfaces

Contracts that pluggable pipeline components must satisfy. Uses structural
(duck) typing — implement the right method signature and it works, no
inheritance required.

| Protocol | Method | Purpose |
|----------|--------|---------|
| `EventLoader` | `load(source) → DataFrame` | Parse raw data into events |
| `TradeSource` | `load(events, source) → DataFrame` | Build the canonical trades DataFrame |
| `DataWriter` | `write(data, dest)` | Serialize pipeline outputs |
| `Format` | factory methods | Bundle loader, trade source, and writer for a venue |

A `Format` also declares a `FeedType` (`matched_book` vs `diff_feed`); see
[Data quality: matched book vs diff feed](../data-quality.md).

::: ob_analytics.protocols.FeedType

::: ob_analytics.protocols.EventLoader

::: ob_analytics.protocols.TradeSource

::: ob_analytics.protocols.DataWriter

::: ob_analytics.protocols.Format
