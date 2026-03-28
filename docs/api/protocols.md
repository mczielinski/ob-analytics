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
| `MatchingEngine` | `match(events) → DataFrame` | Pair bid/ask fills |
| `TradeInferrer` | `infer_trades(events) → DataFrame` | Build trade records |
| `DataWriter` | `write(data, dest)` | Serialize pipeline outputs |
| `Format` | factory methods | Bundle all of the above for a venue |

::: ob_analytics.protocols.EventLoader

::: ob_analytics.protocols.MatchingEngine

::: ob_analytics.protocols.TradeInferrer

::: ob_analytics.protocols.DataWriter

::: ob_analytics.protocols.Format
