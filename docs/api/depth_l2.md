---
title: L2 depth
---

# L2 (price-level) depth

Price-level (L2 / market-by-price) components: the depth loader, trade reader,
CSV writer, and `DepthCsvFormat`. For the guide, see
[Process L2 feeds](../howto/l2-depth.md).

A price-level feed carries `[price, quantity]` levels and diffs with no order
IDs, so the loader yields the depth frame directly and the pipeline skips the
per-order stages (see [`Level`](protocols.md)).

::: ob_analytics.depth_l2.L2DepthLoader

::: ob_analytics.depth_l2.L2TradeReader

::: ob_analytics.depth_l2.DepthCsvWriter

::: ob_analytics.depth_l2.DepthCsvFormat
