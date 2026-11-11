---
title: Depth CSV
---

# Depth CSV (L2 price-level)

Price-level (L2 / market-by-price) components: the depth loader, trade reader,
CSV writer, and `DepthCsvSource`. For the guide, see
[Process L2 feeds](../howto/l2-depth.md).

A price-level feed carries `[price, quantity]` levels and diffs with no order
IDs, so the loader yields the depth frame directly and the pipeline skips the
per-order stages (see [`Level`](protocols.md)).

::: ob_analytics.depth_csv.DepthCsvLoader

::: ob_analytics.depth_csv.DepthCsvTradeReader

::: ob_analytics.depth_csv.DepthCsvWriter

::: ob_analytics.depth_csv.DepthCsvSource

::: ob_analytics.depth_csv.DepthCsvSettings

A capture's `meta.json` is read by
[`read_record`](capture_record.md#ob_analytics.capture_record.read_record).
