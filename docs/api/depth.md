---
title: Depth Metrics
---

# Depth Metrics

Order book depth computation: price-level volume tracking, depth summary
metrics in basis-point bins, spread extraction, and depth filtering.

`PriceLevelBook` is the book at L2: the size at each price level, kept one
depth row at a time. `DepthMetricsEngine` builds the depth summary on it, and
`price_level_snapshots` replays it to give the book at many instants, so a
snapshot's touch always equals the depth summary's at that instant.

::: ob_analytics.depth.PriceLevelBook

::: ob_analytics.depth.price_level_snapshots

::: ob_analytics.depth.DepthMetricsEngine

::: ob_analytics.depth.depth_metrics

::: ob_analytics.depth.price_level_volume

::: ob_analytics.depth.filter_depth

::: ob_analytics.depth.get_spread

::: ob_analytics.depth.readable_quotes
