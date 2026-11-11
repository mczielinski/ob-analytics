---
title: Depth
---

# Depth

Order book depth: price-level volume tracking, the depth summary in
basis-point bins, the signals read from it, spread extraction, and depth
filtering.

`PriceLevelBook` is the book at L2: the size at each price level, kept one
depth row at a time. `DepthSummaryEngine` builds the depth summary on it, and
`price_level_snapshots` replays it to give the book at many instants, so a
snapshot's touch always equals the depth summary's at that instant.

::: ob_analytics.depth.PriceLevelBook

::: ob_analytics.depth.BookLevels

::: ob_analytics.depth.price_level_snapshots

::: ob_analytics.depth.DepthSummaryEngine

::: ob_analytics.depth.summarize_depth

::: ob_analytics.depth.price_level_volume

::: ob_analytics.depth.filter_depth

::: ob_analytics.depth.get_spread

::: ob_analytics.depth.book_signals

::: ob_analytics.depth.micro_price

::: ob_analytics.depth.book_imbalance

::: ob_analytics.depth.readable_quotes
