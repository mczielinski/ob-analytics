---
title: Order book
---

# Order book

The book at an instant and the per-order lifecycles, as DataFrames.  The
reconstructions themselves live in the [order-book engine](engine.md); these
are their frame-level faces.

Both functions are also exported from the package root
(`from ob_analytics import order_book, order_lifecycles`). On a run,
`result.book(at)` and `result.lifecycles()` return the same tables and keep
them, so a second call costs nothing (see
[`PipelineResult`](pipeline.md#ob_analytics.pipeline.PipelineResult)).

::: ob_analytics.book.order_book

::: ob_analytics.book.BookSnapshot

::: ob_analytics.book.order_lifecycles
