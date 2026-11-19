---
title: Queue positions
---

# Queue positions

Frame-level functions for the engine's queue reconstruction: each resting
order's place in its price level's queue. The arrays underneath are on the
[Order-book engine](engine.md) page.

`queue_positions` is also exported from the package root
(`from ob_analytics import queue_positions`). On a run, `result.queue()`
returns the same table and keeps it (see
[`PipelineResult`](pipeline.md#ob_analytics.pipeline.PipelineResult)).

::: ob_analytics.queue.queue_positions

::: ob_analytics.queue.queue_age_grid
