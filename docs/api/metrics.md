---
title: Metrics
---

# Metric registry

A metric measures a finished run and draws as a level-less plot. Every metric
registers here under a name; third-party metrics load through the
`ob_analytics.metrics` entry-point group (see
[Extending](../extending.md#4-a-new-metric)). A registered value is a
[`Metric`](protocols.md) instance — a metric needs no per-run construction, so
the object registered is the object called.

The name is also the plot concept the metric draws under, so a renderer
registered at `(name, None, backend)` is the metric's face. A registered metric
shows up in `available_concepts()`, renders through
`result.plot(name)`, and gets its own gallery card — with no edit to
ob-analytics.

Metrics run on demand, not during `Pipeline.run`: use
[`PipelineResult.metric`](pipeline.md) for one, `PipelineResult.metrics()` for
every metric that applies to the run's resolution.

::: ob_analytics.metrics.register_metric

::: ob_analytics.metrics.list_metrics

::: ob_analytics.metrics.get_metric

::: ob_analytics.metrics.load_metric_plugins
