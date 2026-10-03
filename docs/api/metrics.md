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
every metric that applies to the run's resolution. Settings are keyword
arguments: `result.metric("vpin", bucket_volume=5.0)`, or
`result.plot("vpin", bucket_volume=5.0, threshold=0.8)`, where each keyword
goes to `compute` or `prepare`, whichever names it.

## Built-in metrics

| Name | Measures | Wraps |
|---|---|---|
| `l1_ticker` | Best bid, best ask and last trade. Draws the three over time, or the quote card at `at=`. | the depth summary and the trades |
| `vpin` | VPIN per volume bucket | [`compute_vpin`](flow_toxicity.md) |
| `kyle_lambda` | Kyle's λ regression | [`compute_kyle_lambda`](flow_toxicity.md) |
| `order_flow_imbalance` | Order flow imbalance per window | [`order_flow_imbalance`](flow_toxicity.md) |
| `ofi_horizon` | Order flow imbalance over several horizons | [`ofi_by_horizon`](flow_toxicity.md) |

Each uses the defaults of the function it wraps. The gallery and `result.plot`
give a metric the result in display units (quote currency, base asset);
`result.metric` gives it the result as stored (integer ticks and lots), so a
size setting such as `bucket_volume` is in those units there.

::: ob_analytics.metrics.L1TickerMetric

::: ob_analytics.metrics.VpinMetric

::: ob_analytics.metrics.KyleLambdaMetric

::: ob_analytics.metrics.OrderFlowImbalanceMetric

::: ob_analytics.metrics.OfiHorizonMetric

## Registry

::: ob_analytics.metrics.register_metric

::: ob_analytics.metrics.list_metrics

::: ob_analytics.metrics.get_metric

::: ob_analytics.metrics.load_metric_plugins
