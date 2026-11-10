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
every metric that applies to the run: its resolution is one of the metric's
`levels`, and each table in the metric's `needs` has rows, so a run with no
trades skips the trade metrics. Settings are keyword
arguments: `result.metric("vpin", n_buckets=20)`, or
`result.plot("vpin", n_buckets=20, threshold=0.8)`, where each keyword
goes to `compute` or `prepare`, whichever names it.

## Built-in metrics

| Name | Measures | Wraps |
|---|---|---|
| `l1_ticker` | Best bid, best ask and last trade. Draws the three over time, or the quote card at `at=`. | the depth summary and the trades |
| `book_signals` | Mid, micro-price and order-book imbalance | [`depth_signals`](depth.md) |
| `vpin` | VPIN per volume bucket | [`compute_vpin`](flow_toxicity.md) |
| `kyle_lambda` | Kyle's λ regression | [`compute_kyle_lambda`](flow_toxicity.md) |
| `order_flow_imbalance` | Order flow imbalance per window | [`order_flow_imbalance`](flow_toxicity.md) |
| `ofi_horizon` | Order flow imbalance over several horizons | [`ofi_by_horizon`](flow_toxicity.md) |
| `transaction_costs` | Effective and realized spread and price impact per trade | [`transaction_costs`](cost.md) |
| `amihud` | Amihud illiquidity per window | [`amihud`](cost.md) |
| `roll_spread` | Roll's implied spread per window | [`roll_spread`](cost.md) |

Each uses the defaults of the function it wraps, and the run's own
`depth_summary` wherever that function takes quotes.

Every metric reports in the instrument's own units: prices in the quote
currency and sizes in BTC, shares or contracts. `result.metric`,
`result.plot` and the gallery all compute on the run as stored (integer ticks
and lots), and each measure converts with the `tick_size` and `lot_size` the
run's tables carry, so the three give the same numbers. A setting means the
same everywhere too: a size such as `bucket_volume` is in the instrument's own
units and a price in the quote currency, so
`result.metric("vpin", bucket_volume=0.5)` and
`result.plot("vpin", bucket_volume=0.5)` use the same 0.5 BTC buckets.

::: ob_analytics.metrics.L1TickerMetric

::: ob_analytics.metrics.BookSignalsMetric

::: ob_analytics.metrics.VpinMetric

::: ob_analytics.metrics.KyleLambdaMetric

::: ob_analytics.metrics.OrderFlowImbalanceMetric

::: ob_analytics.metrics.OfiHorizonMetric

::: ob_analytics.metrics.TransactionCostsMetric

::: ob_analytics.metrics.AmihudMetric

::: ob_analytics.metrics.RollSpreadMetric

## Registry

::: ob_analytics.metrics.register_metric

::: ob_analytics.metrics.list_metrics

::: ob_analytics.metrics.get_metric

::: ob_analytics.metrics.load_metric_plugins

::: ob_analytics.metrics.skip_reason

::: ob_analytics.metrics.applicable_metrics
