---
title: Flow Toxicity
---

# Flow Toxicity Metrics

Market microstructure measures for detecting informed trading and
quantifying price impact.

## Functions

::: ob_analytics.flow_toxicity.compute_vpin

::: ob_analytics.flow_toxicity.vpin_bucket_volume

::: ob_analytics.flow_toxicity.compute_kyle_lambda

::: ob_analytics.flow_toxicity.order_flow_imbalance

::: ob_analytics.flow_toxicity.ofi_by_horizon

## Models

::: ob_analytics.flow_toxicity.KyleLambdaResult

## Thresholds

::: ob_analytics.flow_toxicity.KYLE_MIN_T_STAT

::: ob_analytics.flow_toxicity.KYLE_MIN_WINDOWS

::: ob_analytics.flow_toxicity.VPIN_BUCKETS_PER_DAY

`compute_vpin` refuses a bucket size that would make more than
[`MAX_VOLUME_BUCKETS`](trade_sign.md#ob_analytics.trade_sign.MAX_VOLUME_BUCKETS)
buckets.

::: ob_analytics.flow_toxicity.OFI_HORIZONS
