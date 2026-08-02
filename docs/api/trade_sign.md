---
title: Trade Signs
---

# Trade-Sign Classification

Infer the aggressor side of each trade when the feed doesn't label it.
L3 crypto (Bitstamp) ships `buy_order_id` / `sell_order_id`, so the trades
frame carries a real `direction`; L2 / aggregated feeds and many CCXT
sources don't, so the signed-flow metrics
([`compute_vpin`](flow_toxicity.md#ob_analytics.flow_toxicity.compute_vpin),
[`order_flow_imbalance`](flow_toxicity.md#ob_analytics.flow_toxicity.order_flow_imbalance))
have nothing to work with. These classifiers fill that gap and are wired in
as an automatic fallback.

## Functions

::: ob_analytics.trade_sign.classify_trade_sign

::: ob_analytics.trade_sign.tick_rule

::: ob_analytics.trade_sign.lee_ready

::: ob_analytics.trade_sign.bulk_volume_classification
