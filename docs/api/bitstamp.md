---
title: Bitstamp
---

# Bitstamp

Bitstamp-specific components: event loading, trade reading from companion
`trades.csv`, CSV writing, and `BitstampFormat` as the default pipeline format.

Modern workflows pair `orders.csv` with `trades.csv` (for example from
`scripts/collect_bitstamp_btcusd.py`). The pipeline resolves
`trades.csv` from the same directory as the `orders.csv` path passed to
`Pipeline.run`.

::: ob_analytics.bitstamp.BitstampLoader

::: ob_analytics.bitstamp.BitstampTradeReader

::: ob_analytics.bitstamp.BitstampWriter

::: ob_analytics.bitstamp.BitstampFormat
