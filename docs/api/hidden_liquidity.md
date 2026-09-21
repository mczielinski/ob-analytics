---
title: Hidden Liquidity
---

# Hidden liquidity

Size a venue will trade that the visible book does not show. An L3 stream
records two footprints of it: the refills of an iceberg order, and trades that
print inside the visible spread.

See [Find hidden liquidity](../howto/hidden-liquidity.md) for a worked example
and the measured recall and precision, and
[Glossary: hidden liquidity](../glossary.md#hidden-liquidity) for the terms.

## Iceberg orders

::: ob_analytics.hidden_liquidity.detect_icebergs

::: ob_analytics.hidden_liquidity.ICEBERG_MAX_DELAY

## Trades against hidden orders

::: ob_analytics.hidden_liquidity.hidden_trades

## Models

::: ob_analytics.hidden_liquidity.IcebergDetection
