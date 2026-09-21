---
title: Feature table
---

# Feature table

One tidy table: a point in time on each row, a microstructure feature in each
column. Two decisions make it, and they are separate — where the rows fall,
and what each column measures.

Where the rows fall is a [bar rule](bars.md), so `features()` takes the same
sampling arguments `bars()` does and the same arguments give the same cut in
both. What each column measures is a **feature**, and ten ship with the
package:

| Feature | Columns | Reads |
|---|---|---|
| `price` | `open`, `high`, `low`, `close`, `vwap` | the bar's trades |
| `returns` | `log_return`, `realized_vol` | the bar's trades |
| `flow` | `volume`, `turnover`, `n_trades`, `signed_volume`, `trade_imbalance` | the bar's trades |
| `spread` | `spread`, `spread_bps` | the book at the close |
| `mid_price` | `mid_price` | the book at the close |
| `micro_price` | `micro_price`, `micro_price_offset` | the book at the close |
| `imbalance` | `obi`, `obi_depth` | the book at the close |
| `depth` | `best_bid_vol`, `best_ask_vol`, `bid_depth`, `ask_depth` | the book at the close |
| `vpin` | `vpin` | the last 20 bars |
| `kyle_lambda` | `kyle_lambda` | the last 20 bars |

Each row is stated as of the close of its bar. The trade columns hold what
happened inside the bar and the book columns hold the book as it stood when it
closed, so nothing from later reaches the row and the table carries no
look-ahead. It carries no target either: a target looks forward, and building
one is a shift the caller makes.

See the ["Build a feature table"](../howto/feature-table.md) how-to for a
worked example and a baseline model, and
[Extending](../extending.md#6-a-new-feature) for writing a feature of your own.

## Functions

::: ob_analytics.features.features

::: ob_analytics.features.register_feature

::: ob_analytics.features.list_features

::: ob_analytics.features.get_feature

## The built-in features

::: ob_analytics.features.PriceFeature

::: ob_analytics.features.ReturnsFeature

::: ob_analytics.features.FlowFeature

::: ob_analytics.features.SpreadFeature

::: ob_analytics.features.MidPriceFeature

::: ob_analytics.features.MicroPriceFeature

::: ob_analytics.features.ImbalanceFeature

::: ob_analytics.features.DepthFeature

::: ob_analytics.features.VpinFeature

::: ob_analytics.features.KyleLambdaFeature
