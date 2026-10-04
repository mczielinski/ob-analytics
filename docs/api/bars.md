---
title: Bars
---

# Bars

A bar summarises a run of consecutive trades: open, high, low, close, volume,
VWAP, and the buy/sell split of its volume. What differs between bar types is
only **where the boundaries fall**, so that decision is a *rule* and everything
else is shared.

Five rules ship with the package:

| Rule | A new bar every… | Threshold |
|---|---|---|
| `time` | fixed span of the clock | a duration (`"1min"`, `pd.Timedelta`) |
| `tick` | N trades | a trade count |
| `volume` | N units of traded size | an amount |
| `dollar` | N units of price × size | an amount |
| `imbalance` | N units of drift in signed size | an amount |

The last four sample the market by activity rather than by the clock. That is
why the quant literature prefers them: a quiet hour and a busy minute produce
the same number of bars, so bar returns come much closer to being independent
and identically distributed.

See the ["Build bars from trades"](../howto/bars.md) how-to for a worked
example, and [Extending](../extending.md#5-a-new-bar-rule) for writing a rule
of your own.

## Functions

::: ob_analytics.bars.bars

::: ob_analytics.bars.register_bar_rule

::: ob_analytics.bars.list_bar_rules

::: ob_analytics.bars.get_bar_rule

## The built-in rules

::: ob_analytics.bars.ClockRule

::: ob_analytics.bars.TickRule

::: ob_analytics.bars.AccumulationRule
