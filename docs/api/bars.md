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
| `volume` | N units of traded size | a size, in the instrument's units |
| `dollar` | N units of price × size | an amount of the quote currency |
| `imbalance` | N units of drift in signed size | a size, in the instrument's units |

A threshold means the same amount on a pipeline result (whole ticks and lots)
and on a display frame: `bars(result.trades, "volume", 0.5)` and
`bars(display_result(result).trades, "volume", 0.5)` both close a bar every
0.5 BTC. The integer frame adds whole lots exactly. On the display frame a
float sum can land just short of a threshold that a run of trades meets
exactly, so the two can put one trade in a different bar.

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

::: ob_analytics.bars.load_bar_rule_plugins

## The built-in rules

::: ob_analytics.bars.ClockRule

::: ob_analytics.bars.TickRule

::: ob_analytics.bars.AccumulationRule
