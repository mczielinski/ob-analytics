---
title: Build bars from trades
---

# Build bars from trades

A bar summarises a run of consecutive trades. [`bars()`](../api/bars.md) turns
a trades frame into one row per bar, with open, high, low, close, volume,
VWAP, and the buy/sell split of that volume.

What differs between bar types is only where the boundaries fall, so that
decision is a **rule** you name.

## Clock bars (OHLCV)

The classic: a new bar every fixed span of the clock.

```python
from ob_analytics import Pipeline, bars, sample_csv_path
from ob_analytics.visualization import display_result

result = Pipeline().run(sample_csv_path())

# Pipeline prices are whole ticks and sizes whole lots; display_result
# converts both to the quote currency and the base asset.
trades = display_result(result).trades

ohlcv = bars(trades, "time", "1min")
print(ohlcv[["timestamp_end", "open", "high", "low", "close", "volume", "vwap"]])
```

```text
                   timestamp_end     open     high      low    close   volume         vwap
2026-05-02 02:36:49.271000+00:00  78319.0  78333.0  78319.0  78323.0 1.622609 78324.423605
2026-05-02 02:37:57.363000+00:00  78323.0  78323.0  78322.0  78323.0 0.011084 78322.160485
2026-05-02 02:38:54.818000+00:00  78323.0  78336.0  78323.0  78336.0 0.014561 78335.772925
```

A minute in which nothing traded produces no row: every bar holds at least one
trade.

## Activity bars

The other four rules cut on trading activity instead of the clock. A quiet
hour and a busy minute then produce the same number of bars, which is what
these are for — bar returns come much closer to being independent and
identically distributed.

```python
tick_bars = bars(trades, "tick", 100)       # every 100 trades
vol_bars = bars(trades, "volume", 0.5)      # every 0.5 BTC traded
dollar_bars = bars(trades, "dollar", 50_000)  # every $50,000 of turnover
imb_bars = bars(trades, "imbalance", 0.5)   # every 0.5 BTC of net one-sided flow
```

Imbalance bars close when signed size drifts the threshold away from where the
bar opened, **in either direction**. A burst of one-sided flow ends a bar; a
balanced stretch of trading stays inside one.

The threshold is fixed, not the moving estimate of López de Prado's original
imbalance bars. A fixed threshold gives the same bars every time the same
trades are read, which is what makes a bar table reproducible.

## Let the threshold pick itself

Leave the threshold out and each rule chooses one that yields about
`target_bars` bars. It records what it used:

```python
for rule in ("time", "tick", "volume", "dollar", "imbalance"):
    cut = bars(trades, rule, target_bars=50)
    print(f"{rule:10s} {len(cut):3d} bars  threshold={cut.attrs['bar_threshold']}")
```

```text
time        44 bars  threshold=0 days 00:00:35.809380
tick        48 bars  threshold=6
volume      37 bars  threshold=0.30059678300000003
dollar      37 bars  threshold=23568.4402418964
imbalance   31 bars  threshold=0.30672681552521563
```

It is an aim, not a promise. One huge trade fills several volume bars at once,
and signed flow part-cancels, so the counts land near the target rather than on
it.

## What a bar carries

| Column | Meaning |
|---|---|
| `bar` | 0-based bar number |
| `timestamp_start` / `timestamp_end` | first and last trade of the bar |
| `open` / `high` / `low` / `close` | trade prices |
| `volume` | total size traded |
| `turnover` | total price × size |
| `n_trades` | number of trades |
| `vwap` | `turnover / volume` |
| `buy_volume` / `sell_volume` / `signed_volume` | size by aggressor side, and buys minus sells |

The last bar is whatever trades were left over, so it may not have reached the
threshold. Drop it with `.iloc[:-1]` where an equal-size bar matters.

## Feeds that don't label the aggressor

`buy_volume` and `sell_volume` need the taker's side. L3 crypto ships it; L2
and aggregated feeds don't, so `bars()` classifies it the same way the
[flow-toxicity metrics](flow-toxicity.md) do. Pass book snapshots to use
Lee–Ready instead of the tick rule:

```python
bars(trades, "volume", 0.5, sign_method="lee_ready", quotes=result.depth_summary)
```

## Plot them

```python
from ob_analytics.visualization import plot, prepare, save_figure

fig = plot("bars", **prepare.bars(bars(trades, "volume", 0.5)))
save_figure(fig, "volume_bars.png")
```

Candles are coloured by whether the bar closed up or down, over a volume strip
coloured by which side was the net aggressor.

The x axis follows the rule. Clock bars occupy equal spans of time, so they are
drawn on a real time axis and a stretch with no trading shows as the gap it
was. Activity bars occupy wildly unequal spans — a burst can close several
inside a second — so they get one equal slot each, labelled with their closing
times. That even spacing *is* the point of an activity bar: it is what puts the
same amount of market in each one.

To put bars in a gallery, build a panel and append it:

```python
from ob_analytics.visualization.gallery import bars_panel, build_gallery_model, generate_gallery

model = build_gallery_model(result)
model.analytics.append(bars_panel(bars(trades, "volume", 0.5)))
generate_gallery(result, "gallery/", model=model)
```

## A rule of your own

A bar rule says where the boundaries fall and nothing else. Register one and
`bars()` finds it by name — see
[Extending](../extending.md#5-a-new-bar-rule).
