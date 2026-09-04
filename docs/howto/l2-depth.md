---
title: Process L2 (price-level) feeds
---

# Process L2 (price-level) depth feeds

ob-analytics was built around an **L3 per-order** stream — every event is a
`created` / `changed` / `deleted` action keyed by an order `id`, and the
reconstruction stages (order classification, aggressiveness, queue position)
all depend on that identity. Many venues — **Binance, Kalshi, Polymarket**, and
most CCXT sources — publish only **L2 aggregated** data: price-level
`[price, quantity]` snapshots and diffs with **no order IDs**.

This guide covers the **L2 depth-native ingestion path**: those feeds produce
valid depth / spread / trade analytics without faking per-order state. It is
the counterpart to the L3 [Bitstamp](your-own-data.md) / [LOBSTER](lobster.md)
paths.

## L2 vs L3, in one line

A [`Source`](../api/protocols.md) declares its
[`level`](../api/protocols.md) — a `Level`:

| `Level` | Feed | What each row is | Per-order stages |
|---------|------|------------------|------------------|
| `L3` | Market-by-order (Bitstamp, LOBSTER, Databento) | one primitive per resting order | run |
| `L2` | Market-by-price (Binance, Kalshi, Polymarket, CCXT) | one price level's aggregate size | **skipped** |

On the L2 path the loader yields the depth frame **directly** — a price-level
feed *is* a depth stream, so there is nothing to reconstruct — and the pipeline
skips the per-order stages (`set_order_types`, `order_aggressiveness`, queue
reconstruction) because there is no order identity to key on.

## The L2 CSV schema

One row per price-level update, each carrying the level's **new absolute**
resting size (`0` removes the level) — *not* a signed delta. A book snapshot is
simply the opening block of rows.

```text
timestamp,side,price,volume
1700000000000,bid,99.0,5.0     # ─┐ opening snapshot
1700000000000,bid,98.0,8.0     #  │
1700000000000,ask,101.0,4.0    #  │
1700000000000,ask,102.0,7.0    # ─┘
1700000005000,ask,101.0,2.0    # level 101 now holds 2 (was 4)
1700000015000,ask,101.0,0.0    # level 101 removed
```

| column | meaning |
|--------|---------|
| `timestamp` | receive time — integer epoch (`config.timestamp_unit`) or any string pandas can parse |
| `side` | `bid` / `ask` (`buy` / `sell` and `b` / `a` are also accepted) |
| `price` | the price level (scaled by `config.price_divisor`, rounded to `price_decimals`) |
| `volume` | the level's **new absolute** resting size (`0` = removed) |

Column names are flexible: `side` / `direction` and `volume` / `size` /
`amount` / `quantity` are all accepted.

Point the pipeline at the file (or a directory containing `depth.csv`); a
companion `trades.csv` next to it is picked up automatically.

```python
from ob_analytics import Pipeline

result = Pipeline.from_source("depth_csv").run("my_l2_run/")

result.level             # Level.L2
result.depth             # the price-level book (timestamp, price, volume, direction)
result.depth_summary     # best bid/ask + BPS-bin depth over time
result.events            # empty — the per-order stages did not run
```

## What you get — and what's skipped

The depth analytics run exactly as they do for L3, because
`DepthMetricsEngine` already consumes an absolute price-level book:

```python
from ob_analytics.depth import get_spread

get_spread(result.depth_summary)      # best bid/ask through time
```

The per-order stages are **skipped with a clear reason** — `result.events` is
an empty (but schema-valid) frame, and `result.level is Level.L2` records
why. Anything that needs order identity — order-type classification, order
aggressiveness, order lifecycles, queue position — has nothing to work with on
an aggregated feed and is not attempted.

## Trade signs

L3 crypto ships the taker side for free; price-level feeds usually don't. When
`trades.csv` has no `side` column, the pipeline classifies the aggressor with
**Lee–Ready** against the reconstructed BBO (falling back to the tick rule at
the mid) — see [trade signs](../api/trade_sign.md). A native `side` column is
honored as-is.

```python
result.trades["direction"]    # buy / sell taker side, native or classified
```

Because there are no order IDs, the `maker` / `taker` (and event-id) columns are
`<NA>` — that is expected for L2, not a data-quality failure.

## Visualizing an L2 result

The gallery adapts to the resolution: an L2 result renders the depth/trades
faces (the depth heatmap, price view, depth percentiles, trade tape, trade
size) and **skips every per-order face** rather than erroring.

```python
from ob_analytics.visualization import available_concepts

result.plot("depth_heatmap")          # the hero L2 face
sorted(available_concepts(result))    # only the L2-supported concepts
```

## Checking data quality

`ob-analytics audit <source> --source depth_csv` (and
`data_quality_summary`) work on an L2 result: the crossed-book % is read from
the price-level book, and the per-order metrics report zero.

## Try it on the synthetic fixture

The bundled `toy_l2_depth()` / `toy_l2_trades()` are a tiny, hand-verifiable
snapshot + delta stream — the L2 counterpart to `toy_events` / `toy_trades`:

```python
from ob_analytics.datasets import toy_l2_depth, toy_l2_trades
from ob_analytics.depth import depth_metrics, get_spread

summary = depth_metrics(toy_l2_depth())
get_spread(summary)
```

## Related

- [Trade signs](../api/trade_sign.md) — Lee–Ready / tick / BVC for unlabelled feeds
- [Custom components](custom-components.md) — write a loader for any other format
- [Data quality](../data-quality.md) — matched book vs diff feed
- [L2 API reference](../api/depth_l2.md) — `L2DepthLoader`, `DepthCsvSource`, …

!!! note "Live L2 capture"
    This path is for **offline** L2 files. A live depth-update event kind for
    the [capture runner](live-capture.md) — so `LiveSource`s for L2 venues
    (Binance, Kalshi, Polymarket) can stream straight into this schema — is a
    planned follow-up.
