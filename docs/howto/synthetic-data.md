---
title: Generate synthetic L3 data
---

# Generate synthetic L3 data

`ob_analytics.synth` produces a per-order (L3) event stream — `created`,
`changed`, and `deleted` events plus matching trades — without any captured
market data. The output uses the same columns and volume/fill rules as a real
loader, so it passes the schema validators and runs through the full pipeline.

Use it for deterministic tests and CI fixtures, for benchmarks with known
ground truth, and for teaching examples that build their own order books.

## Minimal example

```python
from ob_analytics.synth import generate_session

session = generate_session(seed=1)
print(session.events.shape, session.trades.shape)
print(sorted(session.events["type"].unique().astype(str)))
```

`generate_session` returns a `SynthSession` with three attributes: `events`,
`trades`, and the `config` used. The same seed and config always produce
byte-identical frames.

## Run it through the pipeline

`SyntheticLoader` and `SyntheticTradeSource` serve a generated session to the
pipeline, so no files are written to disk:

```python
from ob_analytics import Pipeline
from ob_analytics.synth import (
    generate_session,
    SynthConfig,
    SyntheticLoader,
    SyntheticTradeSource,
)

session = generate_session(SynthConfig(seed=1, duration=120))
result = Pipeline(
    loader=SyntheticLoader(session),
    trade_source=SyntheticTradeSource(session),
).run(source=None)

print(result.depth_summary[["best_bid_price", "best_ask_price"]].tail())
```

The result is an ordinary `PipelineResult`: `events` (classified, with
`aggressiveness_bps`), `trades`, `depth`, and `depth_summary`.

## Override individual parameters

Pass field overrides directly, or build a `SynthConfig`:

```python
from ob_analytics.synth import generate_session, SynthConfig

# keyword overrides on top of the defaults
fast = generate_session(seed=7, duration=60, market_rate=3.0)

# or an explicit config
cfg = SynthConfig(
    seed=7,
    duration=600,
    mid_price=250.0,
    tick_size=0.05,
    mean_size=2.5,
)
session = generate_session(cfg)
```

## The model

Three independent arrival processes run over `duration` seconds:

- **Limit orders** rest passively on the book. Each picks a side, a price a
  whole number of ticks behind the touch (never crossing), and a size.
- **Cancellations** remove one resting order chosen uniformly at random.
- **Market orders** cross the spread and consume resting liquidity best price
  first, oldest order first. Each maker consumed produces one trade; the
  aggressor is fully filled and never rests.

Arrivals are Poisson (constant rate) or Hawkes (self-exciting, generated with
Ogata thinning using numpy only). Everything is driven by one seeded
`numpy.random.Generator`.

Under the pipeline's order-type classifier the orders come out as
`resting-limit`, `flashed-limit`, and `market`. The generator does not produce
`market-limit` or `pre-existing` orders: every order has a `created` event and
no aggressor rests. This is a deliberate simplification.

## Parameters

### Seed and timing

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `seed` | `0` | Seed for the random generator. Same seed and config give identical output. |
| `duration` | `300.0` | Length of the simulated session, in seconds. |
| `start_time` | `2020-01-01 00:00:00` | Wall-clock anchor for the first event. Timestamps are tz-naive. |

### Arrival process

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `arrival_process` | `"poisson"` | `"poisson"` for constant rates, `"hawkes"` for self-exciting arrivals. |
| `limit_rate` | `6.0` | New limit orders per second (Hawkes background rate). |
| `cancel_rate` | `4.0` | Cancellations per second. |
| `market_rate` | `1.5` | Market orders per second. |
| `hawkes_excitation` | `0.5` | Mean offspring per event (branching ratio). Must be below 1. Hawkes only. |
| `hawkes_decay` | `1.0` | Rate at which one event's excitation fades, per second. Hawkes only. |

### Book structure

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `mid_price` | `100.0` | Reference mid price the book is anchored to. |
| `tick_size` | `0.01` | Smallest price increment. Prices are whole multiples of this. |
| `half_spread_ticks` | `2` | Ticks from the mid to the best price on each side. The resting spread is twice this. |
| `depth_levels` | `10` | Number of distinct price levels available on each side. |
| `level_decay` | `0.5` | Geometric probability for how far behind the touch an order sits; higher concentrates orders at the touch. In `(0, 1]`. |

### Order sizes and sides

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `mean_size` | `1.0` | Mean of the exponential order-size distribution. |
| `min_size` | `0.01` | Lower bound applied to every size. |
| `limit_bid_prob` | `0.5` | Probability a new limit order is a bid. |
| `market_buy_prob` | `0.5` | Probability a market order is a buy. |

### Precision

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `price_decimals` | `2` | Decimal places for emitted prices. |
| `volume_decimals` | `8` | Decimal places for emitted volumes. |

### Optional injections (off by default)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `iceberg_fraction` | `0.0` | Probability a new limit order is an iceberg. |
| `iceberg_size_multiple` | `5.0` | An iceberg's total size as a multiple of its displayed peak; the remainder refills one peak at a time, at the back of the price queue. |
| `toxic_fraction` | `0.0` | Probability a market order is toxic (informed). |
| `toxic_size_multiple` | `3.0` | Size-mean multiplier applied to a toxic market order. |

Icebergs and toxic flow are simplifications for teaching and stress tests. Turn
them on by setting the fractions above zero:

```python
session = generate_session(
    seed=1,
    iceberg_fraction=0.2,
    toxic_fraction=0.1,
)
```

## Related

- [Custom components](custom-components.md) — write a loader for any other format
- [Check data quality](audit.md) — the matched-book / diff-feed checks a session should pass
- [Configuration API](../api/config.md) — the pipeline's `PipelineConfig`
