---
title: Measure transaction costs
---

# Measure effective spread, price impact, and illiquidity

What a trade cost is not what the book advertised. The quoted spread is the
distance between the best bid and the best ask; the **effective spread** is
what the taker actually paid, measured from the mid-price the trade crossed.
Part of that payment is the liquidity provider's compensation — the
**realized spread** — and part is the price moving against them because the
trade carried information — the **price impact**. The three add up:

```
effective spread = realized spread + price impact
```

Run the pipeline first, then measure `result.trades` against
`result.depth_summary`.

## The decomposition

```python
from ob_analytics import Pipeline, sample_csv_path
from ob_analytics import transaction_costs, cost_summary

result = Pipeline().run(sample_csv_path())
costs = transaction_costs(result.trades, result.depth_summary, horizon="5s")

summary = cost_summary(costs)
print(f"effective {summary.effective_spread_bps:.2f} bps"
      f" = realized {summary.realized_spread_bps:.2f}"
      f" + impact {summary.price_impact_bps:.2f}")
```

On the bundled Bitstamp capture:

```
effective 1.45 bps = realized -0.41 + impact 1.86
```

Read that as: the average unit traded paid 1.45 basis points to cross, the
market then moved 1.86 basis points in the taker's favour, and the liquidity
provider was left 0.41 basis points down. A *negative* realized spread means
the flow was, on average, informed — the same story VPIN and Kyle's λ tell,
measured in the currency a taker pays.

`costs` is one row per trade, so the distribution is there too:

```python
costs["effective_spread_bps"].describe()
costs[costs["effective_spread_bps"] < 0]      # trades that printed inside the touch
```

On this capture 72 of the 284 trades come back with a negative effective
spread. That is a fact about the capture, not about the market — see
[below](#trust-the-book-before-you-trust-the-cost).

`cost_summary` weights every figure by trade size, which is why it reports
what the average *unit traded* cost rather than what the average *trade*
cost. It also reports `n_trades` and `n_realized` separately: a trade in the
last horizon of the capture has no future mid to measure against, so it
carries an effective spread but no realized spread.

## Choosing the horizon

The realized spread is read one *horizon* after the trade, and there is no
neutral choice. Too short and the price has not finished reacting, so impact
is understated. Too long and unrelated moves are charged to this trade. Five
minutes is the equity convention; on a fast crypto tape the price moves
further in five minutes than the whole spread is wide, so the estimate turns
to noise. On the bundled capture:

| horizon | effective | realized | impact | spread of realized |
| ------- | --------- | -------- | ------ | ------------------ |
| 1s      | 1.45      | -0.41    | 1.85   | 2.8                |
| 5s      | 1.45      | -0.41    | 1.86   | 2.4                |
| 30s     | 1.45      | -2.95    | 4.40   | 4.6                |
| 1min    | 1.45      | -1.85    | 3.29   | 5.6                |
| 5min    | 1.45      | -0.26    | 1.74   | 11.6               |

All in basis points. The effective spread does not depend on the horizon at
all; the split does, and its noise grows steadily with it. Pick a horizon
short enough that the spread of the realized figure is small beside the
effective spread, and report the horizon beside the number.

## Illiquidity from the trade prices alone

Two measures need no quotes and no aggressor side, so they run on any tape —
including an aggregated feed that labels nothing.

**Amihud illiquidity** is the price move a unit of turnover buys. A market
where heavy turnover barely moves the price scores low:

```python
from ob_analytics import amihud

illiq = amihud(result.trades, window="5min")
illiq[["timestamp", "abs_return", "turnover", "amihud"]]
```

It carries a unit — one over the turnover unit of the input. On a raw
pipeline frame that is ticks times lots; pass a display-unit result (see
[`display_result`](../api/gallery.md)) to get turnover in the quote currency,
which is what published figures use.

**Roll's implied spread** reads the spread out of the bounce between hitting
the bid and lifting the ask, which makes successive price changes negatively
correlated:

```python
from ob_analytics import roll_spread

roll_spread(result.trades)
```

On the bundled capture this returns `NaN`, with an autocovariance of
`+47795` beside it. That is not a failure to work around — it is the
estimator saying its model does not fit. Roll assumes an efficient price plus
a bounce of constant half-spread, with order flow that carries no
information. This capture trends over its thirty minutes, so the drift swamps
the bounce and the autocovariance comes out positive, where the formula has
no real root. The sign is returned so the reason is visible rather than
hidden behind a number. Where quotes exist, measure the spread with
`transaction_costs` instead; Roll is for the tapes where they do not.

## Plotting it

The decomposition draws as a level-less face on either backend — two lines
with the price impact as the band between them:

```python
from ob_analytics.visualization import plot, prepare, save_figure

fig = plot("transaction_costs", **prepare.transaction_costs(costs, window="1min"))
save_figure(fig, "transaction_costs.png")
```

To put it in a gallery, build the panel and append it to the model:

```python
from ob_analytics.visualization.gallery import (
    build_gallery_model, generate_gallery, transaction_costs_panel,
)

model = build_gallery_model(result)
model.analytics.append(transaction_costs_panel(costs))
generate_gallery(result, "output/gallery/", model=model)
```

The Bitstamp and LOBSTER demos already do this, so `ob-analytics
bitstamp-demo` writes the face without any extra work.

## Feeds without a native aggressor side

The effective spread needs to know which side crossed. L3 crypto labels it;
L2 and aggregated feeds do not, so `transaction_costs` classifies with the
same machinery the flow-toxicity metrics use — Lee–Ready against the quotes
you passed in, by default:

```python
costs = transaction_costs(l2_trades, depth_summary, sign_method="tick")
```

A native `direction` is honored as-is unless `sign_method` overrides it. See
[Compute flow toxicity](flow-toxicity.md#feeds-without-a-native-aggressor-side).

## Trust the book before you trust the cost

Every number here inherits the quality of the book it is measured against. On
a diff feed a trade can print through resting orders the venue never withdrew,
which reads as a *negative* effective spread — the taker apparently paying
less than the mid. That is a finding about the capture, not about the market.
Crossed quotes are already skipped, because a book whose best bid is above its
best ask has no midpoint; stale orders are not, because removing them is a
judgement the library leaves to you. Run the audit first:

```bash
ob-analytics audit orders.csv
```

See [Check data quality](audit.md) and
[Matched book vs diff feed](../data-quality.md).

## Related

- [Transaction Costs API](../api/cost.md) — parameters and return types
- [Glossary: transaction cost](../glossary.md#transaction-cost) — what each measure means, with citations
- [Compute flow toxicity](flow-toxicity.md) — VPIN, Kyle's λ and order-flow imbalance
