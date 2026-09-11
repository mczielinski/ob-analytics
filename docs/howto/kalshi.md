---
title: Capture Kalshi markets
---

# Capture Kalshi prediction markets

Kalshi lists event contracts. Each market asks a yes-or-no question, and a Yes
contract pays $1 if the answer is Yes. ob-analytics captures a Kalshi market
through the [ccxt source](ccxt.md) and replays it through the
[L2 path](l2-depth.md). You do not need a Kalshi account or an API key: the
order book and the trades are public.

```bash
pip install "ob-analytics[ccxt]"

ob-analytics capture ccxt --exchange kalshi --pair KXPRESNOMD-28-MK --minutes 10 --out /tmp/kalshi
ob-analytics process /tmp/kalshi --source depth_csv --gallery --output /tmp/kalshi_out
```

## Find a market ticker

`--pair` takes a Kalshi market ticker, such as `KXPRESNOMD-28-MK`. Each
market's page on kalshi.com shows its ticker, and Kalshi's public API lists
the open markets:

```bash
curl -s "https://api.elections.kalshi.com/trade-api/v2/markets?status=open&limit=100"
```

Pick a market that trades. A quiet market gives a book that seldom changes and
few trades. Even busy Kalshi books change only a few times a minute, so
capture for 10 minutes or more.

## The book is the Yes book

Kalshi publishes two lists of bids, one for Yes and one for No, and no asks. A
Yes contract and a No contract together always pay $1, so a bid to buy No at
`p` is the same as an offer to sell Yes at `1 - p`. ccxt uses this to build one
book, seen from the Yes side:

| Kalshi publishes | The captured book holds |
|------------------|-------------------------|
| a Yes bid at 0.05 for 30 contracts | a bid at 0.05 for 30 |
| a No bid at 0.91 for 306 contracts | an ask at 0.09 for 306 |

Trades follow the same rule. Every trade price is the Yes price. The trade's
`side` is `buy` when the taker bought Yes and `sell` when the taker bought No.

Prices are in dollars, between 0 and 1, so a price is also the market's
probability of Yes. Sizes are numbers of contracts, and can be fractional.

!!! warning "Use the plain ticker"
    ccxt also accepts `<ticker>-NO` for the No side. It shows the book in No
    prices but still prices trades in Yes, so the trades and the book do not
    line up. Capture the plain ticker.

## Tick sizes

Kalshi markets do not all use the same price step. Most use whole cents. Many
of the most traded markets use tenths of a cent below 0.10 and above 0.90, and
whole cents in between. A few use tenths of a cent everywhere.

A capture records the market's price step as `tick_size` in `meta.json`.
`ob-analytics process` and `ob-analytics audit` read it from there. From
Python, set it yourself:

```python
from ob_analytics import DepthCsvSource, Pipeline, PipelineConfig

config = PipelineConfig(tick_size=0.001, price_decimals=3)
result = Pipeline(config, source=DepthCsvSource()).run("/tmp/kalshi")
```

`recorded_tick_size("/tmp/kalshi")` returns the value the capture recorded. If
the tick size in use is coarser than the prices in the file, the loader raises
`ConfigError` instead of rounding the prices to fit.

## Depth bins

The depth summary adds up the size resting in rings around the mid-price. Each
ring is `depth_bps` basis points wide: 25 by default, 20 rings a side. That
suits a price of 100,000, but not a price of 0.04, where 25 basis points is
0.0001, a tenth of the smallest tick. Most rings are then empty. Use wider
rings for a prediction market:

```python
config = PipelineConfig(tick_size=0.001, price_decimals=3, depth_bps=500)
```

The depth heatmap and the price view draw the book directly and do not use the
rings.

## What a capture can and cannot see

The capture polls Kalshi's public REST API, once a second by default. Change
the rate with `--poll-interval`. Because it polls:

- A change that appears and goes away between two polls is not seen.
- Book timestamps are the time each poll returned, not Kalshi's time. Trades
  carry Kalshi's time.
- There is no sequence number, so `audit` cannot check for dropped data.
- Trades from before the opening book are left out. Kalshi's first answer is
  its recent history, which can reach back hours.

Kalshi's WebSocket feed sends every change with a sequence number, but it needs
an API key. Capturing from it is
[#240](https://github.com/mczielinski/ob-analytics/issues/240), and using API
keys is [#239](https://github.com/mczielinski/ob-analytics/issues/239).

## See also

- [Capture CCXT venues](ccxt.md): the source this capture uses
- [Process L2 (price-level) feeds](l2-depth.md): what the captured `depth.csv`
  goes through
- [Check data quality](audit.md): run `audit` on the capture
