---
title: Glossary
---

# Glossary

Brief definitions of the market-microstructure jargon used throughout
ob-analytics. Each entry links to the relevant API or concept. For a
from-scratch introduction, start with the tutorial chapter
[From a price to an order book](tutorial/01_from_price_to_book.md).

## Exchange mechanics

**Exchange / bourse**
: A venue where strangers trade a standardized instrument by posting
firm, standing offers instead of haggling pairwise. The public list of
those offers is the order book.

**Matching engine**
: The exchange's neutral component that pairs compatible buy and sell
orders under fixed, published rules and reports the resulting trades.

**Continuous double auction**
: The market design run by modern exchanges: both sides post offers
("double"), and matching happens the moment offers become compatible,
all session long ("continuous") — rather than at a scheduled auction
time.

**Price–time priority**
: The standard matching rule: better-priced orders trade first, and at
the same price, earlier arrivals trade first. The within-price queue it
creates is what Level 3 data (and the
[queue engine](api/analytics.md)) lets you reconstruct.

## Market data levels

**Level 1 (L1)**
: The top-of-book summary: best bid, best ask, and last trade. What
brokerage apps and tickers show as "the price".

**Level 2 (L2) — market by price**
: The full ladder of price levels with *aggregate* size at each level.
Individual orders are summed away.

**Level 3 (L3) — market by order**
: Every individual order with its own identity and queue position —
enough to replay arrivals, cancellations, and fills exactly. Also
called market-by-order (MBO) data. This is the resolution ob-analytics
reconstructs from Bitstamp and LOBSTER feeds.

**Best bid / best ask**
: The highest standing buy price and lowest standing sell price. The
ask minus the bid is the spread; their average is the mid-price.

**Last trade**
: The most recent execution's price — the number headlines call "the
price", which can move without any trade (see the flash example in the
[tutorial](tutorial/01_from_price_to_book.md)).

## Order book mechanics

**Limit order**
: An instruction to buy (bid) or sell (ask) at a specified price or better.
Sits in the book until matched, modified, or cancelled.

**Market order**
: An instruction to execute immediately against the best available
counter-side liquidity. Modeled in this package as a limit order whose
price crosses the spread on arrival.

**Maker / taker**
: The maker is the resting side of a trade (the limit order that was
already in the book); the taker is the aggressive side that crossed the
spread to consume it. See the `maker_event_id` / `taker_event_id` columns
documented in [Data Contracts](api/schemas.md).

**Spread**
: Best ask price minus best bid price. Extracted from the depth summary
via [`get_spread`](api/depth.md#ob_analytics.depth.get_spread).

**Mid-price**
: `(best_bid + best_ask) / 2`. Reference price for measuring order
aggressiveness in basis points.

**Basis point (BPS)**
: 1/100 of a percent. The depth summary bins liquidity into rings of
`depth_bps` width around the mid-price; see
[`PipelineConfig.depth_bps`](api/config.md#ob_analytics.config.PipelineConfig).

## Order classifications

Produced by [`set_order_types`](api/analytics.md#ob_analytics.analytics.set_order_types).

**Resting limit**
: A passive limit order that sits in the book and is eventually filled or
cancelled without ever crossing the spread.

**Market**
: An order that crosses the spread on arrival and executes immediately.

**Market-limit**
: A limit order that crosses on arrival but, after partial fills, comes
to rest as a passive order at a price inside the book.

**Flashed-limit**
: A limit order that is created and cancelled within a very short window
without ever filling. Common in HFT quote-stuffing patterns.

## Flow toxicity

Implemented in [`flow_toxicity`](api/flow_toxicity.md).

**VPIN — Volume-Synchronized Probability of Informed Trading**
: Easley, López de Prado, & O'Hara (2012). Bucket trades by equal volume,
classify each bucket as buy- or sell-driven, and report the rolling
absolute imbalance. High values (≳0.7) signal informed-trader pressure.

**Kyle's lambda (λ)**
: Kyle (1985). Slope of `Δprice ~ signed_volume` regression over a rolling
window. Higher λ → less liquid market (more adverse-selection cost per unit
of order flow). Returned as a [`KyleLambdaResult`](api/flow_toxicity.md#ob_analytics.flow_toxicity.KyleLambdaResult)
with the regression DataFrame attached.

**Order flow imbalance (OFI)**
: Per-window net buy-minus-sell volume normalised by total traded volume.
A short-horizon proxy for directional pressure.

## Data formats

**Bitstamp CSV**
: One row per order event with columns `id, timestamp, exchange_timestamp,
price, volume, action, direction`. The pipeline also expects a sibling
`trades.csv` with live-trade columns (`trade_id`, `timestamp`, `price`,
`amount`, `buy_order_id`, `sell_order_id`, `side`, …) — see
`scripts/collect_bitstamp_btcusd.py`.

**LOBSTER**
: Lim-Order-Book-System-The-Efficient-Reconstructor data set
([lobsterdata.com](https://lobsterdata.com/)). Provides paired
*message* and *orderbook* CSV files; integer prices in ten-thousandths of
a dollar; timestamps in seconds-after-midnight. Event types 1–7 cover
submissions, cancellations, executions of visible/hidden liquidity, cross
trades, and trading halts.
