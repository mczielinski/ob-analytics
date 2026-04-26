---
title: Glossary
---

# Glossary

Brief definitions of the market-microstructure jargon used throughout
ob-analytics. Each entry links to the relevant API or concept.

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
on the [`Trade`](api/models.md#ob_analytics.models.Trade) model.

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

**Pacman**
: A limit order that walks through multiple price levels via successive
modifications, consuming liquidity as it goes — named for the arcade
character that eats dots.

**Zombie order**
: An order that should have been filled or cancelled but was never
explicitly closed in the source feed (a data-quality artefact). Detected
by [`get_zombie_ids`](api/data.md#ob_analytics.data.get_zombie_ids) and
removed by the pipeline unless `skip_zombie_detection` is set.

## Flow toxicity

Implemented in [`flow_toxicity`](api/flow_toxicity.md).

**VPIN — Volume-Synchronized Probability of Informed Trading**
: Easley, López de Prado, & O'Hara (2012). Bucket trades by equal volume,
classify each bucket as buy- or sell-driven, and report the rolling
absolute imbalance. High values (≳0.7) signal informed-trader pressure.

**Kyle's lambda (λ)**
: Kyle (1985). Slope of `Δprice ~ signed_volume` regression over a rolling
window. Higher λ → less liquid market (more adverse-selection cost per unit
of order flow). Returned as a [`KyleLambdaResult`](api/models.md#ob_analytics.models.KyleLambdaResult)
with the regression DataFrame attached.

**Order flow imbalance (OFI)**
: Per-window net buy-minus-sell volume normalised by total traded volume.
A short-horizon proxy for directional pressure.

## Algorithms

**Needleman–Wunsch**
: Dynamic-programming sequence-alignment algorithm originally from
bioinformatics. ob-analytics' Bitstamp matcher uses it to pair simultaneous
bid and ask fills under a configurable time cutoff
(`match_cutoff_ms`). See
[`BitstampMatcher`](api/bitstamp.md#ob_analytics.bitstamp.BitstampMatcher).

## Data formats

**Bitstamp CSV**
: One row per order event with columns `id, timestamp, exchange_timestamp,
price, volume, action, direction`. The bundled sample is parsed from raw
websocket logs by `scripts/parse_bitstamp_log.sh`.

**LOBSTER**
: Lim-Order-Book-System-The-Efficient-Reconstructor data set
([lobsterdata.com](https://lobsterdata.com/)). Provides paired
*message* and *orderbook* CSV files; integer prices in ten-thousandths of
a dollar; timestamps in seconds-after-midnight. Event types 1–7 cover
submissions, cancellations, executions of visible/hidden liquidity, cross
trades, and trading halts.
