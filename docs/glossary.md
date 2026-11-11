---
title: Glossary
---

# Glossary

Brief definitions of the market-microstructure terms used throughout
ob-analytics. Each term here has the same name as in the code. For a
from-scratch introduction, start with the tutorial chapter
[From a price to an order book](tutorial/01_from_price_to_book.md).

## Exchange mechanics

**Venue**
: The exchange or market a feed comes from, also called a bourse. A venue
is where strangers trade a standardized instrument by posting firm,
standing offers instead of haggling pairwise. The public list of those
offers is the order book. Named `venue` in the code.

**Matching engine**
: The venue's neutral component that pairs compatible buy and sell
orders under fixed, published rules and reports the resulting trades.

**Continuous double auction**
: The market design run by modern venues: both sides post offers
("double"), and matching happens the moment offers become compatible,
all session long ("continuous") — rather than at a scheduled auction
time.

**Price–time priority**
: The standard matching rule: better-priced orders trade first, and at
the same price, earlier arrivals trade first. The queue at each price is
what L3 data (and the [queue engine](api/engine.md)) lets you
reconstruct.

## Market data levels

**Resolution**
: The detail a feed or a plot works at: L2 or L3. The value is always
named `level` and typed `Level`, so a bare "level" means a resolution, and
one price on one side of the book is always called a *price level*.

**L1**
: The top of the book only: best bid, best ask, and last trade. What
brokerage apps and tickers show as "the price". No source is read at L1,
and `Level` has no such value.

**L2 — market by price (MBP)**
: The total resting size at each price level, with no order identity.
Individual orders are summed away. Most CCXT venues, Binance, Kalshi and
Polymarket are L2.

**L3 — market by order (MBO)**
: Every individual order with its own identity and its rank in the queue —
enough to replay arrivals, cancellations, and fills exactly. Bitstamp,
LOBSTER and Databento are L3.

**Last trade**
: The price of the most recent trade — the number headlines call "the
price". The book can change a lot while the last trade stays the same (see
the flash example in the [tutorial](tutorial/01_from_price_to_book.md)).

## Instruments and runs

**Instrument**
: The traded thing a run covers, written the readable way, for example
`BTC/USD`. Named `symbol`, except in `CaptureConfig.pair`, which holds the
form the venue itself uses (`btcusd` on Bitstamp).

**Price**
: A whole number of ticks (`int64`), never a float in the quote currency.
Named `price` on every table; see *Tick size* under
[Order book mechanics](#order-book-mechanics).

**Lot size**
: The instrument's smallest size increment, in the instrument's own units.
Every stored size (`volume`) is a whole number of lots (`int64`); multiply
by `lot_size` to get the size in the instrument's units.

**Instrument units**
: The units you give a setting in: BTC, shares or contracts for a size, the
quote currency for a price. Every size or price setting is in these units,
also when the frame holds whole lots and ticks. A frame's
`attrs["lot_size"]` and `attrs["tick_size"]` tell the package how to convert.

**Session**
: The venue's own trading period, in the venue's local time zone. Named
`session_tz` where that zone matters.

**Run**
: One pass of the pipeline over one source, for one instrument at one
venue.

**Run context**
: The parameters that change per run rather than per source: trading date,
session time zone, symbol and venue. Typed `RunContext`.

## Order book mechanics

**Order book (book)**
: All orders resting at a venue for one instrument at one instant.

**Side**
: Bid or ask. Named `direction` on the events, depth and book snapshot
tables.

**Price level**
: One price on one side of the book, and the size resting there.

**Limit order**
: An instruction to buy (bid) or sell (ask) at a specified price or better.
Sits in the book until matched, modified, or cancelled.

**Market order**
: An instruction to execute immediately against the best price on the other
side. Modeled in this package as a limit order whose price crosses the spread
on arrival.

**Maker / taker**
: The maker is the resting side of a trade (the limit order that was
already in the book); the taker is the aggressive side that crossed the
spread to take it. See the `maker_event_id` / `taker_event_id` columns
documented in [Data Contracts](api/schemas.md).

**Best bid / best ask**
: The highest resting buy price and lowest resting sell price. Named
`best_bid_price` / `best_ask_price`, with `best_bid_vol` / `best_ask_vol`
for the size resting there.

**Touch**
: The best bid and the best ask together: the innermost price level on each
side. Also called the top of book.

**Spread**
: Best ask price minus best bid price, in ticks — see *Tick size* below.
Extracted from the depth summary via
[`get_spread`](api/depth.md#ob_analytics.depth.get_spread).

**Mid-price**
: `(best_bid_price + best_ask_price) / 2`, in ticks — see *Tick size* below.
The reference price for distances in basis points, such as order
aggressiveness.

**Crossed book**
: A book whose best bid is at or above its best ask. Expected in a diff
feed; a fault in a matched book (see *Feed type* under
[Data formats](#data-formats)).

**Tick size**
: The instrument's minimum price increment, in quote currency. Every
`price` column is stored as a whole number of ticks (`int64`), not a float
in the quote currency; multiply by
[`PipelineConfig.tick_size`](api/config.md#ob_analytics.config.PipelineConfig)
to recover the quote-currency price.

**Basis point (BPS)**
: 1/100 of a percent, the unit for distances from the mid-price. Named
`bps` in identifiers.

**Depth**
: The record of price level size changing over time. The `depth` table holds
one row per change to one price level.

**Depth summary**
: One row per event holding the touch and the size resting in each depth bin
out from the mid-price. The `depth_summary` table, made by
[`summarize_depth`](api/depth.md).

**Depth bin**
: One range of distances from the mid-price in the depth summary, on each
side of the book. Its width in basis points is
[`PipelineConfig.depth_bps`](api/config.md#ob_analytics.config.PipelineConfig)
and the number of bins is `depth_bins`.

**Micro-price**
: The mid-price weighted by the size resting on the opposite side. Named
`micro_price`.

**Book imbalance**
: The signed share of resting volume on the bid side. Computed by
`book_imbalance` and carried in the `obi` and `obi_depth` columns. Not the
same as *trade imbalance*, which measures traded size.

**Book snapshot**
: The resting book at one point in time, as separate bid and ask tables.

**Book replay**
: Book snapshots at one instant per interval across part of a run, shown one
after another beside the trades. Named `book_replay`. Each step on screen is a
frame, which holds one book snapshot.

**Venue book**
: The book as the venue itself states it, beside the order events: for
LOBSTER, the orderbook file, which holds the best N price levels after every
message. `audit` compares the rebuilt book with it. Most sources have none.

## Order events

**Event**
: One change to one order, and the unit of the L3 model.

**Order id**
: The venue's identity for an order, stable across its whole life. Named
`id`.

**Event id**
: The identity of one event, unique across the run. Named `event_id`, and
the key a trade's `maker_event_id` and `taker_event_id` point at.

**Receive time**
: When ob-analytics saw the event, on the local clock. Named `timestamp`,
and always tz-aware UTC.

**Exchange time**
: When the venue's matching engine saw the event. Named
`exchange_timestamp`, and always tz-aware UTC.

**Action**
: What an event did to the order: `created`, `changed` or `deleted`. Named
`action`; the venue's own code is kept in `raw_event_type`.

**Volume**
: Size, stored as a whole number of lots. Its meaning depends on the table: the size still
resting after the event on the events table, the size removed without
trading on a delete, the price level's new resting size on the depth table,
and the executed size on the trades table.

**Fill**
: The size executed at one event, and `0` when nothing traded. Named
`fill`. On a delete, `volume` plus `fill` is what the order had resting just
before.

**Outstanding**
: The size of an order still resting, after everything executed or
cancelled so far.

**Resting price**
: The price an order rests at: the price of its latest `created` row, moved
only by a `changed` row with no fill. The depth table, the per-order book
and the queues all place an order at its resting price.

**Pre-existing order**
: An order first seen part-way through the stream, with no `created` row —
the opening book at capture start, or a hidden execution. It cannot be
classified, and that is not a failure.

**Aggressiveness**
: How far an order was placed from the best price on its own side, in basis
points. Named `aggressiveness_bps`; a positive value improved the touch.

**Lifecycle**
: One row per order, from placement to end.

**Outcome**
: How an order's life ended: `filled`, `partial`, `cancelled` or `resting`.
Named `outcome`.

## Order classifications

**Order type**
: The class
[`set_order_types`](api/analytics.md#ob_analytics.analytics.set_order_types)
gives an order from how it behaved over its whole life. Named `type`, with
one of the six values below.

**Unknown**
: The starting value. It stays on any order that fits none of the other
types once classification finishes. `set_order_types` issues a warning
when this happens, because it shows a gap in the classifier, not a normal
outcome.

**Pre-existing**
: A pre-existing order (see [Order events](#order-events)). It gets its own
value instead of *unknown*, because it cannot be classified by design.

**Resting limit**
: A passive limit order that sits in the book and is eventually filled or
cancelled without ever crossing the spread. The value is `resting-limit`.

**Market**
: An order that crosses the spread on arrival and executes immediately.

**Market-limit**
: A limit order that crosses on arrival but, after partial fills, comes
to rest as a passive order at a price inside the book.

**Flashed-limit**
: A limit order that is created and cancelled within a very short time
without ever filling. Common in HFT quote-stuffing patterns.

## Trades

**Trade direction**
: The taker's side of a trade: `buy` or `sell`. Named `direction` on the
trades table — the same column name as the book side, but with different
values. Prose may call it the aggressor side where a bare "direction" would
read as the book side.

**Trade sign**
: A trade direction worked out after the fact, for a feed that does not
label the taker's side: by the tick rule, Lee-Ready, or bulk volume
classification. See [`trade_sign`](api/trade_sign.md).

**Unmatched trade**
: A trade that could not be tied back to a resting order.

## Queue

Implemented in the [queue engine](api/engine.md).

**Queue**
: The orders resting at one price level, in the arrival order that decides
which executes first.

**Rank**
: An order's place in its queue, counted from the front, starting at 1.
Named `rank`.

**Ahead volume**
: The total size of the orders in front of an order in its queue. Named
`ahead_volume`.

**Queue length**
: The number of orders resting at a price level. Named `queue_len`.

**Age**
: How long an order has rested, in seconds. Named `age_s`.

## Hidden liquidity

Implemented in [`hidden_liquidity`](api/hidden_liquidity.md).

**Hidden order**
: An order the venue will match but does not show in the visible book.
LOBSTER marks an execution against one as event type 5, with order id `0`.

**Iceberg order**
: An order that shows a small displayed *peak* and keeps the rest in
reserve. Each visible order it shows is a *slice*. When a slice is filled
out, the venue shows the next one as a new order at the same price and side.

**Filled out**
: A slice whose last fill as a maker left nothing outstanding.

**Refill**
: The new slice that appears after an iceberg's slice is filled out.
[`detect_icebergs`](api/hidden_liquidity.md#ob_analytics.hidden_liquidity.detect_icebergs)
looks for refills within `max_delay` (one millisecond by default) and chains
them into one suspected iceberg.

**Trade against a hidden order**
: A trade that printed strictly inside the visible spread, so no visible order
rested at its price. Found by
[`hidden_trades`](api/hidden_liquidity.md#ob_analytics.hidden_liquidity.hidden_trades).

## Flow toxicity

Implemented in [`flow_toxicity`](api/flow_toxicity.md).

**VPIN — Volume-Synchronized Probability of Informed Trading**
: Easley, López de Prado, & O'Hara (2012). Bucket trades by equal volume,
classify each bucket as buy- or sell-driven, and report the rolling
absolute imbalance. High values (≳0.7) signal informed-trader pressure.
The usual bucket size is average daily volume ÷ 50
([`vpin_bucket_volume`](api/flow_toxicity.md#ob_analytics.flow_toxicity.vpin_bucket_volume)).

**Kyle's lambda (λ)**
: Kyle (1985). Slope of `Δprice ~ signed_volume` regression over a rolling
window. Higher λ → less liquid market (more adverse-selection cost per unit
of signed volume). Returned as a [`KyleLambdaResult`](api/flow_toxicity.md#ob_analytics.flow_toxicity.KyleLambdaResult)
with the regression DataFrame, a bootstrap confidence interval, and a
`significant` flag that is `False` on too few windows or `|t| < 2`.

**Order flow imbalance (OFI)**
: Signed volume divided by total traded volume, per time window, from `-1`
(all sells) to `+1` (all buys). A short-horizon proxy for directional
pressure. Computed by `order_flow_imbalance`, or by `ofi_by_window` for
several trailing windows at once.

## Bars

Implemented in [`bars`](api/bars.md).

**Bar**
: A summary of a run of consecutive trades: open, high, low, close, volume,
and the microstructure columns that go with them. Produced by `bars()`.

**Bar rule**
: What decides where the bar boundaries fall: `time`, `tick`, `volume`,
`dollar` or `imbalance`. Named `rule` and typed `BarRule`.

**Threshold**
: How much of the rule's own quantity closes a bar. Named `threshold`; its
unit is the rule's (a duration, a trade count, or an amount in instrument
units).

**Time bar (OHLCV)**
: A bar covering a fixed span of the clock — the familiar 1-minute or daily
candle.

**Tick bar**
: A bar covering a fixed number of trades. "Tick" here is the market-data
sense of one printed trade, not the price increment.

**Volume bar / dollar bar**
: A bar covering a fixed amount of traded size, or of price × size. Sampling
by activity rather than by the clock puts the same amount of market in each
bar, which brings bar returns much closer to being independent and
identically distributed.

**Imbalance bar**
: A bar that ends when signed size drifts a set amount away from where the bar
opened, in either direction (López de Prado). A burst of one-sided flow closes
a bar; balanced trading stays inside one.

**VWAP**
: Volume-weighted average price — a bar's turnover divided by its volume. The
average price actually paid, rather than the average of the prices printed.

**Turnover**
: Price × size summed over a bar: the value that changed hands, in the quote
currency. Also called notional or dollar volume.

**Signed volume**
: Buyer-initiated volume minus seller-initiated volume. The net direction of
the flow in a bar, and what an imbalance bar accumulates. Named
`signed_volume`, with `buy_volume` / `sell_volume` for the two halves.

## Transaction cost

Implemented in [`cost`](api/cost.md).

**Effective spread**
: What a taker actually paid to cross, measured from the mid-price the trade
crossed: `2 * D * (price - mid)`, where `D` is `+1` for a buy and `-1` for a
sell. Doubled so it compares with a quoted spread, which also spans both
sides of the mid. Unlike the quoted spread it is a property of trades, not of
the book, so it reflects where in the book people actually traded.

**Realized spread**
: The part of the effective spread the liquidity provider kept, measured one
horizon after the trade: `2 * D * (price - mid_later)`. A negative realized
spread means the provider lost money on the trade — the flow was informed.

**Price impact**
: The rest of the effective spread: how far the trade moved the market,
`2 * D * (mid_later - mid)`. Effective spread = realized spread + price
impact, exactly, trade by trade.

**Horizon**
: The wait between a trade and the mid-price the realized spread is read
against. Five minutes is the equity convention; a fast tape needs less,
because unrelated price moves are charged to the trade as impact. There is no
neutral value, so the horizon is reported with the number.

**Amihud illiquidity**
: Amihud (2002). The price move a unit of turnover buys, `|return| / turnover`
over a window. High means thin: a small amount of trading swings the price.
Needs no quotes and no aggressor side.

**Roll's implied spread**
: Roll (1984). The spread implied by bid-ask bounce, `2 * sqrt(-cov)` over the
lag-1 autocovariance of trade price changes. It assumes the bounce is the only
thing moving the price, which fixes the lag-1 **autocorrelation** of those
changes at `-0.5` — the diagnostic returned beside the estimate. On a tape
that is sparse relative to how fast the instrument moves, the price change
between trades is mostly efficient-price movement rather than bounce; the
autocovariance then comes out positive, the estimate does not exist, and it is
reported as `NaN` rather than hidden. When it lands negative by chance the
estimate exists but means nothing, which is why the autocorrelation matters
more than the root.

## Feature table

Implemented in [`features`](api/features.md).

**Feature table**
: One tidy table for a model or a study: a point in time on each row, a
microstructure feature in each column. Every row is stated as of one instant,
and everything in it was known at that instant.

**Feature**
: One measured column set of that table. Registered under a name, so a
measurement of your own becomes a column with no edit to the package.

**Look-ahead**
: Using data from after a row's instant to compute that row. It flatters a
model in testing and cannot be repeated in trading, which is the most common
way a backtest comes out wrong. The feature table has none by construction:
the trade columns cover the bar only, and the book columns are a backward
as-of join.
: The test for it is truncation. Cut the inputs short and recompute; a table
free of look-ahead reproduces the rows that survive exactly, because none of
them ever read the data that was removed.

**As-of join**
: Matching each row to the last observation at or before its instant, rather
than to one sharing its key. How the book reaches a row whose instant falls
between two book snapshots.

**Target**
: What a model predicts, usually the return over the bar after the row. It
looks forward, so it is not a feature and the table does not hold one: build
one with a negative shift.

**Trade imbalance**
: The signed share of a bar's volume, `signed_volume / volume`, from `-1`
(every trade a sell) to `+1` (every trade a buy).

**Realized volatility**
: The standard deviation of returns over a trailing window. Per bar rather
than annualized: only time bars span equal amounts of time, so there is no
one factor that would scale it to a year.

**Trailing window**
: The stretch a value looks back over, ending at its own row. A feature
counts it in bars (`n_bars`, the row's own included). A `window` argument is
always a span of time, given as a pandas offset string.

**Close location**
: Where a bar's close fell within its own high-low range. Not a built-in
feature; the worked example of writing one.

## Pipeline concepts

**Source**
: A venue ob-analytics can read, with its name, resolution, feed type and
settings. It replays stored files (`OfflineSource`), captures a running venue
(`LiveSource`), or both. See [Sources](api/sources.md).

**Canonical**
: Matching the shared schema. The canonical events and trades frames are
the venue-independent DataFrames every pipeline stage consumes,
defined by the column contracts in [Data Contracts](api/schemas.md).
Once frames pass the validators, downstream code cannot tell which
venue they came from.

**Shared schema**
: The one column contract every stage reads and writes, and the versioned
Parquet format that carries it between runs. See [Schema](schema.md).

**Loader**
: Produces the canonical frame for a source, via `create_loader`. An L3
source's `create_loader` returns an [`EventLoader`](api/protocols.md),
whose `load()` returns a validator-passing events frame; an L2 source
returns a [`DepthLoader`](api/protocols.md), whose `load()` returns the
depth frame directly, since a price-level feed has no per-order events to
fold. One per venue dialect.

**Trade reader**
: The companion protocol for executions: `load(events, path)` returns
the canonical trades frame with maker/taker attribution.

**Pipeline**
: The run that takes a source to its four output tables: `events`,
`trades`, `depth` and `depth_summary`.

**Declarations**
: What a run's data is declared to show: feed type, trade attribution,
sequence kind and clocks. They come from the source, and a capture record
replaces any it holds. A result carries them (`FeedDeclarations`), and
`audit` holds the run to them.

**Credential**
: A key a source needs to read a venue's data, such as an API key. Read from
an environment variable, and never written to a capture's files.

**Window**
: One stretch of time a windowed run (`Pipeline.run_windows`) cuts its input
into, from one boundary up to, but not including, the next. The windows cover
the whole input.

**Carry**
: Starting a window from the book the previous window ended with, so that the
cut does not show in the output. Named `carry`.

**Metric**
: A measurement taken from a finished run — registered under a name in
`METRICS`, computed on demand by `result.metric(name)`, and drawn as a
level-less plot under that same name. See
[Metrics](api/metrics.md) and [Extending](extending.md#4-a-new-metric).

## Capture

**Capture**
: Recording a live venue to files the pipeline can later replay.

**Segment**
: One part of a capture, from a snapshot to its close-out, in its own folder
(`seg-0001`, ...). A disconnect, a roll or a restart ends one segment and
starts the next, so every segment replays alone.

**Capture gap**
: A stretch of a capture that no segment covers, recorded in `manifest.json`
with its cause. Not a sequence gap, which is a message missing inside a
segment.

**Capture record**
: The `meta.json` a capture writes beside each segment's files. It says which
source made the capture, what that source declares, and the instrument's tick
and lot size. Typed `CaptureRecord`, read with `read_record`.

**Origin**
: Which part of a capture wrote a row: `snapshot` (the opening book),
`stream` (a live message) or `shutdown` (a synthetic close-out at the end).
Named `origin`.

## Ordering and data quality

See the [Data quality explainer](data-quality.md).

**Sequence**
: The venue's own per-event number, carried only by sources that publish
one. Named `sequence`.

**Ingest sequence**
: A local counter that records the order in which rows arrived. Named
`ingest_seq`.

**Source row number**
: Which row of the source file an event came from. Named `original_number`.

**Sequence gap**
: A skipped venue sequence number, which means a dropped message.

**Data quality summary**
: The per-run health report on a feed, covering crossing, unmatched trades,
duplicate ids, pre-existing orders and sequence faults.

**Faithful**
: Replayed exactly as the feed states it, crossed resting orders included.
The default, except on the L3 face of a book replay (see *Uncross*).

**Uncross**
: Remove crossed resting orders so a book reads `best_bid < best_ask`. A
display choice (`uncross=`), never the default, except on the L3 face of a
book replay.

**Stale order**
: A resting order that a trade printed through, and that the venue did not
report again within a grace period (`STALE_GRACE`, one second). The trade
shows the order has gone, but the faithful book still holds it. Reported by
`audit`, never removed. Found by `detect_stale_orders`.

## Plots

See [Visualization](api/visualization.md) and [Gallery](api/gallery.md).

**Concept**
: What a plot shows, independent of resolution — for example
`depth_heatmap` or `trade_tape`. A concept holds up to one variant per
resolution.

**Face**
: One rendered plot: a concept at one resolution on one backend.

**Comparable**
: A concept with both an L2 and an L3 variant, so the two can sit side by
side.

**Backend**
: The drawing library a face is rendered with: matplotlib, plotly, or bokeh.

**Gallery**
: The generated HTML page that holds a run's faces.

**Card**
: One entry in a gallery: a title, a note on how to read it, and one panel
per column.

**Panel**
: One column within a gallery card, holding a face and how to display it.

**View**
: Which faces a gallery shows: `l2`, `l3`, `both` or `comparison`.

## Data formats

**Feed type**
: Whether a feed's resting book can be crossed. Named `feed_type` and typed
`FeedType`, with the four values below. It is a property of the source, not
of the rebuild.

**Matched book**
: An L3 feed produced by the venue's own matching engine (LOBSTER,
exchange MBO feeds): bids can never rest above asks, so an uncrossed
book is a guaranteed invariant of the data.

**Diff feed**
: An L3 feed reconstructed from a public placement/cancellation
stream (the Bitstamp feed used here). It can contain genuinely crossed
*resting* orders; `order_book()` replays such feeds faithfully — a
crossed book in the output is a property of the feed, not a
reconstruction bug. See the pitfall note in
[L1 → L2 → L3](tutorial/02_three_resolutions.md) and the
[Data quality explainer](data-quality.md), which shows how to measure the
crossing (`audit`) and uncross it for display (`uncross=`).

**Price levels**
: An L2 feed: the venue's total size at each price, with no order identity
(Binance, Coinbase, Kraken, Kalshi, Polymarket, and cryptofeed at L2). The
venue does not publish a crossed book, so a crossed one means the capture's copy
is wrong: most often, the capture kept a price level the venue removed. See
[What each feed shows](feeds.md).

**Unknown feed type**
: A source that does not declare a feed type. `FeedType.UNKNOWN`, the
default.

**Trade attribution**
: Which orders of a trade a feed's order events can name. Named
`trade_attribution` and typed `TradeAttribution`. `BOTH` when the feed
reports every order, takers included (Bitstamp). `MAKER_ONLY` when it shows
resting orders only (LOBSTER, Databento, cryptofeed). `NONE` for L2, which
has no order identity.

**Bitstamp CSV**
: One row per order event with columns `id, timestamp, exchange_timestamp,
price, volume, action, direction`. The pipeline also expects a sibling
`trades.csv` with live-trade columns (`trade_id`, `timestamp`, `price`,
`amount`, `buy_order_id`, `sell_order_id`, `side`, …), as
`ob-analytics capture bitstamp` writes it.

**LOBSTER**
: Lim-Order-Book-System-The-Efficient-Reconstructor data set
([lobsterdata.com](https://lobsterdata.com/)). Provides paired
*message* and *orderbook* CSV files; integer prices in ten-thousandths of
a dollar; timestamps in seconds-after-midnight. Event types 1–7 cover
submissions, cancellations, executions of visible/hidden liquidity, cross
trades, and trading halts.
