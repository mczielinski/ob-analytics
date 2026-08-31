# ob-analytics

ob-analytics rebuilds a limit order book from a venue's raw event stream and
measures it. This file is the glossary of the words the code already uses, so
that new code, tests, issues and documents name things the same way. Where it
disagrees with the published glossary at `docs/glossary.md`, that file wins and
this one is corrected.

## Venue and instrument

**Venue**:
The exchange or market a feed comes from. Named `venue`.
_Avoid_: exchange, bourse, market, provider

**Instrument**:
The traded thing a run covers, written the readable way, for example `BTC/USD`.
Named `symbol`, except in `CaptureConfig.pair`, which holds whatever form the
venue itself uses (`btcusd` on Bitstamp, `BTC/USDT` on CCXT).
_Avoid_: asset, market, product, ticker

**Price**:
A whole number of ticks, never a float in the quote currency. Named `price` on
every table.
_Avoid_: raw price, float price

**Tick size**:
The instrument's smallest price increment, in the quote currency. Named
`tick_size`, and what a price is multiplied by to get the quote currency.

**Basis point (BPS)**:
One hundredth of a percent, the unit for distances from the mid-price. Named
`bps` in identifiers.

**Session**:
The venue's own trading period, in the venue's local time zone. Named
`session_tz` where that zone matters.

**Run**:
One pass of the pipeline over one source, covering one instrument at one venue.
_Avoid_: session, job, execution

## Resolution

**Resolution**:
The detail a feed or a plot works at: L2 or L3. Called resolution in prose;
the value is always named `level` and typed `Level`.
_Avoid_: granularity, depth, `resolution` as an identifier

**L1**:
The top of the book only: best bid, best ask and last trade. No source is
ingested at L1; `Level` has no such value.

**L3 / market-by-order (MBO)**:
One record per individual order, with a stable order identity. Bitstamp,
LOBSTER and Databento are L3.

**L2 / market-by-price (MBP)**:
Aggregate resting size per price, with no order identity. Most CCXT venues,
Binance, Kalshi and Polymarket are L2.

**Price–time priority**:
The matching rule that gives the queue its order: better prices execute first,
and at one price, earlier arrivals execute first.

**Feed type**:
Whether an L3 feed's resting book can be crossed. Named `feed_type` and typed
`FeedType`, with the values below; it is a property of the source, not of the
rebuild.

**Matched book**:
An L3 feed emitted by the venue's own matching engine. Its book is never
crossed. `FeedType.MATCHED_BOOK`.

**Diff feed**:
An L3 feed rebuilt from a public placement and cancellation stream. It can hold
genuinely crossed resting orders. `FeedType.DIFF_FEED`.

**Unknown feed type**:
A source that does not declare a feed type. `FeedType.UNKNOWN`, the default.

**Faithful**:
Replayed exactly as the feed states it, crossed resting orders included.
_Avoid_: raw, unmodified, as-is

**Uncross**:
Remove crossed resting orders so a book reads `best_bid < best_ask`. A display
choice, never the default.

## The book

**Order book** (**book**):
All orders resting at a venue for one instrument at one instant.

**Side**:
Bid or ask. Named `direction` on the events, depth and book snapshot tables.
_Avoid_: `side` as a column name

**Price level**:
One price rung on one side of the book, and the size resting there. Always
written in full, because a bare "level" means resolution.
_Avoid_: rung, bucket, tier, bare "level"

**Touch**:
The best bid and the best ask — the innermost price level on each side.
_Avoid_: top of book, BBO, inside market

**Best bid** / **best ask**:
The highest resting buy price and the lowest resting sell price. Named
`best_bid_price` / `best_ask_price`, with `best_bid_vol` / `best_ask_vol` for
the size resting there.

**Spread**:
The best ask price minus the best bid price. Named `get_spread` where it is
extracted.

**Mid-price**:
The average of the best bid and the best ask, and the reference point distances
in basis points are measured from.
_Avoid_: mid market, fair value

**Crossed book**:
A state where the best bid is at or above the best ask. Expected in a diff
feed; a fault in a matched book.

**Book snapshot**:
The resting book at one point in time, as separate bid and ask tables.
_Avoid_: state, image, frame

## Order events

**Event**:
One change to one order, and the unit of the L3 model.

**Order id**:
The venue's identity for an order, stable across its whole life. Named `id`.

**Event id**:
The identity of a single event, unique across the run. Named `event_id`, and
the key a trade's `maker_event_id` and `taker_event_id` point at.
_Avoid_: `id` when you mean `event_id`

**Receive time**:
When ob-analytics saw the event, on the local clock. Named `timestamp`, and
always tz-aware UTC.

**Exchange time**:
When the venue's matching engine saw the event. Named `exchange_timestamp`,
and always tz-aware UTC.
_Avoid_: venue time, server time, event time

**Action**:
What an event did to the order: `created`, `changed` or `deleted`. Named
`action`; a venue's own code is `raw_event_type`.
_Avoid_: kind, op, message type

**Volume**:
Size, in the base asset or in shares. Its meaning is per table: outstanding
size after the event, the size removed on a delete, the level's new resting
size on the depth table, and the executed size on trades.
_Avoid_: size, quantity, amount

**Fill**:
The size executed at one event, and `0` when nothing traded. Named `fill`.

**Outstanding**:
The size of an order still resting, after everything executed or cancelled so
far.
_Avoid_: open, live

**Pre-existing order**:
An order first seen part-way through the stream, with no `created` event. It
cannot be classified, which is not a failure.

**Aggressiveness**:
How far an order was placed from the best price on its own side, in basis
points. Named `aggressiveness_bps`; a positive value improved the touch.

**Lifecycle**:
One row per order, from placement to end.

**Outcome**:
How an order's life ended: `filled`, `partial`, `cancelled` or `resting`.
Named `outcome`.
_Avoid_: status, state, result

## Order classification

**Order type**:
The class the classifier gives an order from how it behaved over its whole
life. Named `type`, with values `unknown`, `pre-existing`, `flashed-limit`,
`resting-limit`, `market-limit` and `market`.
_Avoid_: category, class, label

**Maker** / **taker**:
The maker is the resting side of a trade; the taker is the side that crossed
the spread to take it. Named `maker` / `taker` for the order ids and
`maker_event_id` / `taker_event_id` for the events.

**Trade direction**:
The taker's side of a trade: `buy` or `sell`. Named `direction` on the trades
table — the same column name as the book side, but different values.
_Avoid_: aggressor, sign, initiator

**Trade sign**:
A direction worked out after the fact for a feed that does not label the taker
side, by the tick rule, Lee-Ready or bulk volume classification.
_Avoid_: inferred direction, classified side

## Depth

**Depth**:
The record of price level size changing over time. The `depth` table holds one
row per change to one price level.
_Avoid_: liquidity, book state, ladder

**Depth summary**:
One row per event holding the touch and the size resting in each basis-point
ring out from the mid-price. Named `depth_summary`.
_Avoid_: metrics, stats, aggregates

**Depth bin**:
One basis-point ring of the depth summary. Its width is `depth_bps` and their
count is `depth_bins`.
_Avoid_: bucket, band, ring

**Micro-price**:
The mid-price weighted by the size resting on the opposite side. Named
`micro_price`.

**Book imbalance**:
The signed share of resting volume on the bid side. Computed by
`book_imbalance` and carried in the `obi` and `obi_depth` columns.

## Queue

**Queue**:
The orders resting at one price level, in the arrival order that decides which
executes first.

**Rank**:
An order's position in its price level queue, counted from the front, starting
at 1. Named `rank`.
_Avoid_: position, index, slot

**Ahead volume**:
The total size of the orders in front of an order in its queue. Named
`ahead_volume`.

**Queue length**:
The number of orders resting at a price level. Named `queue_len`.

**Age**:
How long an order has rested, in seconds. Named `age_s`.

## Sources and the pipeline

**Source**:
A venue ob-analytics can read, declaring its name, resolution, feed type and
settings. It replays stored files (`OfflineSource`), captures a running venue
(`LiveSource`), or both.
_Avoid_: format, connector, adapter, driver, backend

**Loader**:
The component that turns one source's stored files into a canonical table —
`EventLoader` at L3, `DepthSource` at L2.

**Trade source**:
The component that builds the trades table for a run. Typed `TradeSource`: a
loader role, not a venue.

**Canonical**:
Matching the shared schema every stage reads and writes, so no downstream code
can tell which venue the data came from.
_Avoid_: normalized, standard, internal

**Shared schema**:
The one column contract every layer reads and writes, and the versioned
Parquet format that carries it between runs.

**Pipeline**:
The run that takes a source to its four output tables: `events`, `trades`,
`depth` and `depth_summary`.

**Run context**:
The parameters that change per run rather than per source — trading date,
session time zone, symbol and venue. Typed `RunContext`, named `ctx`.
_Avoid_: options, params

**Capture**:
Recording a live venue to files the pipeline can later replay.
_Avoid_: stream, ingest, collect, record

## Ordering and data quality

**Sequence**:
The venue's own per-event number, carried only by sources that publish one.
Named `sequence`.

**Ingest sequence**:
A local counter recording the order rows arrived in. Named `ingest_seq`.
_Avoid_: row number, index

**Source row number**:
Which row of the source file an event came from. Named `original_number`, and
not the same thing as an ingest sequence.

**Sequence gap**:
A skipped venue sequence number, meaning a dropped message. A step that does
not advance means a reordered or repeated message.

**Unmatched trade**:
A trade that could not be tied back to a resting order.

**Data quality summary**:
The per-run health report on a feed, covering crossing, unmatched trades,
duplicate ids, pre-existing orders and sequence faults.

## Plots

**Concept**:
What a plot shows, independent of resolution — for example `depth_heatmap` or
`trade_tape`. A concept holds up to one variant per resolution.
_Avoid_: chart type, plot kind, view

**Face**:
One rendered plot: a concept at one resolution on one backend.

**Panel**:
One column within a gallery card, holding a face and how to display it.
_Avoid_: panel for a face, or for a level-less analytic plot

**Comparable**:
A concept with both an L2 and an L3 variant, so the two can sit side by side.

**Backend**:
The drawing library a face is rendered with: matplotlib or plotly. A renderer
is the registered function, not the library.
_Avoid_: engine, renderer library

**View**:
Which faces a gallery shows: `l2`, `l3`, `both` or `comparison`.

**Gallery**:
The generated HTML page holding a run's faces.
