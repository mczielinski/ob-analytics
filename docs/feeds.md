---
title: What each feed shows
---

# What each feed shows

Two captures of the same market can tell you different things. One source
shows every order and another only the top 100. One reports every change and
another polls once a second. One names the taker of a trade and another cannot.
These properties belong to the source and the venue, not to the analysis, and
they decide which results you can trust.

This page describes each property once, then sets out its value for every
source and venue in four tables:

- [What the book shows](#what-the-book-shows)
- [Timing and integrity](#timing-and-integrity)
- [Trades](#trades)
- [Venue rules](#venue-rules)

Each how-to page for a source opens with its row of these tables.

The values were checked in September 2026 with ccxt 4.5.84 and cryptofeed
2.4.1, by reading their code and by short live captures of public data. A value
that could not be checked says why. [How the values were
checked](#how-the-values-were-checked) gives the details.

## The properties

Each property is described in the same way:

- **What it is.**
- **Values:** the values it can take.
- **Sources:** which sources have which value (the tables below give every
  source).
- **Lets you conclude:** what a value allows.
- **Stops you concluding:** what a value rules out, and how that shows up.
- **In the package:** where ob-analytics declares or checks it.

### Level

Whether the feed shows single orders or only the total size at each price.

- **Values:** L3, one row per order, with the order's id. L2, one row per price
  level, with no ids.
- **Sources:** L3 is native Bitstamp, cryptofeed on its four per-order venues,
  LOBSTER and Databento. L2 is everything else.
- **Lets you conclude:** at L3, order lifetimes, queue position, order types,
  and the order behind each trade.
- **Stops you concluding:** at L2, anything about single orders. A level that
  grows could be one new order or ten.
- **In the package:** `Source.level`. The pipeline skips the per-order stages at
  L2 (see [Process L2 feeds](howto/l2-depth.md)).

### Depth shown

How much of the book the feed shows.

- **Values:** the whole book; the top N orders a side; the top N price levels a
  side; whatever each poll returns.
- **Sources:** most feeds show the whole book. cryptofeed's Bitstamp and
  Bitfinex books show the top 100 orders a side. ccxt's Kraken book shows the
  top `--depth-limit` levels, and its Binance book 1,000 levels by default.
- **Lets you conclude:** depth, the depth summary's rings and the depth heatmap
  are right out to the edge of what is shown.
- **Stops you concluding:** anything past that edge. With a window of the top
  N, a level or order that leaves the window looks the same as one that was
  cancelled.
- **In the package:** `--depth-limit` for ccxt. The cryptofeed source keeps an
  order that leaves Bitstamp's top-100 window, and records it `deleted` only
  once a snapshot shows its price without it.

**Why Binance stops at 5,000 levels.** Binance's REST book returns at most 5,000
levels a side, and past its opening book ccxt learns a level only when it
changes. A deeper book would have gaps at the bottom, so the capture refuses a
`--depth-limit` above 5,000. For BTC/USDT, 5,000 levels reach about 1% from the
price. See [Binance](howto/binance.md#depth-request-more-levels).

### Orders shown

At L3: whether the feed shows the orders that trade on arrival, the takers.

- **Values:** every order, takers included; resting orders only.
- **Sources:** native Bitstamp shows every order. All other L3 sources show
  resting orders only.
- **Lets you conclude:** with every order, which order took liquidity in each
  trade, so orders can be labelled `market` or `market-limit`.
- **Stops you concluding:** with resting orders only, anything about takers. It
  shows up in the order types: no order is labelled `market` or
  `market-limit` from the feed alone.
- **In the package:** `Source.trade_attribution` (below).

### Update form

How the feed reports the book changing.

- **Values:** every change, one message each; changes merged, where one update
  can hold several; repeated snapshots of the book; polls.
- **Sources:** native Bitstamp, LOBSTER, Databento and cryptofeed's Bitfinex,
  Blockchain.com and Independent Reserve books report every change. ccxt's
  websocket venues merge changes. cryptofeed's Bitstamp L3 book is a snapshot
  about 10 times a second. Kalshi is polled once a second.
- **Lets you conclude:** with every change, short-lived orders, cancel rates and
  every flicker of the spread.
- **Stops you concluding:** with merged changes, snapshots or polls, anything
  that starts and ends between two updates. It shows up as too few changes: a
  cancel rate from such a feed is too low, and short lifetimes are missing.
- **In the package:** not declared. The how-to page for each source describes
  it.

**Why ccxt merges changes.** ccxt applies every message the venue sends to its
own copy of the book. The capture reads that copy when it asks for the next
update, so all messages that arrived since the last read reach the capture as
one update.

### What can be missed

What a correct capture can still leave out.

- **Values:** nothing; changes between snapshots or polls; changes merged into
  one; a lost message that goes unnoticed.
- **Sources:** a file (LOBSTER, Databento) misses nothing. Snapshots and polls
  miss what happens between them. ccxt's Coinbase and Kraken books, and the
  native Bitstamp feed, cannot notice a lost message.
- **Lets you conclude:** where nothing can be missed, a quiet book was quiet.
- **Stops you concluding:** elsewhere, that a quiet book was quiet. A lost
  message shows up only later, as a stale order or level, or not at all.
- **In the package:** the sequence check (below), where the venue sends a
  sequence number.

### After a lost message

What happens to the capture's book when a message is lost.

- **Values:** drifts, where the book stays wrong until the capture ends or the
  level changes again; resyncs, where the source takes a new opening book;
  corrected by the next snapshot or poll.
- **Sources:** native Bitstamp drifts. ccxt's Binance book and cryptofeed's
  per-change L3 books resync. Snapshots and polls correct themselves.
- **Lets you conclude:** with resync or snapshots, a long capture stays right.
- **Stops you concluding:** with drift, that a level or order in the book is
  still there. It shows up as stale orders, or as a price level the venue
  removed long ago.
- **In the package:** `audit`'s `stale_orders` check at L3 and the crossing
  check at L2. `meta.json` counts ccxt's and cryptofeed's `book_resyncs`, and
  `audit` warns about them for a capture of several segments.

**Why snapshots correct themselves and change streams drift.** A snapshot states
the whole book, or the whole window, so it replaces whatever the capture held.
A change stream states only what changed. A lost removal is never sent again,
so the order or level stays. On an L2 stream each change carries the level's
new size, so a level that is wrong is put right the next time it changes, but
a level that should have gone stays until something else touches it.

### Sequence

Whether the venue numbers its messages, and what a skipped number means.

- **Values:** contiguous, where every message adds one and a skip is a lost
  message; only rises, where skips are normal; none.
- **Sources:** cryptofeed's per-change L3 venues only rise; see the note
  below. ccxt's Binance book only rises. Databento only rises within one
  instrument. Most other feeds send no number.
- **Lets you conclude:** with a contiguous number, whether a message was lost.
- **Stops you concluding:** with none, anything about lost messages. `audit`
  then reports `0 row(s) numbered` for the venue sequence.
- **In the package:** `Source.sequence_kind` and the `sequence_gaps` check (see
  [Check data quality](howto/audit.md)).

cryptofeed's numbers on Bitfinex and Blockchain.com count every message on the
connection, trades and heartbeats included. On Independent Reserve, cryptofeed
passes on no row for a change to an order it does not hold. So the book rows
skip numbers although nothing was lost: 60 skips in 5 minutes on Bitfinex, 20
in 13 minutes on Independent Reserve. The cryptofeed source declares these
numbers as only rising, so `audit` checks only that they never go back.
cryptofeed itself checks every message on the connection, and on a lost one it
reconnects and takes a new opening book, which the capture counts as a
`book_resyncs`. On Bitfinex and Blockchain.com the count starts again on the
new connection; the capture records that step back as a `sequence_restarts`,
and `audit` does not count it as out of order.

### Clocks

Which times each row carries.

- **Values:** `both`, venue and receive; `receive_only`; `venue_only`.
- **Sources:** most live feeds carry both. Kalshi books and cryptofeed's
  Bitfinex, Blockchain.com and Kraken books carry receive time only. LOBSTER
  carries venue time only.
- **Lets you conclude:** with both clocks, latency, and whether messages
  arrived out of order.
- **Stops you concluding:** with one clock, either of those.
- **In the package:** the [`Clocks`](api/protocols.md) declaration, and
  `audit`'s `exchange_time_after_receive` and `exchange_time_reordered` checks,
  at L2 and L3.

The schema has a column for each clock, so a row with one clock copies it into
the other column, as LOBSTER does. Where a live venue sends no time with its
book, `exchange_timestamp` holds the receive time. A live capture counts the
books that came without a venue time (`books_without_venue_time` in
`meta.json`). When none came with one, it records `"clocks": "receive_only"`.
With one clock the two columns are equal, so `audit` does not run the clock
checks. It says why in their place.

Independent Reserve and Coinbase send a time with every message except the
opening book. The opening book's rows carry the receive time in both columns,
and the capture keeps both clocks. Those rows are marked as the opening book
(`origin` is `snapshot`), and the clock checks leave them out.

Where cryptofeed fetches the opening book over REST (Binance, Independent
Reserve), it does so while it handles the first live message, and hands the
book over first. The book is stamped with a time taken after that message
arrived. So the capture holds a book that has no changes listed until the next
book arrives. If the next one was received earlier, the held book is placed
1 ms before it and marked as the opening book. The message keeps its own
receipt time, and replay applies the book first.

### Crossing

Whether the book can show a bid at or above an ask.

- **Values:** `matched_book`, never crossed; `diff_feed`, can be crossed and
  still faithful; `price_levels`, an L2 book, which the venue does not publish
  crossed.
- **Sources:** native Bitstamp is a diff feed. The other L3 sources are matched
  books. Every L2 source is `price_levels`.
- **Lets you conclude:** in a matched book or price levels, a crossed book is a
  fault in the data or the capture.
- **Stops you concluding:** in a diff feed, that a crossed book is a fault.
  `audit` reports it as information, not as an error.
- **In the package:** `Source.feed_type` and the `crossed_book` check. See
  [Data quality](data-quality.md) for more, and for `uncross=`, which removes
  a crossing for display.

Three cryptofeed L3 captures failed the crossing check in testing:

- **Bitstamp:** crossed for 0.125% of the time, above the 0.05% allowed.
  Bitstamp's own snapshots can show a bid at the price of a resting ask for a
  moment, as its public order feed does. In the first case, the bid was there
  for 87 ms.
- **Bitfinex:** crossed for 58% of the time. cryptofeed reports every change to
  a Bitfinex order, of price or of size, as a delete and a create with the same
  id, so the capture holds ids that are created more than once. The cause of
  the crossing was not found.
- **Independent Reserve:** crossed for 58% of a 20-minute capture on 1 October
  2026. `audit` reported 321 stale resting orders. 223 of them were still in
  the venue's book a day later: trades from the venue's other markets, at
  prices in another currency, made them look stale (see the note under
  [Trades](#trades)). Without the other 98, the book was never crossed. Of
  those 98:
  - 52 came from the opening book, which cryptofeed fetches from the venue's
    REST interface, and the stream never mentioned them again.
  - 7 had a change, and cryptofeed then ignored their cancel (see the note
    under [Timing and integrity](#timing-and-integrity)).
  - 39 were deleted more than a second after a trade at a price past them.

### Trade sides named

Which orders of a trade the order events can name: the maker, which was
resting, and the taker, which arrived.

- **Values:** both; maker only; neither.
- **Sources:** native Bitstamp names both. cryptofeed's Bitstamp and
  Independent Reserve L3 captures, LOBSTER and Databento name the maker only.
  cryptofeed's Bitfinex and Blockchain.com captures name neither, although the
  cryptofeed source declares maker only for all four venues. So `audit`
  reports every Bitfinex trade as unmatched. L2 names neither.
- **Lets you conclude:** with both, maker–taker links and order types.
- **Stops you concluding:** with maker only, which order took liquidity.
  `audit` then looks only for makers, and the order types have no `market`
  orders.
- **In the package:** `Source.trade_attribution`. The `unmatched_trades` check
  looks only for the orders the feed can name.

**Why ITCH does not name the aggressor.** Nasdaq's ITCH feed, from which
LOBSTER is built, reports each execution against the resting order it hit. It
does not identify the incoming order. This protects traders who split a large
order into small ones: publishing the id would let others link the pieces.
Databento's MBO data follows the same exchange convention. The LOBSTER reader
guesses the taker; see [LOBSTER](howto/lobster.md#takers-are-guessed).

**Why Bitstamp publishes both ids.** Bitstamp's trade messages carry
`buy_order_id` and `sell_order_id`, and its `live_orders` channel reports every
order the venue accepts, takers included. So both orders of a trade can be
found. Independent Reserve's trade messages also carry both ids, `BidGuid` and
`OfferGuid`, and the cryptofeed source writes them to `trades.csv`. But its
book does not show every taker: some takers appear as new limit orders, and
most never do. So the source declares maker only.

### Taker side

Whether a trade's buy or sell comes from the venue or is worked out.

- **Values:** from the venue; classified with Lee–Ready.
- **Sources:** every live source reports it. A `depth_csv` file without a
  `side` column is classified.
- **Lets you conclude:** trade signs and flow toxicity (VPIN, order flow
  imbalance).
- **Stops you concluding:** with a classified side, that each sign is right.
  Lee–Ready is wrong for some trades, and a wrong sign looks like any other.
- **In the package:** the trades `direction` column, and
  [trade signs](api/trade_sign.md) when there is no side.

**ccxt's Coinbase sides are reversed.** Coinbase reports the side of the
**maker**, and ccxt passes it on as the taker's side. In a test capture, 93% of
trades marked `buy` printed at or below the best bid, and 96% of trades marked
`sell` at or above the best ask. Every other source's side named the taker.
Reverse the signs of a Coinbase capture before you use them.

### Fills

Where the capture learns that an order was filled rather than cancelled.

- **Values:** per fill, from the order feed; from execution rows; from the trade
  tape; not linked.
- **Sources:** native Bitstamp reports each fill. LOBSTER has execution rows,
  Databento fill records. cryptofeed's Bitstamp and Independent Reserve L3
  captures take fills from the trade tape. On Bitfinex and Blockchain.com an
  order's size drops, and nothing links that to the trade.
- **Lets you conclude:** where fills are linked, filled against cancelled, and
  order lifetimes.
- **Stops you concluding:** where they are not, whether an order that went away
  was filled or cancelled. Both show up as `deleted`, so filled orders count
  as cancelled.
- **In the package:** the order `changed` and `deleted` events, and the order
  types built from them.

### Trade tape gaps

Whether the trade tape can miss or repeat trades.

- **Values:** none; polled; repeated trades removed; trades from before the
  capture.
- **Sources:** Kalshi is polled, and trades from before the opening book are
  dropped. Polymarket can send a trade twice, and the capture drops the repeat.
  ccxt's Coinbase tape and cryptofeed's Bitfinex tape start with trades from
  before the capture.
- **Lets you conclude:** where there are no gaps, trade counts and volumes.
- **Stops you concluding:** that a trade near the start of a capture happened
  after the book was captured. Check its `exchange_timestamp`.
- **In the package:** `meta.json` counts `duplicate_trades`.

### Price grid

The smallest step between two prices.

- **Values:** fixed; mixed, where the step depends on the price; changes during
  the capture.
- **Sources:** most crypto books use a fixed step. Bitfinex quotes five
  significant figures, so its step changes with the price. Kalshi steps depend
  on the market. Polymarket makes its step finer near 0 and 1, during the
  capture.
- **Lets you conclude:** with the right tick size, every price is stored
  exactly.
- **Stops you concluding:** that a depth ring in basis points holds the same
  number of ticks on every market. On a coarse grid, a narrow ring can hold no
  tick at all.
- **In the package:** `PipelineConfig.tick_size`. A ccxt capture records it in
  `meta.json`, and the loaders refuse prices off the grid. ccxt ignores
  Polymarket's messages that change the step. The capture finds the finer
  step from the prices instead, records the final one, and counts the changes
  in `tick_size_changes`.

### What the book means

How to read a price in this book.

- **Values:** normal; built from Yes and No bids; one book per outcome.
- **Sources:** Kalshi's book is built from Yes and No bids. Polymarket has one
  book per outcome. All others are normal.
- **Lets you conclude:** on a prediction market, the price is the probability of
  the outcome.
- **Stops you concluding:** that depth in basis points means the same at a
  price of 0.04 as at 100,000. At 0.04, a ring 25 basis points wide holds no
  tick, so most rings are empty.
- **In the package:** see [Kalshi](howto/kalshi.md#the-book-is-the-yes-book)
  and [Polymarket](howto/polymarket.md#each-outcome-is-its-own-book).

**Why the Kalshi book is built from Yes and No bids.** A Yes contract and a No
contract together pay $1. So a bid to buy No at `p` is an offer to sell Yes at
`1 - p`. Kalshi publishes only bids on both sides, and ccxt turns the No bids
into Yes asks.

### Access

What you need to capture the feed.

- **Values:** public; an API key; a supported location; a paid subscription.
- **Sources:** most crypto venues are public. Binance refuses some locations.
  LOBSTER and Databento are paid. Kalshi's websocket and Coinbase's per-order
  feed need a key.
- **Lets you conclude:** whether you can capture it at all.
- **Stops you concluding:** that one venue id means one market everywhere.
  From a refused location, `binanceus` is a separate venue with its own book.
  A feed that needs a key stops at the start with an authentication error.
- **In the package:** ob-analytics does not use API keys yet.

## What the book shows

| Source | Level | Depth shown | Orders shown |
|---|---|---|---|
| `bitstamp` (native) | L3 | whole book | every order, takers included |
| cryptofeed, Bitstamp | L3 | top 100 orders a side | resting only |
| cryptofeed, Bitfinex | L3 | top 100 orders a side | resting only |
| cryptofeed, Blockchain.com | L3 | not checked: the BTC-USD book held between 1 and 34 orders | resting only |
| cryptofeed, Independent Reserve | L3 | whole book | resting only |
| `lobster` | L3 | whole displayed book | resting only |
| `databento` | L3 | whole displayed book | resting only |
| cryptofeed, Bitstamp L2 | L2 | whole book | — |
| cryptofeed, Kraken L2 | L2 | top 1,000 levels a side; levels that leave it are kept | — |
| cryptofeed, Coinbase L2 | L2 | not checked: cryptofeed 2.4.1 needs an API key for Coinbase | — |
| cryptofeed, other L2 venues | L2 | not checked | — |
| ccxt, Binance | L2 | 1,000 levels a side by default, up to 5,000 with `--depth-limit` (about 0.2–0.3% and 1.0–1.4% from the price on BTC/USDT) | — |
| ccxt, Coinbase | L2 | whole book | — |
| ccxt, Kraken | L2 | top `--depth-limit` levels (100 by default), kept filled by Kraken | — |
| ccxt, Kalshi | L2 | whole book, each poll | — |
| ccxt, Polymarket | L2 | whole book | — |
| `depth_csv` | L2 | what the file holds | — |

## Timing and integrity

| Source | Update form | What can be missed | After a lost message | Sequence | Clocks | Crossing |
|---|---|---|---|---|---|---|
| `bitstamp` | every change | a lost message | drifts: the order stays until the capture ends ([stale orders](data-quality.md#stale-resting-orders)) | none | venue + receive | diff feed |
| cryptofeed, Bitstamp | snapshots, about 10 a second | orders that come and go between snapshots | the next snapshot corrects it | none | venue + receive | matched book; failed the check in testing |
| cryptofeed, Bitfinex | opening book, then every change | nothing | cryptofeed reconnects and takes a new opening book | counts every message on the connection; only rises | receive only | matched book; failed the check in testing |
| cryptofeed, Blockchain.com | opening book, then every change | nothing | cryptofeed reconnects and takes a new opening book | counts every message on the connection; only rises | receive only | matched book |
| cryptofeed, Independent Reserve | REST book, then every change | any change or cancel after an order's first change, which cryptofeed ignores; the trades restore the fills | cryptofeed reconnects and takes a new opening book | skips the changes cryptofeed ignores; only rises | venue + receive; the opening book receive only | matched book; failed the check in testing (58% crossed) |
| `lobster` | every change | nothing (a file) | — | none | venue only | matched book |
| `databento` | every change | nothing (a file) | — | venue's where sent; only rises | venue + Databento receive | matched book |
| cryptofeed, Bitstamp L2 | REST book after 5 s, then changes | a lost message | drifts until the level changes again | none | venue + receive | price levels |
| cryptofeed, Kraken L2 | changes | a lost message | drifts until the level changes again | none | receive only | price levels |
| cryptofeed, Coinbase L2 | not checked: needs an API key | not checked | not checked | not checked | not checked | price levels |
| cryptofeed, other L2 venues | not checked | not checked | not checked | not checked | not checked | price levels |
| ccxt, Binance | changes, merged | merged changes | ccxt takes a new opening book, up to 10 times | only rises | venue + receive | price levels |
| ccxt, Coinbase | changes, merged | merged changes; a lost message | drifts until the level changes again | none | venue + receive | price levels |
| ccxt, Kraken | changes, merged | merged changes; a lost message | drifts until the level changes again | none | venue + receive | price levels |
| ccxt, Kalshi | polls, 1 a second | anything shorter than a poll | the next poll corrects it | none | receive only | price levels |
| ccxt, Polymarket | changes, merged, and a whole-book snapshot every second or two while the market trades | merged changes | the next snapshot corrects it | none | venue + receive | price levels |
| `depth_csv` | as recorded | as recorded | — | if present | as recorded | price levels |

Notes on this table:

- **Kraken:** Kraken sends a checksum of its book, but neither ccxt nor
  cryptofeed checks it by default, so a lost message is not noticed.
- **Coinbase:** Coinbase numbers its messages, but ccxt does not pass the number
  on.
- **Polymarket:** each snapshot carries a hash of the book, which ccxt does not
  check.
- **cryptofeed, Independent Reserve:** after an order's first change,
  cryptofeed 2.4.1 forgets the order, so it ignores any later change or cancel.
  In 10 minutes of the venue's own messages on 1 October 2026, 14 orders
  changed, and 10 of them had a later message that cryptofeed ignored: 9
  cancels and 3 changes. The source takes the fills from the trades, so it does
  not lose them. A cancelled order stays in the book until the capture ends.
- **cryptofeed, Bitstamp L2:** cryptofeed waits 5 seconds, then fetches the REST
  book. It drops changes stamped in any second before the REST book's second,
  and applies those from the same second, even ones older than the book.
- **Kalshi:** a poll that times out ends the capture segment, and the capture
  starts a new one from a new opening book.

## Trades

| Source | Maker named | Taker named | Taker side | Fills | Tape gaps |
|---|---|---|---|---|---|
| `bitstamp` | yes | yes | venue | per fill | none |
| cryptofeed, Bitstamp | yes, from the tape | never | venue | from the tape | a few trades from the 5 s before the first snapshot |
| cryptofeed, Bitfinex | never (the venue's trades carry no order ids) | never | venue | not linked | starts with the last 30 trades before the capture |
| cryptofeed, Blockchain.com | never (the venue's trades carry no order ids) | never | venue (not checked: 1 trade in 20 minutes) | not linked | not checked (1 trade in 20 minutes) |
| cryptofeed, Independent Reserve | yes, from the tape | only when the book shows it (6 of 26 takers) | venue | from the tape | none found |
| `lobster` | yes | guessed | from the execution row | execution rows | — |
| `databento` | yes | never | venue | fill records | — |
| cryptofeed, Bitstamp L2 | — | — | venue | — | a few trades from the 5 s before the REST book |
| cryptofeed, Kraken L2 | — | — | venue | — | none |
| cryptofeed, Coinbase L2 | — | — | not checked | — | not checked |
| cryptofeed, other L2 venues | — | — | not checked | — | not checked |
| ccxt, Binance | — | — | venue (the `m` flag) | — | none |
| ccxt, Coinbase | — | — | **reversed**: Coinbase names the maker's side; reverse the signs before use | — | starts with trades from before the capture |
| ccxt, Kraken | — | — | venue | — | none |
| ccxt, Kalshi | — | — | venue (buy = the taker bought Yes) | — | polled; trades from before the opening book dropped |
| ccxt, Polymarket | — | — | venue | — | repeated trades removed |
| `depth_csv` | — | — | the `side` column, or Lee–Ready | — | as recorded |

Note on this table:

- **cryptofeed, Independent Reserve:** the trade channel for one market also
  carries trades from the venue's other markets for the same coin, at prices in
  their own currency. In 40 minutes of BTC-AUD on 1 October 2026, 3 of 26
  trades came from BTC-NZD or BTC-SGD, at about 148,700 and 107,000, while
  BTC-AUD traded near 120,000. The venue keeps one book for all its currencies,
  so these trades name orders in the capture, and their fills are right. Their
  prices are not: `audit` reads them as trades through the book and reports
  orders that are still resting as stale.

## Venue rules

| Source | Price grid | What the book means | Access |
|---|---|---|---|
| `bitstamp` | fixed (0.01 on BTC/USD) | normal | public |
| cryptofeed, Bitstamp | fixed (0.01 on BTC/USD) | normal | public |
| cryptofeed, Bitfinex | five significant figures: 1 on BTC/USD at 83,000 | normal | public |
| cryptofeed, Blockchain.com | fixed (0.01 on BTC/USD) | normal | public |
| cryptofeed, Independent Reserve | fixed (0.01 on BTC/AUD); some trade prices are not on it | normal | public |
| cryptofeed, Kraken L2 | fixed (0.1 on BTC/USD) | normal | public |
| cryptofeed, Bitstamp L2 | fixed (0.01 on BTC/USD) | normal | public |
| cryptofeed, Coinbase L2 | not checked | normal | API key, in cryptofeed 2.4.1 |
| cryptofeed, other L2 venues | not checked | normal | not checked |
| `lobster` | from the file | normal | paid files |
| `databento` | from the file | normal | paid, API key |
| ccxt, Binance | fixed (0.01 on BTC/USDT) | normal | refused in some locations: use `binanceus` or `--market-data-mirror` |
| ccxt, Coinbase | fixed (0.01 on BTC/USD) | normal | `coinbase` is public; `coinbaseexchange` and the per-order feed need a key |
| ccxt, Kraken | fixed (0.1 on BTC/USD) | normal | public |
| ccxt, Kalshi | by market: cents; tenths of a cent near 0 and 1; tenths or hundredths of a cent throughout | Yes book, from Yes and No bids; price = probability of Yes | REST public; websocket needs a key |
| ccxt, Polymarket | 0.01 or 0.001, finer near 0 and 1; can change during the capture, and the capture finds the finer step from the prices | one book per outcome | public |
| `depth_csv` | you supply it | as recorded | — |

## Bitstamp at L3: which source

Two sources capture Bitstamp orders, and each has a limit the other does not:

- **Native `bitstamp`** shows every order, takers included, and every fill. But
  it is a stream of changes with no sequence number, so a lost message leaves an
  order in the book until the capture ends.
- **cryptofeed's Bitstamp book** shows only the top 100 orders a side and no
  takers. But each snapshot states that window in full, so an error there is
  corrected within a tenth of a second.

Use native `bitstamp` for analysis. Use a cryptofeed capture of the same period
as an independent check of the top of the native book.

## How the values were checked

On 28 September 2026 we made captures of about 5 minutes on every live source
above, and of 13 to 20 minutes on the quieter venues (Kalshi, Polymarket,
Blockchain.com and Independent Reserve). We ran `audit` on each, and read the
ccxt 4.5.84 and cryptofeed 2.4.1 code for each venue. The table says how each
source's values were found. A value not listed for a source came from reading
the code.

| Source | Measured in a capture | Found another way |
|---|---|---|
| `bitstamp` | crossing, clocks, tape gaps | this package's code, tests and the bundled sample |
| cryptofeed, Bitstamp | crossing, clocks, tape gaps, price grid | the window and fills: this package's code and tests |
| cryptofeed, Bitfinex | depth, repeated ids, crossing, sequence skips, clocks, taker side, tape gaps, price grid | why ids repeat: cryptofeed's code |
| cryptofeed, Blockchain.com | number of orders and trades | — |
| cryptofeed, Independent Reserve | opening-book size, sequence skips, clocks, taker side, price grid; on 1 October 2026, crossing and trades linked to their maker | the ignored changes and cancels, and which takers the book shows: a recording of Independent Reserve's websocket messages |
| `lobster`, `databento` | — | this package's code, tests and docs |
| cryptofeed, Bitstamp L2 | depth, taker side, clocks, tape gaps, price grid, crossing | — |
| cryptofeed, Kraken L2 | depth, taker side, clocks, tape gaps, price grid, crossing | — |
| cryptofeed, Coinbase L2 | — | the capture stopped with HTTP 401: it needs a key |
| ccxt, Binance | depth, taker side, sequence, crossing | reach from the price and the check against Binance's REST book: the [Binance](howto/binance.md) page's own test |
| ccxt, Coinbase | depth, taker side, tape gaps, sequence, crossing | — |
| ccxt, Kraken | depth, taker side, tape gaps, price grid, crossing | — |
| ccxt, Kalshi | depth, compared with Kalshi's REST book; clocks, taker side, timeouts, crossing | price grids: Kalshi's market list |
| ccxt, Polymarket | depth, compared with Polymarket's REST book; taker side, repeated trades, crossing | snapshot frequency and hash: a direct listen to Polymarket's websocket |

The measurements:

- **Taker side:** each trade's price against the best bid and ask just before
  it, on the venue's clock. A taker who buys pays the ask. Every trade agreed
  with its side on Binance (2,372 trades), Kraken through ccxt (265), Kalshi
  (71) and Polymarket (36). Through cryptofeed, 96% to 100% agreed on Bitstamp
  L2 and Kraken L2, and 92% to 96% on Bitfinex. On Independent Reserve, 18 of
  22 trades agreed. On Coinbase, 93% to 96% disagreed.
- **Depth shown:** the most price levels or orders the capture held at once.
- **Crossing:** the share of time `audit` found the book crossed. No L2 capture
  was crossed for any measurable time; see
  [Data quality](data-quality.md#price-level-l2-feeds).
- **Sequence:** skipped numbers counted in `orders.csv`.
- **Trades linked:** trades that `audit` matched to their maker in
  `orders.csv`. On Independent Reserve, all 13 trades of a 20-minute capture on
  1 October 2026 were matched.
- **Takers the book shows:** in 40 minutes of Independent Reserve's own
  messages on 1 October 2026, 6 of 26 takers appeared in the book, each as a
  new limit order. The other 20 never did.

Blockchain.com's BTC-USD market held at most 34 orders and traded once in 20
minutes, so its depth and trade values are not checked. cryptofeed's other L2
venues were not captured.

## See also

- [Data quality: matched book, diff feed and price levels](data-quality.md)
- [Check data quality](howto/audit.md)
- [Process L2 (price-level) feeds](howto/l2-depth.md)
