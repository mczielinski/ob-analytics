---
title: Process Databento MBO files
---

# Process Databento MBO files

[Databento](https://databento.com/) publishes normalized market data for many
venues in its own binary format, DBN. Its **market-by-order (MBO)** schema is a
per-order feed: every record carries an `order_id`, a `price`, a `size` and a
`side`, which is what ob-analytics reconstructs from. A DBN MBO file therefore
replays through the standard pipeline with order lifetimes, queue position and
order classification all available.

`databento` is an optional dependency:

```bash
pip install "ob-analytics[databento]"
```

Databento data needs your own API key and subscription, so none of it ships
with the package.

## Run a file

```python
from ob_analytics import Pipeline
from ob_analytics.databento import DatabentoSource

result = Pipeline(source=DatabentoSource()).run("aapl-20240212.mbo.dbn.zst")

print(result.events.shape, result.trades.shape)
```

Or by name, through the source registry:

```python
from ob_analytics import Pipeline

result = Pipeline.from_source("databento").run("aapl-20240212.mbo.dbn.zst")
```

From the command line:

```bash
ob-analytics process aapl-20240212.mbo.dbn.zst --source databento --output out/
```

There is no trading date to supply: DBN records carry absolute UTC timestamps.

## Set the instrument's tick and lot

The defaults suit a US equity — a one-cent tick and whole shares. Another
instrument needs its own, or prices are quantised onto the wrong grid:

```python
from ob_analytics import Pipeline, PipelineConfig
from ob_analytics.databento import DatabentoSource

# E-mini S&P 500 future: a quarter-point tick.
config = PipelineConfig(tick_size=0.25)
result = Pipeline(config, source=DatabentoSource()).run("es-20240212.mbo.dbn.zst")
```

The fixed-point scale of the raw prices is a property of DBN, not of the
instrument, and is already set for you (`price_divisor=1_000_000_000`).

## Pick one instrument out of a file

A DBN file can hold several instruments, and a consolidated dataset such as
`DBEQ.BASIC` can hold several venues quoting the same one. Each
(instrument, publisher) pair is a separate book, so a run covers exactly one.
Name it, or the loader raises and lists what the file holds:

```python
from ob_analytics.databento import DatabentoSettings, DatabentoSource

source = DatabentoSource(settings=DatabentoSettings(raw_symbol="GOOGL"))

# by numeric id instead, for a file with no symbol mapping:
source = DatabentoSource(
    settings=DatabentoSettings(instrument_id=1108, publisher_id=41)
)
```

## How the records map

Databento's `action` field says what a record does. Only the first four change
the book.

| `action` | meaning | canonical event |
|---|---|---|
| `A` add | insert a new order | `created`, `volume` = the new size |
| `M` modify | change price and/or size | `changed`, `volume` = the new size |
| `C` cancel | remove some or all of an order | `changed` while size is left, `deleted` when it empties |
| `R` clear | remove every resting order | a `deleted` row for each order still resting |
| `T` trade | an aggressing order traded | a trade row (see below) |
| `F` fill | a resting order was filled | the `fill` on the next book event, and a trade row |
| `N` none | flags only | nothing |

`size` means a different thing per action, which is what makes the mapping
work. On `A` and `M` it is the order's new total size, which is the canonical
`volume` directly. On `C` it is the amount removed, so the loader carries the
outstanding size per order and subtracts from it. On `F` it is the amount
executed.

A venue reports an execution as a fill and then a separate cancel or modify
that takes the size off the book. The loader charges each `F` to the next
`A`/`M`/`C` record for the same order, which is what tells a cancel that was
really an execution apart from a cancel the trader asked for. A modify for an
order this window never saw added is treated as an add, the same way
Databento's own reference book builder treats it.

## Trades and the aggressor

Databento sends two records for a trade: a print (`T`) for the execution and
a fill (`F`) for each resting order it hit. `DatabentoSettings(trades_from=...)`
chooses which one the trades frame is built from.

- **`"fills"`, the default.** Each fill becomes a trade row with a `maker` and
  a `maker_event_id` — one row per resting order a sweep took out. That is what
  order classification and queue analysis read. A file with no fills at all
  falls back to its prints.
- **`"prints"`.** The venue's whole tape, one row per print, but no maker on
  any row.

The difference matters for trades the publisher sent no fill for: an opening
or closing auction, a trade against a non-displayed order, an off-exchange
print. With fills they are not in the frame, and the loader warns with the
volume left out. On a US equity day the auctions alone can be a large share of
the volume, so for a question about the tape — VWAP, bars, flow toxicity,
transaction costs — use prints:

```python
from ob_analytics.databento import DatabentoSettings, DatabentoSource

source = DatabentoSource(settings=DatabentoSettings(trades_from="prints"))
```

The two are not mixed. A print and the fills behind it describe one execution,
and nothing in a DBN record ties them together reliably — a fill and the modify
it causes can carry different receive times — so joining them would risk
counting the same volume twice.

Where the venue states the aggressor, `direction` is its answer rather than an
estimate.

Databento states no side at all for some trades — an opening or closing
auction, a trade against a non-displayed order, an implied order, an
off-exchange print. Those reach the pipeline unlabelled and are then classified
with Lee-Ready against the reconstructed quotes, so they end up with a
direction like any other trade. A side the venue did state is never overwritten:
a classifier is an estimate and the venue's answer is not.

The taker's own order is **not** identified: a DBN trade record does not
reliably carry the aggressing order's id, so `taker` and `taker_event_id` are
NA. `set_order_types` reads those, so on a Databento run it labels executed
resting orders `resting-limit` and never `market` or `market-limit`. Everything
that reads the maker side — order lifetimes, queue position, depth, effective
spread — is unaffected.

## One thing the depth reconstruction cannot represent

Databento's `M` can move an order to another price or make it bigger. Both lose
queue priority, so both are really a new queue entry, and the shared schema has
no event for that: it has `created`, `changed` and `deleted`, and the
price-level rebuild counts every one of an order's later rows on the price its
`created` row carried, so that an order's volume can only cancel on the level it
was added to.

The loader records the new price and size on a `changed` event, so the events,
order lifetimes and trades are right. The depth is not: after a move the volume
stays counted on the old level and is missing from the new one, and after a
growth the added size is not counted anywhere. The loader warns with the number
of rows affected, so the size of the error is visible rather than silent. Most
feeds never hit this, because most venues report an amendment as a cancel and a
new order.

## What the loader refuses, and what it drops

A feed the loader does not understand is **refused**:

- A publisher that only sends top-of-book or price-level data. Databento
  normalizes that into MBO records with `F_TOB` or `F_MBP` set and an
  `order_id` that means nothing, so reconstructing per-order state would invent
  identity the feed never had. Read those publishers through the
  [L2 path](l2-depth.md) instead.
- A file holding a price-level schema (`mbp-1`, `mbp-10`, `bbo`, `tbbo`).
- A file covering more than one instrument or publisher with no filter saying
  which book to read.
- An `action` outside DBN's own alphabet (`A M C R T F N`). A record whose
  meaning is unknown has no safe reading, and dropping it would take its
  liquidity out of the book unannounced.
- An `order_id` above 2⁶³−1. DBN order ids are unsigned 64-bit and the shared
  schema's is signed, so a cast would wrap to a negative id and could merge two
  distinct orders into one.

A malformed record inside a feed it does understand is **dropped and counted**
in a warning:

- A book action, trade or fill with no price (`UNDEF_PRICE`, which is
  `INT64_MAX`). Kept, it would sit at about nine billion: read back as the
  best bid or ask, or as a trade at that price. A clear is exempt, because its
  price is never read.
- A book action with no side. There is no side of the book to put it on, and
  keeping it would leave the events frame and the depth frame disagreeing.

Refusing a whole session over a handful of malformed records would be worse
than saying how many went, which is why these two are a count rather than an
error.

The loader also warns when records carry `F_MAYBE_BAD_BOOK`, which means the
feed reported a gap it could not recover from.

## Size a query before you download it

MBO volumes are large. One instrument over a session-length window sits inside
the in-memory envelope; a whole feed for a whole day does not. Databento bills
by record, so it will tell you the size and the cost of a query before anything
is downloaded:

```python
import databento as db

client = db.Historical()  # reads DATABENTO_API_KEY from the environment

n = client.metadata.get_record_count(
    dataset="XNAS.ITCH",
    symbols=["AAPL"],
    schema="mbo",
    start="2024-02-12T14:30",
    end="2024-02-12T16:00",
)
print(n)  # compare against the ~5M-event envelope
```

`scripts/databento_window.py` is this as a runnable tool: it sizes each window,
downloads the ones you ask for, runs them one at a time and concatenates the
results.

```bash
export DATABENTO_API_KEY=db-...
uv run --extra databento python scripts/databento_window.py \
    --dataset XNAS.ITCH --symbol AAPL \
    --window 2024-02-12T14:30 2024-02-12T16:00
```

Windows are run separately so peak memory is bounded by the largest one.
[Scale and chunking](../scale-and-chunking.md) sets out what concatenates
cleanly across a cut and what does not.

## Write a file back out

`DatabentoWriter` inverts the loader record for record, so a window you have
worked on can be handed to another DBN reader:

```python
from ob_analytics.databento import DatabentoWriter

DatabentoWriter(result.config).write({"events": result.events}, "out/window.dbn")
```

The metadata says `OB.ANALYTICS` rather than claiming to be Databento's own
data, and the fills are written back at the timestamp of the event they were
charged to, so the file is a faithful reconstruction rather than a byte copy of
the original.
