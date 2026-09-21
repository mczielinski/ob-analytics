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

Which record makes a trade depends on what the publisher sends.

- **Fills, when the file has them.** A fill names the resting order, so each
  one becomes a trade row with a `maker` and a `maker_event_id` — one row per
  resting order a sweep took out.
- **Trade prints otherwise.** Some publishers report no passive side; then the
  print is all there is, and the row carries the volume and the aggressor's
  side but no maker.

Either way the aggressor's side comes from the venue rather than a classifier,
so `direction` is the real taker side and not an estimate.

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

## What the loader refuses

Some Databento publishers only send top-of-book or price-level data, and
Databento normalizes that into MBO records with `F_TOB` or `F_MBP` set and an
`order_id` that means nothing. The loader raises rather than reconstruct
per-order state that the feed never had; read those publishers through the
[L2 path](l2-depth.md) instead.

It also raises on a file holding a price-level schema (`mbp-1`, `mbp-10`,
`bbo`, `tbbo`), and warns when records carry `F_MAYBE_BAD_BOOK`, which means
the feed reported a gap it could not recover from.

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
