---
title: Process LOBSTER files
---

# Process LOBSTER files

[LOBSTER](https://lobsterdata.com/) message and orderbook files are
supported out of the box via `LobsterLoader`, `LobsterTradeReader`,
`LobsterWriter`, and `LobsterSource`. Depth is read from the official
orderbook file (ground-truth) when present.

!!! info "What this feed shows"

    | Property | Value |
    |---|---|
    | Level | L3 |
    | Depth shown | the requested price levels |
    | Orders shown | resting only |
    | Update form | every change |
    | What can be missed | events outside the requested levels |
    | After a lost message | — |
    | Sequence | none |
    | Clocks | venue only |
    | Crossing | matched book; the per-order book crosses when the levels are too few |
    | Trade sides named | maker only; the taker is guessed |
    | Taker side | from the execution row |
    | Fills | execution rows |
    | Trade tape gaps | — |
    | Price grid | from the file |
    | What the book means | normal |
    | Access | paid files |

    Use it for depth on US equities, and for order lifetimes and queue
    position when the file has enough levels (see
    [Request enough levels](#request-enough-levels)). Don't use it for
    anything that depends on the taker (market-order labels, trade impact per
    taker) or on latency: the taker is guessed and there is one clock.
    [What each feed shows](../feeds.md) explains each property and compares
    every source.

<!-- docs-test: skip needs LOBSTER files -->

```python
from ob_analytics import LobsterSource, Pipeline
from ob_analytics.protocols import RunContext

source = LobsterSource()
ctx = RunContext(trading_date="2012-06-21")
result = Pipeline(source=source, ctx=ctx).run(
    "/path/to/extracted_lobster_folder"
)

# equivalent shorthand via the source registry:
result = Pipeline.from_source(
    "lobster", ctx=RunContext(trading_date="2012-06-21"),
).run("/path/to/extracted_lobster_folder")
```

A context given to `Pipeline` is the default for every run. A context given to
`run` is for that run alone, so one pipeline reads several days:

<!-- docs-test: skip needs LOBSTER files -->

```python
pipeline = Pipeline(source=LobsterSource())
monday = pipeline.run("/data/AAPL_monday", ctx=RunContext(trading_date="2012-06-18"))
tuesday = pipeline.run("/data/AAPL_tuesday", ctx=RunContext(trading_date="2012-06-19"))
```

The source keeps nothing from a run, so one `LobsterSource()` can also serve
several pipelines at once.

## Per-source extras

Some sources expose auxiliary event tables that don't fit the universal
events schema — LOBSTER trading halts, cross trades, and hidden
executions, for example. These no longer ride on `PipelineResult`; a
`LobsterLoader` splits them out during `load()` and exposes them as a public
attribute (`None` when absent). Build the loader yourself and give it to the
pipeline, and read the tables off it after the run:

<!-- docs-test: skip needs LOBSTER files -->

```python
from ob_analytics import Pipeline
from ob_analytics.lobster import LobsterLoader, LobsterSource
from ob_analytics.visualization.gallery import (
    build_gallery_model,
    display_result,
    generate_gallery,
    trading_halts_card,
)

loader = LobsterLoader(trading_date="2015-05-01")
result = Pipeline(source=LobsterSource(), loader=loader).run("/data/AAPL_2015-05-01")
halts = loader.trading_halts  # pd.DataFrame | None

if halts is not None:
    model = build_gallery_model(result)
    # The halts card draws trade price, so give it display-unit trades
    # (quote-currency floats), like the rest of the gallery.
    model.analytics.append(trading_halts_card(display_result(result).trades, halts))
    generate_gallery(result, "out/gallery", model=model)
```

The loader keeps two such tables, each `None` when the file has no rows of
its type:

- `loader.trading_halts`: the trading-halt messages (event type 7).
- `loader.cross_trades`: the cross-trade messages (event type 6), which
  LOBSTER uses for auction trades such as the opening and closing cross.
  They are not in the events frame, so the trades frame leaves them out too.

A `LobsterLoader` built by hand reads LOBSTER's own grid, a cent tick and whole
shares, and knows that a price in the file is in ten-thousandths of a dollar.
So it reads the same prices and sizes as a pipeline run. Give it a
`PipelineConfig` only for another grid.

Bitstamp runs have no such tables (`loader.trading_halts is None`). Hidden
executions are detected automatically by the gallery builder
(`build_gallery_model`) when the events frame contains LOBSTER
hidden-execution rows.

!!! note
    The orderbook file has one row for each message row, cross trades (event
    type 6) and trading halts (event type 7) included. The events keep only
    types 1 to 5, so each event reads the orderbook row of its own message,
    named by its `original_number`. An orderbook file with more or fewer
    rows than the message file raises `ConfigError`. Sizes in the orderbook
    file are converted to whole lots of `lot_size`, as the message sizes are, so
    `depth` and `depth_summary` hold `int64` lots.

## Request enough levels

A LOBSTER file covers the number of price levels you ask LOBSTER for. The
message file holds only the events inside those levels. An order that moves
outside them, because the touch moved away, and is then cancelled or filled
there gets no closing row. The book rebuilt from the messages keeps that order
to the end of the file.

When the folder has an orderbook file, the depth table is read from it, so it
is right at every level the file covers, and so are the L2 book snapshot and
depth chart, which read the depth table. The per-order views are not: the L3
book snapshot and depth chart, queue positions, order lifecycles and order
outcomes. They are complete only when the requested levels cover every price
an order reaches. Without an orderbook file, the depth table is rebuilt from
the same messages and keeps the same stale orders.

The AAPL sample of 21 June 2012, at three level counts. The touch of the book
rebuilt from the messages was compared with the orderbook file at 58 instants:

| Requested levels | Touch matches the file | Book crossed or locked |
|---|---|---|
| 50 | 58 of 58 | 0 of 58 |
| 30 | 28 of 58 | 27 of 58 |
| 10 | 0 of 58 | 58 of 58 |

When the book rebuilt from the messages is crossed for more than 0.05% of the
session time, the run warns (a `UserWarning`) and gives the share. On the
10-level sample it is crossed 99.2% of the time; on the 50-level sample, never.
Request more levels if the warning appears and you need the per-order views.

[`audit`](audit.md) reports the same share on its **per-order book crossed**
line. When the folder has an orderbook file, the depth table is not crossed,
so the `crossed_book` check passes and the `per_order_book_crossed` check
warns. With `--strict`, that warning fails the run.

`audit` also compares the book rebuilt from the messages with the orderbook
file, row by row, on every price and size of the price levels the file
states. Its **venue book** line gives the number of rows that differ and the
first difference, and the `venue_book` check warns when any row differs. Each event
is matched with the orderbook row of its own message, so the rows of cross
trades and halts do not shift the comparison.

On the AAPL samples, the two books differ in almost every row, at every
number of requested levels:

| Requested levels | Rows compared | Rows that differ | Touch differs |
|---|---|---|---|
| 50 | 91,997 | 91,821 (99.81%) | 0.49% |
| 30 | 86,616 | 86,608 (99.99%) | 55.51% |
| 10 | 400,391 | 400,391 (100.00%) | 99.18% |

The 50- and 30-level samples cover the first hour of the day and the 10-level
sample the whole day. The rebuilt book is right only for the orders the
message file covers. Three kinds of order are not covered:

- **Orders resting before the file starts.** The first orderbook row already
  holds the opening book, but the first message adds one bid, so the rebuilt
  book starts with that bid only. Those orders appear in the messages only
  when they are changed or deleted (the orphan orders `audit` counts).
- **Orders that arrive outside the requested levels.** They have no message
  when they arrive, so when the touch moves to their prices, the rebuilt book
  has less size there, or does not have the level.
- **Orders cancelled or filled outside the requested levels.** They have no
  closing message, so the rebuilt book keeps them. On the 10-level sample,
  these cause most of the differences.

On the 50-level sample, the differences are mostly in the deep price levels: 0.49%
of rows differ at the touch and 9.6% within the first 10 levels. The depth
table and the L2 views read the orderbook file, so they are not affected.

## Takers are guessed

A LOBSTER execution row names the resting order it hit. The order that hit it
is not in the file: LOBSTER is built from Nasdaq ITCH, which does not identify
the aggressor, and an order that trades on arrival never rests. The trade
reader fills the `taker` columns with a guess: the most recent new order on the
other side, before the execution, at a price that could have traded. The guess
changes only the `taker` columns, so book volumes are exact, but
`set_order_types` labels the guessed order `market` or `market-limit`, and a
wrong guess mislabels an order.

The source declares `trade_attribution = maker_only`, so
[`audit`](audit.md) counts a trade as unmatched only when its maker is missing
and does not count a guessed taker as a match.

## LOBSTER round-trip output

To write results back to LOBSTER message + orderbook CSVs, see
[Save, load, and export](output.md#serialisation). The orderbook file is
written from the run's own depth table, so a run from any source can be
written. Reading the files back gives the run's book after every event: the
depth summary read back equals the run's, as far as the written levels reach.

## Related

- [LOBSTER API](../api/lobster.md) — loader, trade reader, writer
- [Glossary: LOBSTER](../glossary.md#data-formats) — format details
