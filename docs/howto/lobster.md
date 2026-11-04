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

## Per-source extras

Some sources expose auxiliary event tables that don't fit the universal
events schema — LOBSTER trading halts, cross trades, and hidden
executions, for example. These no longer ride on `PipelineResult`; a
`LobsterLoader` splits them out during `load()` and exposes them as a public
attribute (`None` when absent):

```python
from ob_analytics import Pipeline, RunContext
from ob_analytics.lobster import LobsterLoader, LobsterSource
from ob_analytics.visualization.gallery import (
    build_gallery_model,
    display_result,
    generate_gallery,
    trading_halts_panel,
)

ctx = RunContext(trading_date="2015-05-01")
result = Pipeline(source=LobsterSource()).run(path, ctx=ctx)

loader = LobsterLoader(trading_date="2015-05-01")
loader.load(path)             # populates loader.trading_halts
halts = loader.trading_halts  # pd.DataFrame | None

if halts is not None:
    model = build_gallery_model(result)
    # The halts panel draws trade price, so give it display-unit trades
    # (quote-currency floats), like the rest of the gallery.
    model.analytics.append(trading_halts_panel(display_result(result).trades, halts))
    generate_gallery(result, "out/gallery", model=model)
```

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
