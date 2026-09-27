---
title: Process LOBSTER files
---

# Process LOBSTER files

[LOBSTER](https://lobsterdata.com/) message and orderbook files are
supported out of the box via `LobsterLoader`, `LobsterTradeReader`,
`LobsterWriter`, and `LobsterSource`. Depth is read from the official
orderbook file (ground-truth) when present.

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
    When message files contain cross trades (event type 6) or trading halts
    (event type 7), filtered rows may not align one-to-one with orderbook
    rows; the implementation logs a warning and uses the minimum consistent
    length.

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
[Save, load, and export](output.md#serialisation).

## Related

- [LOBSTER API](../api/lobster.md) — loader, trade reader, writer
- [Glossary: LOBSTER](../glossary.md#data-formats) — format details
