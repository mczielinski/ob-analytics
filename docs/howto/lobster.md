---
title: Process LOBSTER files
---

# Process LOBSTER files

[LOBSTER](https://lobsterdata.com/) message and orderbook files are
supported out of the box via `LobsterLoader`, `LobsterTradeReader`,
`LobsterWriter`, and `LobsterFormat`. Depth is read from the official
orderbook file (ground-truth) when present.

```python
from ob_analytics import LobsterFormat, Pipeline
from ob_analytics.protocols import RunContext

fmt = LobsterFormat()
ctx = RunContext(trading_date="2012-06-21")
result = Pipeline(format=fmt, ctx=ctx).run(
    "/path/to/extracted_lobster_folder"
)

# equivalent shorthand via the format registry:
result = Pipeline.from_format(
    "lobster", ctx=RunContext(trading_date="2012-06-21"),
).run("/path/to/extracted_lobster_folder")
```

## Per-format extras

Some formats expose auxiliary event tables that don't fit the universal
events schema — LOBSTER trading halts, cross trades, and hidden
executions, for example. These no longer ride on `PipelineResult`; a
`LobsterLoader` splits them out during `load()` and exposes them as a public
attribute (`None` when absent):

```python
from ob_analytics import Pipeline, RunContext
from ob_analytics.lobster import LobsterFormat, LobsterLoader
from ob_analytics.visualization.gallery import generate_gallery, trading_halts_panel

ctx = RunContext(trading_date="2015-05-01")
result = Pipeline(format=LobsterFormat()).run(path, ctx=ctx)

loader = LobsterLoader(trading_date="2015-05-01")
loader.load(path)             # populates loader.trading_halts
halts = loader.trading_halts  # pd.DataFrame | None

if halts is not None:
    generate_gallery(
        result, "out/gallery",
        extra_panels=[trading_halts_panel(result.trades, halts)],
    )
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

## LOBSTER round-trip output

To write results back to LOBSTER message + orderbook CSVs, see
[Save, load, and export](output.md#serialisation).

## Related

- [LOBSTER API](../api/lobster.md) — loader, trade reader, writer
- [Glossary: LOBSTER](../glossary.md#data-formats) — format details
