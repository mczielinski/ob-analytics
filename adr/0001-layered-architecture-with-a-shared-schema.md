---
status: accepted
date: 2026-08-30
---

# Layered architecture around one shared schema

ob-analytics is organised as five layers that talk to each other only through
one shared, versioned schema. The fast, stateful part — the order-book rebuild —
stays in numpy. The work needed to make the library faster, larger, live, and
open to new venues happens at the **boundaries between the layers**: the data
format, the way sources plug in, and the dataframe library. This record states
that target so every issue aims at the same shape. It does not change what ships
next.

## Context

The rebuild engine was already written in numpy, not pandas, when this plan was
made. It is therefore in the right place: a rewrite would spend a large amount of
work to arrive near where the code already is. What was missing was a stable
format between the stages, one way for data sources to attach, and a dataframe
story that does not force every user onto one library.

Fix those three boundaries and each goal — speed, scale, export, live capture,
more venues — becomes a small addition rather than a rewrite.

## The layers

Top to bottom. Each layer reaches the next only through the shared schema
(#112):

```
Public API     plot_result, prepare, Pipeline, CLI      takes pandas or Polars
Plots / export plots by concept and level; export #113; Parquet #112
Analytics      depth, spread, toxicity, micro-price, iceberg     Polars for big data
Engine         rebuild the book; FIFO queue  (#136)      numpy now; faster later #138 / #139
Sources        files and live feeds, one plug-in  (#137)
               the shared schema (#112) sits under every boundary
```

## The decisions

1. **One shared schema — #112.** Every layer reads and writes it. Versioned
   Parquet / Arrow. The four data-model choices (#146 sequence numbers, #147
   per-row symbol and venue, #154 UTC time model, #155 integer tick prices) sit
   inside it.
2. **Separate the engine — #136.** Give the rebuild engine a clear input and a
   clear output, so it can be made faster later without touching the rest.
3. **One source plug-in — #137.** Files and live feeds use the same shape, with
   typed configuration in place of an untyped dictionary. Build this before new
   sources.
4. **Take pandas or Polars — #104.** Use Polars inside for speed. Let the public
   API accept and return either, through Narwhals. The engine does not change.
   *Superseded by [0002](0002-dataframe-library.md) on 2026-08-31: the public API
   stays pandas, Arrow is the interop promise, and Narwhals, Polars and DuckDB do
   not become dependencies. The last sentence — the engine does not change —
   still stands. The tool list at the end of this record is corrected there too.*
5. **Streaming core — #139.** Process events one at a time, so live capture and
   file replay share one path. Build alongside the live view (#105).
6. **Faster engine — #138.** Use numba first, then possibly Rust. Build only
   after #136, and only after real Databento data (#100) shows the rebuild is
   the slow part.

## Considered and rejected

- **Rewrite the core in Polars.** Rejected. The engine is stateful, row-by-row
  numpy code. A dataframe library is the wrong tool for it, and the rewrite
  would cost far more than fixing the boundaries.
- **Move the whole library to Polars and drop pandas.** Rejected. It would break
  every existing user for an internal gain. Narwhals gives the internal speed
  without changing the public contract.
- **Write the engine in Rust now.** Rejected for now. No benchmark yet shows the
  rebuild is the bottleneck at real data sizes. numba is the cheaper first step.
- **Add each new venue as its own loader.** Rejected. Without one plug-in shape
  (#137) every venue re-invents configuration and capability handling.

## Consequences

Do not:

- rewrite the engine in Polars;
- add Rust before a benchmark asks for it;
- add new sources before #137;
- break the pandas API — Narwhals prevents this;
- make the public API harder to read.

This plan adds these tools on top of the current uv / ruff / ty / hypothesis /
zensical stack: Narwhals, Polars, DuckDB, numba, typed configuration (pydantic
or dataclasses), entry-point plug-ins, a benchmark test in CI, and conda-forge
(#115). Rust tools (maturin, cibuildwheel) only if #138 needs them.

## Allowed upward imports

*Added 2026-11-17 (#495).* A layer imports only from the layers below it.
`tests/test_layer_imports.py` checks every module against this. It puts each
module in a layer (`LAYER_OF`), says which layers each layer may use
(`MAY_IMPORT`), and lists the few imports that go upward on purpose
(`ALLOWED_EXCEPTIONS`). That list and this one must match. To add an upward
import, add it to both, with the reason here.

The schema and the small helpers every layer uses (`schemas`, `config`,
`protocols`, `exceptions`, `_utils`, the `QualityCheck` and `Severity` types,
and a few others) form a shared layer under all five. The diagram puts
`plot_result` and `prepare` in the public API, so the test does too, and puts
the gallery beside them. The rest of
`ob_analytics.visualization`, Parquet I/O (`data`) and the backtest writers
(`interop`) form the plots and export layer. Export may use sources, because
a source can supply the writer for its own format.

The one upward import by design:

- **`metrics` → `visualization._data`.** `Metric.prepare` returns the data a
  renderer draws. One object then holds a metric and its plot, which is the
  short way for a plug-in author to add both (goal #174). The import is inside
  each `prepare` method, so loading `metrics` does not load the plot code.

Upward imports allowed for now. Each uses a function or type that sits one
layer too high for it. Moving them down would touch modules that other work is
changing, so each is left as a later move:

- **`lobster` → `depth` and `quality`.** The LOBSTER loader rebuilds the price
  levels from the messages and warns when that book is crossed for too long. It
  uses `quality.book_crossing` and `CROSSED_TOLERANCE_PCT`, so the loader and
  the audit measure a crossed book the same way. It builds the depth table and
  its summary with `depth.price_level_volume` and `DepthSummaryEngine`, and
  writes the orderbook file's levels with `depth.book_levels`. It uses only
  public names from both modules.
- **`synth` → `analytics`.** A generated session comes back with
  `set_order_types` already applied, so it looks like real input after the
  pipeline's classification step.
- **`databento` → `_engine_frames`.** The loader uses `resting_price`, the
  library's one rule for the price an order rests at, so it does not keep a
  second copy of that rule.
- **`data` → `pipeline`.** Loading a saved run builds a `PipelineResult`, the
  type the pipeline returns. The imports are inside functions or under
  `TYPE_CHECKING`, which breaks the import cycle with `pipeline`.
- **`visualization._data` → `visualization`.** Plot data resolves the level
  of a run with the same function `plot_result` uses, so the two cannot
  disagree.

## Related

- Roadmap and current status: epic #124. That issue tracks *what is done*; this
  record states *the target*.
- `ARCHITECTURE.md` describes the pipeline as it stands today, including the
  shipped `Source` shape from #137.
