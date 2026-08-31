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

## Related

- Roadmap and current status: epic #124. That issue tracks *what is done*; this
  record states *the target*.
- `ARCHITECTURE.md` describes the pipeline as it stands today, including the
  shipped `Source` shape from #137.
