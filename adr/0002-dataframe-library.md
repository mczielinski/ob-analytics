---
status: accepted
date: 2026-08-31
supersedes: "0001 (decision 4)"
---

# The public API stays pandas; Arrow is the interop promise

ob-analytics keeps pandas as the frame type of its public API and its plug-in
shapes. It does not add Narwhals, Polars, or DuckDB as dependencies. Users who
want Polars or DuckDB get them through the versioned Parquet from #112 and
through Arrow accessors on the result object. The performance port of analytics
to Polars is not rejected outright, but it does not start until a measurement
says it is worth doing, and this record states what that measurement must show.

This supersedes decision 4 of [0001](0001-layered-architecture-with-a-shared-schema.md),
which read: "Take pandas or Polars — #104. Use Polars inside for speed. Let the
public API accept and return either, through Narwhals." Nothing else in 0001
changes. The five layers, the shared schema, the separated engine, the source
plug-in, the streaming core, and the faster engine all stand.

## Context

Decision 4 was written on 2026-08-19, from the architecture review in #124. Five
things have changed or come to light since.

**The interop half already shipped.** #112 closed. `docs/schema.md` carries
worked examples of Polars and DuckDB reading the output files with no pandas
step. The re-scope of #104 said the real gain was "interop plus scale
ergonomics, most of it delivered by #112". The interop part is delivered.

**The stated bottleneck is not frame code.** #104 argues from the scale envelope
in `ARCHITECTURE.md`: the depth stages dominate time and memory. Those stages
are numpy kernels — `_interval_sums_sorted` and `_interval_sums_sparse` in
`depth.py`. A dataframe library cannot make a numpy kernel faster. The stated
reason for the work points at #136 and #138, not at the dataframe library.

**No number exists to judge it by.** `scripts/bench_scale.py` prints one figure
for the depth stages as a whole. There is no per-stage split. #144, which would
build one, is open. The gate written into #104 — "at least 2x on loader and
analytics at 5M events or more" — cannot be measured today.

**Narwhals gives portability, not speed.** Narwhals code running over a
pandas-backed frame runs pandas. Speed needs a real conversion to Polars, which
needs Polars as a dependency and a copy at the boundary. Decision 4 joined two
separate choices in one sentence, and only one of them is work Narwhals does.

**Narwhals inside the code is a rewrite, not a translation.** Narwhals has no
index. `analytics.py` holds 26 index-dependent expressions, `lobster.py` 7,
`depth.py` 6. There are four `merge_asof` calls, and Polars `join_asof` differs
from pandas `merge_asof` on tolerance, ties, and nulls — the join-semantics trap
#104 itself warns about.

**The plug-in contract already promises pandas.** #137 merged on 2026-08-30 with
`pd.DataFrame` written into `EventLoader.load`, `TradeSource.load`,
`DepthSource.load`, `DataWriter.write`, and `OfflineSource.compute_depth`.
Decision 4 predates that merge. Widening those signatures now would change a
shape third parties are invited to implement.

## The decision

1. **The public API takes and returns pandas.** Say so as a promise in #141,
   not as an accident of history.

2. **Interop is the file format, not the Python type.** The versioned Parquet
   from #112 is what other tools read. Add `to_arrow()` and `to_polars()`
   accessors on `PipelineResult` so a user can convert in one call. #112 already
   makes that conversion nearly free.

3. **No Narwhals, no Polars, no DuckDB as dependencies.** They stay on the
   user's side of the boundary, where `docs/schema.md` already puts them. This
   corrects the tool list at the end of 0001, which named all three.

4. **Source and metric plug-ins receive pandas.** Do not widen the #137
   protocols. A source builds a frame once from bytes, so the frame type there
   costs nothing measurable, and widening five protocols to buy nothing is how
   #104 grew to XL the first time. One exception is worth looking at on its own:
   `DataWriter.write` feeds Parquet, so it may want Arrow rather than pandas.

5. **Measure before porting anything.** #144 comes first: time and peak memory
   per stage, for load, `price_level_volume`, `depth_metrics`, queue, and
   analytics, on the bundled sample and on a large generated slice (#114). Then
   #136, which turns "which code could move to Polars" from a judgement into a
   module boundary. Decide the port last, with numbers in hand.

6. **What would reopen the port.** Both of these, together:
   - the per-stage profile shows loader plus analytics — counting neither the
     numpy kernels nor the stateful loops — taking **30 percent or more of wall
     time** at 5M events or more; and
   - a spike on one of those stages shows **2x or better**, with output identical
     byte for byte against the golden tests from #143.

   Below 30 percent, even a 2x win is under a 15 percent gain end to end, which
   does not pay for a second dataframe stack in the tree.

7. **The engine does not change.** This part of decision 4 stands. `queue.py`
   and `order_book` stay numpy, behind the boundary #136 draws, and get faster
   only through #138 and only when a benchmark asks.

## Considered and rejected

- **Narwhals as a thin wrapper** — `from_native` on the way in, `to_native` on
  the way out, pandas bodies untouched. Rejected for now. It delivers "accepts
  either" and no speed, and it puts a dependency into the type signature of
  about a dozen public functions. Revisit if a user asks for it.
- **Narwhals throughout the analytics code.** Rejected until the measurement in
  point 6 says otherwise. It is a rewrite of the index-dependent code and the
  four `merge_asof` calls, guarded by oracle tests for null and join semantics.
- **Polars as an optional extra.** Rejected. An extra means the fast path is not
  the default path, so two paths need testing against outputs the #143 tests pin
  byte for byte.
- **Editing 0001 in place.** Rejected. 0001 is one day old, but the value of a
  record is that it shows where the thinking moved and why. Keep both.

## Consequences

Do not:

- add Narwhals, Polars, or DuckDB to `pyproject.toml`;
- widen the #137 plug-in protocols away from pandas;
- start the Polars port before #144 and #136, or before the test in point 6
  passes;
- promise a Polars return type anywhere in the docs or in issue text.

Do:

- carry point 1 into #141 as a written stability promise;
- amend #140 to drop "able to take pandas or Polars (Narwhals, #104)" from the
  metric shape; the metric shape takes pandas.

## Related

- #104 becomes the record of this decision and the frame-type contract that
  #174 and #181 need. It is no longer an XL port.
- #178 no longer depends on #104. Nothing in this record changes peak memory.
  That goal is served by #116 and by a lazy, out-of-core read path over the
  Parquet from #112.
- Roadmap and current status: epic #124.
