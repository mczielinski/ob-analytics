---
status: accepted
date: 2026-11-22
---

# numba is a required dependency, for the rebuild loops

ob-analytics depends on numba. The loops that rebuild a book one event at a
time are compiled with it. The first of these is the L2 depth replay behind
`summarize_depth`. There is one code path: no optional extra, and no numpy
fallback.

## Context

Two loops replay a book one event at a time: the L2 depth replay in
`DepthSummaryEngine.compute`, and the queue engine. Each event costs a dozen or
so small numpy calls, and almost all of that time is the overhead of the calls.
A dataframe library cannot remove that overhead
([ADR 0002](0002-dataframe-library.md)). Compiling the loop can.

#138 set the condition for compiling: a per-stage measurement on real data has
to show that the loops, not file reading or analytics, are the slow part. On a
full day of Databento AAPL MBO data (3.7 million events), `depth_metrics` took
69 percent of a run and `queue_positions` 7 percent. On the bundled sample the
two together took 91 percent.

numba 0.68 ships wheels for CPython 3.11 to 3.14, the versions this package
supports, on Linux (x86_64 and aarch64), macOS (arm64) and Windows. It works
with numpy 2.4 and 2.5. It needs no compiler on the user's machine. The
compiled loops take and return numpy arrays only, so the pandas version does
not matter to them.

## The decision

1. **numba is a required dependency.** It is in `dependencies` in
   `pyproject.toml`, with a floor at the version CI tests.

2. **One code path.** The numpy version of a loop is deleted when the compiled
   version replaces it. ADR 0002 turned down Polars as an optional extra for
   the same reason: an extra means two paths, and both have to give the
   outputs the golden tests (#143) pin.

3. **Compiled code takes numpy arrays and plain numbers.** It holds no pandas
   and no Python objects. The class around it keeps the frames, the
   validation and the public interface, so the compiled part can change
   without changing what a caller sees.

4. **Outputs do not change.** A compiled loop gives the same values as the
   code it replaces, value for value, and it does its float arithmetic in the
   same order to make that true.

5. **Compiled functions are cached on disk** (`cache=True`). The first call in
   a new environment compiles them, which takes a few seconds. Later
   processes load the cached machine code. Where no cache folder can be
   written, as in some containers, each process compiles them again instead
   of failing.

## Considered and rejected

- **An optional extra, with the numpy code as the fallback.** Rejected. Two
  depth paths would need to stay identical, and CI would have to test both.
- **An optional extra, with one plain-Python loop that runs compiled when
  numba is installed.** Rejected. Without numba, that loop would run many
  times slower than the numpy code it replaces, so the fallback would be a
  trap for anyone who does not install the extra.
- **Rust, through PyO3 and maturin.** Not now. #138 keeps it for the case
  where numba is not fast enough. It needs a build for each platform, Rust in
  CI, and someone who knows Rust.

## Consequences

- An install is larger: numba and llvmlite add about 50 MB.
- The first call in a new environment pays the compile time. `bench_scale.py`
  runs the compiled loop once before it times anything, so its timings leave
  compile time out.
- Compiled code is harder to step through in a debugger. Setting
  `NUMBA_DISABLE_JIT=1` runs the same functions as plain Python, very slowly.
- The queue engine can be compiled next, under the same rules.

## Related

- #138 is the plan and holds the measurements.
- [ADR 0002](0002-dataframe-library.md) keeps pandas for the public API and
  explains why a dataframe library does not help these loops.
