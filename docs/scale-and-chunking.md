---
title: Scale and chunking
---

# Scale and chunking

!!! info "Decision record"
    **Status:** Accepted · **Date:** 2026-07-11 · **Context:** WS-8.4b,
    gating the Databento adapter (WS-6.1).

    **Decision:** ob-analytics stays single-shot and in-memory. For data
    beyond the session-scale envelope, **pre-slice by time window** and process
    each slice independently. We do **not** build streaming or chunking
    infrastructure now.

ob-analytics keeps the full event, depth, and trade tables in memory (pandas).
The [scale envelope](architecture.md#scale-envelope) puts the comfortable
ceiling at **~5M events (~5 GiB peak RSS)** — a few hours of a single liquid
instrument. The Databento adapter (WS-6.1) opens the door to venue market-by-order
(MBO) feeds whose *full* volume is far larger, so before building it we have to
decide whether the in-memory model needs a chunked execution mode, or whether a
documented pre-slicing workflow suffices.

## The criterion

> Does a target Databento day for **one symbol** fit within the ~5M-event
> envelope after **time-slicing to a session-length window**?
>
> - **Yes** → document the pre-slice workflow and stop; add no infrastructure.
> - **No** → specify a minimal chunked-run helper (slice → run → concatenate),
>   justified by a memory profile.

The load-bearing words are *one symbol* and *session-length window*. The
alarming totals — the whole Nasdaq feed runs to **several billion messages a
day** — are *every symbol at once*. A single instrument is orders of magnitude
smaller, and a session-length window is a fraction of that instrument's day.

## Evidence

### What fits — the supply side

From the measured [scale envelope](architecture.md#scale-envelope) (WS-8.4a,
`scripts/bench_scale.py`): peak RSS grows roughly linearly at **~1 GiB per 1M
events**, dominated by the depth stages.

| events | peak RSS | depth stages |
|--------|---------:|-------------:|
| 314 k  | ~0.43 GiB | ~14 s |
| 628 k  | ~0.73 GiB | ~25 s |
| 942 k  | ~1.02 GiB | ~38 s |
| 1.26 M | ~1.32 GiB | ~51 s |

The **comfortable ceiling ≈ 5M events / ~5 GiB** on a typical 16 GB machine.
(That 5M point is a linear *extrapolation* from the measured rows above, not a
direct measurement — see `bench_scale.py`. It is conservative: tiling adds
transient overhead the extrapolation carries forward.)

### What a job needs — the demand side

Single-symbol event counts, from published figures:

| Source | Scope | Count |
|--------|-------|------:|
| [LOBSTER sample][lobster] (AAPL/AMZN/GOOG/MSFT/INTC, 2012-06-21) | one symbol, full 6.5 h session | 300 k – 600 k events each |
| [Nasdaq TotalView-ITCH][xnas] (`XNAS.ITCH` MBO) | whole feed, one day | several **billion** messages |
| [Databento example][apidemo] (`ESH4`, `trades` schema, week of 2024-02-12) | one instrument, ~1 week | 1,735,003 trade prints |

Reading these together:

- A **large-cap single symbol** produced **300 k – 600 k events for a full
  session** in 2012 — already an order of magnitude under the 5M ceiling. That
  is the MBO-equivalent count (adds, cancels, executions), not just trades.
- Message rates have grown since (more venues, finer ticks, denser quoting).
  The `ESH4` figure is ~1.7M *trade prints* in a week — a few hundred thousand a
  day for one instrument. An MBO stream counts every add/cancel/modify/execute
  and runs **10× or more** above the print count, so a *very* active single
  instrument today sits in the **low millions of MBO messages per day** — near
  or, on volatile days, above the 5M full-day ceiling.
- Intraday message flow is strongly **U-shaped** (the open and close dominate),
  so **slicing a heavy day into session-length windows keeps each slice
  comfortably under 5M**. The finer the window, the more headroom.

For the realistic target — *one symbol over a session-length window* — the
answer to the criterion is **yes, it fits**, with pre-slicing covering the
busiest-instrument tail.

## Decision

**Document the pre-slice-by-time-window workflow; build no chunking or
streaming infrastructure.** The in-memory, single-shot model remains the whole
design. This keeps the memory profile simple and predictable and matches the
guidance already on the [scale envelope](architecture.md#scale-envelope) page.

## Recommended workflow

### 1. Size the job before you run it

You do not have to guess. Databento bills by record, so it exposes an exact,
metadata-only record count — cheap, no bulk download:

```python
import databento as db

client = db.Historical()  # reads DATABENTO_API_KEY from the environment

n = client.metadata.get_record_count(
    dataset="XNAS.ITCH",
    symbols=["AAPL"],
    schema="mbo",
    start="2024-02-12T14:30",  # one session-length window, UTC
    end="2024-02-12T16:00",
)
print(n)  # events in this window — compare against the ~5M envelope
```

`client.metadata.get_cost(...)` takes the same arguments and returns the dollar
cost, so you can size memory and spend in one step. If you already hold a
`.dbn` / `.dbn.zst` file, count it locally — reading a local file needs no API
key:

```python
from databento import DBNStore

store = DBNStore.from_file("aapl-mbo.dbn.zst")  # local file — no API key needed
n = sum(1 for _ in store)                        # total records in the file
# if the file interleaves symbols, count just one instrument:
# n = sum(1 for r in store if r.instrument_id == my_instrument_id)
print(n)  # compare against the ~5M envelope
```

### 2. Slice, run each window, concatenate

Split the input at time-window boundaries, run the pipeline on each window, and
concatenate the per-slice outputs. Peak memory is bounded by the **largest
single window**, not by the whole day:

```python
import pandas as pd
from ob_analytics import Pipeline

windows = ["s_0930_1100.csv", "s_1100_1230.csv", "s_1230_1400.csv"]  # per-window sources
results = [Pipeline().run(w) for w in windows]

events = pd.concat([r.events for r in results], ignore_index=True)
trades = pd.concat([r.trades for r in results], ignore_index=True)
depth  = pd.concat([r.depth  for r in results], ignore_index=True)
```

!!! warning "Slices are independent books, not one continuous book"
    Each `run()` rebuilds the order book **from scratch** within its window.
    Orders already resting when a window begins are not in that window's input,
    so the first moments of a slice under-count standing liquidity, and
    per-order lifecycles are split across the cut.

    - **Concatenates cleanly:** windowed views — depth over time, the trade
      tape, per-bucket flow toxicity.
    - **Does *not* span slices:** whole-book, per-order questions — queue
      position, order lifetimes, `order_outcome`.

    Slice at natural low-activity boundaries, or treat each slice as an
    independent session. The Databento adapter (WS-6.1) softens the boundary by
    seeding each window from the feed's periodic snapshot (DBN `F_SNAPSHOT`);
    WS-6.0's *pre-existing order* class labels the carried-in orders.

## When we would revisit this

The decision flips only if a workflow needs **whole-day, single-instrument
outputs that a concatenation of independent slices cannot reconstruct** — for
example continuous per-order queue trajectories across an entire session at a
volume no single window can hold.

If that workload appears (and is *measured*, not assumed), the minimal response
is a `chunked_run(source, boundaries)` helper: slice → run each window → carry
the resting book (or a snapshot) into the next window's input → concatenate into
one merged `PipelineResult`. Its peak memory stays bounded by the largest
window; its real cost is boundary-state carry — the same pre-existing-order
handling described above — which is exactly why it is not worth building
speculatively. Until such a workload exists, pre-slicing is sufficient and
simpler.

## References

- Scale envelope and benchmark: [Architecture → Scale envelope](architecture.md#scale-envelope); `scripts/bench_scale.py`.
- [LOBSTER sample files][lobster] — per-symbol daily event counts.
- [Nasdaq TotalView-ITCH on Databento][xnas] — whole-feed daily message volume.
- [Databento Python API demo][apidemo] — a concrete single-instrument record count.
- [`metadata.get_record_count`][getcount] — size any query before downloading it.

[lobster]: https://lobsterdata.com/info/DataSamples.php
[xnas]: https://databento.com/datasets/XNAS.ITCH
[apidemo]: https://databento.com/blog/api-demo-python
[getcount]: https://databento.com/docs/api-reference-historical/metadata/metadata-get-record-count
