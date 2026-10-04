---
title: Scale and chunking
---

# Scale and chunking

!!! info "Decision record"
    **Status:** Accepted, amended · **Date:** 2026-07-11, amended
    2026-10-03 · **Context:** WS-8.4b, gating the Databento adapter (WS-6.1).

    **Decision:** ob-analytics stays in-memory. For data beyond the
    session-scale envelope, **cut the input into time windows**.

    **Amendment:** `Pipeline.run_windows` now does the cutting for one input.
    It runs the depth stages one window at a time, carries the book across
    each cut, and writes each window to disk as it finishes. See
    [Windowed runs](#windowed-runs).

ob-analytics keeps the full event, depth, and trade tables in memory (pandas).
The [scale envelope](architecture.md#scale-envelope) puts the comfortable
ceiling at **~5M events (~5 GiB peak RSS)** — a few hours of a single liquid
instrument. The [Databento adapter](howto/databento.md) opens the door to venue
market-by-order (MBO) feeds whose *full* volume is far larger, so before
building it we had to decide whether the in-memory model needs a chunked
execution mode, or whether a documented pre-slicing workflow suffices.

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
`scripts/bench_scale.py --envelope`): peak RSS grows roughly linearly at **~1 GiB per 1M
events**, dominated by the depth stages.

| events | peak RSS | depth stages |
|--------|---------:|-------------:|
| 314 k  | ~0.43 GiB | ~14 s |
| 628 k  | ~0.73 GiB | ~25 s |
| 942 k  | ~1.02 GiB | ~38 s |
| 1.26 M | ~1.32 GiB | ~51 s |

The **comfortable ceiling ≈ 5M events / ~5 GiB** on a typical 16 GB machine.
(That 5M point is a linear *extrapolation* from the measured rows above, not a
direct measurement — see `bench_scale.py --envelope`. It is conservative: tiling adds
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

The 2026-10-03 amendment adds one piece of infrastructure, a windowed run, and
keeps the rest of this decision. Nothing streams: the loader still reads the
whole input, and each window is an ordinary in-memory run of the depth stages.

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

### 2. Run one window at a time

Give `run_windows` the input, the times to cut it at, and a folder for the
output:

```python
from ob_analytics import Pipeline
from ob_analytics.data import load_data
from ob_analytics.databento import DatabentoSource

cuts = ["2024-02-12T16:00", "2024-02-12T17:30", "2024-02-12T19:00"]  # UTC
out = Pipeline(source=DatabentoSource()).run_windows("aapl.mbo.dbn.zst", cuts, "out/")

tables = load_data(out)  # events, trades, depth, depth_summary
```

Three cuts make four windows, which together cover the whole input. Each window
starts from the book the previous one ended with, so the output is the same as a
single run's. [Windowed runs](#windowed-runs) gives the details and the limits.

### 3. Files already split by window

If the input is already one file per window, for example the files
`scripts/databento_window.py` downloads, run each file and concatenate the
results:

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

    `run_windows` has none of these limits, because it cuts one input and
    carries the book across each cut. With separate files, slice at natural
    low-activity boundaries, or treat each slice as an independent session. A Databento window that includes the feed's periodic
    snapshot softens the boundary: those records are ordinary adds, so they
    seed the window's book, and the *pre-existing order* class labels whatever
    the window carried in without one.

## Windowed runs

`Pipeline.run_windows(source, boundaries, output)` cuts one input at the
`boundaries` and runs the depth stages on one window at a time. The depth
stages hold most of a run's memory, so their peak is set by the largest window,
not by the whole input.

### What it writes

One Parquet file per table in `output`: `events`, `trades`, `depth` and
`depth_summary`. The tests check this against a single run on the bundled
Bitstamp sample, a synthetic session, a Databento record set and an L2 file. This is the same folder `save_data` writes for a single run,
and `load_data` reads it back. Each window is written as soon as it finishes.

The output goes to disk, not into a `PipelineResult`, because the finished
tables are themselves large. On the bundled sample a run peaks at 456 MiB, and
its result alone takes 164 MiB, 110 MiB of that in `depth_summary`. A merged
result held in memory would grow with the whole input and save about half the
memory at best.

### How the book crosses a cut

With `carry=True`, the default, each window starts from the book the previous
window ended with:

- The orders still resting at the cut are put back as `created` rows stamped
  one nanosecond before the window starts. The price-level rebuild then changes
  each level from its right size. These rows are not written out.
- The depth summary keeps one engine for the whole run, so its book, including
  the crossed levels it has removed, carries on into the next window.
- Order types and trade signs are decided once, over the whole input, as in a
  single run.

The result matches a single run row for row in every table. There are three
differences:

- `events` is written in window order, not in the loader's order.
- `aggressiveness_bps` looks up the quote standing before each order by
  `event_id`. Bitstamp numbers its events by order, not by time, so on a
  Bitstamp input the lookup can find a quote from another window. Sources that
  number events in time order (Databento, the synthetic generator) match
  exactly.
- Where the Databento loader warns that the depth is off (a modify that
  carries a fill and also moves the order), the carried order goes onto its new
  price level, so the windowed depth can differ by the same amount.

A run that fails part-way leaves the output folder as it was: the files are
moved into place only once every window is done.

With `carry=False`, each window starts from an empty book, as if it were a
separate input.

An L2 input needs no seed rows. Each depth row states its level's whole size,
so the summary engine alone carries the book.

### Memory, measured

The bundled sample tiled four times (1.26M events), with no trades, one process
per row. "Load only" is the loaded events table and nothing else.

| windows | peak RSS | time |
|--------:|---------:|-----:|
| load only | 386 MiB | — |
| 1 (a single run) | 1,507 MiB | 70 s |
| 2 | 1,259 MiB | 71 s |
| 4 | 953 MiB | 70 s |
| 8 | 771 MiB | 71 s |

More windows bring the peak down toward the load-only floor at almost no cost in
time.

### Limits

- **The loader still reads the whole input.** The `events` and `trades` tables
  are held for the whole run, because order types and trade signs depend on
  every row. These are the fixed part of the memory above. A loader that reads
  one window at a time, for example Databento's `DBNStore.to_df(count=...)`,
  would remove it. Build that when a measured input's events table does not
  fit.
- **A source's own depth is not used.** LOBSTER's order book file states the
  book after each message of the whole session, so it cannot be cut. A windowed
  LOBSTER run rebuilds its depth from the messages instead.

## References

- Scale envelope and benchmark: [Architecture → Scale envelope](architecture.md#scale-envelope);
  `scripts/bench_scale.py --envelope`. The per-stage speed test that CI runs is
  the same script with no arguments.
- [LOBSTER sample files][lobster] — per-symbol daily event counts.
- [Nasdaq TotalView-ITCH on Databento][xnas] — whole-feed daily message volume.
- [Databento Python API demo][apidemo] — a concrete single-instrument record count.
- [`metadata.get_record_count`][getcount] — size any query before downloading it.
- [Process Databento MBO files](howto/databento.md) and
  `scripts/databento_window.py` — this workflow as runnable code.

[lobster]: https://lobsterdata.com/info/DataSamples.php
[xnas]: https://databento.com/datasets/XNAS.ITCH
[apidemo]: https://databento.com/blog/api-demo-python
[getcount]: https://databento.com/docs/api-reference-historical/metadata/metadata-get-record-count
