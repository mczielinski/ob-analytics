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

    **Second amendment (2026-11-22):** `ob_analytics.scan` reads a saved run
    one batch at a time and aggregates it, so a saved run larger than memory
    can be used. See [Reading a saved run in parts](#reading-a-saved-run-in-parts).

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
Since then, the Databento, Bitstamp and L2 CSV loaders have learned to read
one window at a time; see [Limits](#limits).

## Recommended workflow

### 1. Size the job before you run it

You do not have to guess. Databento bills by record, so it exposes an exact,
metadata-only record count — cheap, no bulk download:

<!-- docs-test: skip needs a Databento API key and the network -->

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

<!-- docs-test: skip needs a Databento file -->

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

<!-- docs-test: skip needs a Databento file -->

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

<!-- docs-test: skip needs one file per window -->

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

`Pipeline.run_windows(path, boundaries, output)` cuts one input at the
`boundaries` and runs the depth stages on one window at a time. The depth
stages hold most of a run's memory, so their peak is set by the largest window,
not by the whole input.

### What it writes

One Parquet file per table in `output`: `events`, `trades`, `depth` and
`depth_summary`. The tests check this against a single run on the bundled
Bitstamp sample, a synthetic session, a Databento record set and an L2 file,
and, for the loaders that read one window at a time, with small parts that cut
across orders and windows. This is the same folder `save_data` writes for a single run,
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
  which crossed levels it leaves out of the touch, carries on into the next
  window.
- Order types and trade signs are decided once, over the whole input, as in a
  single run.

The result matches a single run row for row in every table. The one
difference is that `events` is written in window order, not in the loader's
order.

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

### Memory of a long input

Measured on 2026-11-23 on one Linux x86_64 desktop with 15 GiB of memory. The
input is one day of `XNAS.ITCH` AAPL MBO (2026-11-10, 3.87M records), repeated
1, 3, 5 and 10 times in one uncompressed DBN file. Each copy is one day later
than the one before and has new order ids. Each file is run with
`Pipeline(source=DatabentoSource()).run_windows` and one-hour windows, each
run in its own process. Peak memory is `ru_maxrss`. "Before" is the code
before the order types and the trades went to the working folder.

| days | records | windows | peak before | time before | peak after | time after |
|-----:|--------:|--------:|------------:|------------:|-----------:|-----------:|
| 1 | 3.9 M | 25 | 1.41 GiB | 30 s | 1.25 GiB | 30 s |
| 3 | 11.6 M | 73 | 2.92 GiB | 89 s | 1.32 GiB | 88 s |
| 5 | 19.4 M | 121 | 4.52 GiB | 155 s | 1.34 GiB | 145 s |
| 10 | 38.7 M | 241 | not run | — | 1.43 GiB | 304 s |

Before, the peak grew by about 0.78 GiB per day of AAPL, so ten days would
have needed about 8.4 GiB. Most of it was the order types: they were decided
over the whole run in memory, together with the values per order they are
read from. Now the peak grows by about 20 MiB per day, and ten days peak 14%
above one day. What still grows is the trade records the loader keeps and the
trades, both held while the windows are read; the time of each trade, held
while the depth stages run; and a little memory that the Parquet writers and
the memory allocator keep from one window to the next. A single in-memory run
of one day peaks at about 3.8 GiB.

The Bitstamp and L2 CSV loaders were measured on 2026-11-24 on the same
desktop. The Bitstamp input is the bundled sample (314k order rows, 30
minutes) repeated 1, 4 and 8 times in one `orders.csv` and `trades.csv`, each
copy 30 minutes later than the one before and with new order ids. The L2
input is the depth and trades of a single run of the sample, written as an L2
CSV folder (313k depth rows) and repeated the same way. Each file is run
with `run_windows` and five-minute windows, each run in its own process, and
peak memory is `ru_maxrss`. "Before" is the code where these loaders read the
whole input.

| input | copies | windows | peak before | peak after |
|-------|-------:|--------:|------------:|-----------:|
| Bitstamp | 1 | 7 | 540 MiB | 404 MiB |
| Bitstamp | 4 | 25 | 816 MiB | 439 MiB |
| Bitstamp | 8 | 49 | 1,269 MiB | 477 MiB |
| L2 | 1 | 7 | 409 MiB | 369 MiB |
| L2 | 4 | 25 | 516 MiB | 395 MiB |
| L2 | 8 | 49 | 794 MiB | 401 MiB |

Before, the peak grew by about 100 MiB per copy for Bitstamp and 55 MiB per
copy for L2. Now it grows by about 10 MiB and 5 MiB per copy: the trades,
which are still read whole, and the time of each trade. Each run took the
same time before and after, within a few seconds.

### Limits

- **Three loaders read one window at a time:** the Databento, Bitstamp and L2
  CSV loaders, and only when the source builds them, as
  `Pipeline(source=DatabentoSource())`, `Pipeline()` and
  `Pipeline.from_source("depth_csv")` do. A capture replays through one of the
  last two: an L3 capture (Bitstamp, Kraken, the cryptofeed venues) through
  the Bitstamp format, a ccxt or Kalshi capture through the L2 CSV format.
  - The Databento loader reads the file in parts, carries each order's size
    and price across the cuts, and gives the events of one window at a time.
    The records must be in receive-time order, which Databento guarantees for
    one instrument.
  - The Bitstamp loader puts each order's rows in order, and works out its
    fills, from all of its rows, and an order can last over many windows. So
    it reads the file in parts and splits the rows by order id into files in
    the working folder. Then it reads back one group of orders at a time, puts
    their rows in order, matches the trades in `trades.csv` to them, and
    splits the rows by window into files. Last, it reads each window back and
    numbers its events as a whole read does. The file need not be in time
    order.
  - The L2 CSV loader reads the file in parts and splits the rows by window
    into files, then reads each window back and sorts it by time. The file
    need not be in time order.

  Both CSV loaders check the file one part at a time. A bad row's number in an
  error counts from the start of its part, and the error says which row of the
  file that part starts at. A warning that sizes or prices were rounded to the
  grid comes once for each part that has such values, with that part's count.

  For per-order events the run saves each window's events to a working folder
  inside `output`, decides order types and trade signs over the whole run, and
  then reads the windows back for the depth stages. The values per order that
  the order types are read from also go to the working folder, and the types
  are decided one group of orders at a time. There a text order id (a UUID,
  say) stands as a 64-bit hash of its text; for a million orders the chance
  that two share a hash is less than one in ten million, and the run checks:
  if two ids do share one, it stops with an error rather than give wrong
  types. While the depth stages run, the run holds only the time of each
  trade. So the peak is set by the window length, not by the length of the
  input; see
  [Memory of a long input](#memory-of-a-long-input).
- **The LOBSTER loader reads the whole input,** and so does any loader the
  caller passes. The `events` table, or for a price-level source the depth
  rows, is then held for the whole run. A LOBSTER file holds one trading
  day, so its length is bounded.
- **A source's own depth is not used.** LOBSTER's order book file states the
  book after each message of the whole session, so it cannot be cut. A windowed
  LOBSTER run rebuilds its depth from the messages instead.

## Reading a saved run in parts

`load_data` reads a whole saved run into memory, and a windowed run can write
a folder far larger than that. The functions in `ob_analytics.scan` read one
table of the folder one batch of rows at a time, through `pyarrow.dataset`, so
their memory does not grow with the file:

```python
import pandas as pd
from ob_analytics import Pipeline
from ob_analytics.data import save_data
from ob_analytics.scan import aggregate_by_time, read_table, summarize_depth_file

save_data(Pipeline().run("orders.csv"), "out")

# The best bid and ask at the end of each minute, and the mean size at the
# touch over the minute.
minutes = aggregate_by_time(
    "out",
    "depth_summary",
    "1min",
    {
        "best_bid_price": "last",
        "best_ask_price": "last",
        "best_bid_vol": "mean",
        "best_ask_vol": "mean",
    },
)

# Ten minutes of three columns. Only these columns are read, and only the row
# groups that hold rows in the range.
start = minutes["timestamp"].iloc[5]
touch = read_table(
    "out",
    "depth_summary",
    columns=["timestamp", "best_bid_price", "best_ask_price"],
    start=start,
    end=start + pd.Timedelta("10min"),
)

# The depth summary again, with 10 bps bins, replayed from the saved depth.
summarize_depth_file("out", "out-10bps", depth_bps=10)
```

- **The numbers are the same as in memory.** `aggregate_by_time` reduces each
  batch per bucket and combines the parts of a bucket that two batches share,
  so it gives what `frame.groupby(bucket).agg(how)` gives on the whole table.
  `summarize_depth_file` replays the depth rows through one summary engine,
  which keeps the book from one batch to the next, so it gives what
  `summarize_depth` gives on the whole table, row for row. The tests check
  both with batches of under a thousand rows.
- **The time range goes to the reader.** Each file is written in row groups
  of at most 131,072 rows, and Parquet keeps the time range of each one, so a
  `start` and `end` skip the row groups outside them.
- **The result must fit.** `read_table` and `aggregate_by_time` return one
  frame. A short bucket on a long run makes a large result.
- **`summarize_depth_file` needs the depth rows in time order,** as a run
  writes them. It stops with an error on a file that is not, and leaves the
  destination as it was.

### Memory, measured

Measured on 2026-11-23 with `scripts/bench_scale.py --out-of-core`, on one
Linux x86_64 desktop with 15 GiB of memory. The script saves the bundled
sample's run, copies its `depth_summary` and `depth` files end to end to make
longer ones, and measures each size in its own process. "As pandas" is the
memory the table would take if `load_data` read it.

`aggregate_by_time(folder, "depth_summary", "1min", "mean")`:

| rows | on disk | as pandas | peak RSS | time |
|-----:|--------:|----------:|---------:|-----:|
| 314 k | 6 MiB | 0.1 GiB | 590 MiB | 0.2 s |
| 5.0 M | 95 MiB | 1.7 GiB | 1,027 MiB | 1.3 s |
| 20.1 M | 379 MiB | 6.9 GiB | 1,123 MiB | 4.6 s |
| 50.2 M | 948 MiB | 17.2 GiB | 1,160 MiB | 10.8 s |

The largest table would not fit in this machine's memory as a frame. Read in
parts, it takes about 1.1 GiB. From 5M to 50M rows the peak rises by about
130 MiB: Arrow keeps some freed memory for reuse rather than handing it back.

`summarize_depth_file(folder, dest)`:

| depth rows | peak RSS | time |
|-----------:|---------:|-----:|
| 314 k | 830 MiB | 8 s |
| 1.3 M | 830 MiB | 30 s |
| 5.0 M | 830 MiB | 120 s |

The time is the summary engine's, about 25 s per million rows, read in parts
or not. The memory does not grow with the file.

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
