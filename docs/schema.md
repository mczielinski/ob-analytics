# Canonical Parquet / Arrow schema

The pipeline reads and writes a small set of tables. This page is the written
contract for those tables: for every column it gives the Arrow type, the unit,
whether the column can be null, and what the value means. Other tools can read
the Parquet files directly against this spec, without going through
ob-analytics or pandas.

The spec is **versioned**. Each Parquet file carries a schema version in its
metadata, so the on-disk format is a fixed contract, not a detail that can
change without notice.

## Schema version

The current version is **`1.0`** (the constant
[`ob_analytics.schemas.SCHEMA_VERSION`](api/schemas.md)).

`save_data(..., fmt="parquet")` writes the version into each file's Arrow
key-value metadata under the key `ob_analytics_schema_version` (bytes, because
Arrow metadata keys and values are raw bytes). `load_data` reads the key back
and checks it before returning the frame:

- A supported version loads normally.
- An unsupported version raises `ConfigError`.
- A file with **no** version key is treated as legacy data. It loads, with a
  warning. This keeps files written before the version existed — and the
  bundled sample — working.

Bump the version when a change would make an older reader misread a newer file:
a renamed or removed column, a changed unit or dtype, or a new required column.

Read the version from another tool through pyarrow:

```python
import pyarrow.parquet as pq

metadata = pq.read_schema("out/events.parquet").metadata
version = metadata[b"ob_analytics_schema_version"].decode()  # "1.0"
```

## The tables

Two groups of tables exist.

**Persisted tables** — the four core outputs a pipeline run produces and
`save_data` writes to disk:

| File | Produced by | Rows are |
|---|---|---|
| `events.parquet` | the loader + `set_order_types` | one order event (add / change / delete) |
| `trades.parquet` | the trade source | one execution (a trade print) |
| `depth.parquet` | `price_level_volume` | one price level's size after a change |
| `depth_summary.parquet` | `depth_metrics` | the reconstructed book state at one event |

**Derived records** — engine output computed on demand from `events`. They are
not written by default, but they follow a fixed shape and you can save them with
`save_data` like any frame:

| Record | Produced by | Rows are |
|---|---|---|
| order lifecycles | `order_lifecycles` | one order, from placement to outcome |
| book snapshot | `order_book` | one resting order in the book at a point in time |

For a Level-2 (price-level) feed there is no per-order identity, so `events` is
an empty but schema-valid frame and the derived records do not apply. Read
`depth`, `depth_summary`, and `trades` instead.

## Timestamp policy

All timestamps are `timestamp[ns]` and **tz-naive** (no time zone attached).
Each frame is in its venue's native clock: UTC for Bitstamp captures,
exchange-local (US/Eastern) for LOBSTER sessions. A single frame is internally
consistent, but timestamps from different formats are **not comparable**. Do not
join or concatenate events across venues without converting first.

`timestamp` is the local receive time. `exchange_timestamp` is the venue's
matching-engine time. For LOBSTER the two are equal, because only exchange time
exists there.

## Volume and fill

Per-order size follows one convention across every loader:

- `volume` is the order's **outstanding size after the event** for `created` and
  `changed` rows, and the **size removed** (the outstanding size just before the
  delete) for `deleted` rows.
- `fill` is the **executed size at this event** (`0` when nothing traded). A
  `changed` row is either an execution (`fill > 0`, outstanding drops by exactly
  `fill`) or a non-executed reduction (`fill == 0`), never both.

Volume units are the base asset (Bitstamp) or shares (LOBSTER). Price units are
the quote currency.

## Nullable integer columns

pandas has no built-in null for a plain `int64` column, so the loaders carry
nullable integer ids (order and event ids on trades) as an `object` column
holding Python ints or `NA`. On write, pyarrow infers the Arrow type from the
values: a column with no nulls becomes `int64`; a column that contains nulls may
become `int64` with a null mask or `double`. The tables below name the **logical**
type (`int64`) and mark the column nullable. Cleaning this up is one of the
[open decisions](#open-decisions) below.

## events

Required columns (the [`EVENT_COLUMNS`](api/schemas.md) contract) plus the
provenance columns every loader carries.

| Column | Arrow type | Unit | Null? | Meaning |
|---|---|---|---|---|
| `event_id` | `int64` | — | no | Unique id for this event (1-based). Join key for a trade's `maker_event_id` / `taker_event_id`. |
| `id` | `int64` | — | no | Order id. Groups the events of one order over its life. |
| `timestamp` | `timestamp[ns]` | ns, tz-naive | no | Local receive time. |
| `exchange_timestamp` | `timestamp[ns]` | ns, tz-naive | no | Venue matching-engine time (equals `timestamp` for LOBSTER). |
| `price` | `double` | quote currency | no | Limit price of the order. |
| `volume` | `double` | base asset / shares | no | Outstanding size after the event, or size removed on a delete (see [Volume and fill](#volume-and-fill)). |
| `direction` | `dictionary<string>` | — | no | Order side: `bid` or `ask` (ordered categorical). |
| `action` | `dictionary<string>` | — | no | Event kind: `created`, `changed`, or `deleted` (ordered categorical). |
| `fill` | `double` | base asset / shares | no | Executed size at this event (`0` when none). |
| `type` | `dictionary<string>` | — | no | Order class from `set_order_types` (see [Categorical values](#categorical-value-domains)). |
| `original_number` | `int64` | — | no | 1-based source row number. Provenance for round-trip writers. |
| `raw_event_type` | `int64` / `string` / `null` | — | yes | Venue's native event-type code (LOBSTER `1`–`5`); null for Bitstamp. |
| `raw_size` | `double` | base asset / shares | yes | LOBSTER only. Venue's raw per-event quantity, kept for orders with no `created` row and for round-trip writers. |
| `aggressiveness_bps` | `double` | basis points | yes | Placement distance from the best price, added by `order_aggressiveness`. Null where it does not apply. |

`raw_event_type` is present but all-null on a Bitstamp frame; pyarrow writes an
all-null column as the `null` type. `raw_size` and `aggressiveness_bps` are
present only after the stages that add them.

## trades

Required columns (the [`TRADE_COLUMNS`](api/schemas.md) contract) plus maker /
taker provenance.

| Column | Arrow type | Unit | Null? | Meaning |
|---|---|---|---|---|
| `timestamp` | `timestamp[ns]` | ns, tz-naive | no | Trade print time (receive clock). |
| `price` | `double` | quote currency | no | Execution price. |
| `volume` | `double` | base asset / shares | no | Executed size. |
| `direction` | `dictionary<string>` | — | yes | Taker's aggressor side: `buy` or `sell`. Null on an L2 feed until Lee–Ready fills it, then set. |
| `maker_event_id` | `int64` | — | yes | `event_id` of the maker (resting) order's event. Null when the trade is not attributed (L2 feeds have no order identity). |
| `taker_event_id` | `int64` | — | yes | `event_id` of the taker (aggressing) order's event. Null when not attributed. |
| `maker` | `int64` | — | yes | Order id of the maker. |
| `taker` | `int64` | — | yes | Order id of the taker. |
| `maker_og` | `int64` | — | yes | `original_number` of the maker order. |
| `taker_og` | `int64` | — | yes | `original_number` of the taker order. |

See [Nullable integer columns](#nullable-integer-columns) for how the id columns
materialize on disk.

## depth

Output of `price_level_volume`. Required columns are the
[`DEPTH_COLUMNS`](api/schemas.md) contract; the L3 path adds `event_id`.

| Column | Arrow type | Unit | Null? | Meaning |
|---|---|---|---|---|
| `event_id` | `int64` | — | no | The event that produced this level change. Present on the L3 path; absent on an L2 feed. |
| `timestamp` | `timestamp[ns]` | ns, tz-naive | no | Time of the level change. |
| `price` | `double` | quote currency | no | Price level. |
| `volume` | `double` | base asset / shares | no | Resting size at this price level after the change (`0` empties the level). Never negative. |
| `direction` | `dictionary<string>` | — | no | Side of the level: `bid` or `ask` (ordered categorical). |

`volume` here is the level's absolute resting size, not a signed delta. An L2
loader yields this frame directly with the same four required columns.

## depth_summary (book states)

Output of `depth_metrics`. One row per depth event, holding the reconstructed
best bid and offer plus cumulative resting volume in basis-point bins on each
side. This is the reconstructed book state over time.

| Column | Arrow type | Unit | Null? | Meaning |
|---|---|---|---|---|
| `timestamp` | `timestamp[ns]` | ns, tz-naive | no | Time of this book state. |
| `event_id` | `int64` | — | no* | The event this state follows. Present when the input `depth` carried `event_id` (L3). |
| `best_bid_price` | `double` | quote currency | no | Best bid price (`0.0` when the bid side is empty). |
| `best_bid_vol` | `double` | base asset / shares | no | Resting size at the best bid. |
| `bid_vol{N}bps` | `double` | base asset / shares | no | Resting bid volume within `N` bps of the best bid. |
| `best_ask_price` | `double` | quote currency | no | Best ask price (`0.0` when the ask side is empty). |
| `best_ask_vol` | `double` | base asset / shares | no | Resting size at the best ask. |
| `ask_vol{N}bps` | `double` | base asset / shares | no | Resting ask volume within `N` bps of the best ask. |

The bin columns depend on configuration. With the defaults (`depth_bps = 25`,
`depth_bins = 20`) there are 20 bins per side at `N` = 25, 50, …, 500:
`bid_vol25bps` … `bid_vol500bps` and `ask_vol25bps` … `ask_vol500bps`. Each bin
holds the volume in that bps slice of the book.

\* `event_id` is always present on a standard pipeline run, which feeds L3 depth
carrying it. It is omitted only if you call `depth_metrics` on a depth frame
without an `event_id` column.

## Order lifecycles

Output of `order_lifecycles`. One row per order, collapsing its events into a
placement and an outcome. Orders with no `created` row (a pre-existing opening
book, hidden executions) are excluded.

| Column | Arrow type | Unit | Null? | Meaning |
|---|---|---|---|---|
| `id` | `int64` | — | no | Order id. |
| `placed_ts` | `timestamp[ns]` | ns, tz-naive | no | Time of the `created` event. |
| `placed_vol` | `double` | base asset / shares | no | Size at placement. |
| `price` | `double` | quote currency | no | Placement price. |
| `direction` | `dictionary<string>` | — | no | Order side: `bid` or `ask`. |
| `type` | `dictionary<string>` | — | yes | Classifier label. Present when `events` carried a `type` column. |
| `aggressiveness_bps` | `double` | basis points | yes | Placement distance. Present when `events` carried it. |
| `filled_vol` | `double` | base asset / shares | no | Total executed size (sum of `fill`). |
| `end_ts` | `timestamp[ns]` | ns, tz-naive | yes | Termination time. Null (`NaT`) while the order still rests. |
| `outcome` | `string` | — | no | `filled`, `partial`, `cancelled`, or `resting` (see [Categorical values](#categorical-value-domains)). |

## Book snapshot

Output of `order_book`. A snapshot of the resting book at one point in time,
returned as two frames (`bids` and `asks`), each with these columns.

| Column | Arrow type | Unit | Null? | Meaning |
|---|---|---|---|---|
| `id` | `int64` | — | no | Order id of the resting order. |
| `timestamp` | `timestamp[ns]` | ns, tz-naive | no | Receive time of the order's last event. |
| `exchange_timestamp` | `timestamp[ns]` | ns, tz-naive | no | Exchange time of the order's last event. |
| `price` | `double` | quote currency | no | Resting price. |
| `volume` | `double` | base asset / shares | no | Outstanding resting size. |
| `liquidity` | `double` | base asset / shares | no | Cumulative size from the best price down to this order. |
| `bps` | `double` | basis points | no | Distance from the best price on this side, in bps. |

## Categorical value domains

Three categorical columns are **ordered**; the order is the category order, not
alphabetical.

| Column | Table | Values (in order) |
|---|---|---|
| `direction` | events, depth, book snapshot | `bid`, `ask` |
| `direction` | trades | `buy`, `sell` |
| `action` | events | `created`, `changed`, `deleted` |
| `type` | events, lifecycles | `unknown`, `pre-existing`, `flashed-limit`, `resting-limit`, `market-limit`, `market` |
| `outcome` | lifecycles | `filled`, `partial`, `cancelled`, `resting` (plain string, not categorical) |

An ordered categorical is stored as an Arrow `dictionary<string>` (the categories
plus small integer indices). A reader that does not want dictionary encoding can
cast the column to plain strings.

## Zero-copy reads in Polars and DuckDB

The files are plain Parquet, so any Arrow-aware tool reads them directly, with no
pandas step. The examples below are illustrative. **Polars and DuckDB are not
dependencies of ob-analytics and are not installed by default**; install them
yourself to run these.

### Polars

```python
import polars as pl

# Eager read — one file, Arrow-native, no pandas.
events = pl.read_parquet("out/events.parquet")

# Lazy scan — push filters and column selection into the Parquet reader,
# so only the needed row groups and columns are read.
touch = (
    pl.scan_parquet("out/depth_summary.parquet")
    .select(["timestamp", "best_bid_price", "best_ask_price"])
    .filter(pl.col("best_bid_price") > 0)
    .collect()
)

# The ordered categoricals arrive as Polars Categorical / Enum columns.
```

### DuckDB

```python
import duckdb

# Query the Parquet files in place; DuckDB reads them through Arrow.
spread = duckdb.sql(
    """
    SELECT timestamp,
           best_ask_price - best_bid_price AS spread
    FROM 'out/depth_summary.parquet'
    WHERE best_bid_price > 0
    """
).arrow()  # hand back an Arrow table, zero-copy

# Filter by an ordered categorical value using its string form.
bids = duckdb.sql(
    "SELECT price, volume FROM 'out/depth.parquet' WHERE direction = 'bid'"
).df()
```

## Open decisions

The issue that defined this schema lists four cross-cutting changes to the data
model. Each one **reverses a documented current choice** and touches every
loader, so it is not made here. This section states each precisely — current
state, proposed change, and blast radius — so the maintainer can decide.

### 1. Time model: tz-aware UTC nanoseconds and event order

- **Current.** Timestamps are `timestamp[ns]`, tz-naive, in each venue's native
  clock (UTC for Bitstamp, US/Eastern for LOBSTER). Frames from different venues
  are declared not comparable. `timestamp` is receive time, `exchange_timestamp`
  is matching-engine time. Events that share a timestamp have no defined
  tie-break beyond the loaders' stable sort by arrival.
- **Proposed.** Store tz-aware UTC (`timestamp[ns, tz=UTC]`) so frames from
  different venues sit on one clock, and define a total order for events that
  share a timestamp (for example by `event_id` or a source sequence number).
- **Blast radius.** Every loader's timestamp construction
  (`_utils.epoch_to_datetime`, `seconds_after_midnight_to_datetime`, the
  Bitstamp trade reader's `tz_convert(None)`, the LOBSTER loader), the timestamp
  policy in `schemas.py`, the depth engine's stable-sort assumption, every
  downstream comparison and join, the cross-venue "not comparable" rule, and
  every test that asserts a tz-naive dtype.

### 2. Source sequence numbers

- **Current.** No per-event source sequence number. Order relies on arrival plus
  timestamp, so a dropped or out-of-order message in a diff feed cannot be
  detected. Tracked as issue #146.
- **Proposed.** Carry the venue's sequence number on each event as a new column,
  and compare consecutive numbers to find gaps.
- **Blast radius.** `EVENT_COLUMNS` and its validator, every loader (each must
  parse and emit a sequence number, and many sources do not provide one, which
  raises a nullability question), the `empty_events` template, the writers, and
  the tests. Making the column required ripples to every loader at once.

### 3. Integer-tick prices instead of floats

- **Current.** Price is a `double` in the quote currency. LOBSTER carries a
  `price_divisor` (10 000) to recover its integer ticks, and the depth engine
  multiplies by `price_multiplier` and rounds to an integer internally before
  binning. Float rounding can produce crossed levels and wrong sums for
  small-tick or 0–1 instruments.
- **Proposed.** Store price as an integer tick count (`int64`) plus a `tick_size`
  per instrument, generalizing LOBSTER's `price_divisor`. Convert to float only
  for display.
- **Blast radius.** The `price` column in every table (events, trades, depth,
  depth_summary, book snapshot), all price arithmetic (spread, mid, bps bins,
  VWAP), the depth engine's multiply-and-round path, every loader, the plots, and
  the schema itself (a new `tick_size` field or file metadata). This reverses the
  float choice and touches nearly every module.

### 4. Instrument identity

- **Current.** No instrument or venue id column. One frame holds one instrument
  by convention, and cross-venue frames are declared not comparable. Tracked as
  issue #147.
- **Proposed.** Add a stable instrument id (and possibly a venue id) column so
  events from different venues can be told apart and matched. Needed before
  multi-symbol and cross-venue work.
- **Blast radius.** Every table gains an id column (required, or carried in file
  metadata), every loader must populate it, analytics and the engine must group
  by it, the "not comparable across venues" rule is lifted, join keys change, and
  the file layout question opens (one file per instrument, or one column that
  partitions the data).
