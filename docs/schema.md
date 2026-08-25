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

The current version is **`2.0`** (the constant
[`ob_analytics.schemas.SCHEMA_VERSION`](api/schemas.md)). Version `2.0`
(issue #154) makes both timestamp clocks tz-aware UTC nanoseconds; `1.0` wrote
them tz-naive in each venue's native clock. Both still read — Parquet is
self-describing, so a `1.0` file loads as the tz-naive frame it stored; re-save
it with this build to move it onto the UTC clock.

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
version = metadata[b"ob_analytics_schema_version"].decode()  # "2.0"
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

All timestamps are `timestamp[ns, tz=UTC]` — **tz-aware UTC nanoseconds** (int64
nanoseconds since the Unix epoch, with the UTC zone attached). Both clocks use
this one type, so frames from different venues sit on the same clock and **are**
comparable: you can join or concatenate them directly. A source in a
venue-local clock (LOBSTER, US/Eastern) is converted to UTC on load, using the
session date and the venue time zone; a source already in epoch-UTC (Bitstamp,
CCXT) keeps its wall-clock instants and only gains the zone and the nanosecond
unit.

`timestamp` is the local receive time. `exchange_timestamp` is the venue's
matching-engine time. For LOBSTER the two are equal, because only exchange time
exists there.

### Same-instant order

A timestamp alone cannot order two events that share an instant, so the schema
defines a **total order**: `timestamp`, then — as tie-breaks, when the frame
carries them — the venue `sequence`, then `event_id` (the dense per-order key on
the L3 path), then the local `ingest_seq`.
[`ob_analytics.schemas.time_order_keys`](api/schemas.md) returns this key list
for a frame; the per-order reconstructions (`queue_positions`) sort by it, and
every loader builds its frame with fixed, stable sorts, so a rebuild is
deterministic run-to-run. On a price-level (L2) feed there is no `event_id`;
ties there fall to `sequence` / `ingest_seq` when tracked, otherwise to the
loader's stable arrival order. Enforcing the full key inside the price-level
depth engine — so an alternate engine reproduces it bit-for-bit — lands with the
engine separation and rewrite (#136 / #104 / #138), when it can be checked
against that second backend.

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
[data-model decisions](#data-model-decisions) below.

## events

Required columns (the [`EVENT_COLUMNS`](api/schemas.md) contract) plus the
provenance columns every loader carries.

| Column | Arrow type | Unit | Null? | Meaning |
|---|---|---|---|---|
| `event_id` | `int64` | — | no | Unique id for this event (1-based). Join key for a trade's `maker_event_id` / `taker_event_id`. |
| `id` | `int64` | — | no | Order id. Groups the events of one order over its life. |
| `timestamp` | `timestamp[ns, tz=UTC]` | ns, UTC | no | Local receive time. |
| `exchange_timestamp` | `timestamp[ns, tz=UTC]` | ns, UTC | no | Venue matching-engine time (equals `timestamp` for LOBSTER). |
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
| `timestamp` | `timestamp[ns, tz=UTC]` | ns, UTC | no | Trade print time (receive clock). |
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
| `timestamp` | `timestamp[ns, tz=UTC]` | ns, UTC | no | Time of the level change. |
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
| `timestamp` | `timestamp[ns, tz=UTC]` | ns, UTC | no | Time of this book state. |
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
| `placed_ts` | `timestamp[ns, tz=UTC]` | ns, UTC | no | Time of the `created` event. |
| `placed_vol` | `double` | base asset / shares | no | Size at placement. |
| `price` | `double` | quote currency | no | Placement price. |
| `direction` | `dictionary<string>` | — | no | Order side: `bid` or `ask`. |
| `type` | `dictionary<string>` | — | yes | Classifier label. Present when `events` carried a `type` column. |
| `aggressiveness_bps` | `double` | basis points | yes | Placement distance. Present when `events` carried it. |
| `filled_vol` | `double` | base asset / shares | no | Total executed size (sum of `fill`). |
| `end_ts` | `timestamp[ns, tz=UTC]` | ns, UTC | yes | Termination time. Null (`NaT`) while the order still rests. |
| `outcome` | `string` | — | no | `filled`, `partial`, `cancelled`, or `resting` (see [Categorical values](#categorical-value-domains)). |

## Book snapshot

Output of `order_book`. A snapshot of the resting book at one point in time,
returned as two frames (`bids` and `asks`), each with these columns.

| Column | Arrow type | Unit | Null? | Meaning |
|---|---|---|---|---|
| `id` | `int64` | — | no | Order id of the resting order. |
| `timestamp` | `timestamp[ns, tz=UTC]` | ns, UTC | no | Receive time of the order's last event. |
| `exchange_timestamp` | `timestamp[ns, tz=UTC]` | ns, UTC | no | Exchange time of the order's last event. |
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

## Data-model decisions

The four cross-cutting choices the schema left open now have a decided direction
(2026-08-23): **follow the Databento / Nautilus conventions where they are the
standard, expressed in Arrow / Parquet, not the DBN binary format.** Sequence
numbers (#146), instrument identity (#147), and the time model (#154) are
implemented. Integer-tick prices (#155) change existing numbers, so they land
next, on their own, behind the correctness gate (#143). See the roadmap (#124).

### 1. Time model: tz-aware UTC nanoseconds and event order — implemented (#154)

- **Decision, built here.** Both clocks — `timestamp` (receive) and
  `exchange_timestamp` (matching engine) — are `timestamp[ns, tz=UTC]`: int64
  nanoseconds since the Unix epoch, with the UTC zone attached (see
  [Timestamp policy](#timestamp-policy)). A venue-local source (LOBSTER,
  US/Eastern) is converted to UTC on load from the session date and venue time
  zone; an epoch-UTC source (Bitstamp, CCXT) keeps its wall-clock instants and
  only gains the zone and the nanosecond unit. The same-instant total order is
  `timestamp`, then the venue `sequence`, then `event_id`, then `ingest_seq`
  (`ob_analytics.schemas.time_order_keys`).
- **Comparable across venues.** Because every frame is on one UTC clock, frames
  from different venues can be joined or concatenated directly — the earlier
  "not comparable across venues" rule is gone.
- **Deferred.** Enforcing the full same-instant key inside the price-level depth
  engine (so an alternate backend reproduces the rebuild bit-for-bit) lands with
  the engine separation and rewrite (#136 / #104 / #138), validated against that
  second backend. Today the per-order reconstructions sort by the total order and
  the engine plays events back in a stable receive-clock order, deterministic
  run-to-run.

### 2. Source sequence numbers — implemented (#146)

- **Decision, built here.** Two optional columns: **`sequence`** (nullable
  `Int64`, the venue's per-event sequence number, populated where a source
  provides one — the CCXT `nonce`, or an optional column in a Bitstamp / L2
  capture CSV) and **`ingest_seq`** (`int64`, a local monotonic counter in arrival
  order, the stable replay key). `detect_sequence_gaps()` reports dropped and
  out-of-order messages, surfaced in `DataQualitySummary` and the `validate`
  command.
- **Non-breaking.** The loader-attached columns are gated behind a default-off
  `PipelineConfig.track_sequence` flag, so existing frames are byte-for-byte
  unchanged.
- **Later.** Live re-sync on a detected gap (refetch a snapshot) is the
  non-additive half and is not built yet.

### 3. Integer-tick prices instead of floats — decided, not yet built (#155)

- **Decision.** Store price as an integer tick count (`int64`) plus a
  **per-instrument `tick_size`** — not float, and not one global scale. Convert to
  a float only for display. This matches the depth engine's internal integer
  binning and the backtester-export targets (#113).
- **Current.** Price is a `double`. LOBSTER carries a `price_divisor` (10000); the
  depth engine multiplies by `price_multiplier` and rounds to an integer
  internally before binning.
- **Why later.** This is the deepest change — the `price` column in every table,
  all price arithmetic, the engine's multiply-and-round path, every loader, and
  the plots. It changes numbers on purpose, so it re-baselines golden output and
  waits for the correctness gate (#143).

### 4. Instrument identity — implemented, first stage (#147)

- **Decision, built here.** Readable per-row **`symbol`** and **`venue`** columns
  (nullable `string`, dictionary-encoded) — not Databento's numeric id plus a
  definitions join, which is built for a far larger feed than this project has.
  Each loader tags its frame when identity is supplied; `venue` defaults to the
  source name (`bitstamp`, `lobster`, the CCXT exchange id).
  `group_by_instrument()` splits a frame by whichever of `(venue, symbol)` are
  present.
- **Non-breaking.** Identity is opt-in, so a default single-instrument run adds no
  columns and existing output is unchanged.
- **Later.** The full multi-symbol, cross-venue pipeline (one run over several
  instruments) is the larger part of #147 and is not built yet.
