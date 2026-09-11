# Changelog

All notable changes to ob-analytics are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- **Polymarket prediction markets, through the ccxt source** (#103).
  `ob-analytics capture ccxt --exchange polymarket --pair <token id>` streams
  one outcome's order book and trades over Polymarket's public websocket, with
  no account or API key. `--pair` is Polymarket's token id for the outcome,
  which the Gamma API lists as `clobTokenIds`. Each outcome is its own book,
  and its trades are priced in that outcome.

  Polymarket makes a market's tick finer as the price nears 0 or 1, so a ccxt
  capture now makes its recorded tick size finer when a price arrives between
  two ticks, and counts each change in `tick_size_changes` in `meta.json`. The
  replay then reads every price exactly. See the ["Capture Polymarket
  prediction markets"
  how-to](https://mczielinski.github.io/ob-analytics/howto/polymarket/).

- **Kalshi prediction markets, through the ccxt source** (#102).
  `ob-analytics capture ccxt --exchange kalshi --pair <market ticker>` records
  a Kalshi market's order book and trades from Kalshi's public API, with no
  account or API key, and `ob-analytics process` replays it through the L2
  path. The ccxt source now looks up CCXT's prediction markets
  (`ccxt.prediction`: Kalshi, Polymarket and others) as well as its crypto
  exchanges; before, `--exchange kalshi` failed with "Unknown CCXT exchange".
  `binance` and `hyperliquid` are in both lists, so the plain id keeps meaning
  the crypto exchange and `prediction/<id>` picks the prediction market.

  The captured book is the market's Yes book: a bid to buy No at `p` is
  recorded as an offer to sell Yes at `1 - p`, and a trade is priced in Yes
  and signed from the Yes side. A ccxt capture also records its market's tick
  size in `meta.json`, which `process` and `audit` use. See the ["Capture
  Kalshi prediction markets"
  how-to](https://mczielinski.github.io/ob-analytics/howto/kalshi/).

- **A capture records which rows came from its opening snapshot** (#237).
  `orders.csv` and `depth.csv` gain an `origin` column: `snapshot` for the
  opening book, `stream` for a live message, `shutdown` for a synthetic
  close-out. The capture runner fills it in, so every live source gets it
  without a change, and the loaders carry it through to `events`. Before this,
  the only way to tell a snapshot row from a live one was to compare its
  `exchange_timestamp` with `snapshot_microtimestamp` in `meta.json`.

  `meta.json` also reports `n_snapshot_unconfirmed`: how many orders in the
  opening book no later order event or trade mentioned. The bundled Bitstamp
  sample has 6,294 of 6,512. Almost all of them sit far from the touch and did
  not trade, but two stale asks among them held the best ask for most of the
  session. See ["Capture live
  data"](https://mczielinski.github.io/ob-analytics/howto/live-capture/).

- **`audit` names stale resting orders** (#234). A trade above a resting ask,
  or below a resting bid, shows that the order has gone. An order the venue
  then does not report again within one second is now reported as a
  `stale_orders` warning, and the worst one is named with its id, side, price
  and how long it held the touch. On the bundled Bitstamp sample this names
  ask `2002347646152704`, which held the ask touch for 27 minutes and causes
  almost all of the 91.6% crossed time. The crossing note no longer calls a
  diff feed's crossing normal when the run has stale orders. Nothing is
  removed: `order_book()` still replays what the feed said. New public names:
  `detect_stale_orders`, `StaleOrder`, `DataQualitySummary.stale_orders`, and a
  `tick_size=` argument on `data_quality_summary`.

- **A metric registry, so a user metric runs and plots with no core edit**
  (#140). A metric is a plain object with a `name`, a `title`, the `levels` it
  applies to, `compute(result)` and `prepare(frame)` — no base class to
  inherit, the same structural typing sources and writers use. Register it with
  `register_metric(metric)`, or ship it in your own package under the
  `ob_analytics.metrics` entry-point group and `load_metric_plugins()` finds it
  at `import ob_analytics`.

  A registered metric is a level-less plot concept under its own name, so a
  renderer at `(name, None, backend)` is its face. It then appears in
  `available_concepts(result)`, renders through `result.plot(name)`, and gets
  its own gallery card with no `extra_panels=`. Metrics run when asked for, not
  during `Pipeline.run`: `result.metric(name)` computes one and
  `result.metrics()` computes every metric whose `levels` include the run's
  resolution — so an L3-only metric is skipped on an L2 run instead of failing
  on its empty `events` table, and a metric that raises is logged and its card
  dropped, so one broken metric cannot stop the gallery being built. New public
  names: `Metric`,
  `register_metric`, `list_metrics`, `get_metric`, `load_metric_plugins`,
  `PipelineResult.metric` / `.metrics`. See the ["A new metric"
  how-to](https://mczielinski.github.io/ob-analytics/extending/#4-a-new-metric).

- **`ob-analytics audit`, a data-quality gate** (#108). The old `validate` verb
  is now `audit` (the old name still works), it scores the run against named
  checks, and it **exits non-zero when one fails** — so a script can stop before
  trusting a feed. Five checks are new: orphan orders (changed or deleted with
  no `created` row), non-positive prices, negative volumes or fills, and the two
  clock-order defects — a venue timestamp later than the receive timestamp, and
  messages that arrived out of venue order. `audit` also loads with
  `track_sequence` on, so the dropped-message check (#146) reads a venue
  sequence whenever the feed carries one.

  Each check carries a `Severity`: an **error** fails the run, a **warning**
  fails it only under `--strict`, and **info** never does. A crossed resting
  book is scored by feed type, not by size — an error on a matched book, a
  faithful replay on a diff feed. `--json` emits every check plus an `ok`
  verdict; `--from-parquet` audits a saved `process` output without re-running
  the pipeline. New public names: `Severity`, `QualityCheck`, and
  `DataQualitySummary.ok` / `.errors` / `.warnings` / `.checks`. See the
  ["Check data quality with `audit`" how-to](https://mczielinski.github.io/ob-analytics/howto/audit/).

- **`PipelineResult.to_arrow()` and `PipelineResult.to_polars()`** (#104). Both
  return the run's four tables — `events`, `trades`, `depth`, `depth_summary` —
  keyed by name, with the same keys on every run: on an L2 run `events` is an
  empty table, not a missing key. The Arrow tables carry the schema version and
  tick size in their metadata, the same key-value metadata the Parquet files
  carry, so a reader handed tables in memory is no worse off than one reading
  files. Polars is not a dependency and is not installed; `to_polars()` raises
  `ImportError` with an install hint when it is missing, and Polars keeps no
  schema metadata, so the version and tick size do not survive that conversion.
- **The frame-type contract is written down** in
  [Frame types: pandas in, pandas out](https://mczielinski.github.io/ob-analytics/schema/#frame-types-pandas-in-pandas-out):
  public functions take and return pandas, plug-ins are handed pandas, and the
  versioned Parquet is how other tools read the output. The reasoning is in
  `adr/0002-dataframe-library.md`.
- **cryptofeed source for live L2 *and* L3 capture** (`ob-analytics capture
  cryptofeed --exchange <venue> --pair <symbol>`). The per-order complement to
  the CCXT source: venues publishing an order-by-order book record `orders.csv`
  with the venue's own ids and replay through the full reconstruction pipeline;
  the rest record `depth.csv` for the L2 path. The level is discovered from the
  venue's declared channels rather than a hardcoded list, and `--level` forces
  it — except L3 on a venue that publishes none, which raises. Ships as the
  optional `[cryptofeed]` extra, imported lazily. See the new
  "Capture cryptofeed venues" how-to.
- **`sequence` is now written to `orders.csv`.** The capture sink dropped the
  venue sequence on the L3 path, so `detect_sequence_gaps` had nothing to read;
  L2 already kept it. cryptofeed captures also report `sequence_gaps` /
  `sequence_missing` in `meta.json`.

- **L2 (price-level) depth-native ingestion path.** Price-level feeds
  (Binance, Kalshi, Polymarket, most CCXT sources) publish `[price, quantity]`
  levels and diffs with no order IDs; ob-analytics now ingests them as a
  first-class **L2** resolution instead of faking per-order state. A format
  declares its `resolution` (`Level.L2` / `Level.L3`, exposed as
  `ob_analytics.Level`); an L2 format's loader is a `DepthSource` that yields
  the depth frame directly, and `Pipeline.run` takes the price-level path —
  depth metrics / spread and trade-sign classification run, while the per-order
  stages (`set_order_types`, `order_aggressiveness`, queue reconstruction) are
  skipped. `PipelineResult` gains a `resolution` field and, on an L2 run,
  returns an empty (schema-valid) `events` frame. Ships the `depth_csv` format
  (`L2DepthLoader`, `L2TradeReader`, `DepthCsvWriter`, `DepthCsvFormat`) for the
  canonical L2 CSV schema, a `toy_l2_depth()` / `toy_l2_trades()` synthetic
  snapshot+delta fixture, and `data_quality_summary` + the gallery degrade
  gracefully (L3-only faces skipped, not errored). `ob-analytics
  process|validate --format depth_csv` works from the CLI. Unblocks the
  aggregated venue connectors. Documented in a new "Process L2 feeds" how-to.
- **Trade-sign classification** (`ob_analytics.trade_sign`) for feeds that
  don't label the aggressor side. `tick_rule` (last-price-change sign),
  `lee_ready` (quote-midpoint test with a tick-rule fallback), and
  `bulk_volume_classification` (BVC — the buy fraction of a volume bar via the
  standardized-price-change normal CDF). `classify_trade_sign(trades,
  method=..., quotes=...)` is the per-trade entry point. `compute_vpin` and
  `order_flow_imbalance` gain `sign_method` / `quotes` arguments and now
  synthesize `direction` automatically when the trades frame has none — so
  VPIN and OFI run on L2 / aggregated captures, not just L3. A native
  `direction` is still honored unchanged (`sign_method=None`). On the bundled
  Bitstamp L3 sample the classifiers agree with the true maker/taker side
  ~0.83 (tick) / ~0.79 (Lee–Ready) — validated by a test harness.
- **Feed classification.** Every format declares a `FeedType`
  (`matched_book` vs `diff_feed`) through a `feed_type` attribute —
  `BitstampFormat` → `diff_feed`, `LobsterFormat` → `matched_book` — so
  downstream code reasons about crossed books by coordinate, not by format
  name. Exposed as `ob_analytics.FeedType`.
- **`order_book(..., uncross=True)`** evicts crossed resting orders *for
  display*, mirroring the depth engine's crossed-level eviction. The default
  stays faithful, so a diff feed's genuinely crossed resting orders are
  replayed as-is. Threaded through `prepare.book_snapshot(..., uncross=True)`
  (also drives `depth_chart`) and available frame-level as
  `analytics.uncross_book_sides`.
- **Per-run data-quality summary.** `data_quality_summary()` and the new
  `ob-analytics validate <source>` CLI verb report the crossed-resting %,
  unmatched-trades %, duplicate ids, and pre-existing-order count. A new
  "Data quality: matched book vs diff feed" explanation page and a `validate`
  how-to document the distinction.

### Fixed

- **A price-level file no longer has its prices rounded to the tick size.**
  `L2DepthLoader` and `L2TradeReader` converted each price to the nearest
  whole number of ticks, so a price finer than `tick_size` moved without a
  warning: a Kalshi price of 0.036 loaded as 0.04 at the default 0.01 tick,
  and the most traded Kalshi markets quote in tenths of a cent. Both now raise
  `ConfigError` when a price is not a whole number of ticks, and say which
  tick size to set. A ccxt capture records its market's tick size in
  `meta.json`, and `ob-analytics process` and `ob-analytics audit` read it
  from there (`recorded_tick_size`), so a CLI replay needs no extra option.

- **A streamed ccxt capture no longer writes a trade twice.** ccxt's
  Polymarket websocket handed back a trade it had already delivered, together
  with the next new one, and only a polled capture skipped repeats. Every ccxt
  capture now skips a trade identical to one it has written: same id, time,
  price, size and side. Two fills that share a Polymarket id (the settling
  transaction) are both kept. `meta.json` counts the skipped repeats in
  `duplicate_trades`.

- **A polled ccxt capture no longer records trades from before it started.**
  On a venue without websockets, the first poll of the trade tape returns the
  venue's recent history, which on Kalshi reached back nine hours. Those trades
  were written with the capture's receive time, as if they had just happened.
  The capture now drops trades older than its opening book.

- **A Bitstamp capture no longer starts from a snapshot older than its stream**
  (#237). The capturer subscribes to the WebSocket, then fetches the REST book.
  It assumed the stream already covered the moment the book describes, but it
  often does not: in a live test the first order message came 0.7 s after the
  snapshot's `microtimestamp`, and the bundled sample shows the same 0.74 s gap.
  An order deleted in that gap stayed in the capture until the synthetic
  `deleted` at shutdown. In the bundled sample, one such ask was the best ask for
  89% of the session.

  The capturer now fetches the book again, a second apart and up to 10 times,
  until some buffered order message is at or before the snapshot's
  `microtimestamp`. `meta.json` gains `snapshot_fetches` and
  `snapshot_overlap`. In the live test, the second fetch no longer listed any
  of the 15 orders that were gone. Eight of those were orders that trades
  printed through.

- **A price level now empties when the order resting on it goes away.**
  `price_level_volume` added an order's volume at the price on its `created`
  row and subtracted it at the price on whichever later row removed it. Those
  two prices are not always the same: Bitstamp reports a `deleted` carrying a
  price the order never rested at for 1.3% of orders, and the subtraction then
  landed on a level the volume was never added to, leaving the created level
  holding it for the rest of the session. Every later row now subtracts at the
  order's created price, so `+v` and `-v` always cancel on one level.

  On the bundled Bitstamp sample this removed 104 price levels holding 29.95
  BTC that no order was resting on. They were the reported touch on both sides
  — best bid $78,495.00 against a real best bid of $78,350.00, and best ask
  $78,324.00 against a real best ask of $78,333.00 — so `best_bid_price` moves
  on 35.7% of `depth_summary` rows and `best_ask_vol` on 68.9%. The per-order
  rebuild (`engine.book_state`) tracks orders by id and never had this problem;
  the two rebuilds now agree on how long that book is crossed.

- **`aggressiveness_bps` is NaN, not an infinity, against a zero touch.** The
  depth engine reports a zero price for an empty side, and the Bitstamp sample
  also carries orders priced at zero (`audit` reports these as
  `nonpositive_price`). Dividing by that produced a signed infinity that
  travelled through every downstream mean. A distance from a price that is not
  tradeable has no value, so it is now NaN.

### Changed

- **Sizes are integer lots plus a `lot_size`, not floats** (issue #226).
  **Breaking: the on-disk schema goes 3.0 → 4.0.** Every `volume` and `fill`
  column is now a whole number of lots (`int64`) instead of a `double` in the
  base asset. The base-asset size is `lots * lot_size`, where `lot_size` is the
  instrument's minimum size increment (`PipelineConfig.lot_size`, default
  `1e-8`; LOBSTER sets `1`, whole shares). This is the size half of the
  integer-tick decision (issue #155) and it fixes a real defect rather than
  only re-expressing the data.

  A price level is a running sum of adds, cancels and fills. A float sum does
  not return to exactly zero when the last order leaves, so a level landed on
  residue such as `5.55e-17`, stayed live, and was reported as the best bid or
  ask ahead of the real one. On the bundled Bitstamp sample that corrupted the
  reported best bid on 25,611 of 313,565 rows (8.2%) and the best ask on 30,096
  (9.6%) — the spread on about one row in eleven. Integer lots cancel exactly,
  so a level empties or it does not, and those counts are now zero.

  It was found by the new cross-check against hftbacktest (issue #224), and
  that is what confirms the fix: replaying an exported session through
  hftbacktest's own L3 reconstruction now agrees with `depth_summary` on the
  best bid and ask for every row across five synthetic seeds, and Nautilus'
  book agrees too. Before the fix the two disagreed on up to 78 rows a seed.

  The change reaches every size-valued column — `depth_summary`'s per-bin
  volumes, `placed_vol` and `filled_vol`, the book snapshot's `liquidity`, and
  the queue's `ahead_volume` and `remaining` — so their sums are exact as well.
  Three float-era workarounds went with it: the Kahan compensation behind
  `filled_vol`, the simulator's `_vol_eps` exhaustion tolerance, and the
  LOBSTER book replay's `1e-12` level cutoff. Loaders convert on the way in;
  the plots and the round-trip and export writers convert back, so what a user
  sees and what another tool reads are unchanged. `lot_size` travels in each
  Parquet file's key-value metadata under `ob_analytics_lot_size`, next to
  `ob_analytics_tick_size`, and `load_data` surfaces it as
  `df.attrs["lot_size"]`. Files written at `1.0`–`3.0` still read, as the
  float-size frames they are. Golden outputs were re-baselined on purpose.

- **The export writers leave out orders that never rested** (issue #224). A
  marketable order is recorded as a transient add on its own side at the touch,
  then the fill, then a delete; `ob_analytics.depth` has always excluded these
  from the book, but the hftbacktest and Nautilus writers were sending them.
  A backtesting engine reads an add as real liquidity, so its book crossed at
  the touch and dropped the resting level the order traded against — its
  reconstruction drifted permanently thinner than ours. Both writers now
  exclude them, which is what makes the two books agree.

- **The order-book engine is its own module** (issue #136). The rebuild
  (`order_book`), the per-order lifecycles, and the FIFO queue reconstruction
  moved out of `analytics.py` / `queue.py` into `ob_analytics/engine/`, behind
  one input and one output: order events in, book states and order lifecycles
  out, and nothing else. The engine **imports no pandas** — everything crosses
  its interface as NumPy arrays, with the shared schema (issue #112) as the
  input, timestamps as int64 UTC nanoseconds (issue #154) and prices as integer
  ticks (issue #155). Results carry a *row index* back into the caller's event
  arrays instead of copying columns out, so adding a column to the schema does
  not widen the interface and the engine never learns a vocabulary — order
  types, venue names — belonging to the layer above.
  `ob_analytics/_engine_frames.py` is the one place pandas and the engine meet;
  `analytics.order_book`, `analytics.order_lifecycles`, and the
  `ob_analytics.queue` functions are now its frame adapters and keep their exact
  signatures, dtypes, column order, and index behaviour. Output is **unchanged
  byte for byte** — the golden-output gates from issue #143 pass on their
  recorded fingerprints. Two things did move: the display window (`max_levels`,
  `bps_range`) and the queue sampling window are set by the frame adapters
  rather than the engine, which reconstructs the whole book and replays to the
  instants it is given. A new import test (`tests/test_engine_boundary.py`)
  keeps the engine free of pandas and of every layer above it. This is what lets
  the inside be replaced with a faster implementation (#138) or fed one event at
  a time (#139) without touching anything else. `Direction`, `Action`, and
  `Outcome` are `IntEnum` code vocabularies that derive their schema strings
  from their own member names, so a code and its label cannot drift apart.
  Two details of the frame code are reproduced deliberately rather than
  rewritten: an order's executed total is accumulated with compensated (Kahan)
  summation, as the pandas aggregation it replaced did, and placement values are
  taken per column as the first non-null among an order's `created` rows. The
  lifecycle table is now covered by `tests/test_golden_synth.py`, which it was
  not before.

- **`BitstampTradeReader` no longer requires integer order ids.** It keyed its
  maker/taker lookup on `int(order_id)`, which crashed on a public trade tape
  carrying no ids (`int(NaN)`) and on venues publishing UUIDs. Integer ids
  behave exactly as before; other ids match on their string form, and a missing
  id resolves to `NaN` instead of raising.

- **One `Source` shape for every data source, file or live** (issue #137;
  settled #145 as "optional extras plus entry-point plug-ins"). File loaders
  and live capturers were two separate designs with two registries; they are
  now one `Source` protocol with two capability refinements — `OfflineSource`
  (replay stored files: the loader / trade-source / writer / depth factories)
  and `LiveSource` (capture a venue: `snapshot` / `stream` /
  `shutdown_synthetic_events`). A source states its `level` (L2/L3) and
  `feed_type`, carries typed `settings`, and registers in the single `SOURCES`
  registry via `register_source`. A source can be both: `BitstampSource` now
  covers offline replay and live capture in one descriptor. This is a breaking
  API change with no back-compat shims:
    - `Format` → `OfflineSource`; `LiveCapturer` → `LiveSource`;
      `BitstampFormat` / `LobsterFormat` / `DepthCsvFormat` →
      `BitstampSource` / `LobsterSource` / `DepthCsvSource`;
      `CcxtCapturer` → `CcxtSource`.
    - `Pipeline(format=...)` → `Pipeline(source=...)`; `Pipeline.from_format`
      → `Pipeline.from_source`.
    - The `FORMATS` / `CAPTURERS` registries and their `register_format` /
      `register_capturer` / `list_formats` / `list_capturers` /
      `get_capturer` helpers are replaced by `SOURCES` / `register_source` /
      `list_sources` / `get_source` (in `ob_analytics.sources`).
    - `PipelineResult.resolution` → `PipelineResult.level` (one coordinate
      name across the codebase; `Source.level`, matching the visualization
      layer).
    - `CaptureConfig.extras` (the untyped settings dict) is removed. Per-source
      settings are now typed `SourceSettings` on the source itself, e.g.
      `CcxtSource(settings=CcxtSettings(exchange="binance", depth_limit=100))`.
    - CLI: `process` / `validate` take `--source` (was `--format`), and the
      `formats` verb is now `sources` (it also shows each source's capability
      and required context).
- **Third-party sources load through entry points.** A source can ship in its
  own package and advertise itself under the `ob_analytics.sources`
  entry-point group; `ob_analytics.sources.load_source_plugins()` discovers and
  registers it at import time, with no edit to ob-analytics. The built-in
  sources (bitstamp, lobster, depth_csv, ccxt) self-register on import and stay
  behind today's `[live]` / `[ccxt]` extras.
- **Prices are now integer ticks, not floats** (issue #155). Every `price`
  column — events, trades, depth, depth_summary, book snapshot, and order
  lifecycles — is a whole number of ticks (`int64`); the quote-currency price is
  `ticks * tick_size`, where `tick_size` is the instrument's minimum price
  increment (`PipelineConfig.tick_size`, default `0.01`). Loaders convert a raw
  price to ticks on load; the plots and the round-trip writers convert back for
  display, so figures and CSV output are unchanged. Storing the exact integer
  removes the float rounding that made small-tick and 0-1 instruments show
  crossed levels that were not real, and the depth engine now bins and compares
  levels on exact integers instead of multiplying and rounding each event —
  LOBSTER's `price_divisor` is now just the raw-feed encoding scale.
  `tick_size` is written to each Parquet file's `ob_analytics_tick_size`
  key-value metadata (a JSON map keyed by instrument, ready for per-`(venue,
  symbol)` ticks in #147) and surfaced on `load_data` frames' `attrs`.
  **Breaking:** the dtype of every `price` column changed from `double` to
  `int64` and the stored numbers changed (prices re-expressed as ticks;
  price-valued analytics such as `trade_impacts` VWAP and Kyle's λ are now in
  tick units — multiply by `tick_size` for the quote currency; scale-free
  metrics such as bps depth and order-book imbalance are unchanged). The
  canonical Parquet **schema version is now `3.0`** (a `1.0` / `2.0` file still
  reads — Parquet is self-describing — as the float-price frame it stored, whose
  prices are not directly comparable to a `3.0` file's ticks; re-save it to move
  it onto the tick model). Golden-output baselines were re-recorded behind the
  correctness gate (#143).
- **Timestamps are now tz-aware UTC nanoseconds** (`timestamp[ns, tz=UTC]`) on
  both clocks — `timestamp` (receive) and `exchange_timestamp` (matching
  engine) — across every table, loader, the synthetic generator, and the toy
  datasets (issue #154). Before, they were tz-naive and in each venue's native
  clock (millisecond-resolution UTC for Bitstamp, US/Eastern for LOBSTER), and
  frames from different venues were declared not comparable. Now every frame sits
  on one UTC clock, so cross-venue frames can be joined or concatenated directly.
  LOBSTER's seconds-after-midnight are converted to UTC from the session date and
  a venue time zone (`RunContext(session_tz=...)`, default `America/New_York`);
  Bitstamp / CCXT keep their wall-clock instants and only gain the zone and the
  nanosecond unit, so their values do not move. The schema also documents a
  same-instant **total order** — `timestamp`, then `sequence`, then `event_id`,
  then `ingest_seq` (`ob_analytics.schemas.time_order_keys`), which the per-order
  reconstructions sort by. **Breaking:** the dtype of every timestamp column
  changed, so the canonical Parquet **schema version is now `2.0`** (a `1.0` file
  still reads — Parquet is self-describing — as the tz-naive frame it stored;
  re-save it to move it onto the UTC clock). Consumers that compared pipeline
  timestamps against tz-naive `pandas.Timestamp`s must now use tz-aware (UTC)
  ones.

### Fixed

- **Order lifecycles read every filled order as cancelled when sizes were
  floats** (#226 regression). `order_lifecycles` summed each order's fills and
  cast the total to `int64`. On integer lots that is exact, but the function
  also accepts base-asset floats, and it is handed them on every gallery run:
  `display_result` converts a whole result to display units before any face
  builds. Base-asset sizes are mostly below 1, so a 0.121 BTC fill truncated to
  `0`, the order read as never executed, and the three lifecycle-derived L3
  faces — **Order Activity**, **Order Outcome** and **Queue Position** — drew a
  book of nothing but cancellations. On the bundled Bitstamp sample the Order
  Activity face lost 224 of its 226 filled spans. The sum now keeps the units it
  was given, integer lots summing exactly and base-asset floats with the
  compensation that was dropped as part of #226.

  LOBSTER was never affected: its lot size is 1, so a truncated size equals the
  size. Every LOBSTER face is pixel-identical across the change.

- **LOBSTER's `fill` column was `float64`, not integer lots** (#226). A `0.0`
  literal in the expression that built it widened the whole column, so a
  schema-4.0 LOBSTER run wrote base-asset-looking floats that were really lot
  counts. Nothing raised; the values only differ from the correct ones once the
  lot size is not 1.

---

## [0.1.0] - 2026-06-26

First public release (PyPI). The Python port of the R
[obAnalytics](https://cran.r-project.org/package=obAnalytics) package, reworked
into a pipeline API with pluggable formats, flow-toxicity metrics, L2/L3
visualization, and Matplotlib/Plotly backends — plus the packaging,
documentation, and distribution that make it installable. The sections below also
record how the API was deliberately de-bloated and unified during the port (the
pipeline's numeric output is unchanged — the regression fingerprints pass; only
the *shape* of the public API moved). See
[Extending ob-analytics](https://github.com/mczielinski/ob-analytics/blob/main/docs/extending.md).

### Packaging & distribution

- The bundled Bitstamp sample ships gzip-compressed (`orders.csv.gz`,
  ~23 MB → ~2.9 MB installed); `sample_csv_path()` returns the `.gz` path, read
  transparently by pandas. No API change.
- Published documentation site (GitHub Pages), `CITATION.cff`, an explicit
  GPL-2.0-or-later license section, and a "Scale envelope" doc.
- PyPI release workflow (`release.yml`, trusted publishing), package classifiers
  and project URLs, and `ob_analytics.__version__`.
- Fixed quickstart and API-reference documentation drift.

### Breaking

- **Pydantic models removed.** `ob_analytics.models` (`OrderEvent`, `Trade`,
  `DepthLevel`, `OrderBookSnapshot`) deleted; the data contract is now
  column-list constants + `validate_events_df` / `validate_trades_df` /
  `validate_depth_df` in `ob_analytics.schemas`.
- **`metrics/` package removed.** `ToxicityMetric`, `Vpin`, `Ofi`,
  `KyleLambda`, `register_metric`, and `list_metrics` are gone. Call
  `compute_vpin`, `compute_kyle_lambda`, and `order_flow_imbalance` on
  `result.trades` directly.
- **`Pipeline(metrics=...)` removed.** Metrics are no longer a pipeline stage —
  compute them after the run.
- **`PipelineConfig.vpin_bucket_volume` removed** — pass `bucket_volume=` to
  `compute_vpin`.
- **`PipelineResult` slimmed** to exactly `events`, `trades`, `depth`,
  `depth_summary`, and `config`. The `vpin`, `ofi`, `metrics`, `metadata`, and
  `extras` attributes are gone.
- **The thirteen `plot_*` wrappers removed** → one
  `plot(name, *, backend="matplotlib", ax=None, **data)` dispatcher keyed by
  `(plot_name, backend)`; renderers self-register into `RENDERERS`.
- **Global theme state removed.** `set_plot_theme` / `get_plot_theme` /
  `_current_theme` deleted; pass `theme=PlotTheme(...)` to `plot()`.
- **Exception hierarchy collapsed** to `ObAnalyticsError` + `ConfigError`.
  `InvalidDataError`, `MatchingError`, `InsufficientDataError`, and
  `ConfigurationError` are removed.
- **Top-level `__all__` trimmed** to ~22 orchestration names. Low-level helpers
  now import from their submodules — `ob_analytics.bitstamp`,
  `ob_analytics.lobster`, `ob_analytics.analytics`, `ob_analytics.depth`,
  `ob_analytics.data`, `ob_analytics.visualization`, `ob_analytics.flow_toxicity`.
- **`Format` is now a `typing.Protocol`** — there is no base class to inherit;
  any conforming object is recognised structurally.
- **Low-level helpers no longer re-exported from the package root** (e.g.
  `depth_metrics` is now `from ob_analytics.depth import depth_metrics`).
- **`RunContext.extras` and `Format.collect_extras` removed.** LOBSTER trading
  halts are read from `LobsterLoader.trading_halts` and composed into the
  gallery via `extra_panels=`.
- **`DepthMetricsEngine.update()` removed** → the public hot-path method is
  `update_side(price, volume, side, out)`.

### Added

- **`ob_analytics.schemas`** — the single data contract: column-list constants
  (`EVENT_COLUMNS`, `TRADE_COLUMNS`, `DEPTH_COLUMNS`) plus the `validate_*`
  functions, run at the pipeline's Protocol boundaries. Replaces the Pydantic
  model layer.
- **One generic `Registry[K, V]`** (`ob_analytics._registry`) backs the format,
  writer, capturer, and renderer registries. Register through the public
  helpers `register_format`, `register_writer`, `register_capturer`, and
  `RENDERERS.register` / `register_plot_backend`.
- **Unified `plot()` dispatcher** + `RENDERERS` registry keyed by
  `(plot_name, backend)`, so new plots and backends plug in without a wrapper
  function. The HTML gallery composes custom panels via `extra_panels=`.
- **`ob_analytics.live`** — optional sub-package for live order-book capture:
  the `LiveCapturer` protocol (with an optional `SupportsDiagnostics`
  capability), `CaptureConfig`, `CaptureResult`, `CaptureSink`, and a generic
  asyncio runner. Capture output drops straight into the pipeline (`orders.csv`
  schema unchanged). Install with `pip install "ob-analytics[live]"`.
- **`ob-analytics capture <venue>`** CLI verb with a built-in `bitstamp`
  capturer (`ob_analytics/live/bitstamp.py`); `--list` shows registered
  capturers. `scripts/collect_bitstamp_btcusd.py` is now a thin wrapper around
  it.
- **`TradeSource` protocol** and **`BitstampTradeReader`** — read an
  authoritative companion `trades.csv` and join it to events via the `fill`
  column.
- **`RunContext`** dataclass (`ob_analytics.protocols`, re-exported at the top
  level) for per-run parameters such as LOBSTER `trading_date` that don't
  belong on long-lived `Format` instances.
- **Docs** — `docs/extending.md` (add a data source / writer / plot / metric /
  capturer).
- **Tests** — `test_bitstamp.py`, `test_cli.py` (subprocess smoke tests for all
  CLI subcommands), `test_exceptions.py`, `test_data_registry.py`, a regression
  snapshot suite pinning demo Parquet hashes + the Kyle-λ baseline, and
  `ob_analytics/__main__.py` (`python -m ob_analytics`).

### Changed

- **Bundled sample** — `ob_analytics/_sample_data/` now ships `orders.csv` and
  `trades.csv` from a modern BTC/USD live capture (replaces the legacy 2015
  orders-only slice).
- **Demos consolidated** into `ob_analytics._demos`; `scripts/bitstamp_demo.py`,
  `scripts/lobster_demo.py`, and the `bitstamp-demo` / `lobster-demo` CLI
  subcommands are now thin argparse wrappers. Behaviour unchanged.
- **Performance** — the LOBSTER book is maintained as a `SortedDict` (no
  per-event re-sort), Bitstamp trade→event resolution is indexed, LOBSTER depth
  uses a single strategy, the Plotly import is memoised, and depth metrics sum
  active levels into bps bins. Numeric output is unchanged (pinned by the
  regression snapshots).
- `compute_kyle_lambda` computes its OLS via `np.linalg.lstsq` (was hand-rolled;
  agrees with the prior implementation to `rtol=1e-10`).
- **Internal modules reorganized** (renames from the 0.x line): e.g.
  `event_processing.py` → `bitstamp.py`, validation/time helpers → `_utils.py`,
  and the visualization modules split into a `visualization/` subpackage.
- Type checking is Astral's `ty` (not mypy); lint and format are Ruff.

### Removed

- **`pacman` order type.** A legacy artifact of the 2015 Bitstamp HTTP API,
  where a single `order_id` could appear at multiple prices over its lifetime.
  Modern Bitstamp WS v2 and LOBSTER do not produce this pattern (price-modifies
  become cancel + new id). The `type` Categorical no longer includes
  `"pacman"`, the set-subtraction classification path is gone, and
  `LobsterLoader` no longer renumbers hidden-execution ids (raw type 5 now
  retains the native LOBSTER `id=0`).
- **Bitstamp trade inference.** A companion `trades.csv` next to `orders.csv` is
  now required. Removed: Needleman–Wunsch matching, `BitstampMatcher`,
  `BitstampTradeInferrer`, the `MatchingEngine` / `TradeInferrer` protocols,
  `NeedlemanWunschMatcher`, and the `match_cutoff_ms` / `price_jump_threshold`
  fields on `PipelineConfig`.
- **Zombie detection** — `get_zombie_ids` and the `zombie_offset_seconds` /
  `skip_zombie_detection` config fields.
- **LOBSTER `LobsterMatcher`** — removed; `LobsterTradeInferrer` renamed to
  `LobsterTradeReader` with `load(events, source)`.
- Legacy Bitstamp-only wrappers `load_event_data`, `event_match`,
  `match_trades`, `process_data`, and `plot_price_levels_faster`.
- 12 unused runtime dependencies (scikit-learn, scipy, jupyter, bokeh, …) and
  stale dev dependencies (black, flake8 + plugins, darglint).

### Fixed

- `depth_metrics` no longer overflows for prices > $9,999.99 — dynamic
  `dict[int, int]` state replaces the fixed array.
- `best_bid` / `best_ask` are tracked correctly from the first event (were
  initialised with dataset-wide max/min).
- `datetime_to_epoch` uses `.astype("int64")` instead of the deprecated
  `.view("int64")`.
- All `print()` replaced with `loguru` logging; all bare `assert` statements
  replaced with raised exceptions; `plt.show()` removed from plot functions
  (callers control display).
