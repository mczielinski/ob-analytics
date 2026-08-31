# Changelog

All notable changes to ob-analytics are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

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

### Changed

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
