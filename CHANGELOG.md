# Changelog

All notable changes to ob-analytics are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Breaking

- **`Format` API:** `create_loader`, `create_trade_source`, `create_writer`,
  and `compute_depth` now take a second positional argument `ctx: RunContext`.
  New optional method `collect_extras(loader, events, source, ctx) -> dict`
  contributes per-format auxiliary DataFrames to `PipelineResult.extras`.
- **`LobsterFormat`:** `trading_date` moved from the constructor to
  `RunContext`. Use
  `Pipeline(format=LobsterFormat(), ctx=RunContext(trading_date=...))` or
  `Pipeline.from_format("lobster", ctx=RunContext(trading_date=...))`.
- **`LobsterLoader`:** public properties `trading_halts`, `cross_trades`,
  and `orderbook_path` removed. The data is on `PipelineResult.extras`
  instead (`result.extras["trading_halts"]`,
  `result.extras["cross_trades"]`). Hidden executions surface as
  `result.extras["hidden_executions"]`.
- **`PipelineResult`:** new field `extras: dict[str, pd.DataFrame]` (empty
  for formats that don't produce per-run extras).
- **`save_data`:** `fmt="lobster"` now works. Registered writers are
  factories taking `(config, ctx)`; pass
  `save_data(data, p, fmt="lobster", ctx=RunContext(trading_date=...))`.
- **`plot_hidden_executions` / `plot_trading_halts`:** accept a
  `PipelineResult` via `result=` and read from `result.extras`. Old
  DataFrame-arg form still works.

### Changed

- `scripts/collect_bitstamp_btcusd.py` reduced from ~600 LOC to a ~80-line
  wrapper around `ob_analytics.live.bitstamp.BitstampCapturer`. Same CLI
  flags; behaviour unchanged.
- Demo logic consolidated into `ob_analytics._demos`. Both
  `scripts/bitstamp_demo.py` and `scripts/lobster_demo.py` (and the CLI's
  `bitstamp-demo`/`lobster-demo` subcommands) are now thin argparse wrappers.
  Behaviour unchanged.
- `ob_analytics.depth.DepthMetricsEngine`: internal `_update_bid`/`_update_ask`
  and `_write_bid_metrics`/`_write_ask_metrics` consolidated into one
  side-parameterised path (`_update_side`, `_refresh_best`,
  `_write_side_metrics`). Public API unchanged.
- `ob_analytics.visualization` no longer re-exports the private helpers
  `_apply_theme`, `_create_axes`, or the `prepare_*` data shapers. Import
  these from `ob_analytics.visualization._matplotlib` /
  `ob_analytics.visualization._data` if you need them (internal use only).
  `infer_volume_scale` is still re-exported. An explicit `__all__` now
  documents the public surface.

### Removed

- Dead `df["volume"].cumsum().to_numpy(...)` line in `flow_toxicity.py`.
- Unused `DepthMetricsEngine._initialise_best`.

### Added

- **`ob_analytics.live`** -- new optional sub-package for live order-book
  capture. Introduces `LiveCapturer` protocol, `CaptureConfig`,
  `CaptureResult`, `CaptureSink`, and a generic asyncio runner. Capture
  output drops straight into the pipeline (`orders.csv` schema unchanged).
- **`ob-analytics capture <venue>`** -- new CLI verb. Built-in `bitstamp`
  capturer ships with the `[live]` extra. Use `--list` to see registered
  capturers.
- **`ob_analytics/live/bitstamp.py`** -- `BitstampCapturer` implementing
  the protocol. `scripts/collect_bitstamp_btcusd.py` becomes a thin
  back-compat wrapper around it (same CLI flags).
- **`[project.optional-dependencies] live = ["websockets>=12"]`** --
  install with `pip install "ob-analytics[live]"`.
- `tests/test_bitstamp.py` — dedicated coverage for `BitstampLoader`,
  `BitstampTradeReader`, `BitstampWriter`, and `BitstampFormat`, including
  a round-trip and a missing-companion error-path test. Uses the existing
  `tiny_bitstamp_orders_csv` fixture for the writer round-trip to keep
  the suite fast.
- `tests/test_cli.py` — subprocess smoke tests for all four CLI
  subcommands (`process`, `gallery`, `bitstamp-demo`, `lobster-demo`).
- `tests/test_exceptions.py` — pins the exception hierarchy contract
  (inheritance, isolation between siblings, picklability).
- `tests/test_data_registry.py` — pins `register_writer` /
  `list_writers` / `save_data(fmt=...)` semantics via a stub writer,
  including the explicit-writer override.
- `tests/conftest.py` — new `cli_runner` and `bitstamp_sample_orders_only`
  fixtures.
- `ob_analytics/__main__.py` — enables `python -m ob_analytics` (used by
  the CLI subprocess tests).
- **`ToxicityMetric`** protocol and `ob_analytics.metrics` sub-package.
  Three built-in implementations: `Vpin`, `Ofi`, `KyleLambda`.
- `Pipeline(metrics=...)` accepts any sequence of `ToxicityMetric`.
  Computed outputs land in `PipelineResult.metrics: dict[str, DataFrame]`,
  keyed by `metric.name`.
- `register_metric(name, cls)` / `list_metrics()` registry for plugging in
  user-defined metrics (Amihud, BVC, PIN, Roll, etc.) without modifying
  `Pipeline`.
- `PipelineResult.metrics` field.
- `RunContext` dataclass in `ob_analytics.protocols` (also exported from
  the top-level package) for per-run parameters that don't belong on
  long-lived `Format` instances.

- **`pacman` order type removed.** Legacy artifact of the 2015 Bitstamp HTTP
  API, where a single `order_id` could appear at multiple prices over its
  lifetime. Modern Bitstamp WS v2 and LOBSTER do not produce this pattern
  by design (price-modifies become cancel + new id). The label preempted
  market/maker/taker classification via set subtractions in
  `set_order_types`, and forced `LobsterLoader` to renumber hidden-execution
  ids to dodge misclassification. The `type` Categorical no longer includes
  `"pacman"`. On the bundled live-capture sample, 347 rows previously
  labelled `pacman` are now classified by their maker/taker behaviour
  (337 → `market`, 6 → `unknown`, 4 → `market-limit`); `depth` and
  `depth_summary` correspondingly gain those rows (no longer filtered out
  of `price_level_volume`). `LobsterLoader` no longer renumbers
  hidden-execution event ids (raw event type 5), which now retain the
  native LOBSTER convention of `id=0`.
- **Bitstamp trades** — The pipeline no longer infers trades from matched
  fills. A companion `trades.csv` next to `orders.csv` is required (capture
  format from `scripts/collect_bitstamp_btcusd.py`). Removed: Needleman–Wunsch
  matching, `BitstampMatcher`, `BitstampTradeInferrer`, `MatchingEngine`,
  `TradeInferrer`, and related `PipelineConfig` fields (`match_cutoff_ms`,
  `price_jump_threshold`).
- **Zombie detection** — Removed `get_zombie_ids` and `PipelineConfig` fields
  `zombie_offset_seconds`, `skip_zombie_detection`.
- **`Pipeline`** — Takes `trade_source=` instead of `matcher=` /
  `trade_inferrer=`. `Format` provides `create_trade_source()` only.
- **LOBSTER** — `LobsterMatcher` removed; `LobsterTradeInferrer` renamed to
  `LobsterTradeReader` with `load(events, source)`.
- **Bundled sample** — `ob_analytics/_sample_data/` now ships `orders.csv` and
  `trades.csv` from a modern live capture (replaces the legacy 2015-only
  orders slice).

### Added

- **`TradeSource` protocol** and **`BitstampTradeReader`** — read authoritative
  `trades.csv` and join to events via the `fill` column.

### 0.1.x baseline (historical; several items superseded by **Breaking** above)

- **`Pipeline` class** — composable orchestrator with pluggable `EventLoader`,
  `MatchingEngine`, `TradeInferrer`, optional `Format` descriptors, and
  `PipelineResult` output (events, trades, depth, depth_summary, optional
  VPIN/OFI, metadata).
- **`PipelineConfig`** — frozen Pydantic model centralising all tunable
  parameters (`price_decimals`, `volume_decimals`, `price_divisor`,
  `timestamp_unit`, `match_cutoff_ms`, `price_jump_threshold`, `depth_bps`,
  `depth_bins`, `zombie_offset_seconds`, `skip_zombie_detection`,
  `vpin_bucket_volume`).
- **Protocol interfaces** — `EventLoader`, `MatchingEngine`, `TradeInferrer`,
  `DataWriter`, and abstract `Format` base for venue-specific bundles.
- **`ob_analytics.bitstamp`** — complete Bitstamp-specific stack:
  `BitstampLoader`, `BitstampMatcher`, `BitstampTradeInferrer`,
  `BitstampWriter`, `BitstampFormat`. Replaces the former `event_processing.py`
  and `trades.py` modules.
- **LOBSTER stack** — `LobsterLoader`, `LobsterMatcher`, `LobsterTradeInferrer`,
  `LobsterWriter`, `LobsterFormat`, orderbook-backed depth.
- **`ob_analytics.analytics`** — format-agnostic analytics module with
  `order_aggressiveness()`, `trade_impacts()`, `set_order_types()`, and
  `order_book()`.
- **Timestamp conversion helpers** — `epoch_to_datetime`, `datetime_to_epoch`,
  `seconds_after_midnight_to_datetime`, `datetime_to_seconds_after_midnight`
  (in `_utils.py`).
- **`register_format()` / `register_writer()`** — built-in `"bitstamp"` and
  `"lobster"` registrations; `Pipeline.from_format(name, **kwargs)`.
- **`list_formats()`** in `pipeline.py` — returns sorted list of registered
  format names.
- **`list_writers()`** in `data.py` — returns sorted list of registered writer
  names.
- **`Format.name`** — string attribute on the `Format` base class; subclasses
  set e.g. `name = "bitstamp"`. Used in `PipelineResult.metadata`.
- **`DepthMetricsEngine`** — incremental depth metrics with dynamic price
  support (no fixed array size).
- **Pydantic domain models** — `OrderEvent`, `Trade`, `DepthLevel`,
  `OrderBookSnapshot`, `KyleLambdaResult`.
- **Exceptions** — `ObAnalyticsError` hierarchy (`InvalidDataError`,
  `MatchingError`, `InsufficientDataError`, `ConfigurationError`).
- **Flow toxicity** — `compute_vpin`, `compute_kyle_lambda`,
  `order_flow_imbalance` with matching plot helpers.
- **Visualization** — `PlotTheme`, `save_figure`, Matplotlib + optional Plotly
  (`backend="plotly"`), `register_plot_backend()`, LOBSTER-specific plots
  (`plot_hidden_executions`, `plot_trading_halts`).
- **`visualization.gallery.generate_gallery`** — HTML gallery of standard plots.
- **Parquet I/O** — `load_data()` / `save_data()` via pyarrow.
- **Test suite** — pytest tests (unit, integration, LOBSTER, visualization).
- **Docs** — [Zensical](https://github.com/zensicalHQ/zensical) site with API
  reference from docstrings.
- **Demo scripts** — `scripts/lobster_demo.py` for LOBSTER processing and
  gallery generation; `scripts/bitstamp_demo.py` for Bitstamp CSV processing
  and gallery generation.
- **CLI** — `ob-analytics` entry point with subcommands: `process`, `gallery`,
  `bitstamp-demo`, `lobster-demo`. Registered via `[project.scripts]` in
  `pyproject.toml`.
- **CI/CD** — GitHub Actions workflow (`.github/workflows/ci.yml`): `ruff`
  lint/format, `ty` type check, `pytest` on Python 3.11/3.12/3.13, Codecov
  coverage upload.

### Changed

- `Pipeline` still honours legacy `config.vpin_bucket_volume`: when set and
  no explicit `metrics=` is passed, materialises `Vpin` + `Ofi` for
  back-compat. The `result.vpin` and `result.ofi` attributes still work
  but are mirrors of `result.metrics["vpin"]` / `["ofi"]`.
- Gallery now iterates `result.metrics` instead of hard-coding the
  VPIN/OFI/Kyle panels. Built-ins keep their dedicated plotters; future
  user-registered metrics can plug in their own.
- `event_processing.py` **renamed** to `bitstamp.py`. Hard break — no shim.
- `DefaultTradeInferrer` renamed to `BitstampTradeInferrer`; symmetric with
  `LobsterTradeInferrer`.
- `BitstampFormat` converted to `@dataclass` (was plain class); symmetric with
  `LobsterFormat`.
- `LobsterWriter.__init__` now accepts `config` as first positional argument
  (previously it had no `config` parameter), matching `BitstampWriter(config)`.
- `LobsterFormat.create_writer(config)` now passes `config` to `LobsterWriter`.
- `Format.create_matcher()` and `Format.create_trade_inferrer()` now raise
  `NotImplementedError` instead of silently defaulting to Bitstamp implementations.
- `Pipeline` no-format defaults updated to use `BitstampMatcher` and
  `BitstampTradeInferrer` (previously `NeedlemanWunschMatcher` and
  `DefaultTradeInferrer`).
- `PipelineResult.metadata["format"]` now stores `format.name` (e.g.
  `"bitstamp"`) instead of the Python class name.
- `scripts/bitstamp_demo.py` — uses `Pipeline(format=BitstampFormat())` instead
  of bare `Pipeline()`, matching the `lobster_demo.py` pattern.
- All `print()` statements replaced with `loguru` logging (library, CLI, and
  demo scripts).
- Type checking moved from `mypy` to Astral's `ty`; all `ty` errors resolved.
- `sns.set_theme()` moved from module-level into per-function `_apply_theme()`
  calls (no global side effects on import).
- All bare `assert` statements replaced with proper exception raising.
- `plt.show()` removed from all plot functions (callers control display).
- Internal modules renamed with underscore prefix: `auxiliary.py` →
  `_utils.py`, `needleman_wunsch.py` → `_needleman_wunsch.py`.
- `_validation.py` merged into `_utils.py`.
- `_time_utils.py` merged into `_utils.py`.
- `matching_engine.py` and `_needleman_wunsch.py` merged into `bitstamp.py`.
- `order_types.py` and `order_book_reconstruction.py` merged into `analytics.py`.
- Visualization files (`visualization.py`, `_chart_data.py`, `_matplotlib.py`,
  `_plotly.py`, `gallery.py`) moved into `visualization/` subpackage.
- Docs nav: `api/event_processing.md` → `api/bitstamp.md`; `api/analytics.md`
  added; `api/trades.md`, `api/order_types.md`, `api/order_book.md`, and
  `api/matching_engine.md` merged into their parent module pages.

### Deprecated

- `PipelineResult.vpin` and `PipelineResult.ofi`. Use
  `result.metrics["vpin"]` / `result.metrics["ofi"]` instead. These
  attribute mirrors will be removed in a future release.

### Removed

- `load_event_data()` — legacy Bitstamp-only pipeline wrapper.
- `event_match()` — legacy wrapper around `NeedlemanWunschMatcher.match()`.
- `match_trades()` — legacy wrapper around `DefaultTradeInferrer.infer_trades()`.
- `process_data()` — legacy monolithic Bitstamp-only pipeline.
- `plot_price_levels_faster()` — legacy alias for `plot_price_levels()`.
- `NeedlemanWunschMatcher` removed from public `__all__` (still importable
  directly via `from ob_analytics.bitstamp import NeedlemanWunschMatcher`).
- `register_writer("lobster", LobsterWriter)` registration removed from
  `__init__.py` — it was always broken (required `trading_date`). A clear
  `ValueError` is now raised by `save_data(fmt="lobster", ...)`.
- `ob_analytics/trades.py` deleted; all content moved to `bitstamp.py` or
  `analytics.py`.
- 12 unused runtime dependencies (scikit-learn, scipy, jupyter, bokeh, etc.).
- Stale dev dependencies (black, flake8 and plugins, darglint).
- Commented-out debug code and stale filename header comments.

### Fixed

- `datetime_to_epoch` in `_utils.py` uses `.astype("int64")` instead of
  deprecated `.view("int64")`.
- **BUG-1**: Hardcoded local file paths in debug CSV exports.
- **BUG-2**: `depth_metrics` state array overflow for prices > $9,999.99 — now
  uses dynamic `dict[int, int]` supporting any price range.
- **BUG-3**: `best_bid`/`best_ask` initialized with dataset-wide max/min
  (incorrect for first events) — now tracked correctly from the first event.
- `NeedlemanWunschMatcher` crash on empty match result.

---

## [0.1.0] — 2024-09-03

Initial Python port from the R [obAnalytics](https://cran.r-project.org/package=obAnalytics) CRAN package.

- Line-for-line translation of R functions to Python/pandas
- Bitstamp CSV loading and processing
- Needleman-Wunsch event matching
- Trade inference and order type classification
- Depth metrics computation
- Matplotlib/seaborn visualization suite
- Sample data: Bitstamp BTC/USD limit order events, 2015-05-01 00:00–05:00 UTC
