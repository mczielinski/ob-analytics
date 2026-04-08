# Changelog

All notable changes to ob-analytics are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- **`Pipeline` class** — composable orchestrator with pluggable `EventLoader`,
  `MatchingEngine`, `TradeInferrer`, optional `Format` descriptors, and
  `PipelineResult` output (events, trades, depth, depth_summary, optional
  VPIN/OFI, metadata).
- **`PipelineConfig`** — frozen Pydantic model centralising all tunable
  parameters (`price_decimals`, `match_cutoff_ms`, `depth_bps`,
  `price_divisor`, `timestamp_unit`, `skip_zombie_detection`,
  `vpin_bucket_volume`, …).
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
