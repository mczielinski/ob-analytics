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
- **Bitstamp stack** — `BitstampLoader`, `BitstampWriter`, `BitstampFormat`.
- **LOBSTER stack** — `LobsterLoader`, `LobsterMatcher`, `LobsterTradeInferrer`,
  `LobsterWriter`, `LobsterFormat`, orderbook-backed depth, `download_sample()`.
- **`register_format()` / `register_writer()`** — built-in `"bitstamp"` and
  `"lobster"` registrations; `Pipeline.from_format(name, **kwargs)`.
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
- **`gallery.generate_gallery`** — HTML gallery of standard plots.
- **Parquet I/O** — `load_data()` / `save_data()` via pyarrow.
- **Test suite** — 200+ pytest tests (unit, integration, LOBSTER, visualization).
- **Docs** — Zensical site with API reference from docstrings.
- **Demo scripts** — `scripts/lobster_demo.py` for LOBSTER sample download and
  gallery generation; `scripts/bitstamp_demo.py` for Bitstamp CSV processing
  and gallery generation.
- **CLI** — `ob-analytics` entry point with subcommands: `process`, `gallery`,
  `bitstamp-demo`, `lobster-demo`. Registered via `[project.scripts]` in
  `pyproject.toml`.
- **CI/CD** — GitHub Actions workflow (`.github/workflows/ci.yml`): `ruff`
  lint/format, `ty` type check, `pytest` on Python 3.11/3.12/3.13, Codecov
  coverage upload.

### Changed

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

### Removed

- 12 unused runtime dependencies (scikit-learn, scipy, jupyter, bokeh, etc.).
- Stale dev dependencies (black, flake8 and plugins, darglint).
- Commented-out debug code and stale filename header comments.

### Fixed

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
