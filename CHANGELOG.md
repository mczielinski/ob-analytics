# Changelog

All notable changes to ob-analytics are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- **`Pipeline` class** — composable orchestrator that runs the full 8-step
  processing sequence with dependency injection. Accepts pluggable `EventLoader`,
  `MatchingEngine`, and `TradeInferrer` implementations.
- **`PipelineConfig`** — frozen Pydantic model centralizing all tunable
  parameters (`price_decimals`, `match_cutoff_ms`, `depth_bps`, etc.). Replaces
  hardcoded magic numbers throughout the codebase.
- **Protocol interfaces** — `EventLoader`, `MatchingEngine`, `TradeInferrer`
  (runtime-checkable `Protocol` classes) enabling custom implementations without
  subclassing.
- **Default protocol implementations**:
  - `BitstampLoader` — loads Bitstamp-format CSVs (implements `EventLoader`)
  - `NeedlemanWunschMatcher` — matches bid/ask fills via sequence alignment
    (implements `MatchingEngine`)
  - `DefaultTradeInferrer` — infers trades from matched events (implements
    `TradeInferrer`)
- **`DepthMetricsEngine`** — stateful, incremental depth metrics computation
  replacing the monolithic `depth_metrics()` function. Uses `dict[int, int]`
  instead of fixed `np.zeros(1_000_000)` arrays, supporting arbitrary price
  ranges and precisions.
- **Pydantic domain models** — `OrderEvent`, `Trade`, `DepthLevel`,
  `OrderBookSnapshot` for validated, typed data contracts at package boundaries.
- **Custom exception hierarchy** — `ObAnalyticsError` base with
  `InvalidDataError`, `MatchingError`, `InsufficientDataError`,
  `ConfigurationError`.
- **Input validation** — `validate_columns()` and `validate_non_empty()` wired
  into all public functions. Missing columns now raise `InvalidDataError` instead
  of cryptic `KeyError`.
- **`PlotTheme` dataclass** — configurable visual theme with `set_plot_theme()`
  and `get_plot_theme()` for global theme management.
- **`save_figure()` utility** — convenience wrapper for saving matplotlib
  figures with sensible defaults.
- **All plot functions return `Figure`** — `plot_trades()`,
  `plot_price_levels()`, `plot_event_map()`, `plot_volume_map()`,
  `plot_current_depth()`, `plot_volume_percentiles()`,
  `plot_events_histogram()`, `plot_time_series()` now return
  `matplotlib.figure.Figure` and accept an optional `ax` parameter for subplot
  composition.
- **Parquet serialization** — `load_data()` / `save_data()` default to Parquet
  format via `pyarrow`. Pickle retained as backward-compatible fallback with
  security warnings.
- **`py.typed` marker** — PEP 561 type-checking support for downstream users.
- **Comprehensive test suite** — 106 tests (pytest) covering unit tests,
  R parity tests (ported from original R package), integration tests with
  golden output comparison, and visualization tests. >80% coverage on core
  logic modules.
- **README** — comprehensive documentation with pipeline workflow diagrams,
  architecture overview, configuration reference, and extensibility guide.
- **CHANGELOG** — this file.
- **Module-level docstrings** for all modules.

### Changed

- All `print()` statements replaced with stdlib `logging` module.
- `sns.set_theme()` moved from module-level import into per-function
  `_apply_theme()` calls (no global side effects on import).
- All bare `assert` statements replaced with proper exception raising.
- `plt.show()` removed from all plot functions (callers control display).
- Internal modules renamed with underscore prefix: `auxiliary.py` →
  `_utils.py`, `needleman_wunsch.py` → `_needleman_wunsch.py`.
- `_validation.py` merged into `_utils.py`.

### Removed

- 12 unused runtime dependencies: `scikit-learn`, `scipy`, `jupyter`, `bokeh`,
  `jupyterlab`, `line-profiler`, `ecdsa`, `base58`, `opencv-python`,
  `openpyxl`, `ipython`, `ruff`.
- Stale dev dependencies: `black`, `flake8` and plugins, `darglint`.
- Commented-out debug code and stale filename header comments.

### Fixed

- **BUG-1**: Hardcoded local file paths in debug CSV exports removed from git
  history.
- **BUG-2**: `depth_metrics` state array overflow for prices > $9,999.99. Now
  uses dynamic `dict[int, int]` supporting any price range.
- **BUG-3**: `best_bid`/`best_ask` initialized with dataset-wide max/min
  (incorrect for first events). Now tracked correctly by default
  (`compat_mode=False`); R parity preserved with `compat_mode=True`.
- `NeedlemanWunschMatcher` crash on empty match result (no fills within cutoff
  window).

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
