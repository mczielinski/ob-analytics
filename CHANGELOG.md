# Changelog

All notable changes to ob-analytics are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0] - 2026-06-04

First stable release: a deliberate **breaking** cut that de-bloats the package
and unifies its extension surfaces behind one registry primitive and one data
contract. The **numeric output of the pipeline is unchanged** from 0.1.x — the
regression fingerprints still pass; only the *shape of the public API* moved.
See [the migration guide](docs/migration-to-v1.md) for before/after on every
breaking change (its sections are cited as §N below), and
[Extending ob-analytics](docs/extending.md) for the new extension walkthrough.

### Breaking

- **Pydantic models removed.** `ob_analytics.models` (`OrderEvent`, `Trade`,
  `DepthLevel`, `OrderBookSnapshot`) deleted; the data contract is now
  column-list constants + `validate_events_df` / `validate_trades_df` /
  `validate_depth_df` in `ob_analytics.schemas`.
  (§1)
- **`metrics/` package removed.** `ToxicityMetric`, `Vpin`, `Ofi`,
  `KyleLambda`, `register_metric`, and `list_metrics` are gone. Call
  `compute_vpin`, `compute_kyle_lambda`, and `order_flow_imbalance` on
  `result.trades` directly. (§2)
- **`Pipeline(metrics=...)` removed.** Metrics are no longer a pipeline stage —
  compute them after the run. (§3)
- **`PipelineConfig.vpin_bucket_volume` removed** — pass `bucket_volume=` to
  `compute_vpin`. (§4)
- **`PipelineResult` slimmed** to exactly `events`, `trades`, `depth`,
  `depth_summary`, and `config`. The `vpin`, `ofi`, `metrics`, `metadata`, and
  `extras` attributes are gone. (§5)
- **The thirteen `plot_*` wrappers removed** → one
  `plot(name, *, backend="matplotlib", ax=None, **data)` dispatcher keyed by
  `(plot_name, backend)`; renderers self-register into `RENDERERS`.
  (§6)
- **Global theme state removed.** `set_plot_theme` / `get_plot_theme` /
  `_current_theme` deleted; pass `theme=PlotTheme(...)` to `plot()`.
  (§7)
- **Exception hierarchy collapsed** to `ObAnalyticsError` + `ConfigError`.
  `InvalidDataError`, `MatchingError`, `InsufficientDataError`, and
  `ConfigurationError` are removed. (§8)
- **Top-level `__all__` trimmed** to ~22 orchestration names. Low-level helpers
  now import from their submodules — `ob_analytics.bitstamp`,
  `ob_analytics.lobster`, `ob_analytics.analytics`, `ob_analytics.depth`,
  `ob_analytics.data`, `ob_analytics.visualization`, `ob_analytics.flow_toxicity`.
  (§9)
- **`Format` is now a `typing.Protocol`** — there is no base class to inherit;
  any conforming object is recognised structurally.
  (§10)
- **Low-level helpers no longer re-exported from the package root** (e.g.
  `depth_metrics` is now `from ob_analytics.depth import depth_metrics`).
  (§11)
- **`RunContext.extras` and `Format.collect_extras` removed.** LOBSTER trading
  halts are read from `LobsterLoader.trading_halts` and composed into the
  gallery via `extra_panels=`. (§12)
- **`DepthMetricsEngine.update()` removed** → the public hot-path method is
  `update_side(price, volume, side, out)`.
  (§13)

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
- **Docs** — `docs/migration-to-v1.md` (breaking-change before/after guide) and
  `docs/extending.md` (add a data source / writer / plot / metric / capturer).
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
  and the visualization modules split into a `visualization/` subpackage. The
  resulting public import surface is documented in
  §9](docs/migration-to-v1.md).
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
