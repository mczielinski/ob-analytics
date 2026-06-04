---
title: Migrating to v1.0
---

# Migrating to ob-analytics v1.0

v1.0 is a deliberate **breaking** release. It removes the Pydantic model
layer, the metrics plugin system, the thirteen per-plot wrapper functions, and
the global theme state; it collapses the exception hierarchy to two classes;
and it trims the top-level namespace down to the orchestration surface. The
**numeric output of the pipeline is unchanged** — every regression fingerprint
from the previous release still passes. Only the *shape of the API* moved.

If you read only one paragraph: catch `ObAnalyticsError` (config/contract
problems are now `ConfigError`), import the lower-level helpers from their
submodules instead of the top-level package, and call the flow-toxicity
functions directly on `result.trades`.

Each change below is shown as **Before** (v0.x) → **After** (v1.0), with a
short *why*.

---

## 1. Pydantic models removed → DataFrame contracts + validators

The `ob_analytics.models` module (Pydantic `OrderEvent`, `Trade`,
`DepthLevel`, `OrderBookSnapshot`) is gone. The pipeline already passed plain
`pandas.DataFrame`s internally; the models were a parallel description that
drifted. The contract is now expressed as column-list constants and
validators in `ob_analytics.schemas`.

**Before**

```python
from ob_analytics import OrderEvent, Trade

ev = OrderEvent(id=1, timestamp=..., price=..., volume=...)
```

**After**

```python
from ob_analytics.schemas import (
    validate_events_df,
    validate_trades_df,
    validate_depth_df,
)

# events/trades/depth are pandas DataFrames; validate at your boundary:
validate_events_df(result.events)
validate_trades_df(result.trades)
validate_depth_df(result.depth)
```

`KyleLambdaResult` (the one model worth keeping) now lives in
`ob_analytics.flow_toxicity` — see §2.

**Why:** one source of truth for the data shape, no row-by-row Pydantic
construction on the hot path, and DataFrames are what every downstream
consumer already wanted.

---

## 2. `metrics/` package removed → call the functions directly

The `ob_analytics.metrics` subpackage and its wrapper classes (`Vpin`,
`KyleLambda`, `Ofi`, the `ToxicityMetric` base) are gone. The underlying
functions live in `ob_analytics.flow_toxicity` and are re-exported at the top
level.

**Before**

```python
from ob_analytics.metrics import Vpin, KyleLambda

vpin = Vpin(bucket_volume=10.0).compute(result.trades)
kyle = KyleLambda().compute(result.trades)
```

**After**

```python
from ob_analytics import compute_vpin, compute_kyle_lambda, order_flow_imbalance

vpin = compute_vpin(result.trades, bucket_volume=10.0)   # -> DataFrame
kyle = compute_kyle_lambda(result.trades)                # -> KyleLambdaResult
ofi = order_flow_imbalance(result.trades)                # -> DataFrame
```

**Why:** a metric is a function of a DataFrame. The class hierarchy and
plugin registry added indirection without buying anything.

---

## 3. `Pipeline(metrics=...)` removed → compute after the run

`Pipeline` no longer accepts a `metrics=` list and no longer runs metrics as
a stage.

**Before**

```python
from ob_analytics import Pipeline, Vpin

result = Pipeline(metrics=[Vpin(bucket_volume=10.0)]).run(source)
vpin = result.metrics["vpin"]
```

**After**

```python
from ob_analytics import Pipeline, compute_vpin

result = Pipeline().run(source)
vpin = compute_vpin(result.trades, bucket_volume=10.0)
```

**Why:** metrics are cheap, optional, and parameter-heavy. Computing them
post-pipeline keeps the pipeline focused on producing `events`/`trades`/
`depth` and lets you tune metric parameters without re-running the load.

---

## 4. `PipelineConfig.vpin_bucket_volume` removed → pass it to `compute_vpin`

**Before**

```python
from ob_analytics import Pipeline, PipelineConfig

result = Pipeline(config=PipelineConfig(vpin_bucket_volume=10.0)).run(source)
```

**After**

```python
from ob_analytics import Pipeline, compute_vpin

result = Pipeline().run(source)
vpin = compute_vpin(result.trades, bucket_volume=10.0)
```

**Why:** the bucket size is a property of the VPIN calculation, not of the
pipeline. It belongs on the call that uses it.

---

## 5. `PipelineResult` slimmed to five fields

`PipelineResult` now carries exactly `events`, `trades`, `depth`,
`depth_summary`, and `config`. The `vpin`, `ofi`, `metrics`, `metadata`, and
`extras` attributes are gone.

**Before**

```python
result = Pipeline(metrics=[...]).run(source)
vpin = result.vpin
halts = result.extras["trading_halts"]
ran_at = result.metadata["timestamp"]
```

**After**

```python
result = Pipeline().run(source)
events, trades = result.events, result.trades
depth, depth_summary = result.depth, result.depth_summary

# Metrics: compute post-run (see §2).
# Run-state (timestamps, provenance): keep it in your own dataclass that
# wraps PipelineResult — the library no longer carries an open-ended bag.
```

LOBSTER trading halts moved out of `result.extras`; see §12.

**Why:** an open-ended `extras` dict and optional metric fields made
`PipelineResult` impossible to reason about. Five well-typed DataFrames (plus
the config that produced them) is the whole contract.

---

## 6. The thirteen `plot_*` wrappers removed → one `plot()` dispatcher

Every `plot_<name>(...)` function is gone. Prepare the data with the matching
`prepare_<name>_data` helper and render through the unified `plot()`
dispatcher, choosing a `backend`.

**Before**

```python
from ob_analytics import plot_trades, plot_price_levels

fig = plot_trades(result.trades)
```

**After**

```python
from ob_analytics.visualization import plot
from ob_analytics.visualization import _data

fig = plot("trades", backend="matplotlib", **_data.prepare_trades_data(result.trades))
# backend="plotly" returns an interactive Plotly figure instead.
```

For the standard multi-plot gallery, the flow-toxicity and LOBSTER panels are
built with helper functions and appended via `extra_panels=`:

```python
from ob_analytics import compute_vpin, compute_kyle_lambda
from ob_analytics.visualization.gallery import (
    generate_gallery,
    vpin_panel,
    kyle_panel,
)

vpin = compute_vpin(result.trades, bucket_volume=10.0)
kyle = compute_kyle_lambda(result.trades)
generate_gallery(
    result,
    "out/gallery",
    extra_panels=[vpin_panel(vpin), kyle_panel(kyle)],
)
```

The matching builders are `vpin_panel`, `ofi_panel`, `kyle_panel`, and
`trading_halts_panel` (all in `ob_analytics.visualization.gallery`). Hidden
executions are detected automatically by `default_specs` when the events frame
contains LOBSTER hidden-execution rows.

**Why:** thirteen near-identical wrappers were thirteen things to keep in sync.
One dispatcher keyed by `(plot_name, backend)` is the single extension point —
see [Extending ob-analytics](extending.md).

---

## 7. Global theme state removed → `theme=` per call

`set_plot_theme()` / `get_plot_theme()` are gone. There is no global mutable
theme. Pass a `PlotTheme` to `plot()` instead; it applies only when the
renderer creates its own figure.

**Before**

```python
from ob_analytics import PlotTheme, set_plot_theme

set_plot_theme(PlotTheme(style="whitegrid", font_scale=1.0))
fig = plot_trades(result.trades)
```

**After**

```python
from ob_analytics.visualization import plot, PlotTheme
from ob_analytics.visualization import _data

fig = plot(
    "trades",
    theme=PlotTheme(style="whitegrid", font_scale=1.0),
    **_data.prepare_trades_data(result.trades),
)
```

**Why:** global theme state is a hidden, order-dependent side effect. A
per-call `theme=` is explicit and thread-safe.

---

## 8. Exception hierarchy collapsed to two classes

`InvalidDataError`, `MatchingError`, `InsufficientDataError`, and
`ConfigurationError` are removed. Everything raised by the library is now an
`ObAnalyticsError`; configuration and data-contract problems are the
`ConfigError` subclass.

**Before**

```python
from ob_analytics import InvalidDataError, ConfigurationError

try:
    result = Pipeline().run(source)
except (InvalidDataError, ConfigurationError) as e:
    ...
```

**After**

```python
from ob_analytics import ObAnalyticsError, ConfigError

try:
    result = Pipeline().run(source)
except ConfigError as e:      # bad config / violated DataFrame contract
    ...
except ObAnalyticsError as e:  # everything else from the library
    ...
```

**Why:** five exception types with overlapping meanings encouraged
over-specific `except` clauses. Two — a base and a config/contract subclass —
cover the real branch points.

---

## 9. Top-level `__all__` trimmed → import from submodules

The package root now exports only the orchestration surface (≈22 names). The
lower-level helpers still exist; import them from their home modules.

| Removed from `ob_analytics` (top level) | Import from in v1.0 |
| --- | --- |
| `BitstampLoader`, `BitstampTradeReader`, `BitstampWriter` | `ob_analytics.bitstamp` |
| `LobsterLoader`, `LobsterTradeReader`, `LobsterWriter` | `ob_analytics.lobster` |
| `lobster_depth_from_orderbook` | `ob_analytics.lobster` |
| `order_aggressiveness`, `trade_impacts`, `set_order_types`, `order_book` | `ob_analytics.analytics` |
| `DepthMetricsEngine`, `price_level_volume`, `depth_metrics`, `filter_depth`, `get_spread` | `ob_analytics.depth` |
| `register_writer`, `list_writers` | `ob_analytics.data` |
| `PlotTheme`, `save_figure`, `infer_volume_scale`, `register_plot_backend` | `ob_analytics.visualization` |
| `plot_*` (13 wrappers), `set_plot_theme`, `get_plot_theme` | removed — use `ob_analytics.visualization.plot` (§6, §7) |
| `OrderEvent`, `Trade`, `DepthLevel`, `OrderBookSnapshot` | removed — DataFrame contracts in `ob_analytics.schemas` (§1) |
| `ToxicityMetric`, `Vpin`, `KyleLambda`, `Ofi`, `register_metric`, `list_metrics` | removed — call functions directly (§2) |
| `KyleLambdaResult` | `ob_analytics.flow_toxicity` (still re-exported at top level) |
| `InvalidDataError`, `MatchingError`, `InsufficientDataError` | removed — use `ObAnalyticsError` (§8) |
| `ConfigurationError` | renamed → `ConfigError` (§8) |

Still exported at the top level: `Pipeline`, `PipelineResult`,
`PipelineConfig`, `register_format`, `list_formats`, `BitstampFormat`,
`LobsterFormat`, the four Protocols (`Format`, `EventLoader`, `TradeSource`,
`DataWriter`), `RunContext`, `save_data`, `load_data`, `compute_vpin`,
`compute_kyle_lambda`, `order_flow_imbalance`, `KyleLambdaResult`,
`sample_csv_path`, `sample_data_dir`, `ObAnalyticsError`, `ConfigError`.

**Why:** a 70-name root namespace hid the handful of entry points that 95% of
users need. Submodule imports document where each helper actually lives.

---

## 10. `Format` is now a Protocol — drop the base class

`Format` is a `typing.Protocol`. You no longer inherit from it; any object
with the right methods satisfies the contract (structural typing).

**Before**

```python
from ob_analytics import Format

class MyVenueFormat(Format):
    ...
```

**After**

```python
class MyVenueFormat:           # no base class
    name = "myvenue"
    def create_loader(self, config, ctx): ...
    def create_trade_source(self, config, ctx): ...
    def create_writer(self, config, ctx): ...
    def config_defaults(self): ...
```

See [Extending ob-analytics](extending.md) for the full walkthrough.

**Why:** structural typing means third-party formats don't have to import and
subclass an internal base just to be recognised.

---

## 11. Low-level helpers moved namespace (e.g. `depth_metrics`)

The convenience wrapper `depth_metrics(depth, bps=25, bins=20)` is unchanged in
behaviour and signature — but, like the other low-level helpers in §9, it is no
longer re-exported from the package root.

**Before**

```python
from ob_analytics import depth_metrics
```

**After**

```python
from ob_analytics.depth import depth_metrics
```

---

## 12. `RunContext.extras` / `Format.collect_extras` removed (LOBSTER halts)

`RunContext` no longer has an `extras` field and `Format` no longer has a
`collect_extras` hook. LOBSTER trading halts are read from the loader and
plotted explicitly.

**Before**

```python
result = Pipeline(format=LobsterFormat()).run(source, ctx=RunContext(...))
halts = result.extras["trading_halts"]
```

**After**

```python
from ob_analytics import Pipeline, RunContext
from ob_analytics.lobster import LobsterFormat, LobsterLoader
from ob_analytics.visualization.gallery import generate_gallery, trading_halts_panel

ctx = RunContext(trading_date="2015-05-01")
result = Pipeline(format=LobsterFormat()).run(source, ctx=ctx)

# Halts are no longer on PipelineResult. A LobsterLoader splits them out
# during load() and exposes them as a public attribute (None if absent):
loader = LobsterLoader(trading_date="2015-05-01")
loader.load(source)               # populates loader.trading_halts
halts = loader.trading_halts      # pd.DataFrame | None

if halts is not None:
    generate_gallery(
        result, "out/gallery",
        extra_panels=[trading_halts_panel(result.trades, halts)],
    )
```

**Note on `trading_date`:** LOBSTER requires a `trading_date` on the
`RunContext`. A missing date raises `ValueError`; a wrong type (not `str` or
`pandas.Timestamp`) raises `TypeError`. This behaviour is unchanged — v1.0
only consolidated the three duplicate checks into one.

**Why:** `extras` was an untyped escape hatch that only LOBSTER used. Reading
`loader.trading_halts` is explicit and typed; the gallery composes it via
`extra_panels=`.

---

## 13. `DepthMetricsEngine.update()` → `update_side()`

The thin `update()` passthrough on `DepthMetricsEngine` is gone; the single
hot-path method is now public as `update_side()`. The argument order is
`(price, volume, side, out)`.

**Before**

```python
engine.update(price, volume, side, out)
```

**After**

```python
engine.update_side(price, volume, side, out)
```

**Why:** there was one real method behind two names. `update_side` is the one
that does the work, so it became the public one.

---

## Quick checklist

- Replace `from ob_analytics import OrderEvent, Trade, ...` with DataFrame
  validators from `ob_analytics.schemas`.
- Replace metric classes / `Pipeline(metrics=...)` with `compute_vpin` /
  `compute_kyle_lambda` / `order_flow_imbalance` on `result.trades`.
- Replace `plot_<name>(...)` with `plot("<name>", backend=..., **prepare_<name>_data(...))`.
- Replace `set_plot_theme(...)` with `theme=PlotTheme(...)` passed to `plot()`.
- Replace `except InvalidDataError/ConfigurationError` with `except
  ConfigError` / `except ObAnalyticsError`.
- Move low-level helper imports to their submodules (see §9).
- Read LOBSTER halts from `LobsterLoader.trading_halts`, not `result.extras`.
