---
title: Extending
---

# Extending ob-analytics

Every pluggable surface in ob-analytics follows the same shape: implement a
small **Protocol** by structural typing (no base class to inherit), then
**register** it under a name. Things that are genuinely swappable at runtime —
data sources, export formats, plot backends, live capturers — live in
name-keyed registries. Things that aren't — metrics, themes — are plain
functions and values you call directly.

## How extension works

| Want to add… | Implement | Register with | Use via |
|---|---|---|---|
| **A data source** (new venue, file and/or live) | `Source` + `OfflineSource` and/or `LiveSource` | `register_source(name, cls)` (or an entry point) | `Pipeline.from_source(name)` · CLI `process --source name` / `capture name` |
| **An export format** | `DataWriter` | `register_writer(name, factory)` | `save_data(data, path, fmt=name)` |
| **A plot** | a `prepare_*` function + a renderer | `RENDERERS.register((name, backend), fn)` | `plot(name, backend=...)` |
| **A metric** | a plain function | — *(no registry)* | `my_metric(result.trades, ...)` |

Registration is an import side-effect: the module that calls
`register_*` must be imported before the name is looked up. Built-ins register
themselves when `ob_analytics` is imported — see
[Making registration fire](#making-registration-fire).

The Protocol contracts referenced below are documented on the
[Protocols](api/protocols.md) page; the DataFrame column contracts are on
[Data Contracts](api/schemas.md).

Every protocol here takes and returns `pandas.DataFrame` — see
[Frame types: pandas in, pandas out](schema.md#frame-types-pandas-in-pandas-out)
for the contract and what to use when you want Arrow or Polars.

---

## 1. A new source

Every data source — file or live — has the same shape. A **`Source`** states
two coordinates, `level` (`Level.L2` / `Level.L3`) and `feed_type`
(`FeedType`), and carries typed `settings` (a `SourceSettings`). It then
implements one or both **capabilities**:

- **`OfflineSource`** — replay stored files. The per-venue factories the
  pipeline needs: a loader (`EventLoader` for L3, `DepthSource` for L2), a trade
  source (`TradeSource`), and — optionally — a writer (`DataWriter`), plus
  `config_defaults()` and `compute_depth(...)`.
- **`LiveSource`** — capture a venue's live feed: `snapshot` / `stream` /
  `shutdown_synthetic_events` (see [the live capability](#the-live-capability)).

A venue can do both — `BitstampSource` replays CSV *and* captures live. None of
this requires a base class: any object whose attributes match the Protocol
satisfies it. The built-in [`BitstampLoader`](api/bitstamp.md) /
[`LobsterLoader`](api/lobster.md) are the reference implementations for the
parsing work; the skeleton below shows the contracts for an offline source.

```python
from __future__ import annotations

from typing import Any

import pandas as pd

from ob_analytics import (
    FeedType,
    Level,
    PipelineConfig,
    RunContext,
    SourceSettings,
    register_source,
)


class CoinbaseLoader:
    """Satisfies the EventLoader Protocol."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def load(self, source: Any) -> pd.DataFrame:
        # Parse the venue feed into the canonical event columns
        # (event_id, timestamp, price, volume, action, direction, ...).
        # See ob_analytics.bitstamp.BitstampLoader for a full implementation
        # and docs/api/schemas.md for the column contract.
        raw = pd.read_json(source)
        ...
        return events


class CoinbaseTradeReader:
    """Satisfies the TradeSource Protocol."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def load(self, events: pd.DataFrame, source: Any) -> pd.DataFrame:
        # Project explicit trade records into the canonical trades schema
        # (timestamp, price, volume, direction, maker/taker ids, ...).
        ...
        return trades


class CoinbaseSource:
    """An offline OfflineSource — no base class needed."""

    name = "coinbase"
    level = Level.L3  # per-order (Coinbase `full` channel)
    feed_type = FeedType.MATCHED_BOOK  # exchange matching engine: never crossed
    settings = SourceSettings()  # empty: no per-source knobs

    def create_loader(self, config: PipelineConfig, ctx: RunContext) -> CoinbaseLoader:
        return CoinbaseLoader(config)

    def create_trade_source(
        self, config: PipelineConfig, ctx: RunContext
    ) -> CoinbaseTradeReader:
        return CoinbaseTradeReader(config)

    def create_writer(self, config: PipelineConfig, ctx: RunContext):
        return None  # no venue-specific writer; use save_data(fmt="parquet")

    def compute_depth(self, events, config, source, ctx):
        return None  # use the standard price-level depth pipeline

    def config_defaults(self) -> dict:
        return {}  # e.g. {"price_decimals": 2, "timestamp_unit": "ms"}

    def required_context(self) -> list[str]:
        return []  # e.g. ["trading_date"] if filenames carry no date


register_source("coinbase", CoinbaseSource)
```

!!! warning "`compute_depth` must be defined"
    The pipeline calls `source.compute_depth(...)` unconditionally. **Return
    `None`** to use the standard price-level depth pipeline (what almost every
    venue wants). Only return a `(depth, depth_summary)` tuple if your venue
    ships ground-truth depth — as LOBSTER does from its orderbook file.

Using it — programmatically and from the CLI:

```python
from ob_analytics import Pipeline, list_sources

result = Pipeline.from_source("coinbase").run("coinbase_book.json")
print(list_sources())  # [..., "coinbase", ...]
```

```bash
ob-analytics process coinbase_book.json --source coinbase
```

Per-run parameters that vary across runs of the *same* source (LOBSTER's
`trading_date` is the canonical example) belong on
[`RunContext`](api/protocols.md), not the constructor:
`Pipeline.from_source("coinbase", ctx=RunContext(trading_date="2024-01-02"))`.

### Typed settings

Fixed per-source configuration — an exchange id, a depth limit — is a typed
`SourceSettings` (a frozen pydantic model), not an untyped dict. Subclass it and
declare the fields; the source carries an instance as its `settings`:

```python
from ob_analytics import SourceSettings


class CoinbaseSettings(SourceSettings):
    channel: str = "full"
    depth_limit: int = 100


source = CoinbaseSource(settings=CoinbaseSettings(depth_limit=50))
```

The built-in `CcxtSource` is the worked example
(`CcxtSettings(exchange=..., depth_limit=..., poll_interval=...)`).

### The live capability

A `LiveSource` translates a venue's WebSocket (or REST-poll) feed into the same
event dicts the pipeline reads. It only **parses**: persistence, raw-frame
archival, reconnect/rate-limiting, signal handling, and `meta.json`
finalisation are all handled generically by the runner
(`ob_analytics.live._runner.run_capturer`). Add the three async methods to your
source (alongside the offline factories, if it does both):

```python
from collections.abc import AsyncIterator

from ob_analytics.live import CaptureConfig, EventDict


class CoinbaseSource:  # ... plus the offline members above
    async def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        # Yield the opening book. L3: one action="created" per resting order;
        # L2: one absolute-size depth row per level.
        book = await self._fetch_rest_snapshot(config.pair)
        for level in book:
            yield {
                "id": level["order_id"],
                "timestamp": level["ts"],
                "exchange_timestamp": level["ts"],
                "price": level["price"],
                "volume": level["size"],
                "action": "created",
                "direction": level["side"],
            }

    async def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        # Yield (kind, event, raw_frame) until config.minutes elapse. kind is
        # "order" (L3) / "depth" (L2) / "trade"; raw_frame archives to raw.jsonl.
        async for raw in self._ws_messages(config):
            kind, event = self._parse(raw)
            yield (kind, event, raw)

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        # L3: one action="deleted" per still-resting order, so every id gets a
        # complete created -> ... -> deleted lifecycle. L2: usually nothing.
        for level in self._open_orders.values():
            yield {**level, "action": "deleted"}

    # Optional — satisfies SupportsDiagnostics; merged into meta.json.
    def diagnostics(self) -> dict[str, Any]:
        return {"reconnects": self._reconnects}
```

Capturing from the CLI (requires the `[live]` extra):

```bash
ob-analytics capture coinbase --pair btcusd --minutes 10 --out capture/
ob-analytics capture --list   # show live-capable sources
```

The runner writes `orders.csv` (L3) or `depth.csv` (L2) plus `trades.csv`, in
the same schema the pipeline reads, so a capture feeds straight back in:
`Pipeline.from_source("coinbase").run("capture/")`.

### Shipping a source as its own package

A source can live entirely outside ob-analytics and load through the
`ob_analytics.sources` **entry-point group** — no edit to the core. In your
package's `pyproject.toml`:

```toml
[project.entry-points."ob_analytics.sources"]
coinbase = "my_package.coinbase:CoinbaseSource"
```

`ob_analytics.sources.load_source_plugins()` (run at `import ob_analytics`)
discovers and registers every advertised source, so
`Pipeline.from_source("coinbase")` and `ob-analytics process --source coinbase`
resolve it with nothing else installed.

---

## 2. A new export format

A writer satisfies the `DataWriter` Protocol — a single `write(data, dest)`
method, where `data` is a **dict of DataFrames** keyed by name. You register a
*factory* `(config, ctx) -> DataWriter`, not the class, so writers that need
run state (e.g. [`LobsterWriter`](api/lobster.md) reads `trading_date` from
`ctx`) can pull it.

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ob_analytics.data import register_writer, list_writers


class DuckDBWriter:
    """Satisfies the DataWriter Protocol."""

    def write(
        self, data: dict[str, pd.DataFrame], dest: str | Path, **kwargs: Any
    ) -> Path:
        import duckdb

        dest = Path(dest)
        con = duckdb.connect(str(dest))
        for name, df in data.items():
            con.register("_df", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
            con.unregister("_df")
        con.close()
        return dest


# Factory signature: (config, ctx) -> DataWriter
register_writer("duckdb", lambda config, ctx: DuckDBWriter())
print(list_writers())  # [..., "duckdb", ...]
```

Using it — `save_data` takes a **dict of DataFrames**, not a `PipelineResult`:

```python
from ob_analytics import Pipeline, save_data

result = Pipeline().run("orders.csv")
save_data(
    {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
    },
    "out/analysis.duckdb",
    fmt="duckdb",
)
```

The built-in `"parquet"` and `"pickle"` formats need no registration; named
formats become available as soon as their factory is registered.

---

## 3. A new plot

A plot is two pieces, deliberately split so the data layer never imports a
plotting library:

1. a **`prepare_*` function** that returns a plain dict of plot-ready data, and
2. a **renderer** registered under the coordinate `(concept, level, backend)`.

The **level** is the order-book resolution the plot renders at: `Level.L2`
(Market-By-Price aggregate) or `Level.L3` (Market-By-Order, per order), or
`None` for a level-less plot such as a derived metric. A concept registered at
a single level dispatches without naming it; registering the *same* concept at
both `L2` and `L3` makes it *comparable*, and callers then pass `level=`.

The matplotlib backend calls `renderer(data, ax)` (or
`renderer(data, ax, theme=theme)` when a theme is passed); other backends call
`renderer(data)`.

```python
from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes

from ob_analytics.visualization import RENDERERS, DEFAULT_THEME, Level, PlotTheme, plot


def prepare_cumvol_data(trades: pd.DataFrame) -> dict:
    """Pure data prep — no matplotlib imports here."""
    df = trades[["timestamp", "volume", "direction"]].sort_values("timestamp").copy()
    # `direction` is a categorical; cast to str before mapping to numbers.
    sign = df["direction"].astype(str).map({"buy": 1, "sell": -1}).fillna(0)
    df["signed_cumvol"] = (sign * df["volume"]).cumsum()
    return {"series": df}


def mpl_cumvol(data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    df = data["series"]
    ax.plot(df["timestamp"], df["signed_cumvol"])
    ax.axhline(0, lw=0.5)
    ax.set_ylabel("signed cumulative volume")
    return ax.figure


RENDERERS.register(("cumvol", Level.L2, "matplotlib"), mpl_cumvol)  # None = level-less metric
```

Using it:

```python
from ob_analytics import Pipeline
from ob_analytics.visualization import PlotTheme, plot

result = Pipeline().run("orders.csv")

fig = plot("cumvol", backend="matplotlib", **prepare_cumvol_data(result.trades))

# Override the theme per call (matplotlib only):
fig = plot(
    "cumvol",
    theme=PlotTheme(style="darkgrid"),
    **prepare_cumvol_data(result.trades),
)
```

**A custom backend.** Register renderers under your backend name, then point
the dispatcher at the module so it imports lazily on first use:

```python
from ob_analytics.visualization import register_plot_backend

# In your package, e.g. my_pkg/_bokeh.py, call at import time:
#     RENDERERS.register(("cumvol", Level.L2, "bokeh"), bokeh_cumvol)  # def bokeh_cumvol(data): ...
register_plot_backend("bokeh", "my_pkg._bokeh")

fig = plot("cumvol", backend="bokeh", **prepare_cumvol_data(result.trades))
```

**In the gallery.** There is no panel registry. To put a custom plot in the
HTML gallery, pass it through `extra_panels=` — see the
[Gallery API](api/gallery.md).

---

## 4. A new metric

Metrics are not swapped at runtime, so they have no registry, no Protocol, and
no wrapper class. A metric is a plain function over a DataFrame — almost always
`result.trades`. The built-ins
([`compute_vpin`](api/flow_toxicity.md), [`compute_kyle_lambda`](api/flow_toxicity.md),
[`order_flow_imbalance`](api/flow_toxicity.md)) follow exactly this convention.

```python
import numpy as np
import pandas as pd


def amihud_illiquidity(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Amihud (2002) illiquidity: |return| per unit of traded value, resampled."""
    df = trades.set_index("timestamp").sort_index()
    abs_ret = df["price"].resample(freq).last().pct_change().abs()
    value = (df["price"] * df["volume"]).resample(freq).sum()
    illiq = (abs_ret / value.replace(0, np.nan)).rename("amihud")
    return illiq.to_frame()
```

Using it:

```python
from ob_analytics import Pipeline

result = Pipeline().run("orders.csv")
illiq = amihud_illiquidity(result.trades, freq="5min")
```

To visualise a metric, prepare its data and register a renderer (§3), or pass a
panel to the gallery via `extra_panels=`.

---

## Making registration fire

`register_*` runs as an import side-effect, so the registering module must be
imported before the name is used. Built-ins register themselves when
`ob_analytics` (and `ob_analytics.live`) are imported. For your own
extensions you have two options:

- **Ship as a package** with an entry point (§1, [Shipping a source as its own
  package](#shipping-a-source-as-its-own-package)). This is the cleanest path
  for a source: `load_source_plugins()` finds it at `import ob_analytics`, no
  wiring needed.
- **Import the module once at startup** — most cleanly from your package's
  `__init__.py` — for a plot, a writer, or a source you do not package
  separately:

  ```python
  # my_pkg/__init__.py
  from my_pkg import coinbase  # noqa: F401  — fires register_source / register_writer
  ```

After that, `Pipeline.from_source("coinbase")`, `plot("cumvol", ...)`,
`save_data(..., fmt="duckdb")`, and `ob-analytics capture coinbase` all resolve
your registrations with no further wiring.
