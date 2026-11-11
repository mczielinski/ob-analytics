---
title: Extending
---

# Extending ob-analytics

Every pluggable surface in ob-analytics follows the same shape: implement a
small **Protocol** by structural typing (no base class to inherit), then
**register** it under a name. Things that are genuinely swappable at runtime —
data sources, metrics, export formats, plot backends, live capturers — live
in name-keyed registries. Things that aren't — themes — are plain values you
pass directly.

## How extension works

| Want to add… | Implement | Register with | Use via |
|---|---|---|---|
| **A data source** (new venue, file and/or live) | `Source` + `OfflineSource` and/or `LiveSource` | `register_source(name, cls)` (or an entry point) | `Pipeline.from_source(name)` · CLI `process --source name` / `capture name` |
| **An export format** | `DataWriter` | `register_writer(name, factory)` (or an entry point) | `save_data(data, path, fmt=name)` |
| **A plot** | a `prepare_*` function + a renderer | `RENDERERS.register((name, level, backend), fn)` (a whole backend: `register_plot_backend` or an entry point) | `plot(name, backend=...)` |
| **A metric** | `Metric` | `register_metric(metric)` (or an entry point) | `result.metric(name)` · `result.plot(name)` · its own gallery card |
| **A bar rule** | `BarRule` | `register_bar_rule(rule)` (or an entry point) | `bars(trades, rule=name)` |
| **A feature** | `Feature` | `register_feature(feature)` (or an entry point) | a column of `features(trades, quotes)` |

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

- **`OfflineSource`** — replay stored files. Two factories are required: a
  loader (`EventLoader` for L3, `DepthLoader` for L2) and a trade reader
  (`TradeReader`). The pipeline calls them once for each run, with that run's
  config and `RunContext`, so a source keeps nothing from one run to the next.
  Three more capabilities are optional, each its own small protocol:
  `SupportsWriter` (`create_writer`, a writer in the venue's own format),
  `SupportsComputeDepth` (`compute_depth`, depth the venue ships, as LOBSTER
  does) and `SupportsRequiredContext` (`required_context`, the `RunContext`
  fields a run needs). Leave out the ones you do not use.
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
    InstrumentGrid,
    Level,
    PipelineConfig,
    RunContext,
    SourceSettings,
    register_source,
)
from ob_analytics.schemas import cast_events


class CoinbaseLoader:
    """Satisfies the EventLoader Protocol."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def load(self, path: Any) -> pd.DataFrame:
        # Parse the venue feed into the canonical event columns (listed
        # below this example). See ob_analytics.bitstamp.BitstampLoader for
        # a full implementation.
        raw = pd.read_json(path)
        ...
        return cast_events(events)


class CoinbaseTradeReader:
    """Satisfies the TradeReader Protocol."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def load(self, events: pd.DataFrame, path: Any) -> pd.DataFrame:
        # Project explicit trade records into the canonical trades schema
        # (listed below this example). Return a frame, never None: a run
        # with no trades returns an empty frame with those columns.
        ...
        return trades


class CoinbaseSource:
    """An offline OfflineSource — no base class needed."""

    name = "coinbase"
    level = Level.L3  # per-order (Coinbase `full` channel)
    feed_type = FeedType.MATCHED_BOOK  # exchange matching engine: never crossed
    # The tick and lot size the venue's data is usually on; the run's config
    # starts from it, and a field the caller sets wins.
    grid = InstrumentGrid(tick_size=0.01, lot_size=1e-8)

    def __init__(self, settings: SourceSettings | None = None) -> None:
        self.settings = settings or SourceSettings()  # empty: no per-source knobs

    def create_loader(self, config: PipelineConfig, ctx: RunContext) -> CoinbaseLoader:
        return CoinbaseLoader(config)

    def create_trade_reader(
        self, config: PipelineConfig, ctx: RunContext, loader: Any
    ) -> CoinbaseTradeReader:
        # loader is this run's loader, for trades that come off the same read.
        return CoinbaseTradeReader(config)

    # Optional: define required_context to name RunContext fields a run
    # needs, e.g. ["trading_date"] if file names carry no date.  With no
    # compute_depth, the standard price-level depth stage runs.


register_source("coinbase", CoinbaseSource)
```

A source that writes its own format also has a `create_writer(config, ctx)`
method that returns a `DataWriter`, and its name then works as
`save_data(..., fmt="coinbase")`. Leave the method out when the source has no
format of its own, as here; `save_data`'s default Parquet format still saves
its runs.

How the venue's files encode prices and timestamps (a price in hundredths of a
cent, an epoch in microseconds) is the loader's business: write it into the
loader as a constant, or, when the format leaves it open, as a field of the
source's typed settings. `PipelineConfig` has no field for it.

The pipeline checks both frames on the way in and names any column that is
missing or has the wrong dtype. Build the events with
`cast_events(events)` and the trades with `cast_columns(trades, TRADE_DTYPES)`,
all three from `ob_analytics.schemas`, as the bundled loaders do, and the dtypes
come out right. The loader's events need these columns:

| Column | Type | What it holds |
|---|---|---|
| `event_id` | `int64` | A unique id for each row, in the order the rows happened |
| `id` | `int64`, or `object` for text ids | The order's id, the same on every row of one order |
| `timestamp` | `datetime64[ns, UTC]` | When you received the message |
| `exchange_timestamp` | `datetime64[ns, UTC]` | The venue's own time; repeat `timestamp` if the feed has none |
| `price` | `int64` | Integer ticks: the price divided by `config.tick_size` |
| `volume` | `int64` | Integer lots: the size divided by `config.lot_size` |
| `action` | categorical | `created`, `changed` or `deleted` |
| `direction` | categorical | `bid` or `ask` |
| `fill` | `int64` | The size executed at this event, in lots; `0` when nothing traded |

The pipeline adds `type` itself. The trade reader's frame needs `timestamp`,
`price`, `volume`, `direction` (`buy` or `sell`, the taker's side),
`maker_event_id` and `taker_event_id`. [The schema](schema.md) says what each
column means, including how `volume` and `fill` change over an order's life.

!!! note "Define `compute_depth` only for depth the venue ships"
    Without it the pipeline runs its standard price-level depth stage, which
    is what almost every venue wants. Define it only if your venue ships
    ground-truth depth, as LOBSTER does from its orderbook file. Return a
    `(depth, depth_summary)` tuple, or `None` to fall back to the standard
    stage for one run.

Using it — programmatically and from the CLI:

<!-- docs-test: skip needs a Coinbase file and a finished loader -->

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
`Pipeline.from_source("coinbase", ctx=RunContext(trading_date="2024-01-02"))`
sets it for every run, and `pipeline.run(path, ctx=...)` for one run.

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

### API keys

A source that needs an API key declares it as a settings field, marked with
`Credential`. The field must be typed `SecretStr | None` with a default of
`None`; any other type or default raises `TypeError` when the class is defined.

```python
from typing import Annotated

from pydantic import SecretStr

from ob_analytics import Credential, SourceSettings


class CoinbaseSettings(SourceSettings):
    channel: str = "full"
    api_key: Annotated[
        SecretStr | None,
        Credential(
            env="COINBASE_API_KEY",
            issued_at="the API page of your Coinbase account",
        ),
    ] = None
    private_key: Annotated[
        SecretStr | None,
        Credential(
            env="COINBASE_PRIVATE_KEY_PATH",
            issued_at="the API page of your Coinbase account",
            from_file=True,
        ),
    ] = None
```

What the declaration gives you:

- An unset field is read from its environment variable. With `from_file=True`,
  the variable holds a file path and the field holds the file's contents.
- `run_capture` checks every credential before it creates any output. A
  missing one stops the capture with the variable to set and `issued_at`.
- `ob-analytics sources` lists the variables.
- The key is removed from every file the capture writes and from the CLI log.

In the source, read the key with `self.settings.api_key.get_secret_value()`
only where you sign or send it. Do not yield your login message as a raw
frame: it holds no market data, and the runner would only mask the key in it.
[Use an API key](howto/api-keys.md) is the page for users.

### The live capability

A `LiveSource` translates a venue's WebSocket (or REST-poll) feed into the same
event dicts the pipeline reads. It only **parses**: persistence, raw-frame
archival, signal handling, and `meta.json` finalisation are handled by the
runner (`ob_analytics.live._runner.run_capturer`, one segment), and
reconnecting, rolling and restarting by `ob_analytics.live.run_capture`. So
`stream` does not reconnect: when the connection drops it raises, and
`run_capture` starts a new segment from a fresh snapshot (see
[Running for days](howto/live-capture.md#running-for-days)). Add the three
async methods to your source (alongside the offline factories, if it does
both):

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
        return {"dropped": self._dropped}

    # Optional — satisfies SupportsPreflight; runs before any output exists.
    # Raise ImportError with the install hint when an optional extra is missing.
    def preflight(self) -> None:
        import websockets  # noqa: F401
```

Capturing from the CLI (requires the `[live]` extra):

```bash
ob-analytics capture coinbase --pair btcusd --minutes 10 -o capture/
ob-analytics capture --list   # show live-capable sources
```

Each segment of the capture holds `orders.csv` (L3) or `depth.csv` (L2) plus
`trades.csv`, in the same schema the pipeline reads, so a segment feeds
straight back in: `Pipeline.from_source("coinbase").run("capture/seg-0001/orders.csv")`.

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
method, where `data` is a **dict of DataFrames** keyed by name, and a `target`
that says whether `dest` is a folder (`WriteTarget.FOLDER`) or one file
(`WriteTarget.FILE`). `save_data` checks the path against `target` before it
calls `write`, and refuses the other kind. You register a
*factory* `(config, ctx) -> DataWriter`, not the class, so writers that need
run state (e.g. [`LobsterWriter`](api/lobster.md) reads `trading_date` from
`ctx`) can pull it.

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ob_analytics.data import register_writer, list_writers
from ob_analytics.protocols import WriteTarget


class DuckDBWriter:
    """Satisfies the DataWriter Protocol."""

    target = WriteTarget.FILE  # one .duckdb file

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

<!-- docs-test: skip needs duckdb, which ob-analytics does not install -->

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

The built-in `"parquet"` format is a registered writer too, so it appears in
`list_writers()` and follows exactly these rules. Registering your own under
that name **replaces** the built-in one — there is no special case in
`save_data` for it. `list_writers()` also lists each source that writes its
own format (one with a `create_writer` method), because `save_data` accepts a
source name as `fmt`.

### Asking for Arrow instead of pandas

`data` is a mapping of pandas frames, and every writer above treats it as one.
It also offers `.arrow()`, for a writer whose target is columnar:

```python
class ArrowIpcWriter:
    """Satisfies the DataWriter Protocol."""

    def write(
        self, data: dict[str, pd.DataFrame], dest: str | Path, **kwargs: Any
    ) -> Path:
        import pyarrow.feather as feather

        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for name, table in data.arrow().items():
            feather.write_feather(table, dest / f"{name}.arrow")
        return dest
```

Each table from `.arrow()` carries the schema version and, when the run declared
a `tick_size`, the tick-size metadata — the same key-value metadata a canonical
Parquet file carries. Calling `pyarrow.Table.from_pandas` yourself instead
drops both, and a reader then treats the result as legacy data with prices left
as raw integer ticks.

The frame type is therefore not fixed in the protocol: the CSV writers want
pandas, a Parquet or Arrow writer wants Arrow, and each asks for what it needs.

### Exporting to a backtesting engine

Two formats ship for the engines next door (issue #113). Neither needs its
engine installed:

```python
from ob_analytics import Pipeline, save_data
from ob_analytics.config import PipelineConfig

config = PipelineConfig(tick_size=0.01, lot_size=1e-8)
result = Pipeline(config=config).run("orders.csv")
data = {"events": result.events, "trades": result.trades}

save_data(data, "out/session.npz", fmt="hftbacktest", config=config)
save_data(data, "out/deltas.parquet", fmt="nautilus", config=config)
```

Both scale the integer tick prices back to the quote currency using the run's
`tick_size`, **and** the integer lot sizes back to the base asset using its
`lot_size`, so pass the same `config` the run used. Both matter: a `config`
whose `lot_size` does not match the data writes sizes that are wrong by that
ratio. hftbacktest takes the file without complaint, and Nautilus'
`OrderBookDeltaDataWrangler` raises `ValueError: 'size' not a positive integer,
was 0` once the sizes round to zero at the instrument's `size_precision`.

Exporting the bundled toy data therefore means passing both toy constants, not
the `PipelineConfig` defaults — whose `lot_size` of `1e-8` would turn a size of
2 into `2e-08`:

```python
from ob_analytics import save_data
from ob_analytics.config import PipelineConfig
from ob_analytics.datasets import LOT_SIZE, TICK_SIZE, toy_events, toy_trades

config = PipelineConfig(tick_size=TICK_SIZE, lot_size=LOT_SIZE)
data = {"events": toy_events(), "trades": toy_trades()}

save_data(data, "out/toy.npz", fmt="hftbacktest", config=config)
save_data(data, "out/toy-deltas.parquet", fmt="nautilus", config=config)
```

The toy book's prices and sizes are already whole ticks and whole lots, so both
constants are `1.0` and the export carries the same numbers the toy script
lists. A Nautilus book built from `out/toy-deltas.parquet` then reproduces our
own touch at the end of the stream: best bid 99, best ask 102.

**hftbacktest** gets its feed-event array under the `data` member of the npz,
which is what `np.load(path)["data"]` and `BacktestAsset.data([...])` read. The
engine replays two clocks, an exchange one and a local one, and requires each to
run forwards. Those two orders are not the same order — on the bundled Bitstamp
sample the exchange stamp leads the receive stamp by 0.08 to 2.7 seconds, and
they disagree about 14,966 of 314,057 events — so an event whose clocks disagree
is written twice, once per timeline, exactly as the engine's own
`correct_event_order` would. hftbacktest's `validate_event_order` accepts the
result.

**Nautilus** gets a Parquet file holding the frame its
`OrderBookDeltaDataWrangler` takes: a UTC index and the columns `action`,
`side`, `price`, `size`, `order_id`, `flags`, `sequence`.

<!-- docs-test: skip needs nautilus-trader, which the docs job does not install -->

```python
import pandas as pd
from nautilus_trader.persistence.wranglers import OrderBookDeltaDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

instrument = TestInstrumentProvider.btcusdt_binance()  # see size_precision below
deltas = OrderBookDeltaDataWrangler(instrument).process(
    pd.read_parquet("out/toy-deltas.parquet")
)
```

Writing the file does not need Nautilus, but reading it back does, and
nautilus-trader publishes no build for Python 3.11 that works with pandas 3,
which ob-analytics uses. Read the export from a Python 3.12 or later
environment. The tests that compare our book with Nautilus' run there for the
same reason:

```bash
UV_PROJECT_ENVIRONMENT=$HOME/.cache/ob-analytics-py312 uv run --python 3.12 --group backtest-engines pytest tests/test_backtest_parity.py
```

The `instrument` you pass must declare a `size_precision` fine enough for your
data. Nautilus converts each size to a fixed-point `Quantity` and rejects one
that rounds to zero, so a BTC feed carrying single-satoshi orders needs
`size_precision=8`; the bundled `TestInstrumentProvider.btcusdt_binance()`, at
6, refuses them.

An order that was fully filled leaves a `deleted` event whose canonical volume
is zero, because the volume on a delete is the size removed without trading and
a filled order had nothing left to cancel. Nautilus rejects a zero-size delta, so the export
gives that delete the size the order last rested at. An order that never rested
at a positive size is dropped: there is nothing truthful to say about it.

---

## 3. A new plot

A plot is two pieces, deliberately split so the data layer never imports a
plotting library:

1. a **`prepare_*` function** that returns a plain dict of plot-ready data, and
2. a **renderer** registered under the coordinate `(concept, level, backend)`.

The **level** is the order-book resolution the plot renders at: `Level.L2`
(Market-By-Price aggregate) or `Level.L3` (Market-By-Order, per order). A plot
with no L2 and L3 variants of its own can instead be **level-less**, registered
at `None`: a metric's chart is, and so is the example below, which reads only
trades. The kind decides how the plot is called and where the gallery shows it,
not what data it reads: the built-in `trade_size` reads only trades and is
registered at `L2`. A concept is one kind or the other, on every backend:
registering it both at `None` and at a level raises `ConfigError`.

A concept registered at a single level dispatches without naming it;
registering the *same* concept at both `L2` and `L3` makes it *comparable*, and
callers then pass `level=`.

The matplotlib backend calls `renderer(data, ax)`; other backends call
`renderer(data)`. When the caller passes a theme, `plot()` adds `theme=theme`
for any renderer that accepts it, so a renderer should take a keyword-only
`theme: PlotTheme = DEFAULT_THEME` and read its colours from `theme.palette`.
A renderer without a `theme` parameter still works; `plot()` logs a warning
and draws it without the theme.

```python
from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes

from ob_analytics.visualization import RENDERERS, DEFAULT_THEME, PlotTheme, plot


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
    ax.plot(df["timestamp"], df["signed_cumvol"], color=theme.palette.series)
    ax.axhline(0, lw=0.5, color=theme.palette.rule)
    ax.set_ylabel("signed cumulative volume")
    return ax.figure


# Level-less: it has no L2 and L3 variants of its own.
RENDERERS.register(("cumvol", None, "matplotlib"), mpl_cumvol)
```

Using it:

```python
from ob_analytics import Pipeline
from ob_analytics.visualization import PlotTheme, plot

result = Pipeline().run("orders.csv")

fig = plot("cumvol", backend="matplotlib", **prepare_cumvol_data(result.trades))

# Override the theme per call:
fig = plot(
    "cumvol",
    theme=PlotTheme(style="darkgrid"),
    **prepare_cumvol_data(result.trades),
)
```

**A custom backend.** Register renderers under your backend name, then point
the dispatcher at the module so it imports lazily on first use:

<!-- docs-test: skip needs the reader's own backend package -->

```python
from ob_analytics.visualization import register_plot_backend

# In your package, e.g. my_pkg/_altair.py, call at import time:
#     RENDERERS.register(("cumvol", None, "altair"), altair_cumvol)
#     # def altair_cumvol(data, *, theme=DEFAULT_THEME): ...
register_plot_backend("altair", "my_pkg._altair")

fig = plot("cumvol", backend="altair", **prepare_cumvol_data(result.trades))
```

Matplotlib (static, default), Plotly, and Bokeh already ship first-party this
way — `backend="bokeh"` covers the core concepts (`trade_tape`,
`depth_heatmap`, `book_snapshot`, `depth_chart`) for Bokeh / Panel server
dashboards and streaming views (`pip install ob-analytics[bokeh]`).

**In the gallery.** There is no panel registry. Build the model, add a card
for the plot, and render that model instead of a bare result — see the
[Gallery API](api/gallery.md). The gallery draws the plot through the renderers
already registered, so nothing is registered a second time. It draws each
backend in `backends=`, by default Plotly (when it is installed) and Matplotlib.
A backend with no renderer for the plot shows "Not available" on its card.
The example registers only a Matplotlib renderer, so it asks for that backend
alone. A level-less plot is a `PlotSpec` appended to the model's `analytics`
list. If it goes in the wrong list, its card shows "Not available" and says
which list it belongs in:

```python
from ob_analytics.visualization.gallery import (
    PlotSpec,
    build_gallery_model,
    generate_gallery,
)

model = build_gallery_model(result)
model.analytics.append(
    PlotSpec("cumvol", "Cumulative Volume", "cumvol", prepare_cumvol_data, {"trades": result.trades})
)
generate_gallery(result, "output/gallery/", model=model, backends=["matplotlib"])
```

A plot registered at a level is a `PlotConcept` appended to the model's
`concepts` list instead, before `generate_gallery` is called. Its `variants` map
each level it is registered at to a `PlotSpec`, and the gallery gives it one
card per level:

<!-- docs-test: skip prepare_my_face is the reader's own function -->

```python
from ob_analytics.visualization import Level
from ob_analytics.visualization.gallery import PlotConcept

spec = PlotSpec("my_face", "My Face", "my_face", prepare_my_face, {"depth": result.depth})
model.concepts.append(PlotConcept("my_face", "My Face", {Level.L2: spec}))
```

---

## 4. A new metric

A metric measures a finished run and draws as a level-less plot. It is a plain
object with six members — no base class to inherit, the same structural typing
the other surfaces use:

- `name` — the registry key, and the plot concept the metric draws under.
- `title` — the title of its gallery card.
- `levels` — the resolutions it applies to. A metric that reads per-order
  events declares `(Level.L3,)` and is skipped on an L2 run instead of failing
  on the empty `events` table.
- `needs` — the run's tables it cannot be computed without, such as
  `("trades",)`. The metric is skipped on a run where one of them has no rows.
- `compute(result)` — the measurement: takes a `PipelineResult`, returns a
  `pandas.DataFrame`.
- `prepare(frame)` — turns that table into the payload the renderer takes,
  exactly as a `prepare_*` function does for a plot (§3).

Both methods can take keyword-only settings after their first argument, each
with a default. Put the settings of the measurement (a window, a bucket size)
on `compute` and the settings of the picture (a threshold line, a time window)
on `prepare`. `result.plot(name, **kwargs)` sends each keyword to the method
that names it, so the two must not share a name.

Nine metrics ship registered: `l1_ticker`, `book_signals`, `vpin`,
`kyle_lambda`, `order_flow_imbalance`, `ofi_by_window`, `transaction_costs`,
`amihud` and `roll_spread` (see [the metric list](api/metrics.md)). Each wraps
a plain function you can still call directly, such as
[`vpin`](api/flow_toxicity.md) or [`amihud`](api/cost.md).
Registering is what makes a function run and plot from a result.

```python
from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes

from ob_analytics import Level, PipelineResult, register_metric
from ob_analytics.visualization import DEFAULT_THEME, RENDERERS, PlotTheme


def price_range(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Highest minus lowest trade price per window, in the quote currency."""
    # A run stores whole ticks, and its frames carry the tick size.
    integer = pd.api.types.is_integer_dtype(trades["price"])
    price = trades["price"] * (trades.attrs["tick_size"] if integer else 1.0)
    windows = price.groupby(trades["timestamp"].dt.floor(freq))
    out = (windows.max() - windows.min()).rename("price_range")
    return out.rename_axis("timestamp").reset_index()


class PriceRangeMetric:
    """Satisfies the Metric Protocol."""

    name = "price_range"
    title = "Price Range per Window"
    levels = (Level.L2, Level.L3)  # trades only: both resolutions have them
    needs = ("trades",)  # skipped on a run with no trades

    def compute(self, result: PipelineResult, *, freq: str = "1min") -> pd.DataFrame:
        return price_range(result.trades, freq=freq)

    def prepare(self, frame: pd.DataFrame) -> dict:
        return {"series": frame}


def mpl_price_range(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    df = data["series"]
    ax.plot(df["timestamp"], df["price_range"])
    ax.set_ylabel("high - low")
    return ax.figure


register_metric(PriceRangeMetric())
RENDERERS.register(("price_range", None, "matplotlib"), mpl_price_range)  # None = level-less
```

Note what is registered: an *instance*, not a class. A metric needs no per-run
construction, so the object registered is the object called. Its settings, such
as `freq` above, are keyword arguments with defaults.

Using it:

```python
from ob_analytics import Pipeline
from ob_analytics.visualization import available_concepts

result = Pipeline().run("orders.csv")

result.metric("price_range")              # the table
result.metric("price_range", freq="5min") # the table, with a setting changed
result.metrics()                          # every metric that applies to this run
result.plot("price_range")                # the face, through the renderer above
result.plot("price_range", freq="5min")   # freq goes to compute
available_concepts(result)                # lists "price_range" with an empty level list
```

Metrics run when asked for, not during `Pipeline.run`, so a run pays only for
the metrics it uses and a metric that raises cannot break the pipeline.
`result.metrics()` runs every registered metric whose `levels` include the
run's resolution and whose `needs` tables all have rows.

**In the gallery.** A registered metric becomes a gallery card on its own —
`generate_gallery(result, ...)` draws it beside the built-in faces with no
extra step needed. A metric is computed when its card is drawn, so listing
concepts or plotting another face does not run it. A metric that raises is
logged and its card says why, so one broken metric does not stop the gallery
being built.

**Units.** `result.metric`, `result.plot` and the gallery all hand a metric
the result as stored, with prices in integer ticks and sizes in integer lots
(see [the schema](schema.md)), so the table and the picture always agree.
Report the table in the instrument's own units, as the built-in metrics do:
prices in the quote currency and sizes in BTC, shares or contracts. Each
run's tables carry `attrs["tick_size"]` and `attrs["lot_size"]` for this, as
`price_range` reads above. A setting is in the instrument's own units too: a
size such as VPIN's `bucket_volume` in BTC, shares or contracts, and a price
in the quote currency. Hand a size setting to a function that converts it,
such as `vpin` or `bars`, or divide it by `lot_size` yourself when
the frame's sizes are integers.

**Shipping a metric as its own package.** Advertise it under the
`ob_analytics.metrics` entry-point group and `load_metric_plugins()` finds it at
`import ob_analytics`, the same as a source:

```toml
# pyproject.toml of your package
[project.entry-points."ob_analytics.metrics"]
price_range = "my_pkg.price_range:PriceRangeMetric"
```

The entry point names the metric *class*; discovery constructs it with no
arguments and registers it under its own `name`. A metric with required
settings should either default them or register itself on import instead.

---

## 5. A new bar rule

[`bars()`](api/bars.md) resamples a trade stream into open/high/low/close rows.
Everything a bar carries — the OHLCV columns, VWAP, the signed-volume split —
is shared, so what a bar *type* actually decides is one thing: where the
boundaries fall. That decision is a **`BarRule`**, and a rule of your own is
the boundary decision and nothing else.

A rule states three members:

- `name` — what `bars(trades, rule=name)` looks it up by.
- `normalize(threshold)` — read the threshold in the rule's own type, or raise
  `ConfigError`. Called once before `assign`, so reading and checking the
  threshold live in one place and cutting in another.
- `assign(frame, threshold)` — return the 0-based bar index of each trade.

A rule may also state `units`, the trade columns its threshold is measured in:
`("volume",)` for a size, `("volume", "price")` for a turnover, `("price",)`
for a price distance. A user then gives the threshold in the instrument's own
units and the quote currency, and `bars()` converts it to the frame's units
(whole lots and ticks on a pipeline result) before `assign` sees it. Leave it
out for a threshold with no units, such as a duration or a count.

`assign` is handed a normalized frame in trade order, with `timestamp`,
`price`, `volume`, `turnover` (price × size) and `sign` (`+1` buyer-initiated,
`-1` seller-initiated) — so a rule never has to sort trades or work out the
aggressor side itself.

`default_threshold(frame, target_bars)` supplies a threshold when the caller
passes none, aiming at about `target_bars` bars.

Here is a rule that starts a new bar whenever price moves a set distance from
where the bar opened — a "range bar":

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from ob_analytics import register_bar_rule
from ob_analytics.exceptions import ConfigError


class RangeRule:
    """A new bar every time price travels `threshold` from the bar's open."""

    name = "range"
    units = ("price",)

    def default_threshold(self, frame: pd.DataFrame, target_bars: int) -> float:
        span = frame["price"].max() - frame["price"].min()
        return float(span) / target_bars or 1.0

    def normalize(self, threshold: object) -> float:
        size = float(threshold)
        if not size > 0:
            raise ConfigError(f"bars: the 'range' rule needs a positive move, got {threshold!r}")
        return size

    def assign(self, frame: pd.DataFrame, threshold: float) -> np.ndarray:
        price = frame["price"].to_numpy(dtype=float)
        index = np.empty(price.size, dtype=np.int64)
        bar, opened_at = 0, price[0]
        for i, p in enumerate(price):
            index[i] = bar
            if abs(p - opened_at) >= threshold:
                bar += 1
                opened_at = p
        return index


register_bar_rule(RangeRule())
```

```python
from ob_analytics import bars

bars(result.trades, "range", 25)  # a new bar every $25 of price travel
```

The bar table, the gallery face and the plot all work unchanged: they never
knew which rule cut the bars.

---

## 6. A new feature

[`features()`](api/features.md) builds one tidy table: a point in time on each
row, a microstructure feature in each column. Where the rows fall is a bar
rule, decided before any measuring starts, so what a **`Feature`** decides is
one thing: what its columns read on each row.

A feature states four members:

- `name` — what `features(..., include=[name])` looks it up by.
- `columns` — the columns it writes, in order. Declared rather than
  discovered, so the table's shape is known before anything is measured.
- `requires` — the columns of the prepared frame it reads. A feature named
  explicitly whose requirement is missing raises; one selected by default is
  skipped, which is how the book features drop out of a run with no quotes.
- `compute(frame)` — return one array per declared column, in row order.

`compute` is handed one row per bar, in time order, carrying the bar's own
columns — `open`, `high`, `low`, `close`, `volume`, `turnover`, `n_trades`,
`vwap`, `buy_volume`, `sell_volume`, `signed_volume` — and the book as it
stood at the bar's close, joined on from the quotes frame. **That frame holds
nothing from after a row's close**, so a feature that works row by row is
free of look-ahead already. One that looks along the frame has to look
backwards: `shift(1)`, or a trailing `rolling` window.

Here is a feature that measures how far the bar's close sits inside its own
range — a "close location value":

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from ob_analytics import features, register_feature


class CloseLocationFeature:
    """Where the close fell in the bar's range: -1 at the low, +1 at the high."""

    name = "close_location"
    columns = ("close_location",)
    requires = frozenset({"high", "low", "close"})

    def compute(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        high = frame["high"].to_numpy(dtype=float)
        low = frame["low"].to_numpy(dtype=float)
        close = frame["close"].to_numpy(dtype=float)
        span = high - low
        with np.errstate(invalid="ignore", divide="ignore"):
            located = np.where(span > 0, (2 * close - high - low) / span, 0.0)
        return {"close_location": located}


register_feature(CloseLocationFeature())
```

```python
features(result.trades, result.depth_summary, "volume", 0.5)["close_location"]
```

It is measured with the built-in ten and its column lands after theirs. Name
it in `include` to get it on its own, or to put its column somewhere else.

### Changing a built-in

The features that look back over a trailing window carry its length in bars,
`n_bars`, on the instance, so a different one is a second registration, not an argument to thread through
`features()`. Registering under the same name replaces the built-in; under a
new name, both run:

```python
from ob_analytics.features import VpinFeature

register_feature(VpinFeature(n_bars=50))                # the paper's 50 buckets
register_feature(VpinFeature(name="vpin_5", n_bars=5))  # and a fast one beside it
```

The two one-column features name their column after themselves, so the second
registration above writes `vpin_5` and leaves `vpin` alone. Elsewhere the
columns are fixed, and two features that would write the same one are an
error rather than a silent overwrite — so a second `ReturnsFeature` under a
new name is refused, and the way to change that one is to register it under
`returns` and replace it.

---

## Making registration fire

`register_*` runs as an import side-effect, so the registering module must be
imported before the name is used. Built-ins register themselves when
`ob_analytics` (and `ob_analytics.live`) are imported. For your own
extensions you have two options:

- **Ship as a package** with an entry point (§1, [Shipping a source as its own
  package](#shipping-a-source-as-its-own-package)). This is the cleanest path:
  ob-analytics finds it on import, with no wiring needed. Each registry has its
  own entry-point group:

  | Group | The entry names | Registered under | Found at |
  |---|---|---|---|
  | `ob_analytics.sources` | a `Source` class | the entry name | `import ob_analytics` |
  | `ob_analytics.metrics` | a `Metric` class, constructed with no arguments | its `name` | `import ob_analytics` |
  | `ob_analytics.bar_rules` | a `BarRule` class, constructed with no arguments | its `name` | `import ob_analytics` |
  | `ob_analytics.features` | a `Feature` class, constructed with no arguments | its `name` | `import ob_analytics` |
  | `ob_analytics.writers` | a writer factory `(config, ctx) -> DataWriter` | the entry name | `import ob_analytics` |
  | `ob_analytics.plot_backends` | the backend module, imported on first use | the entry name | `import ob_analytics.visualization` |

  A plug-in that fails to load is skipped with a warning; the others still
  load.
- **Import the module once at startup** — most cleanly from your package's
  `__init__.py` — for a plot, a writer, or a source you do not package
  separately:

  <!-- docs-test: skip needs the reader's own package -->

  ```python
  # my_pkg/__init__.py
  from my_pkg import coinbase  # noqa: F401  — fires register_source / register_writer
  ```

After that, `Pipeline.from_source("coinbase")`, `plot("cumvol", ...)`,
`save_data(..., fmt="duckdb")`, and `ob-analytics capture coinbase` all resolve
your registrations with no further wiring.
