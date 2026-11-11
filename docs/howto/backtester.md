---
title: Export to a backtester
---

# Export to a backtester

Two `save_data` formats write files that two backtesting engines read:
`fmt="hftbacktest"` for [hftbacktest](https://github.com/nkaz001/hftbacktest)
and `fmt="nautilus"` for [Nautilus Trader](https://nautilustrader.io).
Neither format needs its engine installed:

```python
from ob_analytics import Pipeline, save_data

result = Pipeline().run("orders.csv")
data = {"events": result.events, "trades": result.trades}

save_data(data, "out/session.npz", fmt="hftbacktest", config=result.config)
save_data(data, "out/deltas.parquet", fmt="nautilus", config=result.config)
```

Both scale the integer tick prices back to the quote currency using the run's
`tick_size`, **and** the integer lot sizes back to the base asset using its
`lot_size`, so pass the config the run used, `result.config`. Both matter: a `config`
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

## Related

- [Save, load, and export](output.md) — every `save_data` format
- [`HftbacktestWriter`, `NautilusWriter`](../api/interop.md) — the API
- [Extending ob-analytics](../extending.md#2-a-new-export-format) — write a
  format of your own
