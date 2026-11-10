---
title: Use your own data
---

# Use your own Bitstamp-format data

How to point the pipeline at your own capture instead of the bundled sample,
and tune the configuration for other instruments. For non-Bitstamp,
non-LOBSTER sources, see [Custom components](custom-components.md).

## Custom Bitstamp-format CSV

Place `orders.csv` and `trades.csv` in the same directory (a capture made with
`ob-analytics capture bitstamp` shows the expected `trades.csv` schema).
Point the pipeline at the **orders** path; it resolves sibling `trades.csv`
automatically.

<!-- docs-test
import os
import shutil

os.mkdir("my_run")
for name in ("orders.csv", "trades.csv"):
    shutil.copy(name, "my_run")
-->

```python
from ob_analytics import Pipeline, PipelineConfig

config = PipelineConfig(
    tick_size=0.01,
    price_decimals=2,
    lot_size=1e-8,
    volume_decimals=8,
)
result = Pipeline(config=config).run("my_run/orders.csv")
```

## Configuration presets

Set the instrument's price step (`tick_size`) and size step (`lot_size`), with
the decimals that show one step (`price_decimals`, `volume_decimals`). A
`lot_size` finer than `volume_decimals` can show is refused, because sizes
would be shown rounded.

=== "BTC/USD (default)"

    ```python
    config = PipelineConfig(
        tick_size=0.01,
        price_decimals=2,
        lot_size=1e-8,
        volume_decimals=8,
    )
    ```

=== "ETH/USD"

    ```python
    config = PipelineConfig(
        tick_size=0.01,
        price_decimals=2,
        lot_size=1e-6,
        volume_decimals=6,
    )
    ```

=== "FX (EUR/USD)"

    ```python
    config = PipelineConfig(
        tick_size=0.0001,
        price_decimals=4,
        lot_size=0.01,
        volume_decimals=2,
        depth_bps=5,
        depth_bins=100,
    )
    ```

=== "High-Price Equity"

    ```python
    config = PipelineConfig(
        tick_size=0.01,
        price_decimals=2,
        lot_size=1.0,
        volume_decimals=0,
    )
    ```

## Related

- [LOBSTER data](lobster.md) — message + orderbook files
- [Databento MBO files](databento.md) — DBN per-order data for many venues
- [Custom components](custom-components.md) — write a loader for any other format
- [Configuration API](../api/config.md) — every `PipelineConfig` field
