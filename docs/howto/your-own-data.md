---
title: Use your own data
---

# Use your own Bitstamp-format data

How to point the pipeline at your own capture instead of the bundled sample,
and tune the configuration for other instruments. For non-Bitstamp,
non-LOBSTER sources, see [Custom components](custom-components.md).

## Custom Bitstamp-format CSV

Place `orders.csv` and `trades.csv` in the same directory (see
`scripts/collect_bitstamp_btcusd.py` for the expected `trades.csv` schema).
Point the pipeline at the **orders** path; it resolves sibling `trades.csv`
automatically.

```python
from ob_analytics import Pipeline, PipelineConfig

config = PipelineConfig(price_decimals=2, volume_decimals=8)
result = Pipeline(config=config).run("my_run/orders.csv")
```

## Configuration presets

=== "BTC/USD (default)"

    ```python
    config = PipelineConfig(price_decimals=2, volume_decimals=8)
    ```

=== "ETH/USD"

    ```python
    config = PipelineConfig(price_decimals=2, volume_decimals=6)
    ```

=== "FX (EUR/USD)"

    ```python
    config = PipelineConfig(
        price_decimals=4,
        volume_decimals=2,
        depth_bps=5,
        depth_bins=100,
    )
    ```

=== "High-Price Equity"

    ```python
    config = PipelineConfig(price_decimals=2, volume_decimals=0)
    ```

## Related

- [LOBSTER data](lobster.md) — message + orderbook files
- [Custom components](custom-components.md) — write a loader for any other format
- [Configuration API](../api/config.md) — every `PipelineConfig` field
