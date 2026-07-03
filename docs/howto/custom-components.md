---
title: Plug in custom components
---

# Plug in custom components

Every pipeline stage is a [Protocol](../api/protocols.md). Implement the right
method signature on any object and pass it in — no inheritance required.
This page shows the two most common cases: a custom event loader and a custom
trade source. For the full extension surface (formats, writers, plot
backends, live capturers, registries), see
[Extending ob-analytics](../extending.md).

## Custom event loader

For non-Bitstamp, non-LOBSTER data, implement the `EventLoader` protocol.
Your `load` method must return a DataFrame with the columns subsequent
stages consume:

| Column | Type | Notes |
|--------|------|-------|
| `id` | int / str | Exchange-assigned order identifier |
| `event_id` | int | Sequential, 1-based, unique per event |
| `original_number` | int | Original input row number (stable event order) |
| `timestamp` | datetime64 | Local receive time |
| `exchange_timestamp` | datetime64 | Server time stamp |
| `price` | float | Order price |
| `volume` | float | Remaining size |
| `action` | category | `created` / `changed` / `deleted` |
| `direction` | category | `bid` / `ask` |
| `fill` | float | Volume executed by this event (0 for non-fills) |

The columns above are the contract the pipeline reads; see
[Data Contracts](../api/schemas.md) for the canonical column lists and the
`validate_*` helpers.

### Generic CSV loader

```python
from pathlib import Path
import pandas as pd
from ob_analytics import Pipeline, PipelineConfig
from ob_analytics.bitstamp import BitstampTradeReader

class GenericCsvLoader:
    """Load events from a CSV with different column names."""

    COLUMN_MAP = {
        "order_id": "id",
        "event_time": "timestamp",
        "server_time": "exchange_timestamp",
        "side": "direction",
        "type": "action",
    }
    ACTION_MAP = {"new": "created", "partial_fill": "changed",
                  "fill": "changed", "cancel": "deleted"}
    DIRECTION_MAP = {"buy": "bid", "sell": "ask"}

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def load(self, source: str | Path) -> pd.DataFrame:
        df = pd.read_csv(source)
        df = df.rename(columns=self.COLUMN_MAP)
        df["action"] = df["action"].map(self.ACTION_MAP)
        df["direction"] = df["direction"].map(self.DIRECTION_MAP)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["exchange_timestamp"] = pd.to_datetime(df["exchange_timestamp"])
        # Compute fill deltas and event IDs (see BitstampLoader for reference)
        ...
        return df

result = Pipeline(
    loader=GenericCsvLoader(),
    trade_source=BitstampTradeReader(),
).run("my_exchange_data/orders.csv")  # requires sibling trades.csv
```

### Cryptofeed L3 adapter (conceptual)

```python
from pathlib import Path
import pandas as pd
from ob_analytics import PipelineConfig

class CryptofeedLoader:
    """Load events from a cryptofeed L3 order book log.

    Size of 0 means deletion. This adapter tracks state to infer
    actions and compute fills.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def load(self, source: str | Path) -> pd.DataFrame:
        raw = pd.read_parquet(source)
        known_orders: dict[str, float] = {}
        rows = []

        for _, row in raw.iterrows():
            oid = row["order_id"]
            size = row["size"]
            prev_size = known_orders.get(oid)

            if prev_size is None:
                action, fill = "created", 0.0
            elif size == 0:
                action, fill = "deleted", prev_size
            else:
                action, fill = "changed", prev_size - size

            if size > 0:
                known_orders[oid] = size
            elif oid in known_orders:
                del known_orders[oid]

            rows.append({
                "id": oid,
                "timestamp": row["receipt_timestamp"],
                "exchange_timestamp": row["exchange_timestamp"],
                "price": row["price"],
                "volume": size,
                "action": action,
                "direction": "bid" if row["side"] == "BID" else "ask",
                "fill": fill,
            })

        events = pd.DataFrame(rows)
        events["event_id"] = range(1, len(events) + 1)
        events["original_number"] = events["event_id"]
        return events
```

!!! note
    A production adapter would handle out-of-order messages, reconnection
    gaps, and exchange-specific quirks. The key point: any object with a
    `load` method returning the right DataFrame works — no subclassing
    required.

## Custom trade source

Implement `TradeSource` (a `load(events, source) -> DataFrame` method) when
trades come from an API, a database, or a non-CSV layout:

```python
import pandas as pd
from ob_analytics import Pipeline, PipelineConfig, sample_csv_path

class ApiTradeSource:
    def load(self, events: pd.DataFrame, source: object) -> pd.DataFrame:
        # Build the canonical trades DataFrame (timestamp, price, volume,
        # direction, maker_event_id, taker_event_id, maker, taker, …)
        raise NotImplementedError

result = Pipeline(
    config=PipelineConfig(),
    trade_source=ApiTradeSource(),
).run(sample_csv_path())
```

Bundle defaults in a `Format` subclass — see [Protocols](../api/protocols.md).

## Related

- [Extending ob-analytics](../extending.md) — formats, writers, plot backends, capturers, registries
- [Data Contracts](../api/schemas.md) — canonical column lists + validators
