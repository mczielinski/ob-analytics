---
title: Databento
---

# Databento

Support for [Databento](https://databento.com/) DBN files in the
**market-by-order (MBO)** schema: a per-order feed with an `order_id` on every
record, so it replays through the full L3 reconstruction.

Use via `Pipeline(source=DatabentoSource(...))` or
`Pipeline.from_source("databento")`. See the
[how-to](../howto/databento.md) for the workflow, including how to size a query
before downloading it.

Key differences from LOBSTER:

- **Fixed-point prices** — raw prices are integers where one unit is 1e-9 of
  the quote currency (`price_divisor=1_000_000_000`).
- **Two clocks per record** — `ts_recv` (Databento's receive time, monotonic)
  becomes `timestamp`; `ts_event` (the venue's own) becomes
  `exchange_timestamp`.
- **Fills are separate records** — an execution is an `F` record that does not
  change the book, followed by the `C` or `M` that takes the size off it. The
  loader pairs the two, so `fill` tells an execution apart from a cancel.
- **No date to supply** — DBN carries absolute UTC timestamps, so
  `required_context()` is empty.

`databento` is an optional dependency:

```bash
pip install "ob-analytics[databento]"
```

::: ob_analytics.databento.DatabentoLoader

::: ob_analytics.databento.DatabentoTradeReader

::: ob_analytics.databento.DatabentoWriter

::: ob_analytics.databento.DatabentoSource

::: ob_analytics.databento.DatabentoSettings

::: ob_analytics.databento.read_mbo_frame
