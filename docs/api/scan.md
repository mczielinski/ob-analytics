---
title: Reading saved runs in parts
---

# Reading saved runs in parts

`load_data` reads every table of a saved run into memory. The functions on this
page read the same folder one batch of rows at a time, so their memory is set
by the batch size and the size of the result, not by the size of the file. Use
them on a run too large to load. [Scale and
chunking](../scale-and-chunking.md#reading-a-saved-run-in-parts) shows a worked
example and the memory measured.

| Function | What it gives |
|---|---|
| `scan_table` | the rows of one table as pandas frames, one batch at a time |
| `read_table` | the selected rows and columns of one table as one frame |
| `aggregate_by_time` | one row per fixed time bucket: first, last, min, max, sum, count or mean |
| `summarize_depth_file` | a new `depth_summary` from a saved `depth` table, for example with other bins |

The columns and the time range you give go to the Parquet reader, so it reads
only those columns and skips each row group whose times are all outside the
range. Each frame has the dtypes `load_data` gives and the tick size and lot
size on `attrs`.

::: ob_analytics.scan.scan_table

::: ob_analytics.scan.read_table

::: ob_analytics.scan.aggregate_by_time

::: ob_analytics.scan.summarize_depth_file

::: ob_analytics.scan.AGGREGATIONS

::: ob_analytics.scan.BATCH_ROWS
