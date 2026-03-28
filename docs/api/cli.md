---
title: CLI
---

# Command-Line Interface

The `ob-analytics` command is registered as a console entry point when the
package is installed. It provides subcommands for processing data, generating
galleries, and running demo pipelines.

```bash
ob-analytics process orders.csv -o results/
ob-analytics gallery results/parquet/ -o my_gallery/
ob-analytics bitstamp-demo --input orders.csv
ob-analytics lobster-demo --ticker AAPL
```

Pass `-v` / `--verbose` for debug-level logging.

::: ob_analytics.cli
