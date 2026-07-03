---
title: Run from the command line
---

# Run from the command line

Installing the package registers the `ob-analytics` command. All
subcommands accept `-v` / `--verbose` for debug-level logging.

| Subcommand | Description |
|------------|-------------|
| `process` | Run the pipeline on a data source, save Parquet (optional `--gallery`) |
| `gallery` | Generate an HTML plot gallery from saved Parquet data |
| `bitstamp-demo` | Run the Bitstamp demo (pipeline + gallery) |
| `lobster-demo` | Run the LOBSTER demo on local data (pipeline + gallery) |
| `capture` | Live-capture market data from a registered venue (`[live]` extra) |

```bash
# Process a data source
ob-analytics process orders.csv -o results/
ob-analytics process data/ --format lobster --trading-date 2012-06-21
ob-analytics process orders.csv -o results/ --gallery

# Build a gallery from saved Parquet
ob-analytics gallery results/parquet/ -o my_gallery/
ob-analytics gallery results/parquet/ --volume-scale 1e-8 --title "My Analysis"

# End-to-end demos (pipeline + gallery)
ob-analytics bitstamp-demo --input orders.csv -o demo_out/
ob-analytics lobster-demo /path/to/lobster_data --trading-date 2012-06-21 -o demo_out/
```

The `bitstamp-demo` and `lobster-demo` subcommands are equivalent to
running `scripts/bitstamp_demo.py` / `scripts/lobster_demo.py` from a
clone — both run the pipeline, save Parquet, verify round-trip I/O, and
build an HTML plot gallery.

## Related

- [CLI Reference](../api/cli.md) — argparse-level docs for every flag
- [Capture live data](live-capture.md) — the `capture` verb in depth
