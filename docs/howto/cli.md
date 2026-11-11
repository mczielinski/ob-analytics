---
title: Run from the command line
---

# Run from the command line

Installing the package registers the `ob-analytics` command. Put
`-v` / `--verbose` before the subcommand for debug-level logging. Every
subcommand calls its input `path` and its output `-o` / `--output`. The
[CLI flag reference](../api/cli-flags.md) lists every flag of every
subcommand.

| Subcommand | Description |
|------------|-------------|
| `process` | Run the pipeline on a data source and save the result (optional `--gallery`) |
| `audit` | Check a run's data quality; exit non-zero when a check fails (old name: `validate`) |
| `gallery` | Generate an HTML plot gallery from saved Parquet data |
| `bitstamp-demo` | Run the Bitstamp demo (pipeline + gallery) |
| `lobster-demo` | Run the LOBSTER demo on local data (pipeline + gallery) |
| `capture` | Live-capture market data from a registered source (`[live]` extra) |
| `sources` | List the registered sources, what each can do, and the keys it needs |

```bash
# Process a data source
ob-analytics process orders.csv -o results/
ob-analytics process data/ --source lobster --trading-date 2012-06-21
ob-analytics process orders.csv -o results/ --gallery --backend plotly

# Check data quality (crossed book, dropped messages, impossible values, ...)
ob-analytics audit orders.csv
ob-analytics audit orders.csv --json
ob-analytics audit results/ --from-parquet --strict

# Build a gallery from saved Parquet
ob-analytics gallery results/ -o my_gallery/
ob-analytics gallery results/ --volume-scale 1e-8 --title "My Analysis"

# End-to-end demos (pipeline + gallery)
ob-analytics bitstamp-demo orders.csv -o demo_out/
ob-analytics lobster-demo /path/to/lobster_data --trading-date 2012-06-21 -o demo_out/

# Live capture
ob-analytics capture bitstamp --pair btcusd --minutes 10 -o capture/
ob-analytics capture ccxt --venue kraken --pair BTC/USD -o kraken/
```

`process` saves each table with the run's config and declarations. `gallery` and
`audit --from-parquet` read the run back with
[`load_result`](../api/data.md#ob_analytics.data.load_result), so a saved run
keeps its tick size, lot size and display precision: a LOBSTER run still shows
whole shares. Given a live capture, `process` and `audit` use the tick size and
lot size in its `meta.json`, as `Pipeline.run` does in Python.

The `bitstamp-demo` and `lobster-demo` subcommands run the pipeline, save
Parquet, verify round-trip I/O, and build an HTML plot gallery.

## Set how the pipeline runs

`process` and `audit` take the settings that `PipelineConfig` and
`RunContext` take in Python:

| Flag | Python | What it sets |
|------|--------|--------------|
| `--tick-size` | `PipelineConfig(tick_size=...)` | The price step, in the quote currency, and the decimals shown |
| `--lot-size` | `PipelineConfig(lot_size=...)` | The size step, in the base asset, and the decimals shown |
| `--depth-bins` | `PipelineConfig(depth_bins=...)` | Number of depth bins on each side of the book |
| `--depth-bps` | `PipelineConfig(depth_bps=...)` | Width of each depth bin, in basis points |
| `--trading-date` | `RunContext(trading_date=...)` | The date of a session-relative feed (LOBSTER) |
| `--session-tz` | `RunContext(session_tz=...)` | The venue's time zone for a session-relative feed |

```bash
ob-analytics process capture/ --tick-size 0.001 --lot-size 0.0001
ob-analytics process data/ --source lobster --trading-date 2012-06-21 --session-tz Europe/London
```

A flag left out keeps the source's default. A capture's tick size and lot size
come from its `meta.json` unless `--tick-size` or `--lot-size` is given.
`audit --from-parquet` refuses these flags: a saved result keeps the settings
of its own run.

## Save in another format

`process --format` saves in any format
[`save_data`](../api/data.md#ob_analytics.data.save_data) writes. A format
for this package (`parquet`, the default, `depth_csv`, `bitstamp`, `lobster`)
writes a folder; an export to another tool (`hftbacktest`, `nautilus`,
`databento`) writes one file, so give `-o` a file name with its extension:

```bash
ob-analytics process orders.csv --format bitstamp -o bitstamp_out/
ob-analytics process orders.csv --format hftbacktest -o session.npz
```

With `--gallery`, a folder holds its gallery in `gallery/`, and a file has its
gallery next to it, in `<name>_gallery/`. A segmented capture is saved as one
folder per segment, so it takes only a format that writes a folder.

## Run one time window at a time

`process --cut TIME` cuts the input at `TIME` and runs the depth stages on one
window at a time, as
[`Pipeline.run_windows`](../api/pipeline.md#ob_analytics.pipeline.Pipeline.run_windows)
does. Peak memory is then set by the largest window rather than by the whole
input. Give `--cut` once for each cut; a time with no zone is read as UTC:

```bash
ob-analytics process orders.csv -o windows/ --cut 2015-05-01T02:00 --cut 2015-05-01T04:00
```

Each window starts from the book the window before ended with, so the cuts do
not show in the output. `--no-carry` starts each window from an empty book
instead. A windowed run writes Parquet, and builds no gallery: run
`ob-analytics gallery windows/` on its output.

## Set up a capture source

The flags after `capture <source>` are the capture's own (`--pair`,
`--minutes`, `-o` / `--output`, `--roll-minutes`, `--roll-mb`, `--no-raw`)
and one for each field of the source's settings class. The
[CLI flag reference](../api/cli-flags.md#ob-analytics-capture-bitstamp) lists
them for each built-in source, and `--help` lists them for any source:

```bash
ob-analytics capture ccxt --help
```

For `ccxt` and `cryptofeed`, `--venue` names the venue. A source from a
plug-in package gets its flags the same way, so it is set up from the command
line as from Python. A field gets a flag named after it when its type is one
the command line reads (`str`, `int`, `float`, `bool`, a `Literal`, an `Enum`
or `Path`); a field marked with [`CliOption`](../api/sources.md#ob_analytics.sources.CliOption) takes its
help text, flag name or value type from the marker. An API key is never a
flag: it is read from its environment variable (see
[Use an API key](api-keys.md)).

## Related

- [CLI flag reference](../api/cli-flags.md) — every flag of every subcommand
- [Check data quality with `audit`](audit.md) — the `audit` verb in depth
- [Capture live data](live-capture.md) — the `capture` verb in depth
