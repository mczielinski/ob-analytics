---
title: CLI flag reference
---

<!-- Made by scripts/cli_reference.py from the command's parser. Do not edit
by hand: run `uv run python scripts/cli_reference.py` after a flag changes. -->

# CLI flag reference

Every flag of every `ob-analytics` subcommand. Each subcommand also takes
`-h` / `--help`, which prints the same list. Every subcommand accepts the
top-level `-v` / `--verbose` flag before its name, for debug-level logging:
`ob-analytics -v process orders.csv`.

For examples, see [Run from the command line](../howto/cli.md).

## `ob-analytics process`

| Flag | Description |
|------|-------------|
| `path` | Path to data file or directory |
| `-s SOURCE`, `--source SOURCE` | Data source (default: for a capture, the source its meta.json names; otherwise bitstamp. Registered: bitstamp, ccxt, cryptofeed, databento, depth_csv, kalshi, kraken, lobster). Run 'ob-analytics process PATH --source NAME --help' to see the flags of that source's settings |
| `--trading-date TRADING_DATE` | Trading date, for a source that needs one (LOBSTER), as YYYY-MM-DD |
| `--session-tz SESSION_TZ` | The venue's time zone, for a feed whose times count from the session's midnight (LOBSTER; default: America/New_York) |
| `--tick-size TICK_SIZE` | The instrument's price step, in the quote currency (e.g. 0.001). Default: the source's, or a capture's own from its meta.json |
| `--lot-size LOT_SIZE` | The instrument's size step, in the base asset (e.g. 0.0001). Default: the source's, or a capture's own from its meta.json |
| `--depth-bins DEPTH_BINS` | Number of depth bins on each side of the book (default: 20) |
| `--depth-bps DEPTH_BPS` | Width of each depth bin, in basis points (default: 25) |
| `--symbol SYMBOL` | The instrument the data covers (e.g. BTC/USD): tags each row with a symbol column |
| `--venue VENUE` | The venue the data comes from (e.g. kraken): tags each row with a venue column (default: the source's name, when --symbol is given) |
| `-o OUTPUT`, `--output OUTPUT` | Where to save the results (default: ./output): a folder, or a file for a --format that writes one |
| `--format FORMAT` | Format to save the results in (default: parquet). One of `bitstamp`, `databento`, `depth_csv`, `hftbacktest`, `lobster`, `nautilus`, `parquet`. |
| `--cut TIME` | Run one time window at a time, cut at TIME (e.g. 2015-05-01T02:00; a time with no zone is UTC). Give it more than once for more windows. Peak memory is set by the largest window. Writes Parquet |
| `--no-carry` | With --cut: start each window from an empty book, as if it were a separate input, instead of from the book the window before ended with |
| `--gallery` | Also generate an HTML plot gallery |
| `--view VIEW` | Gallery view: resolution level(s) to render (l2\|l3\|both\|comparison; default: both) |
| `--backend BACKEND` | Plot backend to draw the gallery with (matplotlib, plotly, bokeh, or a registered plug-in); give it more than once for more columns. Default: plotly and matplotlib when plotly is installed, else matplotlib. |
| `--title TITLE` | Gallery page title (default: names the input) |

## `ob-analytics audit`

| Flag | Description |
|------|-------------|
| `path` | Path to data file or directory |
| `-s SOURCE`, `--source SOURCE` | Data source (default: for a capture, the source its meta.json names; otherwise bitstamp. Registered: bitstamp, ccxt, cryptofeed, databento, depth_csv, kalshi, kraken, lobster). Run 'ob-analytics audit PATH --source NAME --help' to see the flags of that source's settings |
| `--trading-date TRADING_DATE` | Trading date, for a source that needs one (LOBSTER), as YYYY-MM-DD |
| `--session-tz SESSION_TZ` | The venue's time zone, for a feed whose times count from the session's midnight (LOBSTER; default: America/New_York) |
| `--tick-size TICK_SIZE` | The instrument's price step, in the quote currency (e.g. 0.001). Default: the source's, or a capture's own from its meta.json |
| `--lot-size LOT_SIZE` | The instrument's size step, in the base asset (e.g. 0.0001). Default: the source's, or a capture's own from its meta.json |
| `--depth-bins DEPTH_BINS` | Number of depth bins on each side of the book (default: 20) |
| `--depth-bps DEPTH_BPS` | Width of each depth bin, in basis points (default: 25) |
| `--symbol SYMBOL` | The instrument the data covers (e.g. BTC/USD): tags each row with a symbol column |
| `--venue VENUE` | The venue the data comes from (e.g. kraken): tags each row with a venue column (default: the source's name, when --symbol is given) |
| `--from-parquet` | Read a saved 'process' output directory instead of running the pipeline. --source then only declares the feed of an output that records no declarations (one written by an older version); omit it to leave those undeclared |
| `--strict` | Fail on warnings too, not only on errors |
| `--json` | Emit the summary as JSON instead of text |

## `ob-analytics process --source databento`

The flags 'process' and 'audit' add when the 'databento' source reads the data: given with --source, or named by a capture's meta.json.

| Flag | Description |
|------|-------------|
| `--instrument-id INSTRUMENT_ID` | Keep only records with this instrument_id |
| `--publisher-id PUBLISHER_ID` | Keep only records from this publisher |
| `--raw-symbol RAW_SYMBOL` | Keep only records whose mapped symbol is this (needs a file with Databento's symbol mapping) |
| `--trades-from {fills,prints}` | Build trades from the fill records (each names the resting order) or the print records (the whole tape, no maker named) (default: fills) |
| `--dataset DATASET` | The dataset id written into a DBN file this run saves (default: OB.ANALYTICS) |

## `ob-analytics process --source depth_csv`

The flags 'process' and 'audit' add when the 'depth_csv' source reads the data: given with --source, or named by a capture's meta.json.

| Flag | Description |
|------|-------------|
| `--timestamp-unit {ms,us,ns}` | The unit of an integer epoch timestamp (default: ms) |
| `--price-divisor PRICE_DIVISOR` | What a price in the file is divided by to reach the quote currency (default: 1) |

## `ob-analytics gallery`

| Flag | Description |
|------|-------------|
| `path` | Path to Parquet directory (output of 'process') |
| `-o OUTPUT`, `--output OUTPUT` | Output directory for the gallery (default: ./gallery_output) |
| `--view VIEW` | Gallery view: resolution level(s) to render (l2\|l3\|both\|comparison; default: both) |
| `--backend BACKEND` | Plot backend to draw the gallery with (matplotlib, plotly, bokeh, or a registered plug-in); give it more than once for more columns. Default: plotly and matplotlib when plotly is installed, else matplotlib. |
| `--title TITLE` | Gallery page title (default: names the input) |

## `ob-analytics bitstamp-demo`

| Flag | Description |
|------|-------------|
| `path` | Path to a Bitstamp orders.csv, or the folder holding it and its trades.csv (default: bundled sample data) |
| `-o OUTPUT`, `--output OUTPUT` | Output directory (default: ./bitstamp_output) |
| `--roundtrip` | Write then re-read the result to verify the round-trip (slower) |
| `--view VIEW` | Gallery view: resolution level(s) to render (l2\|l3\|both\|comparison; default: both) |
| `--backend BACKEND` | Plot backend to draw the gallery with (matplotlib, plotly, bokeh, or a registered plug-in); give it more than once for more columns. Default: plotly and matplotlib when plotly is installed, else matplotlib. |
| `--title TITLE` | Gallery page title (default: names the input) |

## `ob-analytics lobster-demo`

| Flag | Description |
|------|-------------|
| `path` | Path to LOBSTER data directory (containing message + orderbook CSVs) |
| `--trading-date TRADING_DATE` | Required. Trading date for LOBSTER format (YYYY-MM-DD) |
| `-o OUTPUT`, `--output OUTPUT` | Output directory (default: ./lobster_output) |
| `--view VIEW` | Gallery view: resolution level(s) to render (l2\|l3\|both\|comparison; default: both) |
| `--backend BACKEND` | Plot backend to draw the gallery with (matplotlib, plotly, bokeh, or a registered plug-in); give it more than once for more columns. Default: plotly and matplotlib when plotly is installed, else matplotlib. |
| `--title TITLE` | Gallery page title (default: names the input) |

## `ob-analytics capture`

Live-capture market data from a registered source. Run 'ob-analytics capture <source> --help' for that source's flags: the capture's own (--pair, --minutes, -o/--output, ...) and one for each of its settings (for ccxt: --venue, --depth-limit, ...).

| Flag | Description |
|------|-------------|
| `source` | Source name (e.g. 'bitstamp', 'ccxt'). Use --list to see live sources. |
| `--list` | List live-capable sources and exit |

## `ob-analytics capture bitstamp`

Live-capture market data from the 'bitstamp' source.

| Flag | Description |
|------|-------------|
| `--pair PAIR` | Required. The venue's pair symbol (e.g. btcusd for bitstamp, BTC/USDT for ccxt) |
| `--minutes MINUTES` | Capture duration in minutes (default: 10.0) |
| `-o OUTPUT`, `--output OUTPUT` | Required. Capture directory: manifest.json plus one folder per segment (orders.csv for L3 or depth.csv for L2, plus trades.csv, raw.jsonl, meta.json). A directory that already holds a capture of the same source and pair is continued. |
| `--roll-minutes ROLL_MINUTES` | Start a new segment every this many minutes (default: never) |
| `--roll-mb ROLL_MB` | Start a new segment once one has written this many MB since its first live event (default: never) |
| `--no-raw` | Don't write raw.jsonl (saves disk for long runs) |

## `ob-analytics capture ccxt`

Live-capture market data from the 'ccxt' source.

| Flag | Description |
|------|-------------|
| `--pair PAIR` | Required. The venue's pair symbol (e.g. btcusd for bitstamp, BTC/USDT for ccxt) |
| `--minutes MINUTES` | Capture duration in minutes (default: 10.0) |
| `-o OUTPUT`, `--output OUTPUT` | Required. Capture directory: manifest.json plus one folder per segment (orders.csv for L3 or depth.csv for L2, plus trades.csv, raw.jsonl, meta.json). A directory that already holds a capture of the same source and pair is continued. |
| `--roll-minutes ROLL_MINUTES` | Start a new segment every this many minutes (default: never) |
| `--roll-mb ROLL_MB` | Start a new segment once one has written this many MB since its first live event (default: never) |
| `--no-raw` | Don't write raw.jsonl (saves disk for long runs) |
| `--venue VENUE` | The CCXT venue id (e.g. 'binance', 'kraken', 'coinbase'); give --pair in CCXT notation, e.g. 'BTC/USDT' |
| `--depth-limit DEPTH_LIMIT` | Order-book depth (levels per side) to request: a floor, not a cap, since most venues record more (see the ccxt how-to page) (default: 100) |
| `--poll-interval POLL_INTERVAL` | Seconds between REST polls (REST-only venues) (default: 1.0) |
| `--market-data-mirror`, `--no-market-data-mirror` | Read from the venue's market-data-only endpoints (binance spot). Check the venue's terms allow it where you are (default: off) |

## `ob-analytics capture cryptofeed`

Live-capture market data from the 'cryptofeed' source.

| Flag | Description |
|------|-------------|
| `--pair PAIR` | Required. The venue's pair symbol (e.g. btcusd for bitstamp, BTC/USDT for ccxt) |
| `--minutes MINUTES` | Capture duration in minutes (default: 10.0) |
| `-o OUTPUT`, `--output OUTPUT` | Required. Capture directory: manifest.json plus one folder per segment (orders.csv for L3 or depth.csv for L2, plus trades.csv, raw.jsonl, meta.json). A directory that already holds a capture of the same source and pair is continued. |
| `--roll-minutes ROLL_MINUTES` | Start a new segment every this many minutes (default: never) |
| `--roll-mb ROLL_MB` | Start a new segment once one has written this many MB since its first live event (default: never) |
| `--no-raw` | Don't write raw.jsonl (saves disk for long runs) |
| `--venue VENUE` | The cryptofeed venue id (e.g. 'bitstamp', 'binance'); give --pair in cryptofeed notation, e.g. 'BTC-USD' |
| `--level {L2,L3}` | Force the capture resolution. Leave it out to read it from the venue: L3 where the venue publishes a per-order book, L2 otherwise |
| `--queue-size QUEUE_SIZE` | Messages the buffer between cryptofeed and the capture holds (default: 10000) |

## `ob-analytics capture kalshi`

Live-capture market data from the 'kalshi' source.

| Flag | Description |
|------|-------------|
| `--pair PAIR` | Required. The venue's pair symbol (e.g. btcusd for bitstamp, BTC/USDT for ccxt) |
| `--minutes MINUTES` | Capture duration in minutes (default: 10.0) |
| `-o OUTPUT`, `--output OUTPUT` | Required. Capture directory: manifest.json plus one folder per segment (orders.csv for L3 or depth.csv for L2, plus trades.csv, raw.jsonl, meta.json). A directory that already holds a capture of the same source and pair is continued. |
| `--roll-minutes ROLL_MINUTES` | Start a new segment every this many minutes (default: never) |
| `--roll-mb ROLL_MB` | Start a new segment once one has written this many MB since its first live event (default: never) |
| `--no-raw` | Don't write raw.jsonl (saves disk for long runs) |

## `ob-analytics capture kraken`

Live-capture market data from the 'kraken' source.

| Flag | Description |
|------|-------------|
| `--pair PAIR` | Required. The venue's pair symbol (e.g. btcusd for bitstamp, BTC/USDT for ccxt) |
| `--minutes MINUTES` | Capture duration in minutes (default: 10.0) |
| `-o OUTPUT`, `--output OUTPUT` | Required. Capture directory: manifest.json plus one folder per segment (orders.csv for L3 or depth.csv for L2, plus trades.csv, raw.jsonl, meta.json). A directory that already holds a capture of the same source and pair is continued. |
| `--roll-minutes ROLL_MINUTES` | Start a new segment every this many minutes (default: never) |
| `--roll-mb ROLL_MB` | Start a new segment once one has written this many MB since its first live event (default: never) |
| `--no-raw` | Don't write raw.jsonl (saves disk for long runs) |
| `--depth {10,100,1000}` | Price levels a side Kraken sends (default: 1000) |
| `--on-checksum-failure {restart,count}` | What a checksum failure does: restart starts a new segment from a new opening book; count only counts it (default: restart) |

## `ob-analytics sources`

This subcommand takes no flags.
