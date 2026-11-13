---
title: Capture Kalshi markets
---

# Capture Kalshi prediction markets

Kalshi lists event contracts. Each market asks a yes-or-no question, and a Yes
contract pays $1 if the answer is Yes. ob-analytics captures a Kalshi market
in one of two ways, and replays either through the [L2 path](l2-depth.md):

- The [ccxt source](ccxt.md) polls Kalshi's public REST API. You do not need a
  Kalshi account or an API key. The rest of this section is about it.
- The `kalshi` source reads Kalshi's WebSocket feed and records every change.
  It needs an API key: see
  [Capture every change, with an API key](#capture-every-change-with-an-api-key).

!!! info "What this feed shows"

    | Property | Value |
    |---|---|
    | Level | L2 |
    | Depth shown | whole book, each poll |
    | Update form | polls, 1 a second |
    | What can be missed | anything shorter than a poll |
    | After a lost message | the next poll corrects it |
    | Sequence | none |
    | Clocks | receive only |
    | Crossing | price levels |
    | Taker side | venue (buy = the taker bought Yes) |
    | Trade tape gaps | polled; trades from before the opening book dropped |
    | Price grid | by market: cents; tenths of a cent near 0 and 1; tenths or hundredths of a cent throughout |
    | What the book means | Yes book, from Yes and No bids |
    | Access | REST public; websocket needs a key |

    Use it for the market's probability of Yes and its trades over minutes and
    hours. Don't use it for anything shorter than a poll, or for book latency:
    the book has only the poll time. [What each feed shows](../feeds.md)
    explains each property and compares every source.

```bash
pip install "ob-analytics[ccxt]"

ob-analytics capture ccxt --venue kalshi --pair KXPRESNOMD-28-MK --minutes 10 -o /tmp/kalshi
ob-analytics process /tmp/kalshi --source depth_csv --gallery --output /tmp/kalshi_out
```

## Find a market ticker

`--pair` takes a Kalshi market ticker, such as `KXPRESNOMD-28-MK`. Each
market's page on kalshi.com shows its ticker, and Kalshi's public API lists
the open markets:

```bash
curl -s "https://api.elections.kalshi.com/trade-api/v2/markets?status=open&mve_filter=exclude&limit=100"
```

`mve_filter=exclude` leaves out Kalshi's combined markets, which are many and
seldom trade.

Pick a market that trades. A quiet market gives a book that seldom changes and
few trades. Many Kalshi books change only a few times a minute, so capture for
10 minutes or more.

## The book is the Yes book

Kalshi publishes two lists of bids, one for Yes and one for No, and no asks. A
Yes contract and a No contract together always pay $1, so a bid to buy No at
`p` is the same as an offer to sell Yes at `1 - p`. ccxt uses this to build one
book, seen from the Yes side:

| Kalshi publishes | The captured book holds |
|------------------|-------------------------|
| a Yes bid at 0.05 for 30 contracts | a bid at 0.05 for 30 |
| a No bid at 0.91 for 306 contracts | an ask at 0.09 for 306 |

Trades follow the same rule. Every trade price is the Yes price. The trade's
`side` is `buy` when the taker bought Yes and `sell` when the taker bought No.

Prices are in dollars, between 0 and 1, so a price is also the market's
probability of Yes. Sizes are numbers of contracts, and can be fractional.

!!! warning "Use the plain ticker"
    ccxt also accepts `<ticker>-NO` for the No side. It shows the book in No
    prices but still prices trades in Yes, so the trades and the book do not
    line up. Capture the plain ticker.

## Tick sizes

Kalshi markets do not all use the same price step. Most use whole cents. Many
of the most traded markets use tenths of a cent below 0.10 and above 0.90, and
whole cents in between. A few use tenths of a cent everywhere.

A capture records the market's price step as `tick_size` in `meta.json`, and
its size step as `lot_size`. The pipeline reads both from there, in Python and
in `ob-analytics process` and `ob-analytics audit`:

<!-- docs-test: skip needs a Kalshi capture -->

```python
from ob_analytics import Pipeline

result = Pipeline.from_source("depth_csv").run("/tmp/kalshi")
result.config.tick_size  # 0.001, for example
```

A tick size you set in `PipelineConfig` is used instead of the recorded one. If
the tick size in use is coarser than the prices in the file, the loader raises
`ConfigError` instead of rounding the prices to fit.

## Depth bins

The depth summary adds up the size resting in rings around the mid-price. Each
ring is `depth_bps` basis points wide: 25 by default, 20 rings a side. That
suits a price of 100,000, but not a price of 0.04, where 25 basis points is
0.0001, a tenth of the smallest tick. Most rings are then empty. Use wider
rings for a prediction market:

```python
from ob_analytics import PipelineConfig

config = PipelineConfig(tick_size=0.001, price_decimals=3, depth_bps=500)
```

The depth heatmap and the price view draw the book directly and do not use the
rings.

## A depth heatmap of flat lines

The depth heatmap leaves out each price level that does not change during the
capture, because it would only draw a flat line. If no level changes at all,
the heatmap draws every level instead, and a note on the chart says that no
level changed. A short capture of a quiet market often gives this result,
because a Kalshi book can go minutes without a change. Capture for longer to
see the book move.

To draw the unchanged levels when some levels do change, pass
`show_all_depth=True`:

<!-- docs-test: skip needs a Kalshi capture -->

```python
from ob_analytics.visualization import plot_result

fig = plot_result(result, "depth_heatmap", show_all_depth=True)
```

## What a capture can and cannot see

The capture polls Kalshi's public REST API, once a second by default. Change
the rate with `--poll-interval`. Because it polls:

- A change that appears and goes away between two polls is not seen.
- Book timestamps are the time each poll returned, not Kalshi's time. Trades
  carry Kalshi's time.
- There is no sequence number, so `audit` cannot check for dropped data.
- Trades from before the opening book are left out. Kalshi's first answer is
  its recent history, which can reach back hours.

Kalshi's WebSocket feed sends every change with a sequence number. It needs an
API key: see the next section.

## Capture every change, with an API key

The `kalshi` source reads Kalshi's WebSocket feed instead of polling. It
records every change to the book, with Kalshi's time and a sequence number, and
every trade as Kalshi sends it. It needs a Kalshi account and an API key.

!!! info "What this feed shows"

    | Property | Value |
    |---|---|
    | Level | L2 |
    | Depth shown | whole book |
    | Update form | opening book, then every change |
    | What can be missed | nothing; a lost change is found by its number |
    | After a lost message | a new segment from a new opening book |
    | Sequence | Kalshi's `seq`; every book message adds one |
    | Clocks | venue + receive |
    | Crossing | price levels |
    | Taker side | venue (buy = the taker bought Yes) |
    | Trade tape gaps | trades from before the opening book dropped; a lost trade is counted |
    | Price grid | by market, from Kalshi's market data |
    | What the book means | Yes book, from Yes and No bids |
    | Access | API key |

    Use it for how a book changes from second to second, and for book latency.
    [What each feed shows](../feeds.md) explains each property and compares
    every source.

### Get a key

1. Log in to kalshi.com and open your profile page,
   <https://kalshi.com/account/profile>.
2. Under **API Keys**, choose **Create New API Key**. For the access, choose
   **Read all data** only: the capture never trades or moves money. An Ed25519
   key (Kalshi's default) and an RSA key both work.
3. Kalshi shows the private key once. Save it in a file that only you can
   read, such as `~/.config/kalshi/private-key.pem` (`chmod 600`).
4. Note the **Key ID** that Kalshi shows with the key.

### Capture

Set the Key ID and the path of the key file, then capture. [Use an API
key](api-keys.md) explains how keys are kept out of the shell history and out
of the capture's files.

```bash
pip install "ob-analytics[kalshi]"

export KALSHI_API_KEY_ID=<your key id>
export KALSHI_PRIVATE_KEY_PATH=~/.config/kalshi/private-key.pem

ob-analytics capture kalshi --pair KXPRESNOMD-28-MK --minutes 10 -o /tmp/kalshi_ws
ob-analytics audit /tmp/kalshi_ws --source depth_csv
ob-analytics process /tmp/kalshi_ws --source depth_csv --gallery --output /tmp/kalshi_ws_out
```

The capture writes the same files as the polled capture, and the book is the
same Yes book. What is different:

- Every book row carries Kalshi's `seq` in the `sequence` column. `audit`
  checks that no number is missing.
- If a number is missing, the capture cannot know what changed, so it ends the
  segment and starts a new one from a new opening book. `manifest.json`
  records the gap. A change that would leave a size below zero does the same.
- Book rows carry Kalshi's time in `exchange_timestamp`, and the time the
  capture received them in `timestamp`. The opening book carries the time
  Kalshi sent it.
- Trades carry their own numbers. `meta.json` counts any that never arrived as
  `trades_missed`.

From Python, build the settings once and give them to each segment:

<!-- docs-test: skip needs a Kalshi API key -->

```python
import asyncio
from pathlib import Path

from ob_analytics.live import CaptureConfig, KalshiSettings, KalshiSource, run_capture

settings = KalshiSettings()  # reads KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH
config = CaptureConfig(pair="KXPRESNOMD-28-MK", out_dir=Path("/tmp/kalshi_ws"), minutes=10)
asyncio.run(run_capture(lambda: KalshiSource(settings=settings), config))
```

If a variable is not set, the capture stops before it creates the output
directory. If Kalshi refuses the key, or has no market with the ticker given,
the capture stops at once with a message that says which: for a refused key,
it names the two variables to check. `manifest.json` records the failed
segment.

## See also

- [Capture CCXT venues](ccxt.md): the source this capture uses
- [Process L2 (price-level) feeds](l2-depth.md): what the captured `depth.csv`
  goes through
- [Check data quality](audit.md): run `audit` on the capture
