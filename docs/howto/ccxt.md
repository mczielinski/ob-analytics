---
title: Capture CCXT venues
---

# Capture crypto & prediction-market data (CCXT)

[CCXT](https://github.com/ccxt/ccxt) is the de-facto venue-normalisation layer
for crypto: one interface to ~100 exchanges *and* the Kalshi / Polymarket
prediction markets. The `ccxt` source wraps it, so any CCXT-supported venue
becomes an ob-analytics source by passing a venue id — no per-venue code.

Install the optional `[ccxt]` extra (CCXT Pro ships inside `ccxt`) and use the
`capture` verb with `--exchange`:

```bash
pip install "ob-analytics[ccxt]"

ob-analytics capture ccxt --exchange binance --pair BTC/USDT --minutes 10 --out /tmp/cap
ob-analytics process /tmp/cap --source depth_csv --gallery --output /tmp/cap_out
```

## It is an L2 (price-level) source

CCXT's unified order book is **price-level (L2)** for every venue — totals per
price, no order IDs. So the `ccxt` source records **`depth.csv`**, not
`orders.csv`, and replays through the [L2 path](l2-depth.md): depth metrics,
spread, and trade-sign classification run; the per-order stages (queue
reconstruction, order-type classification) do not apply. Nothing is faked into
per-order state.

For per-order (L3) capture on the crypto venues that publish it, use the
[cryptofeed source](cryptofeed.md) instead.

Each run produces a self-contained directory:

| File | Contents |
|------|----------|
| `depth.csv` | Price-level updates (`timestamp,exchange_timestamp,side,price,volume`; `volume` = new absolute size, `0` removes the level; `timestamp` is when the capture received the update) |
| `trades.csv` | The trade tape (CCXT taker side; feeds trade-sign) |
| `raw.jsonl` | The whole book CCXT reported once, then the changed levels of each book update with its timestamp and nonce, plus the trades as CCXT gave them (omit with `--no-raw`) |
| `meta.json` | Counts + per-run diagnostics (exchange, tick size, book updates, errors) |

The tick size comes from CCXT's market data. `ob-analytics process` and
`ob-analytics audit` read it from `meta.json`, so a coin quoted in steps finer
than a cent keeps its real prices. If a file's prices are finer than the tick
size in use, the loader raises `ConfigError` instead of rounding them.

## Websockets or REST polling, chosen per venue

The capturer reads each venue's declared capabilities:

- venues with CCXT Pro websockets (`exchange.has['watchOrderBook']` — most
  crypto CEXes) **stream** via `watch_order_book` / `watch_trades`;
- the rest are **polled** via `fetch_order_book` / `fetch_trades` (tune the
  cadence with `--poll-interval`).

CCXT's prediction markets (**Kalshi**, **Polymarket** and others) are in CCXT's
separate `ccxt.prediction` package, and the same `--exchange` id reaches them.
Kalshi is polled; Polymarket streams over its public websocket.

Book updates become depth rows by diffing CCXT's maintained book: a level whose
size changed emits its new absolute size; a level that vanished emits `0`.

## What a size-`0` row means

A `0` row means the level is gone from the book CCXT reports — normally a
cancel. What CCXT reports differs by venue:

- **Coinbase, Bitstamp, OKX** ignore `--depth-limit` and always hand CCXT
  their whole book, so a `0` here is a real cancel.
- **Binance and its family** (`binanceus`, `binanceusdm`, `binancecoinm`) are
  asked for their whole book too — always at least 1,000 levels a side, more
  with a deeper `--depth-limit` (up to the 5,000 Binance sends) — so a `0`
  here is also a real cancel. Below that floor, `--depth-limit` does not
  shrink what is recorded; see [Binance](binance.md#depth-request-more-levels).
- **Kraken** subscribes at exactly `--depth-limit` levels, and Kraken itself
  drops a level from what it sends once the price moves it out of that
  window. A `0` here can be either a real cancel or the level leaving
  Kraken's own window — CCXT reports both the same way, so the capture
  cannot tell them apart. The level is recorded again, with its size, if
  Kraken's window reaches it again.

Earlier captures (before this page's revision) cropped every venue to
`--depth-limit`, so a `0` could also mean a level merely left that smaller,
capture-only window while still resting in the book. That crop is gone
(issue #275); `raw.jsonl` and `depth.csv` now hold whatever CCXT reported.

## A second venue is a config change, not new code

```bash
# Another crypto CEX (websocket):
ob-analytics capture ccxt --exchange kraken --pair BTC/USD --out /tmp/kraken

# Prediction markets:
ob-analytics capture ccxt --exchange kalshi --pair KXPRESNOMD-28-MK --out /tmp/kalshi
ob-analytics capture ccxt --exchange polymarket --pair <token id> --out /tmp/poly
```

| Flag | Meaning |
|------|---------|
| `--exchange` | CCXT venue id (`binance`, `kraken`, `coinbase`, `kalshi`, `polymarket`, …). `binance` and `hyperliquid` are also prediction markets; the plain id is the crypto exchange, and `prediction/binance` is the prediction market |
| `--pair` | Symbol in the venue's CCXT notation (e.g. `BTC/USDT`) |
| `--depth-limit` | Levels per side to request (default 100). Kraken records exactly this many; Coinbase, Bitstamp, OKX, Binance and its family record more — see [what a size-`0` row means](#what-a-size-0-row-means) |
| `--market-data-mirror` | Read from the venue's market-data-only endpoints (Binance spot; see [Binance](binance.md)) |
| `--poll-interval` | Seconds between REST polls (REST-only venues) |

!!! note "Prediction markets"
    A prediction market needs a few more choices than a crypto exchange: which
    side of the Yes/No pair the book shows, what tick size the prices use, and
    how wide the depth bins should be. [Kalshi](kalshi.md) and
    [Polymarket](polymarket.md) each have a page that covers them.

!!! note "Binance"
    Binance refuses some locations, and its book is dense enough that 100
    levels reach only a few hundredths of a percent from the mid-price. The
    [Binance](binance.md) page covers both.

!!! note "Coinbase"
    Use the id `coinbase`, not `coinbaseexchange`, which needs an API key.
    [Coinbase](coinbase.md) has its own page.

## See also

- [Capture Coinbase](coinbase.md) — which ccxt id to use for Coinbase
- [Capture cryptofeed venues](cryptofeed.md) — the per-order (L3) complement, for venues that publish order-by-order data
- [Process L2 (price-level) feeds](l2-depth.md) — what the captured `depth.csv` flows through
- [Capture live data](live-capture.md) — the capture framework and writing a bespoke venue
- [Check data quality](audit.md) — run `audit` on the captured output
