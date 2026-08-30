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

Each run produces a self-contained directory:

| File | Contents |
|------|----------|
| `depth.csv` | Price-level updates (`timestamp,side,price,volume`; `volume` = new absolute size, `0` removes the level) |
| `trades.csv` | The trade tape (CCXT taker side; feeds trade-sign) |
| `raw.jsonl` | Raw book frames (omit with `--no-raw`) |
| `meta.json` | Counts + per-run diagnostics (exchange, book updates, errors) |

## Websockets or REST polling, chosen per venue

The capturer reads each venue's declared capabilities:

- venues with CCXT Pro websockets (`exchange.has['watchOrderBook']` — most
  crypto CEXes) **stream** via `watch_order_book` / `watch_trades`;
- the rest — currently the prediction markets **Kalshi** and **Polymarket**,
  REST-only in CCXT — are **polled** via `fetch_order_book` / `fetch_trades`
  (tune the cadence with `--poll-interval`).

Book updates become depth rows by diffing CCXT's maintained book: a level whose
size changed emits its new absolute size; a level that vanished emits `0`.

## A second venue is a config change, not new code

```bash
# Another crypto CEX (websocket):
ob-analytics capture ccxt --exchange kraken --pair BTC/USD --out /tmp/kraken

# A prediction market (REST poll):
ob-analytics capture ccxt --exchange polymarket --pair <token> --poll-interval 2 --out /tmp/poly
```

| Flag | Meaning |
|------|---------|
| `--exchange` | CCXT venue id (`binance`, `kraken`, `coinbase`, `kalshi`, `polymarket`, …) |
| `--pair` | Symbol in the venue's CCXT notation (e.g. `BTC/USDT`) |
| `--depth-limit` | Order-book depth (levels per side) to request |
| `--poll-interval` | Seconds between REST polls (REST-only venues) |

!!! note "Acquisition vs domain modelling"
    The `ccxt` source handles *acquisition* uniformly across venues. Venue
    *modelling* — Kalshi cents / Yes-No duality, Polymarket 0–1 pricing and
    binary token pairs — is tracked separately and not applied here.

## See also

- [Process L2 (price-level) feeds](l2-depth.md) — what the captured `depth.csv` flows through
- [Capture live data](live-capture.md) — the capture framework and writing a bespoke venue
- [Check data quality](validate.md) — run `validate` on the captured output
