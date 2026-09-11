---
title: Capture Polymarket markets
---

# Capture Polymarket prediction markets

Polymarket runs prediction markets settled in USDC. Each market asks a
question with two outcomes, such as Yes and No, and a share of the winning
outcome pays $1. ob-analytics captures a Polymarket market through the
[ccxt source](ccxt.md) and replays it through the [L2 path](l2-depth.md). You do
not need an account or an API key: the order book and the trades stream over
Polymarket's public websocket.

```bash
pip install "ob-analytics[ccxt]"

ob-analytics capture ccxt --exchange polymarket --pair <token id> --minutes 10 --out /tmp/poly
ob-analytics process /tmp/poly --source depth_csv --gallery --output /tmp/poly_out
```

## Find a token id

Every Polymarket outcome has its own order book, and `--pair` names one
outcome by its token id: a number about 77 digits long. ccxt also has a
readable outcome symbol, but it cannot look one up from the symbol alone, so
the capture needs the token id.

Polymarket's public Gamma API lists the open markets. For each market,
`clobTokenIds` holds the token ids in the same order as `outcomes`:

```bash
curl -s "https://gamma-api.polymarket.com/markets?active=true&closed=false&order=volume24hr&ascending=false&limit=10"
```

To print the token ids of one market, take the market's slug from the end of
its URL on polymarket.com:

```python
import json
import urllib.request

slug = "will-the-fed-increase-interest-rates-by-25-bps-after-the-september-2026-meeting-649"
url = f"https://gamma-api.polymarket.com/markets?slug={slug}"
# The API refuses Python's default User-Agent, so name the client.
request = urllib.request.Request(url, headers={"User-Agent": "ob-analytics"})
market = json.load(urllib.request.urlopen(request))[0]
for outcome, token in zip(json.loads(market["outcomes"]), json.loads(market["clobTokenIds"])):
    print(outcome, token)
```

## Each outcome is its own book

A Yes share and a No share of one market together pay $1, so the two books
mirror each other: a bid for Yes at 0.63 matches an ask for No at 0.37.
Polymarket publishes both books, and prices each outcome's trades in that
outcome. So in one capture the book and the trades always agree, whichever
outcome you pick.

| The Yes book | The No book |
|--------------|-------------|
| best bid 0.63, best ask 0.64 | best bid 0.36, best ask 0.37 |

Capture the Yes outcome to read the book as the market's probability of Yes.
This is the same view a [Kalshi](kalshi.md) capture gives.

Some questions have more than two answers, such as who will win a nomination.
Polymarket lists each answer as a Yes/No market of its own, so you capture each
answer separately.

Prices are in USDC, between 0 and 1. Sizes are numbers of shares, and can be
fractional.

## Tick sizes

A Polymarket market's price step is usually 0.01 or 0.001, and Polymarket makes
it finer as a price nears 0 or 1. A capture starts from the tick size in ccxt's
market data and records it as `tick_size` in `meta.json`. If a later price
arrives between two ticks, the capture divides the tick size by ten until the
price fits, and counts each change in `tick_size_changes`. `ob-analytics
process` and `ob-analytics audit` read the final value, so every price in the
file replays exactly.

From Python, read the recorded value and pass it yourself. Set
`price_decimals` to the number of decimals in the tick size:

```python
from ob_analytics import DepthCsvSource, Pipeline, PipelineConfig
from ob_analytics.depth_l2 import recorded_tick_size

tick_size = recorded_tick_size("/tmp/poly")  # 0.001, for example
config = PipelineConfig(tick_size=tick_size, price_decimals=3)
result = Pipeline(config, source=DepthCsvSource()).run("/tmp/poly")
```

## Depth bins

The depth summary's rings are 25 basis points wide by default, which is too
narrow for prices between 0 and 1. Use wider rings, as for
[Kalshi](kalshi.md#depth-bins).

## What a capture can and cannot see

The capture streams Polymarket's public websocket, so it sees the book change
and the trades as they happen, with Polymarket's own timestamps. It has two
limits:

- There is no sequence number, so `audit` cannot check for dropped data.
- ccxt keeps the book up to date from the websocket, and the capture records
  each price level whose size differs from the last book it saw. Two changes to
  one price level that arrive together are recorded as one.

## See also

- [Capture CCXT venues](ccxt.md): the source this capture uses
- [Capture Kalshi prediction markets](kalshi.md): the other prediction market
- [Process L2 (price-level) feeds](l2-depth.md): what the captured `depth.csv`
  goes through
- [Check data quality](audit.md): run `audit` on the capture
