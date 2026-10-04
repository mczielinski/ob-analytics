---
title: Capture Coinbase
---

# Capture Coinbase

ob-analytics captures a Coinbase market through the [ccxt source](ccxt.md) and
replays it through the [L2 path](l2-depth.md). You do not need a Coinbase
account or an API key: the price-level book and the trades are public.

!!! info "What this feed shows"

    | Property | Value |
    |---|---|
    | Level | L2 |
    | Depth shown | whole book |
    | Update form | changes, merged |
    | What can be missed | merged changes; a lost message |
    | After a lost message | drifts until the level changes again |
    | Sequence | none |
    | Clocks | venue + receive |
    | Crossing | price levels |
    | Taker side | venue; Coinbase names the maker's side, so the capture reverses it |
    | Trade tape gaps | starts with trades from before the capture |
    | Price grid | fixed (0.01 on BTC/USD) |
    | What the book means | normal |
    | Access | `coinbase` public; `coinbaseexchange` needs a key |

    Use it for depth, spread and trade signs over the whole book. Don't use it
    for long captures where a lost message matters. [What each feed
    shows](../feeds.md) explains each property and compares every source.

```bash
pip install "ob-analytics[ccxt]"

ob-analytics capture ccxt --exchange coinbase --pair BTC/USD --minutes 10 --out /tmp/coinbase
ob-analytics process /tmp/coinbase --source depth_csv --gallery --output /tmp/coinbase_out
```

## Use the `coinbase` id

ccxt has two ids for Coinbase:

| ccxt id | Coinbase API | Needs an API key |
|---------|--------------|------------------|
| `coinbase` | Advanced Trade | No |
| `coinbaseexchange` | Exchange | Yes. Without one, the capture stops with `AuthenticationError` |

Use `coinbase`. `--pair` takes a ccxt symbol, such as `BTC/USD`, `ETH/USD` or
`BTC/USDC`.

## Trade sides are reversed for you

Coinbase reports each trade with the side of the maker, the order that was
resting. ccxt passes that side on unchanged, so the capture reverses it:
`trades.csv` names the taker's side, as it does for every other venue, and
`meta.json` records `"trade_side_reversed": true`. `raw.jsonl` keeps each trade
as ccxt handed it over, with the maker's side.

A capture made before ob-analytics did this has no `trade_side_reversed` in its
`meta.json`, and its `trades.csv` holds the maker's side. Reverse its signs
before you use them.

## You get price levels, not orders

The capture holds the total size at each price (L2). It does not hold single
orders, so the per-order stages of the pipeline (queue position, order types)
do not run.

Coinbase has a per-order feed, but it needs an API key. ob-analytics does not
use API keys yet.

## Capture for 5 minutes or more

The gallery leaves out the first minute of the session from its summary
charts, while the book fills. A capture of one minute leaves about one second
for those charts. Capture for 5 minutes or more.

## See also

- [Capture CCXT venues](ccxt.md): the source that this page uses
- [Process L2 (price-level) feeds](l2-depth.md): what the captured `depth.csv` goes through
- [Check data quality](audit.md): run `audit` on the captured output
