---
title: Tutorial
---

# Tutorial

A guided tour of market microstructure through ob-analytics — starting from
absolute basics (what a price *is*, what an exchange does) and building up to
depth analysis, flow toxicity, and per-order (L3) reconstruction.

Every chapter is **executed at docs build time**: the code you read produced
the figures you see, on this exact version of the library. Start with
**[The toy session](00_toy_session.md)** — the 24-event order-book stream
every chapter builds its concepts on before showing real data.

Two runnable notebook tutorials cover the core workflow today.

Both notebooks run on the bundled sample data and are executed in CI, so
they never drift from the API:

- **[A crypto session, end to end](https://github.com/mczielinski/ob-analytics/blob/main/docs/notebooks/01_crypto_session.ipynb)**
  (`01_crypto_session.ipynb`) — load the bundled Bitstamp capture, discover
  what it can plot with `available_concepts`, render one-line plots, and
  finish with flow-toxicity analytics.
- **[L2 vs L3 microstructure](https://github.com/mczielinski/ob-analytics/blob/main/docs/notebooks/02_l3_microstructure.ipynb)**
  (`02_l3_microstructure.ipynb`) — the same book at aggregate (L2) and
  per-order (L3) resolution: book snapshots, order lifecycles and outcomes,
  cancellation behaviour, and the `prepare` escape hatch.

Run them locally with:

```bash
git clone https://github.com/mczielinski/ob-analytics.git
cd ob-analytics
uv sync --group dev --extra interactive
uv run jupyter lab docs/notebooks/
```

## Chapters

The tutorial is growing into a full guide (executed at build time, figures
rendered from the bundled data):

1. **[From a price to an order book](01_from_price_to_book.md)** — no
   finance knowledge assumed: markets, bourses, the L1 ticker, and the
   book built two orders at a time
2. **[L1 → L2 → L3](02_three_resolutions.md)** — three resolutions of the
   same market: what each level can answer, the whale-vs-crowd contrast,
   and the queue engine at toy and industrial scale
3. **[Loading order data](03_loading_data.md)** — the canonical contract,
   the pipeline, and a LOBSTER round trip on the toy session — no external
   data needed
4. **[Order lifecycles & classification](04_lifecycles.md)** — fate vs
   type, the classifier's evidence, and why ~99.7% of a real book's
   orders never trade
5. **[Depth, spread & liquidity](05_depth.md)** — the depth ledger, the
   heatmap finally earned, and the colour scale as a modelling choice
6. **Trades & flow toxicity** *(planned)*
7. **The visualization system** *(planned)*

Until a chapter lands, the [Getting started](../quickstart.md) page and the
how-to guides cover the same ground recipe-style.
