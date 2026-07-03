---
title: Tutorial
---

# Tutorial

A guided tour of market microstructure through ob-analytics — starting from
absolute basics (what a price *is*, what an exchange does) and building up to
depth analysis, flow toxicity, and per-order (L3) reconstruction. Chapters
are being written; today, two runnable notebook tutorials cover the core
workflow.

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

## Planned chapters

The tutorial is growing into a full guide (executed at build time, figures
rendered from the bundled data):

1. **From a price to an order book** — markets, bourses, and L1 from first
   principles
2. **L1 → L2 → L3** — three resolutions of the same market
3. **Loading order data** — Bitstamp, LOBSTER, and your own
4. **Order lifecycles & classification**
5. **Depth, spread & liquidity**
6. **Trades & flow toxicity**
7. **The visualization system**

Until a chapter lands, the [Getting started](../quickstart.md) page and the
how-to guides cover the same ground recipe-style.
