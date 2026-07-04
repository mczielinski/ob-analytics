# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # L1 → L2 → L3: three resolutions of the same market
#
# [Chapter 1](01_from_price_to_book.md) ended with a thesis: everything
# past the ticker is just *refusing to summarize*. This chapter makes
# that precise. Market data comes in three standard resolutions, and
# they nest like map zoom levels — city, street, building. Same
# territory, three levels of detail, and each level is a **lossy
# summary of the next**: you can always compute L1 from L2, and L2 from
# L3, but never the reverse.
#
# ## One instant, three renderings
#
# Here is the toy session's book at t=30s, drawn at all three
# resolutions at once:

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

from _docs_theme import plot_l1_ticker
from ob_analytics import toy_events, toy_trades
from ob_analytics.analytics import order_book, set_order_types
from ob_analytics.visualization import plot, prepare

events = set_order_types(toy_events(), toy_trades())
t30 = events["timestamp"].iloc[0] + pd.Timedelta(seconds=30)
snap = order_book(events, tp=t30)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))
plot_l1_ticker(bid=99, ask=101, last=101, ax=ax1)
ax1.set_title("L1 — the quote", fontsize=10)
plot("book_snapshot", level="L2", ax=ax2, **prepare.book_snapshot(snap))
ax2.set_title("L2 — market by price", fontsize=10)
plot("book_snapshot", level="L3", ax=ax3, **prepare.book_snapshot(snap, per_order=True))
ax3.set_title("L3 — market by order", fontsize=10)
fig.tight_layout()

# %% [markdown]
# Read it left to right, adding detail as you go:
#
# - **L1** is the quote: best bid 99, best ask 101, last 101. Three
#   numbers, nothing else.
# - **L2 (market by price)** unfolds the quote into the full ladder —
#   *every* price level with its total size: 4 at 99, 4 at 98, not just
#   the best. L1 was this picture's top row.
# - **L3 (market by order)** splits each bar into its owners: at 99
#   those 4 units are *two* bids (Alice's 2 and Ivy's 2), at 98 two
#   more. Identity and queue position appear.
#
# Now run it backwards: each panel is a **lossy summary** of the one to
# its right. Summing L3's segments gives you L2 exactly; keeping L2's
# top row gives you L1 exactly. The reverse is impossible — who owns
# the 4 at 99, and who is first in line, cannot be recovered from the
# bar's length.
#
# ## Same L2, different markets
#
# How much does the L3 → L2 summary destroy? Here are two ask books
# that are **identical at L2** — four units offered at 101 — drawn at
# both resolutions:

# %%
whale = {
    "timestamp": t30,
    "bids": pd.DataFrame({"id": [1], "price": [99.0], "volume": [4.0]}).iloc[:0],
    "asks": pd.DataFrame({"id": [10], "price": [101.0], "volume": [4.0]}),
}
crowd = {
    "timestamp": t30,
    "bids": whale["bids"],
    "asks": pd.DataFrame(
        {"id": [21, 22, 23, 24], "price": [101.0] * 4, "volume": [1.0] * 4}
    ),
}

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
for col, (name, book) in enumerate([("one whale", whale), ("a crowd", crowd)]):
    plot("book_snapshot", level="L2", ax=axes[0][col], **prepare.book_snapshot(book))
    axes[0][col].set_title(f"L2 — {name}", fontsize=10)
    plot(
        "book_snapshot",
        level="L3",
        ax=axes[1][col],
        **prepare.book_snapshot(book, per_order=True),
    )
    axes[1][col].set_title(f"L3 — {name}", fontsize=10)
for ax in axes.ravel():
    ax.set_ylim(100.4, 101.6)
    ax.legend().remove()
fig.tight_layout()

# %% [markdown]
# The top row is indistinguishable. The bottom row is two different
# markets: a single 4-lot (one trader's conviction — or one trader who
# can vanish in one cancellation) versus four independent 1-lots. If
# you trade against it, the difference matters: the whale cancelling
# removes *all* the liquidity at once; the crowd thins out one order at
# a time. L2 cannot tell you which market you're in. **That is the
# information L3 buys you.**
#
# ## Why L3 exists: the queue
#
# Chapter 1 showed the 99-bid queue paying off geometrically. The
# package's queue engine turns that picture into a quantitative face:
# each resting order's **FIFO rank at the touch** over time — rank 1 is
# the front of the line — coloured by how the order's story ended.
# Every trajectory below is labeled with its owner, and the book
# ladders underneath show the decisive instants:

# %%
from _docs_theme import plot_queue_story

fig = plot_queue_story(events, toy_trades(), at_s=[6, 45, 56, 57])

# %% [markdown]
# Now the walk is literal:
#
# - **Ivy** (pink) is the story: she joins the 99 queue at t=6 at
#   rank 2 — the ladder below shows her stacked behind Alice — and
#   waits there for *fifty seconds*. When Sam's sweep fills **Alice**
#   at t=56 (×), Ivy steps to rank 1... and is promptly half-filled
#   herself (t=57 ladder: only `Iv 1` remains). Still resting at the
#   end: pink.
# - **Bob** holds rank 1 on the ask side for 46 seconds and fills in
#   two bites. **Frank, Iris, and Sam** blink through rank 1 for a
#   single instant each — market orders technically join the queue
#   too, for the moment it takes to match.
# - **Eve** (yellow ○) jumps straight to rank 1 at a brand-new best
#   price of 100 — front of a queue of one. The t=45 ladder catches
#   her mid-flash; 800 ms later she cancels.
# - **Dana and Chen never appear at all.** This face tracks the queue
#   *at the touch*; their bids at 98 sat one level below it the whole
#   session. A quiet lesson in reading any plot: know what it excludes.
#
# Ivy's fifty seconds at rank 2 and Hana's story are two answers to the
# same question — *how do I get to the front?* Ivy waited. **Hana paid**:
# her order crossed the spread (a taker for 2 units), and the unfilled
# remainder rested at 101, a fresh best bid — instant rank 1, filled
# four seconds later (her × at t=52). That trade-off — queue time
# versus crossing cost — is the daily arithmetic of market making, and
# it is *invisible* below L3.
#
# ## Which level do you need?
#
# | Question | L1 | L2 | L3 |
# |---|---|---|---|
# | Did the price go up today? | ✅ | ✅ | ✅ |
# | What would it cost to buy 500 *right now*? | ❌ | ✅ | ✅ |
# | How much liquidity sits within 10 bps of the mid? | ❌ | ✅ | ✅ |
# | Is that liquidity one whale or a crowd? | ❌ | ❌ | ✅ |
# | Where is *my* order in the queue? | ❌ | ❌ | ✅ |
# | Who cancelled, and how fast? Flash detection? | ❌ | ❌ | ✅ |
# | Full order lifetimes, maker/taker attribution? | ❌ | ❌ | ✅ |
#
# The market-data industry prices along the same ladder: L1 quotes are
# ubiquitous and cheap, L2 depth costs more, and full L3 — usually sold
# as **market-by-order (MBO)** feeds — is the premium product
# ([LOBSTER](https://lobsterdata.com/) for academic equity data,
# Databento and the exchanges' direct feeds commercially). ob-analytics
# is built for that top rung: its loaders reconstruct full per-order
# streams, which is why every question in the table is answerable.
#
# ## The resolution decides what you may ask
#
# This isn't just a data-shopping concern — it's encoded in the API.
# Load the bundled real capture and ask what it can draw:

# %%
from ob_analytics import Pipeline, sample_csv_path
from ob_analytics.visualization import available_concepts

result = Pipeline().run(sample_csv_path())
available_concepts(result)

# %% [markdown]
# Some concepts exist at both resolutions (`trade_tape`,
# `book_snapshot` — the *comparable* pairs you've been looking at);
# some only make sense at one (`depth_heatmap` aggregates by
# construction; `queue_position` and `order_outcome` need identities).
# Here is the queue face again — same code as the toy — on 314,000
# real events:

# %%
t_start = result.events["timestamp"].min()
fig = plot(
    "queue_position",
    level="L3",
    **prepare.queue_position_l3(
        result.events,
        start_time=t_start,
        end_time=t_start + pd.Timedelta(minutes=10),
    ),
)

# %% [markdown]
# Hundreds of orders marching toward rank 1 as the queue ahead of them
# fills or gives up — Ivy's fifty seconds, at industrial scale.
#
# !!! warning "Pitfall: not every L3 feed is a matched book"
#     LOBSTER and exchange MBO feeds are *matched books*: the venue's
#     own engine guarantees bids never cross asks. The Bitstamp public
#     feed reconstructed here is a **placement/cancellation diff
#     stream** — and it genuinely contains crossed *resting* orders
#     (we've verified a bid resting above an ask for ~1.5 minutes,
#     neither ever filling). `order_book()` replays such feeds
#     faithfully rather than silently "fixing" them: a crossed book in
#     your output is a property of the feed, not a reconstruction bug.
#     Know which kind of feed you're holding before you trust an
#     uncrossed-book invariant.
#
# **Next:** loading data — Bitstamp, LOBSTER, and your own — where
# these feed differences become practical. *(Chapter in progress; the
# [how-to guides](../howto/your-own-data.md) cover the recipes today.)*
#
# ---
#
# *Vocabulary introduced here —
# [Level 1 / Level 2 / Level 3, market-by-order](../glossary.md#market-data-levels),
# [price–time priority](../glossary.md#exchange-mechanics), queue rank,
# [matched book vs diff feed](../glossary.md#data-formats) — lives in
# the [Glossary](../glossary.md).*
