# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trades and flow toxicity
#
# Chapters 4 and 5 were about *supply* — the orders standing in the book,
# who placed them, how much rests where. This chapter is about *demand*:
# the trades that cross the spread and consume that supply, and a
# question every liquidity provider cares about — **is this flow
# toxic?**
#
# "Toxic" has a precise meaning. When you post a resting quote, you are
# writing a free option: anyone may trade against you at your price.
# Usually that is fine — you collect the spread from impatient traders
# who have no special information. But sometimes the person hitting your
# quote *knows something you don't* (a fill is coming, a number is about
# to print), and the price is about to move against you. Flow from
# informed traders is **toxic** — it systematically loses you money —
# and three classic measures try to detect it from the trade tape alone.
# We will meet each the same way: the formula, what it *means* in one
# sentence, its source, then the code — on a tape small enough to check
# by hand.
#
# ## A constructed tape
#
# The [toy order book](00_toy_session.md) had only five trades — flow
# metrics are built for thousands, so we construct a busier tape with a
# deliberate arc: a calm, balanced open; then an **informed buyer
# accumulates**, lifting the price from 100 to 103; then calm again at
# the new level. Every metric below should rise in the middle and stay
# low at the edges.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

from _docs_theme import ASK_COLOR, BID_COLOR

base = pd.Timestamp("2026-01-05 10:00:00")
_t = []


def trade(sec, price, volume, side):
    _t.append(
        {
            "timestamp": base + pd.Timedelta(seconds=sec),
            "price": float(price),
            "volume": float(volume),
            "direction": side,
        }
    )


# calm: balanced buys and sells at 100
trade(0, 100, 2, "buy")
trade(10, 100, 2, "sell")
trade(20, 100, 2, "buy")
trade(30, 100, 2, "sell")
# informed accumulation: a run of buys walks the price up
trade(60, 100, 4, "buy")
trade(70, 101, 4, "buy")
trade(80, 101, 4, "buy")
trade(90, 102, 4, "buy")
trade(100, 103, 4, "buy")
# calm again at the new level
trade(130, 103, 2, "sell")
trade(140, 103, 2, "buy")
trade(150, 103, 2, "sell")
trade(160, 103, 2, "buy")

tape = pd.DataFrame(_t)
tape["direction"] = pd.Categorical(tape["direction"], categories=["buy", "sell"])


def draw_tape(ax):
    """The reference tape: price over time, buys green-up, sells orange-down,
    marker area ~ volume, the accumulation window shaded."""
    ax.axvspan(
        base + pd.Timedelta(seconds=45),
        base + pd.Timedelta(seconds=115),
        color="#f2e6c9",
        alpha=0.6,
        zorder=0,
    )
    for side, color, marker in (("buy", BID_COLOR, "^"), ("sell", ASK_COLOR, "v")):
        s = tape[tape["direction"] == side]
        ax.scatter(
            s["timestamp"],
            s["price"],
            s=s["volume"] * 40,
            c=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=side,
        )
    ax.plot(tape["timestamp"], tape["price"], color="#888", lw=0.8, zorder=1)
    ax.set_ylabel("Trade price")
    ax.legend(loc="upper left", framealpha=0.9)


fig, ax = plt.subplots(figsize=(11, 3.6))
draw_tape(ax)
ax.set_title("The reference tape — informed accumulation shaded")

# %% [markdown]
# Thirteen trades: four calm, five in the shaded accumulation, four calm
# again. Three buckets, three metrics, one tape.
#
# ## Order flow imbalance: the direction of the flow
#
# The simplest measure. Over a time window, net the buy and sell volume and
# normalise:
#
# ```
#            buy volume − sell volume
#   OFI  =  ─────────────────────────
#            buy volume + sell volume
# ```
#
# **In one sentence:** OFI is which way the flow leaned, on a scale from
# −1 (everyone sold) through 0 (balanced) to +1 (everyone bought). It is
# the short-horizon pressure on price, and it is a simpler relative of
# the order-book event imbalance studied by Cont, Kukanov and Stoikov
# (2014). On the tape, in 30-second windows:

# %%
from ob_analytics.flow_toxicity import order_flow_imbalance

ofi = order_flow_imbalance(tape, window="30s")
ofi[["timestamp", "buy_volume", "sell_volume", "ofi"]]

# %% [markdown]
# Read it against the tape above: the two calm-open windows are +0.33
# and −1.0 (small, mixed), the two accumulation windows peg at **+1.0**
# (buys only), and the calm close settles back to 0.0. The `ofi` face
# draws exactly this, the tape on top and the imbalance bars below —
# note how the bars swell and turn one-sided precisely under the shaded
# region:

# %%
from ob_analytics.visualization import plot, prepare

fig = plot("order_flow_imbalance", **prepare.ofi(ofi, trades=tape))

# %% [markdown]
# ## VPIN: imbalance on a volume clock
#
# OFI slices time into equal *minutes*. VPIN — Volume-Synchronised
# Probability of Informed Trading — slices into equal *volume* instead,
# which is the key idea. Markets trade in bursts; a **volume clock**
# ticks fast when trading is frenzied and slow when it
# is quiet, so each bucket carries the same economic weight. Within each
# bucket of `V` units, VPIN measures the same imbalance:
#
# ```
#                    | buy volume − sell volume |
#   VPIN_bucket  =  ─────────────────────────────
#                            bucket volume V
# ```
#
# **In one sentence:** VPIN is the average one-sidedness of trading per
# unit of volume — high when buyers and sellers are lopsided, which is
# the footprint of someone trading on information (Easley, López de
# Prado and O'Hara, 2012). Note the absolute value: VPIN does not care
# *which* side is winning, only that the flow is imbalanced. With
# 8-unit buckets:

# %%
from ob_analytics.flow_toxicity import compute_vpin

vpin = compute_vpin(tape, bucket_volume=8.0, n_buckets=3)
vpin[["bucket", "buy_volume", "sell_volume", "vpin", "vpin_avg"]].round(3)

# %% [markdown]
# Bucket 0 gathers the calm open — 4 buys, 4 sells, imbalance
# `|4−4|/8 = 0`. The accumulation buckets are 8 units of pure buying:
# `|8−0|/8 = 1`. The headline number is `vpin_avg`, the trailing mean —
# it climbs from 0 through the accumulation and decays afterward, a
# smoothed toxicity measure. The face plots that climb, with a threshold
# line above which flow is "toxic":

# %%
fig = plot("vpin", **prepare.vpin(vpin, threshold=0.7))

# %% [markdown]
# ## Kyle's λ: the price of a unit of flow
#
# OFI and VPIN measure *imbalance*. Kyle's λ measures its *consequence*:
# how far the price moves per unit of net order flow. Over each window,
# take the price change and the signed volume, then fit a line across
# all windows:
#
# ```
#   ΔP  =  λ · (buy volume − sell volume)  +  noise
# ```
#
# **In one sentence:** λ is the price you pay per unit of net flow — the
# slope of price against signed volume, and the market's *illiquidity*,
# since in a deep liquid market even large imbalances barely nudge the
# price (Kyle, 1985). A big λ means a small trade moves the market a
# lot. Fitting on the tape:

# %%
from ob_analytics.flow_toxicity import compute_kyle_lambda

kyle = compute_kyle_lambda(tape, window="30s")
print(f"λ = {kyle.lambda_:.4f}   t = {kyle.t_stat:.2f}   R² = {kyle.r_squared:.3f}")
kyle.regression_df.round(2)

# %% [markdown]
# The regression table is the anchor: the two calm-open windows carry
# signed volume ±2 with **no price change**, while the accumulation
# windows carry +12 and +8 signed units and lift the price +1 each. A
# line through those points has slope λ ≈ 0.09 — about one price unit
# per eleven net units of flow — with a t-statistic near 6 and R² ≈ 0.89:
# on this constructed tape, signed flow explains nearly all of the price
# moves. The face draws that scatter and its fitted line:

# %%
fig = plot("kyle_lambda", **prepare.kyle_lambda(kyle))

# %% [markdown]
# ## The same three metrics on real data
#
# Now the running Bitstamp capture — the tape we have been building on
# since chapter 3, all 284 of its trades:

# %%
from ob_analytics import Pipeline, sample_csv_path

result = Pipeline().run(sample_csv_path())
kyle_real = compute_kyle_lambda(result.trades, window="5min")
print(
    f"trades: {len(result.trades)}   total volume: {result.trades['volume'].sum():.1f}"
)
print(
    f"λ = {kyle_real.lambda_:.3f}   t = {kyle_real.t_stat:.2f}   "
    f"R² = {kyle_real.r_squared:.3f}   windows = {kyle_real.n_windows}"
)

# %% [markdown]
# Look at the t-statistic before the λ: **1.45**. As a rule of thumb a
# coefficient needs |t| ≳ 2 to be distinguishable from zero, so this λ
# — positive, but weak — is *not statistically significant*. And
# that is the honest headline for this dataset: a quiet thirty-minute
# capture with 284 trades and fifteen units of total volume is far below
# the regime these metrics were designed for. They compute; they just
# cannot conclude.
#
# !!! warning "Pitfall: these metrics need volume, and their knobs are not neutral"
#     Flow-toxicity measures were built for high-frequency equity and
#     futures data — thousands of trades per minute, not a handful per
#     minute. On a thin tape, three problems arise. **(1) Significance:** as
#     above, λ's t-statistic collapses; VPIN's trailing average is taken
#     over too few buckets to mean much. **(2) The knobs move the
#     answer:** `bucket_volume` for VPIN and `window` for Kyle and OFI
#     are not neutral defaults — halve the bucket size and VPIN's whole
#     profile shifts. Choose them from the instrument's typical volume
#     (a common VPIN starting point is average daily volume ÷ 50), and
#     report them alongside the number. **(3) Trade side is itself an
#     inference:** every metric here needs each trade labelled buy or
#     sell, which ob-analytics knows exactly (the taker's `direction`
#     from chapter 4) — but on venues that ship only anonymous prints
#     you must *guess* the side (the tick rule, Lee–Ready), and the
#     guess's errors flow straight into every metric on this page.
#
# ## There is no metrics registry
#
# Unlike loaders or plot backends, a flow metric is not a plugin — it is
# just a function over a trades DataFrame. To add your own, write the
# function. Here is Amihud's (2002) illiquidity, |return| per unit
# volume, in four lines:


# %%
def amihud(trades: pd.DataFrame, freq: str = "30s") -> pd.DataFrame:
    t = trades.set_index("timestamp").sort_index()
    ret = t["price"].pct_change().abs()
    return (ret / t["volume"]).resample(freq).mean().rename("amihud").reset_index()


amihud(tape).dropna().round(5)

# %% [markdown]
# It rises in the accumulation windows for the same reason λ does —
# price moving on volume — and it plugs into the same plotting and
# gallery machinery as the built-ins (wrap it in a panel builder; see
# [Extending ob-analytics](../extending.md)). The point of the whole
# chapter: these are ordinary functions over a trades table, and on a
# small tape you can check every number by hand.
#
# **Next:** [The visualization system](07_visualization_system.md) — the
# concepts, levels and backends behind every figure in this tutorial, and
# how to compose your own.
#
# ---
#
# *Vocabulary introduced here — flow toxicity, informed trading,
# [VPIN](../glossary.md#flow-toxicity),
# [Kyle's lambda](../glossary.md#flow-toxicity),
# [order flow imbalance](../glossary.md#flow-toxicity), volume clock —
# lives in the [Glossary](../glossary.md).*
