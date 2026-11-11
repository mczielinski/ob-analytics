"""Regression snapshots — guard behavior-preserving refactors (PR1).

These hash the Bitstamp demo's in-memory pipeline outputs and pin Kyle's
lambda to a recorded scalar. A refactor that changes any number fails
here. Update the recorded values ONLY when an output change is intended
and reviewed.

The pipeline runs exactly once (module-scoped fixture) and we fingerprint
the resulting DataFrames directly — no Parquet round-trip, no gallery, no
duplicate run — so this stays as cheap as a single pipeline pass.
"""

from __future__ import annotations

import csv
import gzip
from collections import Counter
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from ob_analytics.flow_toxicity import kyle_lambda
from ob_analytics.pipeline import PipelineResult
from tests._golden import df_fingerprint


@pytest.fixture(scope="module")
def demo_result(bundled_sample_result) -> PipelineResult:
    return bundled_sample_result


def test_demo_outputs_present(demo_result):
    for name, df in demo_result.tables().items():
        assert df is not None and len(df) > 0, f"{name} empty"


def test_demo_fingerprints(demo_result):
    # First run prints fingerprints; paste them into EXPECTED below, then
    # the equality assertion locks the numeric baseline.
    fps = {name: df_fingerprint(df) for name, df in demo_result.tables().items()}
    for name, fp in fps.items():
        print(f"FINGERPRINT {name} = {fp}")
    # Update these ONLY when an intended output change lands (separate,
    # labeled commit with a before/after rationale).
    # 2026-06-12 (WS-1.1 PR 1/3): price_level_volume dropped the R-inherited
    # "changed & fill==0 adds full volume" branch.  On this sample it fired on
    # exactly two degenerate unknown-type rows (+0 and +1e-08 at one level),
    # so depth / depth_summary / events (via aggressiveness) shift by at most
    # 1e-08 at a single price level; trades are byte-identical (hash kept).
    # 2026-06-12 (correctness batch, WS-8.2): the 13 sample orders with no
    # created row are now typed "pre-existing" instead of "unknown" — only
    # the events type column moves; trades/depth/depth_summary hashes are
    # unchanged.
    # 2026-08-25 (#154, tz-aware UTC nanosecond time model): both clocks are
    # now datetime64[ns, UTC] instead of tz-naive datetime64[ms]. Bitstamp
    # epoch-milliseconds already count from the UTC epoch, so the reconstructed
    # instants are unchanged and depth / depth_summary values and row order are
    # byte-identical to before — only the timestamp DTYPE moved. The fingerprint
    # hashes dtype + values (tz-aware datetime hashes via its string form), so
    # all four digests changed even though no number did.
    # 2026-08-26 (#155, integer-tick prices): every price column is now int64
    # ticks instead of a double in the quote currency. The loader quantises the
    # raw cent-priced feed at tick_size 0.01, which is exactly the integer the
    # depth engine already computed internally, so the reconstruction is the
    # same book re-expressed: a former 133.30 float is now 13330 ticks. Verified
    # element-wise against the pre-tick float pipeline — ticks × 0.01 reproduces
    # the old events / depth / depth_summary prices to 0.0, and every volume /
    # direction / row is identical. DTYPE moved (double -> int64) and the numbers
    # moved (× 100), so all four digests changed.
    # 2026-09-05 (#226, integer-lot sizes): every volume and fill column is now
    # int64 lots instead of a double in the base asset, on the Bitstamp
    # lot_size of 1e-08, so a former 0.5 float is now 50 000 000 lots. Unlike
    # #155 this is not only a re-expression: it also fixes the book. A price
    # level was a float running sum of adds, cancels and fills, and when the
    # last order left a level that sum landed on residue such as 5.55e-17
    # rather than 0. The level stayed live and was reported as the best bid or
    # ask ahead of the real one — on 25 611 of 313 565 rows for the bid (8.2%)
    # and 30 096 (9.6%) for the ask. Integer lots cancel exactly, so those rows
    # now name the true level and the reported spread changes on them.
    # Confirmed against an independent implementation rather than against
    # ourselves: replaying the exported session through hftbacktest's own L3
    # reconstruction (#224) now agrees with depth_summary on the best bid and
    # ask for every row across five synthetic seeds, where before the fix it
    # disagreed on up to 78 rows per seed. All four digests changed: the dtype
    # moved (double -> int64), the numbers moved (x 1e8), and the depth /
    # depth_summary books are corrected. Every depth_summary volume column is
    # an int64 lot count as well, so the per-bin sums are exact.
    # 2026-09-06 (price-level stranding fix): the depth book is corrected.
    # `price_level_volume` added an order's volume at its created price and
    # subtracted it at the price on whichever row removed it, and Bitstamp
    # reports a `deleted` carrying a price the order never rested at for 2,140
    # rows here. The subtraction landed on a level the volume was never added
    # to, so the created level kept it for the rest of the session. At the last
    # event before the capture's synthetic deletes, 104 levels held 29.95 BTC
    # that no order was resting on; now none do. The phantom was the reported
    # touch on both sides -- best bid $78,495.00 against a real $78,350.00, and
    # best ask $78,324.00 against a real $78,333.00 -- so `best_bid_price`
    # moves on 35.7% of rows and `best_ask_vol` on 68.9%. `depth.price` itself
    # moves on the 0.67% of rows being re-pointed. `events` follows through
    # `aggressiveness_bps`, which is measured against the touch, on 28.5% of
    # rows; it also stops being a signed infinity where the touch is a
    # non-tradeable zero price (414 rows). `trades` is unchanged. Checked
    # against the per-order rebuild, which tracks orders by id and never had
    # this problem: the two now agree on how long this book is crossed, 91.61%
    # of session time against 91.59%, where the price-level path read 92.02%.
    # 2026-10-03 (aggressiveness read at the order's own time): only `events`
    # moved, and only in `aggressiveness_bps`. `order_aggressiveness` took the
    # book before each order from the depth_summary row with the next-lower
    # `event_id`, but the Bitstamp loader numbers events after sorting by order
    # id, so that row could be from any time in the session: 51,805 of 156,718
    # new orders were read against a later book, 7,399 against one more than a
    # minute away (up to 30 minutes). The lookup now follows
    # `schemas.time_order_keys` (timestamp, then event_id at one instant), and
    # the value changes on 45,087 of 314,057 rows. Checked against a
    # brute-force search for the last row before each order in that order.
    # `trades`, `depth` and `depth_summary` are unchanged, as is the synthetic
    # golden, whose event ids are already in time order.
    # 2026-10-03 (depth replayed in event order): the price-level rebuild now
    # follows `schemas.time_order_keys` (timestamp, then event_id at one
    # instant), the order the per-order rebuild uses. It sorted on timestamp
    # alone, so one instant's rows were replayed bids first, then asks, by
    # price, and a depth_summary row was not the book after the event it
    # names: for 44,043 of 156,719 new limit orders, the last row before the
    # order in event order was replayed after it. `depth` keeps the same
    # 313,565 rows; 133,387 move. `volume` changes on 30,670 rows, all at a
    # level that one instant changes more than once: each level's running
    # total is now taken in event order, so every row holds the level after its
    # own event. The level at the end of each instant is unchanged on all
    # 209,536 (instant, level) pairs. (Sorting the replay alone, without the
    # running total, would have ended 3,112 of them on the wrong volume.)
    # `depth_summary`: the crossed-level eviction depended on replay order, so
    # the book at the end of an instant changes at 2,634 of 130,244 instants,
    # the best bid or ask at 2,453. `events` follows through
    # `aggressiveness_bps` (read with the time-order lookup above) on 28,898
    # of 314,057 rows. `trades` is unchanged.
    # Checked against the per-order rebuild: on 1,500 sampled new orders the
    # best price on the order's side just before it agrees with the uncrossed
    # rebuild for 1,320, against 1,160 before; on five synthetic seeds every
    # event's last depth_summary row now matches the rebuild, against 3-17
    # mismatching rows per seed before.
    # 2026-10-04 (the `deleted`-row rule, #341): the Bitstamp loader now puts
    # each order's `created` row first.  It sorted an order's rows by size,
    # largest first, and two instant orders bought by quote amount are created
    # with size 0, so their `created` row came last.  One of them,
    # 2002354918830080 (`created` 0, `changed` 1 lot, `deleted` 1 lot), then
    # had its 1-lot `changed` row read as its first row and a phantom 1-lot
    # fill on its `created` row, so the bid level at 78,360.00 lost 2 lots it
    # never had, which the old clip to zero hid.  Now its rows are `created`,
    # `changed`, `deleted` with no fill, and the level holds the 1 lot from
    # the `changed` row to the `deleted` row.  `events` differs on that
    # order's 3 rows only (their order, and the fill).  `depth` and
    # `depth_summary` differ on 3 rows at that instant: the rows move with
    # the events, and for one row the bid touch is that 1 lot at 78,360.00,
    # locked against the ask, as the feed reports it.  The other order,
    # 2002351232385027, is all size 0 and reads the same.  `trades` is
    # unchanged.  No event overdraws a level, so holding a level at zero
    # changes nothing here.
    # 2026-10-07 (taker found by order id, #352): those two instant orders
    # never report the size they executed, so the trade reader matched no row
    # of theirs on size and trades 136 and 279 had no taker.  It now falls
    # back to the order the trade names and takes its `deleted` row.
    # `trades`: `taker_event_id` and `taker_og` gain a value on those 2 rows
    # (187402 / 180895 and 310552 / 304035), and `taker_og`, no longer
    # holding a NaN, is int64 instead of float64.  `events`: the 6 rows of the
    # two orders change `type` from `unknown` to `market`; nothing else moves.
    # `depth` and `depth_summary`: a market order never rests, so the price
    # level rebuild leaves its rows out, and 4 rows go (313,565 -> 313,561):
    # event 187400 and 310550 (each order's `created` row, 0 lots at the
    # placeholder price 999,999,999.00), 310551 (+1 lot at bid 78,360.00) and
    # 310552 (that lot leaving).  The depth_summary row for 310551 was the one
    # whose best bid was that 1 lot, locked against the ask.  Every other row
    # of both tables is identical.
    # 2026-10-24 (#351, one dtype map): every trade reader casts its output
    # through `schemas.TRADE_DTYPES`.  `trades` moves in dtypes only:
    # `maker_event_id` and `taker_event_id` go from object to nullable `Int64`,
    # `maker_og` and `taker_og` from int64 to `Int64`, and `maker` / `taker`
    # from int64 to object.  None of the six holds a null on this sample, and
    # cast back to the old dtypes the frame hashes to the old digest.
    # 2026-10-24 (the depth summary keeps crossed levels, #343): when a quote
    # crossed the other side, `PriceLevelBook` deleted the crossed levels for
    # good, so a real level crossed by a stale quote never came back unless
    # its own price was updated again.  It now keeps every level, as `depth`
    # does, and leaves the crossed levels out of the touch only while they
    # are crossed, by the rule of `order_book(..., uncross=True)`: the level
    # updated less recently is left out, so a locked touch is uncrossed too.
    # `depth_summary` changes on 177,799 of 313,561 rows: the best bid on
    # 47,185, the best ask on 127,514.  Before, 53,498 rows were locked; now
    # none is.  `events` follows through `aggressiveness_bps` on 30,156 of
    # 314,057 rows.  `trades` and `depth` are unchanged.  Checked against the
    # per-order rebuild: at 300 sampled instants, `order_book(...,
    # uncross=True)` has the summary's best bid at 299 and best ask at 299,
    # against 241 and 165 before.  `hidden_trades` now flags none of the 284
    # trades, where it flagged 29 that all had a visible maker.
    # 2026-11-09 (events numbered in file order across orders, #456): the
    # loader numbered events by order id, so rows of two orders at one instant
    # came in the order their ids sort.  They now keep their file order; each
    # order's own rows are unchanged.  Every `event_id` moves, and with it
    # `maker_event_id` / `taker_event_id` in `trades`; mapped through
    # `original_number`, `trades` is otherwise identical.  The time order of
    # events changes at 61,585 of 314,057 positions (51,956 instants hold
    # more than one order).  `depth` differs on 2,888 rows, in `volume`
    # only: the steps inside one instant; the level sizes at the end of every
    # instant are identical.  `depth_summary` differs on 58,173 rows, the same
    # steps; at the end of an instant it differs on 6 of 130,244, all in one
    # crossed episode (02:48:03.743 to 02:48:04.870), where the touch now
    # agrees with `order_book(..., uncross=True)` at each instant (bid
    # 78,374.00, ask 78,375.00) and did not before.  `events` follows through
    # `aggressiveness_bps` on 3,858 rows.
    EXPECTED: dict[str, str] = {
        "events": "4465fcd2d31b6683228b3cc3807baa846032f7252e87ac78c19f0b83f34a576a",
        "trades": "492a83736f801d08be3c3ebac1e4ca4dde6010a6b324082600d1f3d01b4eb7b7",
        "depth": "2deeef0fe9c009ef0d1dea7bbb1297c4c0523500703e7972040be449b2b8149d",
        "depth_summary": "2ec4be96458bcc6029e00204f3ff72ac6efa0c9a76175493eaba0aec595da781",
    }
    if EXPECTED:
        assert fps == EXPECTED


def test_kyle_lambda_baseline(demo_result):
    res = kyle_lambda(demo_result.trades)
    # Record the baseline scalar; C4 (lstsq rewrite) must stay within rtol.
    print(f"KYLE_LAMBDA_BASELINE = {res.lambda_!r}")
    print(f"KYLE_RSQ_BASELINE = {res.r_squared!r}")
    # 2026-08-26 (#155, integer-tick prices): lambda is ΔPrice per unit signed
    # volume, and ΔPrice is now in ticks (int64) rather than dollars, so lambda
    # is exactly 100× the old 8.651849748125398 (tick_size 0.01). The regression
    # r_squared is scale-free and unchanged. Multiply by tick_size for the
    # quote-currency value.
    # 2026-09-05 (#226, integer-lot sizes): the denominator followed the
    # numerator. Signed volume is now in lots (int64) rather than the base
    # asset, so lambda is exactly 1e-8× the previous 865.184974812539
    # (lot_size 1e-08) and reads as ticks per lot. r_squared is scale-free and
    # is unchanged again, at the same 0.2960289115722834, which is what says
    # the regression itself did not move — only its units. Multiply by
    # tick_size and divide by lot_size for the quote-currency-per-base-asset
    # value.
    # 2026-11-07 (#400, measures in instrument units): kyle_lambda reads
    # tick_size and lot_size off the trades and reports dollars per BTC, so
    # lambda is back at the original 8.651849748125398 (the 1e-6 above times
    # tick_size / lot_size = 1e6), with r_squared unchanged once more.
    BASELINE_LAMBDA: float | None = 8.651849748125398
    if BASELINE_LAMBDA is not None and not np.isnan(BASELINE_LAMBDA):
        assert np.isclose(res.lambda_, BASELINE_LAMBDA, rtol=1e-10, equal_nan=True)


# ── Totals counted from the raw files ─────────────────────────────────
#
# The fingerprints only say the numbers have not moved.  These count the
# sample's own CSV rows with the standard library, not with the package, so a
# loader that drops, doubles or rescales rows fails here even when a
# re-recorded fingerprint would hide it.


def _raw_rows(path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def test_trades_match_the_raw_trades_file(demo_result, bitstamp_sample_dir):
    raw = _raw_rows(bitstamp_sample_dir / "trades.csv")
    trades = demo_result.trades
    tick = Decimal(str(demo_result.config.tick_size))
    lot = Decimal(str(demo_result.config.lot_size))

    assert len(trades) == len(raw)
    assert Counter(trades["direction"].astype(str)) == Counter(r["side"] for r in raw)
    assert int(trades["volume"].sum()) == sum(Decimal(r["amount"]) for r in raw) / lot
    notional = sum(Decimal(r["price"]) * Decimal(r["amount"]) for r in raw)
    ticks_by_lots = sum(
        int(p) * int(v) for p, v in zip(trades["price"], trades["volume"])
    )
    assert ticks_by_lots == notional / tick / lot
    stamps = [int(r["timestamp"]) for r in raw]
    first, last = min(stamps), max(stamps)
    assert trades["timestamp"].min() == pd.Timestamp(first, unit="ms", tz="UTC")
    assert trades["timestamp"].max() == pd.Timestamp(last, unit="ms", tz="UTC")


def test_events_match_the_raw_orders_file(demo_result, bitstamp_sample_dir):
    raw = _raw_rows(bitstamp_sample_dir / "orders.csv.gz")
    events = demo_result.events

    assert len(events) == len(raw)
    assert Counter(events["action"].astype(str)) == Counter(r["action"] for r in raw)
    assert events["id"].nunique() == len({r["id"] for r in raw})
    created = [r for r in raw if r["action"] == "created"]
    lot = Decimal(str(demo_result.config.lot_size))
    on_created = events.loc[events["action"] == "created", "volume"]
    assert int(on_created.sum()) == sum(Decimal(r["volume"]) for r in created) / lot
