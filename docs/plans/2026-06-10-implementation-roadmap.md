# Implementation Roadmap — ob-analytics

**Date:** 2026-06-10 · **Status:** approved working plan (do not commit without the user's explicit ask)
**Audience:** future coding agents. Each item is self-contained: evidence → fix → acceptance.
**Baseline at time of writing:** main @ `8552ac8`, 375 tests passing in 9m29s, ruff/ty clean.

---

## How to use this document

- Work one **workstream phase** at a time (sequencing at the bottom). Don't mix bloat-removal
  commits with behavior changes. Branching/worktree/PR process: see **Development workflow**
  at the end of this document.
- Every claim below was verified on 2026-06-10 by execution or by inspecting rendered output in
  `/home/mcz/Desktop/{bitstamp,lobster}_output_{l2,l3,both}/gallery/` (regenerate with the
  commands below). Re-verify before fixing — line numbers drift.
- **Never commit or push without the user's explicit instruction in the moment.**

### Verification harness

```bash
uv run pytest tests/ -q                      # full suite (slow until WS-2 lands)
uv run ruff check ob_analytics/ tests/ && uv run ty check ob_analytics/

# Visual verification (the only way to validate WS-3):
uv run ob-analytics bitstamp-demo --view both -o /tmp/bs_both
uv run ob-analytics lobster-demo /home/mcz/Downloads/LOBSTER_SampleFile_AAPL_2012-06-21_50 \
    --trading-date 2012-06-21 --view both -o /tmp/lob_both
# then Read the PNGs in <out>/gallery/matplotlib/ and compare against the reference bundle:
#   /home/mcz/Downloads/lob_visualizations/lob_viz_bundle/{images,code,README.md}
```

The bundle README's "Design principles applied" section is the binding style guide:
position/length for quantities (Cleveland–McGill ranks 1–3), hue only for categories,
saturation only for density, step plots for discrete events, small multiples + shared axes,
annotations inside plots.

### LOBSTER ground truth

`/home/mcz/Downloads/LOBSTER_SampleFile_AAPL_2012-06-21_50` (AAPL, 50 levels). The
`*_orderbook_50.csv` companion is authoritative book state after every event — use it to
acceptance-test any per-order reconstruction.

---

## WS-1 · Correctness (P0)

### 1.1 Canonical per-order semantics — THE structural fix

**Evidence (verified by execution).** `order_book(events)` on the LOBSTER sample returns
best bid **587.50** / best ask **584.33** (a crossed book) vs ground-truth **585.69 / 585.95**,
with **2,944 of 3,324 (89%) "active" orders being phantoms** — fully-executed orders. Root
cause is two-fold:

1. LOBSTER never emits a `deleted` event for fully-executed orders (they just stop appearing),
   while `order_book()` computes active = created − deleted ([analytics.py:360-368](../../ob_analytics/analytics.py)).
   Bitstamp *does* emit `deleted` on full fill, which is why the bug is LOBSTER-only.
2. The events-frame `volume` column means **remaining-after-event** for Bitstamp but
   **per-event delta** for LOBSTER `changed` rows (type 2/4/5 Size = cancelled/executed qty;
   see [lobster.py:131-168](../../ob_analytics/lobster.py)). Every consumer that reads `volume`
   off non-`created` rows silently assumes Bitstamp semantics.

**Blast radius:** `order_book()` → `book_snapshot.L2/L3` and `depth_chart.L2/L3` gallery faces
render phantom data for LOBSTER; `prepare_order_activity_l3_data`'s teal-wall bug (§3.2);
`prepare_order_outcome_l3_data`'s censoring test. The depth pipeline itself
(`price_level_volume`) is correct — it uses `fill` deltas.

**Fix (two parts):**

- **(a) Pin the schema.** In [schemas.py](../../ob_analytics/schemas.py), document and enforce:
  `volume` = remaining volume after the event, `fill` = executed delta, for **every** action.
  Make `LobsterLoader.load` derive remaining volume per order id
  (`placed − cumsum(fill or cancel deltas)`, groupby-cumsum, vectorized), keeping the raw Size
  in a new `raw_size` column. Add a loader contract test asserting remaining-volume semantics
  for both formats.
- **(b) One lifecycle table.** Add `order_lifecycles(events, trades=None) -> DataFrame` to
  [analytics.py](../../ob_analytics/analytics.py):
  `id, direction, price, placed_ts, placed_vol, filled_vol, end_ts, outcome`
  with `outcome ∈ {filled, partial, cancelled, flashed, resting}`. Terminal condition =
  `deleted` event **or remaining volume reaches 0**; `resting` = neither by window end.
  Replace the three ad-hoc derivations: `prepare_order_activity_l3_data`
  ([_data.py:404-421](../../ob_analytics/visualization/_data.py)),
  `prepare_order_outcome_l3_data` ([_data.py:514-541](../../ob_analytics/visualization/_data.py)),
  and `order_book()`'s active-set logic. The future queue engine (§4.1) consumes it too.

**Acceptance:** new test comparing `order_book()` best bid/ask + per-level totals against the
last row of the LOBSTER orderbook file (tolerance: exact prices, sizes within visible-only
caveats); phantom count ≈ 0; Bitstamp regression snapshot unchanged.

### 1.2 `Pipeline(config=…, format=…)` silently drops format defaults

**Evidence:** `Pipeline(config=PipelineConfig(depth_bps=50), format=LobsterFormat(), …)`
yields `price_divisor == 1` (must be 10_000) — every LOBSTER price wrong by 10⁴, no warning.
[pipeline.py:141-152](../../ob_analytics/pipeline.py).

**Fix:** merge — format defaults overlaid by the user's *explicitly set* fields only:
`PipelineConfig(**{**fmt.config_defaults(), **{k: getattr(config, k) for k in config.model_fields_set}})`.
Log the effective config at INFO. **Acceptance:** unit test for the exact repro above.

### 1.3 `volume_percentiles` hardcodes 25–500 bps columns

**Evidence:** `prepare_volume_percentiles_data` raises `KeyError` for any
`depth_bps ≠ 25` / `depth_bins ≠ 20` ([_data.py:876-902](../../ob_analytics/visualization/_data.py));
the gallery silently shows "Not available". `PipelineConfig.bps_labels` exists for this and is
never used. **Fix:** derive bin columns by introspecting `depth_summary.columns`
(regex `(bid|ask)_vol(\d+)bps`) — no config plumbing needed; build legend labels and the
N-step palette from the discovered count. **Acceptance:** parametrized test at
`depth_bps=50, depth_bins=10`; rendered face non-empty.

### 1.4 Broken documented examples (`plot("trade_tape", …)`)

`trade_tape` is comparable since the taxonomy merge → bare `plot("trade_tape", …)` raises
`ValueError`. Stale sites: [quickstart.md:71,90,442](../quickstart.md). Fix alongside §5.2
(public prepare API) so docs are only rewritten once.

### 1.5 Mutation hygiene

`order_aggressiveness` writes `events["aggressiveness_bps"]` on the caller's frame
([analytics.py:99](../../ob_analytics/analytics.py)); `set_order_types` writes
`events["type"]` in place ([analytics.py:212](../../ob_analytics/analytics.py)). Copy first
(cheap, CoW) or document loudly. **Acceptance:** test that input frames are unchanged.

### 1.6 Capture `meta.json` undercounts on SIGINT

Counters from the cancelled stream task are discarded
([live/_runner.py:177-184](../../ob_analytics/live/_runner.py)). Accumulate into a shared
mutable counter object written by `_stream` as it goes.

### 1.7 ~~Dead `[tool.pytest]` table~~ — RETRACTED (verified 2026-06-10)

pytest 9 (pinned: `pytest>=9.0.3`) **does** read `[tool.pytest]` — coverage tables print on
local runs, so the config is live. Do not "fix" this. The real issue it hides is performance:
see §2.6 (`--cov` in `addopts` slows every local run).

---

## WS-2 · Performance (P0 — fix before iterating on visuals; demos take ~2.5 min/run)

All numbers measured 2026-06-10 on the dev box. Raw (no profiler) stage timings from the
demo logs — Bitstamp sample (314k events): **load 14.7s · depth_metrics 49.8–68.8s ·
everything else ~3s**; LOBSTER AAPL_50 (92k events): **orderbook-diff loop ~30s ·
engine 4.2s · everything else <1s**; Bitstamp `both` gallery: **45.3s for 15 cards**
(depth_heatmap 18.2s, order_activity L2+L3 16.3s, cancellations 5.7s). Full test suite:
**9m29s** (with coverage on). Package import: 0.63s (fine).

Validated fixes, in order of measured impact:

| # | Target | Now | After (measured/projected) | Status |
|---|---|---|---|---|
| 2.1 | depth engine binning | 49.8–68.8s | ~10–12s numpy tier; ~4s incremental tier | prototyped, identical output |
| 2.2 | Bitstamp loader transform | 19.4s | **0.017s (1,139×), exact-equal verified** | prototyped |
| 2.3 | LOBSTER orderbook-diff loop | ~30s | numpy diff core measured **0.15s** | core prototyped |
| 2.4 | gallery: heatmap card | ~25s | **2.7s (9×)** via one LineCollection | prototyped |
| 2.5 | gallery: `save_figure` double-draw | ~half of every card's cost | drop `bbox_inches="tight"` | verified cause |
| 2.6 | test suite | 9m29s | ~2 min (est.) | causes measured |

### 2.1 Depth-engine binning (the pipeline hot loop)

Measured on the real sample: **active levels per side mean ≈ 1,786, p95 ≈ 2,911** (the code
comment's "few hundred" is wrong — update it), 322,806 side-writes, and `best` changes on
**33.6%** of writes. Each write currently re-iterates the entire levels dict in Python
([depth.py:28-73](../../ob_analytics/depth.py)): 341µs/call measured.

- **Tier 1 (do this):** keep each side's levels in sorted numpy arrays (or `SortedDict` +
  cached key/value arrays); compute bins via masked `cumsum` + `searchsorted`. Measured
  **36µs/call (10×), output identical** → engine ≈ 10–12s, remaining cost is the Python
  per-event loop itself.
- **Tier 2 (optional, after Tier 1):** incremental bin sums — a volume change touches exactly
  one bin when `best` is unchanged (**66.4% of writes → O(1)**); full numpy re-bin only on
  best-moves (33.6%). Projected engine ≈ 4s. More state, more invariants — only do it with the
  regression snapshot green.

**Acceptance:** `depth_metrics(sample_depth)` byte-identical
([tests/test_regression_snapshot.py](../../tests/test_regression_snapshot.py)); pipeline
wall-time on the sample ≤ 20s.

### 2.2 Bitstamp loader timestamp transform

The per-group `lambda x: np.sort(...)` over ~157k groups costs **19.4s**; events are already
id-sorted ([bitstamp.py:114-118](../../ob_analytics/bitstamp.py)), so one global
`np.lexsort((timestamp, id))` gather is equivalent — **verified `Series.equals` on the sample,
0.017s**. Optional bonus: `pd.read_csv(..., engine="pyarrow")` reads orders.csv in 0.04s vs
0.23s (measure dtype fidelity before adopting). **Acceptance:** loader output frame-equal;
load ≤ 2s.

### 2.3 LOBSTER `lobster_depth_from_orderbook` row-loop

The Python row×level diff loop ([lobster.py:653-691](../../ob_analytics/lobster.py)) costs
~30s of the ~35s LOBSTER pipeline. Numpy core measured: `arr[1:] != arr[:-1]` change-mask over
the whole (92k × 200) orderbook array takes **0.15s**; build the depth rows from
`np.nonzero` on the mask (price/size columns, side from column parity) instead of per-row
dicts. Edge cases: dummy-price sentinels, duplicate prices across levels (sum), first row
emits all levels. **Acceptance:** depth frame equal to current implementation on AAPL_50;
LOBSTER pipeline ≤ 6s.

Related, lower priority: `LobsterWriter._reconstruct_orderbook` measured ~7s full-file
(write path only) — same treatment if it starts to matter.

### 2.4 Gallery: batch the heatmap into one `LineCollection`

`mpl_price_levels` builds one `LineCollection` **per price** (~3.5k collections,
[_matplotlib.py:199-214](../../ob_analytics/visualization/_matplotlib.py)). Card apportioning:
prepare 0.2s · artist build 6.2s · **save 18.6s** · plotly build 0.2s · write_html 0.9s.
Prototype with a single vectorized LineCollection (np.roll group-boundary mask, one
`add_collection`): **build 1.9s + save 0.8s, 296,180 segments — ~9× the whole card**.
Same approach for the L3 Gantt's hlines and plotly `_segments_xy` inputs (density degrade in
§3.2 also shrinks N drawn).

### 2.5 Gallery: `save_figure` double-draws every figure

`bbox_inches="tight"` ([_matplotlib.py:112](../../ob_analytics/visualization/_matplotlib.py))
forces a second full draw at save time — for dense figures that's ~half the card cost
(order_activity.L2: render 1.4s vs save 7.9s). Every renderer already calls
`fig.tight_layout()`. Drop `bbox_inches="tight"` from the default (keep as opt-in kwarg);
verify the two faces with outside-axes legends (volume_percentiles, OFI twin-axis) don't clip.
Gallery projection with 2.4+2.5: **45s → ~12s**. Parallelizing panel renders across processes
is possible but unnecessary after this — don't add the complexity.

### 2.6 Test suite (9m29s → ~2 min)

Measured contributors:
- `--cov=ob_analytics` lives in `addopts` ([pyproject.toml:60](../../pyproject.toml)) so **every
  local run pays coverage instrumentation** — multiplies the cost of exactly the hot Python
  loops the suite spends its time in (regression-snapshot module setup alone: 308s under
  cov + contention). CI passes `--cov` explicitly ([ci.yml:54](../../.github/workflows/ci.yml))
  — remove it from `addopts`.
- **9 separate full-sample `BitstampLoader().load()` calls** across
  [test_bitstamp.py](../../tests/test_bitstamp.py) (6×) and
  [test_classes.py](../../tests/test_classes.py) (3×) ≈ 15s each → share one session-scoped
  `loaded_sample_events` fixture (and they all shrink ~7× anyway once §2.2 lands).
- `test_regression_snapshot` fingerprints via `df.to_csv()` — **23.6s** of CSV serialization
  per run ([test_regression_snapshot.py:26-37](../../tests/test_regression_snapshot.py)).
  Hash per-column `ndarray.tobytes()` + names + dtypes instead (IEEE754 bytes are
  pandas-version-independent, unlike CSV float formatting): ~0.1s, stronger stability.
- Add `--durations=15` to CI so regressions in test cost stay visible.

### 2.7 Plotly payloads (size, not CPU)

`depth_heatmap.L2.html` = **25 MB** (313k Scattergl points as JSON; write itself only 0.9s);
a `both` gallery ships ~80 MB across iframes and risks the browser's WebGL-context cap
(~8–16; excess contexts silently fail). Decimate to screen-resolution (drop
sub-pixel-duplicate points per price-row) or bin to `go.Heatmap`; share one plotly.js via
`include_plotlyjs="directory"` instead of per-file CDN tags.

### 2.8 Measured non-issues (don't chase these)

- `compute_vpin`: **1.0s per 1M trades** — the Python loop only iterates bucket boundaries.
- Package import: 0.63s. · Parquet save/load: ~0.5s. · `BitstampTradeReader` matching: ~0.3s.
- `set_order_types`, `order_aggressiveness`, `price_level_volume`: each ≤ 0.5s on the sample.
- `prepare_volume_percentiles_data`'s melt→pivot dance: runs on minute-resampled data (~30
  rows) — aesthetic only (§WS-7 narration sweep may simplify it, not for speed).

---

## WS-3 · Visualization fixes (verified against rendered output)

> Every item below was confirmed by viewing the actual PNGs on 2026-06-10. Reference images:
> bundle `images/`. The shared primitives (3.0) unblock most faces — build them first.

### 3.0 Shared primitives

- **FocusWindow** (the `FUTURE(--focus)` marker in
  [gallery.py:296](../../ob_analytics/visualization/gallery.py)): one value object
  `(t0, t1, price_lo, price_hi)` anchored at the touch/mid (e.g. trades-median ± k·σ, or
  best-bid/ask envelope ± N bps), computed **once** in `build_gallery_model` and passed to every
  prepare. Replaces the per-face ad-hoc clips (q01/q99 here, ±3σ there, `bps_range=150`
  elsewhere) that cause §3.1/§3.2/§3.4/§3.8.
- **Semantic palette module** (`visualization/_palette.py`): today identical hex constants are
  duplicated in [_matplotlib.py:398-417](../../ob_analytics/visualization/_matplotlib.py) and
  [_plotly.py:314-333](../../ob_analytics/visualization/_plotly.py) with "kept identical" comments.
  Single source: side (bid / ask), fate (cancelled / filled / partial), aggressor (buy / sell).
  **Constraint (user-approved 2026-06-10): palettes must be colorblind-safe** — the current
  aggressor pair buy-green `#2e9e5b` / sell-red `#dd4444` fails red–green CVD (the most common
  form); the reference bundle deliberately uses blue/orange. Re-pick pairs that survive
  deuteranopia simulation and grayscale print; verify with a CVD simulator before merging.
- **Time-axis helper:** several faces show raw `"02 02:45"`-style ticks (day-of-month prefix —
  seen on order_activity.L2, volume_percentiles, events others) while others set
  `%H:%M:%S` explicitly. One `format_time_axis(ax)` using `ConciseDateFormatter`; plotly
  equivalent.
- **Theme:** current default (`darkgrid`, `axes.facecolor: "darkgray"`, font_scale 1.5)
  renders gray-on-gray and washes out hairlines (§3.1). Adopt the bundle's style as default
  `PlotTheme`: white background, dotted light grid, despined top/right, title-left-bold,
  smaller fonts. Also resolves the mpl-light vs plotly-dark split inside one gallery page —
  pick one polarity for both backends. **Keep seaborn (user decision 2026-06-10: "wait on
  seaborn removal")** — implement the theme via `PlotTheme`/rcParams on top of seaborn, do not
  rip the dependency out as part of this work.
- **Gallery captions:** `PlotConcept.note` ([gallery.py:86](../../ob_analytics/visualization/gallery.py))
  is plumbed but never set or rendered. Set a 1–2 sentence "how to read this" per concept and
  render under the card title. The bundle's in-image annotations are the model.
- Analytics panels are dropped from the `l3` view ([gallery.py:630](../../ob_analytics/visualization/gallery.py));
  include them in every view (they're level-less).

### 3.1 `book_snapshot` — flagship comparison face is broken

**Evidence:** Bitstamp `book_snapshot.L2.png` and `.L3.png` are **bit-identical** (same md5):
the only L3 cue (white per-order separators) is gated off below 2px bar width
([_matplotlib.py:436-452](../../ob_analytics/visualization/_matplotlib.py)) and at ±150 bps on a
$78k book the bars are sub-pixel. Isolated hairline bars antialias into illegible gray on the
darkgray facecolor. On LOBSTER, `_high_volume_prices` (q99 of **per-row** volumes,
[_data.py:799-803](../../ob_analytics/visualization/_data.py)) emits a forest of full-height
dashed black guide-lines that buries the data (per-order rows ⇒ many rows ≥ q99).

**Fix:**
- Window the snapshot to the FocusWindow (or top-N levels per side, N≈30–50) so bars are
  always ≥ ~3 px; then per-order separators are always on and L3 ≠ L2 by construction.
- Switch to the **horizontal ladder** orientation of the bundle (`cmp_2`/`02b`: price on y,
  size on x, bids below mid, asks above; segments stacked biggest-first within a level, white
  1.3px separators; annotate "one whale vs 15 small" style callouts via `note`).
- Quantile guides: off by default; when on, thin/low-alpha, computed per **level** (not per
  order row), max ~3 per side.
- Optional far-depth context: a small inset or cumulative sparkline, instead of stretching the
  main axis ±150 bps.
**Acceptance:** L2 vs L3 md5s differ on Bitstamp; segments visually countable at the touch;
LOBSTER face has ≤ 6 guide lines; visual check against `images/cmp_2_book_snapshot.png`.

### 3.2 `order_activity.L3` Gantt — saturation wall

**Evidence:** LOBSTER render is a solid teal wall: filled orders never get `deleted`, so
`end_ts` falls back to window end for *every executed order*
([_data.py:420-421](../../ob_analytics/visualization/_data.py)) — fixed at the root by §1.1's
lifecycle table (`end_ts` = fill-exhaustion time). Bitstamp render is a near-solid orange
field: ~10⁴–10⁵ lifelines with fixed `linewidth=1.2, alpha=0.5` — individual lifecycles
unreadable.

**Fix:** consume `order_lifecycles()`; clip to FocusWindow; **degrade by density**: when
spans-in-window > ~2,000, either (a) sample stratified by fate (keep all filled/partial, sample
cancelled) with a "showing n of N" annotation, or (b) switch to a time×price occupancy
heat (saturation = resting liquidity-time, which is then a legitimate density encoding).
Encode size as linewidth (bundle: `lw = 0.5 + size·k`), terminal markers (× fill, ○ cancel) when
N small. **Acceptance:** AAPL render shows distinguishable spans with all three fates; compare
`images/L3_1_order_lifecycle_gantt.png`, `cmp_6`.

### 3.3 `cancellations.L3` — wrong lens for the distribution

**Evidence:** both venues render a red mass crushed against the origin on **linear** axes;
the three latent populations (fleeting/patient/deep) and LOBSTER's striking mass-cancel
stripes at ~15s/18s are invisible. The in-code `FUTURE(--density)` note already names the fix.

**Fix:** log-log axes (clip age at ~1ms floor, distance at 0.05 bps; keep zero-age as a small
"instant" jitter band) + density encoding: mpl `hexbin` per side (or 2D hist with
saturation), plotly `Histogram2d`/binned `Heatmap` instead of 4.8 MB of raw Scattergl points.
Keep the per-side split (small multiples: bid | ask) rather than overplotting blue-on-red.
Annotate the population regions like `images/L3_3_cancel_age_distance.png`. Drop the
bps-quantile clip once axes are log (it currently also clips structure).
**Acceptance:** three clusters identifiable on AAPL; html ≤ 500 KB.

### 3.4 `trade_tape` — don't drop the prints that matter

**Evidence:** Bitstamp L3 tape clips y to q01–q99 of trade prices
([_data.py:602-607](../../ob_analytics/visualization/_data.py)) — the 02:52 spike's extreme
prints are **cut off mid-marker**: precisely the trades a tape exists to show. LOBSTER L3
tape saturates: tens of thousands of executions as area-sized blobs (area = rank-5 encoding;
`normalized_marker_areas` saturates everything ≥ q95 to one size).

**Fix:** never quantile-clip *trades* — pad to data extent within the FocusWindow time range.
Adopt the bundle's **signed lollipop** option (`1_trades_bubbles_vs_lollipops.png`): stem
length = size (rank-3), direction = aggressor side, on the mid/microprice line; keep maker-rest
spans as the L3 differentiator. For dense tapes (LOBSTER), aggregate to per-second VWAP
lollipops or thin markers below a density threshold. Also resolves the in-code `DEFERRED`
note ([_matplotlib.py:742-746](../../ob_analytics/visualization/_matplotlib.py)).
**Acceptance:** spike prints fully visible on Bitstamp; AAPL tape legible; cross-check
`images/cmp_3_trade_tape.png`.

### 3.5 `depth_heatmap` — dead `col_bias`, washed-out ramp

**Evidence:** both venues render almost uniformly dark-purple viridis: volumes are heavy-tailed
and the norm is linear; `col_bias` only switches linear↔log at ≤ 0
([_matplotlib.py:179-193](../../ob_analytics/visualization/_matplotlib.py)) — as a gamma it is
**dead** (any positive value = identical linear plot). The R original used it as a palette
bias. Gallery also never passes `trades` to the heatmap
([gallery.py:258-271](../../ob_analytics/visualization/gallery.py)) so trade markers never
appear.

**Fix:** implement `col_bias` as `mcolors.PowerNorm(gamma=col_bias)` (or default to robust
log/PowerNorm with vmax at q99); wire `trades=result.trades` in the gallery spec; same
treatment in plotly (`Scattergl` marker color array → binned `go.Heatmap` per §2.3).
**Acceptance:** near-touch structure visible at both venues (compare the README hero image
`assets/ob-analytics-price-levels.png`, which the R package produced with biased palette).

### 3.6 `hidden_executions` — ghost blobs

**Evidence:** LOBSTER render: enormous near-white circles swamping the price line. Marker
areas are raw `volume * 2.0` (`volume_marker_areas`,
[_data.py:1110-1113](../../ob_analytics/visualization/_data.py)) — share volumes of 100–1000 ⇒
200–2000 pt² markers; `Reds` cmap with `vmin=0` makes typical volumes near-white.
**Fix:** `normalized_marker_areas`; fixed hue + alpha (size already encodes volume — double
encoding with color is redundant); or color by aggressor side. **Acceptance:** price line
legible, hidden prints read as discrete events.

### 3.7 `volume_percentiles` — palette direction, side identity, legend

**Evidence:** wall of dark navy; the **near-touch** bands (most important) are palest while
far-depth dominates; bid and ask use the *same* ramp (only the y-sign distinguishes);
20-entry legend × 2 sides; `02 02:45` ticks.
**Fix (after §1.3):** invert ramp (darkest at touch, fading outward — importance ↦ salience);
two hue families (asks blue, bids a second hue, shared luminance ramp); collapse legend to
5–6 entries or a labeled colorbar per side; apply the shared time-axis helper.
**Acceptance:** touch band visually dominant; sides distinguishable in grayscale print.

### 3.8 `order_outcome.L3` — clip kills the most informative tail

**Evidence:** Bitstamp render spans only ≤ 0 bps. The symmetric q05–q95 clip
([_data.py:550-556](../../ob_analytics/visualization/_data.py)) removes the rare
touch-improving (> 0) placements — the most aggressive, most informative orders.
**Fix:** asymmetric clip (lo = q05, hi = max(q99, +few bps)), or symlog-x; keep zero-line
annotation. Add the bundle's **binned fate variant** (`images/L3_5_order_fate.png`): stacked
outcome counts by distance bin + common-baseline fill-rate dot plot — as the default L3 face
at high cardinality (the in-code `FUTURE(--density)` note), raw scatter as the low-N variant.
**Acceptance:** positive-bps points visible when present; fill-rate-vs-distance readable.

### 3.9 Smaller polish

- `order_activity.L2` event map: no legend at all (gray created/deleted circles vs colored
  direction dots are unexplained); add one. Seaborn `size=0.1`/`size=0.5` scalar misuse in
  [_matplotlib.py:326-359,370-394](../../ob_analytics/visualization/_matplotlib.py) — replace with
  explicit `s=`.
- `events_histogram`: `multiple="dodge"` comb effect — `step`/overlay reads better; consider
  the bundle's strip-plot variant for *sizes* (§4.3).
- plotly `Bar` traces ignore the computed `bar_width` (vpin) — verify bar widths at sub-second
  cadence.
- Dead statements: [_plotly.py:223](../../ob_analytics/visualization/_plotly.py) (`data["events"]`),
  [_plotly.py:665](../../ob_analytics/visualization/_plotly.py) (`np.zeros(...)`).

---

## WS-4 · New capabilities (the moat)

### 4.1 FIFO queue engine → the two deferred flagship faces

Single pass over normalized events (§1.1 prerequisite), per (side, price) maintaining an
ordered queue of (id, remaining):

- `created` → append (price-time priority); `deleted`/cancel → remove by id; fills consume
  per maker attribution (trades carry `maker_event_id`) or `fill` deltas.
- Emit change-rows: `ts, id, side, price, rank, queue_len, ahead_volume, age`.
- API: `queue_positions(events, trades, *, levels="touch" | "all", config) -> DataFrame`
  in a new `ob_analytics/queue.py`. Visible-only caveat documented (hidden orders absent).

Faces (both already named in the registry plan; bundle references in parentheses):
- **`queue_position.L3`** (`cmp_7`, `L3_2`): rank-vs-time trajectories at the touch, color by
  fate, inverted y (rank 1 = front at top), terminal markers. L2 counterpart = best-size step
  line (`liquidity_at_touch` reuse) ⇒ a comparable pair.
- **`liquidity_at_touch.L3`** (`cmp_1`, `L3_6`): queue composition strip — per time-step
  stacked per-order ticks colored by age. Render as `pcolormesh`/image (one tick per order per
  step is O(huge) as line segments — see §2.3 lesson).

**Acceptance:** on LOBSTER, total queue volume at touch == orderbook-file level-1 sizes at
every event index (visible book); trajectories monotone-nonincreasing in rank except
re-pricing; demo renders for both venues.

### 4.2 Spread ribbon + microprice (`price_view.L2`, bundle fig 4/01b)

`depth_summary` already has best bid/ask price+vol → ribbon = bid↔ask band, microprice
`(bid·ask_vol + ask·bid_vol)/(bid_vol+ask_vol)` line, mid faint. Upgrade or replace the bare
L2 `trade_tape` step line; lollipop trades (§3.4) overlay naturally.

### 4.3 Cheap wins

- **Trade-size strip plot** (bundle fig 5): log-x jittered dots; round-number gridlines.
- **Event rug** under `liquidity_at_touch.L2` (bundle fig 2): adds/cancels/fills ticks.
- **OFI horizon graph** (bundle fig 3): multi-horizon variant of the existing OFI panel (later).

### 4.4 `trade_impacts` panel

[analytics.py:118-173](../../ob_analytics/analytics.py) is currently orphaned (no callers, no
tests). Either expose as an analytics panel (sweep VWAP/price-range per taker) with a test, or
delete. Decide when building WS-4; don't leave it dangling.

---

## WS-5 · API / UX

### 5.1 Concept bindings + one-line plotting

The concept → (prepare fn, result fields, default kwargs) mapping exists only as literals
inside `build_gallery_model` ([gallery.py:239-441](../../ob_analytics/visualization/gallery.py)).
Extract a declarative `CONCEPT_BINDINGS` table; have both the gallery and a new
`plot_result(result, concept, level=None, backend=..., **overrides)` (and/or
`PipelineResult.plot(...)` convenience) consume it. This is the single biggest discoverability
fix: today a figure from a `result` requires knowing private prepare names and their argument
shapes.

### 5.2 Public prepare API + docs repair

Docs reference the private module 9× (`from ob_analytics.visualization import _data`).
Re-export prepares under `ob_analytics.visualization.prepare` (keep `_data` as the impl),
update quickstart (incl. the three broken `trade_tape` calls → `level=` or `plot_result`).

### 5.3 CLI symmetry

- `--format` choices hardcoded ([cli.py:236-240](../../ob_analytics/cli.py)) → `list_formats()`.
- `--trading-date` requirement keyed on the literal name "lobster"
  ([cli.py:56-62](../../ob_analytics/cli.py)) → ask the format (e.g. optional
  `required_context() -> list[str]` with a structural default).
- Add `ob-analytics formats` (mirror of `capture --list`).
- The 4× duplicated `--view` argument block → small helper.

### 5.4 Demos

`run_bitstamp_demo(roundtrip=True)` runs the full pipeline twice
([_demos.py:138-150](../../ob_analytics/_demos.py)) purely to log `match: True` — flip default
to `False`, expose `--roundtrip` on the CLI verb (halves demo latency until WS-2 lands; still
right afterwards).

---

## WS-6 · Ecosystem

1. **`DatabentoFormat` (DBN, MBO schema)** — highest-value format addition. Map
   `action ∈ {A,C,M,T,F}`/`side`/`order_id`/`ts_event` into the canonical events schema
   (post-§1.1 semantics: maintain remaining volume per order). Read `.dbn`/`.dbn.zst` via
   `databento-dbn`. Optional dep `[databento]`. LOBSTER stays the offline/academic path,
   Databento the modern equities/futures L3 path; both exercise the same L3 faces.
2. **cryptofeed capturer** — second `LiveCapturer` implementation (Coinbase L3) to prove the
   live registry generalizes (quickstart already sketches it conceptually).
3. **hftbacktest export writer** — registry writer emitting their normalized L3 input; makes
   ob-analytics the EDA front-end of an established backtester.
4. Arrow/polars ingestion at loader boundaries (zero-copy via pyarrow) — later, demand-driven.

---

## WS-7 · AI-bloat removal (no behavior change; one focused PR)

Delete / trim — all verified unused or narration-only:

| What | Where | Action |
|---|---|---|
| `reverse_matrix` | [_utils.py:81-105](../../ob_analytics/_utils.py) + its test | Delete (R-port leftover; zero callers) |
| `per_order` accepted-then-`del` params | [_data.py:390](../../ob_analytics/visualization/_data.py), [_data.py:702](../../ob_analytics/visualization/_data.py) | Remove param; `_paired` concepts pass distinct prepares, so the "signature symmetry" rationale is hollow |
| Duplicated palette constants ×2 backends | [_matplotlib.py:398-417](../../ob_analytics/visualization/_matplotlib.py), [_plotly.py:314-333](../../ob_analytics/visualization/_plotly.py) | Consolidate into `_palette.py` (also §3.0) |
| PR-narration comments ("the previous version…", "it whited out…", "replaces the hand-rolled…", EMPTY_TRADES naming essay) | [bitstamp.py:291-297](../../ob_analytics/bitstamp.py), [depth.py:36-49](../../ob_analytics/depth.py), [_matplotlib.py:432-436,486-488](../../ob_analytics/visualization/_matplotlib.py), [_utils.py:47-52](../../ob_analytics/_utils.py), [_registry.py:4-7](../../ob_analytics/_registry.py), [_data.py:86-94](../../ob_analytics/visualization/_data.py) | Rewrite to state the *current* contract only (keep genuinely load-bearing notes: bit-exactness contract in depth.py, import-order comments in pipeline.py/bitstamp.py bottom) |
| Dead `else: frac = 0.0` | [flow_toxicity.py:136-139](../../ob_analytics/flow_toxicity.py) | Delete |
| Dead statements | [_plotly.py:223](../../ob_analytics/visualization/_plotly.py), [_plotly.py:665](../../ob_analytics/visualization/_plotly.py) | Delete |
| `FUTURE(--focus/--color-by/--density)` comment blocks | _data.py, _matplotlib.py, _plotly.py, gallery.py | Replace with a pointer to this roadmap §3.0/§3.x once each lands |
| Stale worktree + branches | `.worktrees/l2-l3-visualization-taxonomy`, local+origin `feat/l2-l3-visualization-taxonomy` | `git worktree remove`, delete branches (merged in `8552ac8`) — ask user first |
| Empty dir | `bitstamp_output_live/` in repo root | Delete |
| `trade_impacts` | see §4.4 | Wire up or delete — not status quo |

Explicitly **not** bloat (leave alone): the Registry abstraction, protocols/structural typing,
`Format` descriptors, `RunContext`, the prepare/render split, `SupportsDiagnostics`,
`_demos.py` consolidation — all carry real weight.

---

## WS-8 · Validation, distribution & adoption (user-approved 2026-06-10)

### 8.1 Output validity & property-based testing

- **Hypothesis property tests throughout the logic** (user: "a great idea") — make them part
  of the Phase 2 gate, since they de-risk the §1.1 refactor. Generators: random valid event
  streams (created/changed/deleted with consistent remaining volumes; optional LOBSTER-style
  delta encoding). Invariants to assert:
  - book never crossed after `DepthMetricsEngine` eviction (`best_bid < best_ask` whenever
    both sides non-empty);
  - per-order remaining volume is non-negative and monotone non-increasing;
  - `price_level_volume` cumulative level volume ≥ 0 and returns to 0 when all orders at a
    price terminate;
  - `order_lifecycles()` outcomes partition all order ids; `filled_vol ≤ placed_vol`;
  - depth-engine bin sums equal a brute-force recomputation from active levels (oracle test —
    this is the one that protects WS-2.1's rewrite);
  - loader round-trips (`BitstampWriter` → `BitstampLoader`, `LobsterWriter` →
    `LobsterLoader`) preserve the events frame.
  Add `hypothesis` to the dev group; keep example counts CI-friendly (`max_examples` ~100,
  `deadline=None` for the engine oracle).
- **R-package parity: already done** — external validation against the original R obAnalytics
  was performed on earlier iterations of this port (user statement, 2026-06-10). Do **not**
  rebuild an R comparison harness; rely on the regression snapshots + property tests above.
- **Attribution accuracy + data-quality report** (kept from the review): measure taker
  attribution against LOBSTER's ground-truth trade direction (it's exact there); reclassify
  pre-window orders as an explicit `pre-existing` type instead of `unknown` (Bitstamp: 19,
  LOBSTER: 1,460 today); surface a per-run data-quality summary (crossed-book evictions,
  % unmatched trades, duplicate ids, receive-vs-exchange clock skew) — usable as
  `ob-analytics validate <data>` for the bring-your-own-data path.

### 8.2 Time semantics policy + latency analytics

Bitstamp events are tz-naive **UTC**; LOBSTER events are tz-naive **exchange-local (ET)**.
Each is fine alone; together they are silently incomparable. Decide and document one policy
(recommendation: keep storage tz-naive but record the zone in a result/loader attribute, and
have cross-format tooling refuse to mix zones silently). Separately, `timestamp` (receive)
vs `exchange_timestamp` is carried everywhere and analyzed nowhere — add a feed-latency
analytics panel (distribution + over-time) as a cheap, distinctive analytic; degrade
gracefully where the two are identical (LOBSTER sets them equal).

### 8.3 Distribution / publication readiness

- PyPI name `ob-analytics` is **free** (checked 2026-06-10). Before first publish:
- The bundled sample is **23 MB** (`_sample_data/orders.csv`) and ships in every wheel —
  exclude it from the wheel (`tool.hatch.build.targets.wheel` exclude) and fetch on demand
  (`sample_csv_path()` downloads to a cache dir on first use, e.g. via `pooch` or a plain
  urllib + checksum), or trim the bundled copy to a ~5-minute slice and host the full capture
  as a release asset. Keep `sample_csv_path()`'s API unchanged either way.
- Add `CITATION.cff` (academic audience; cite both this package and the original R
  obAnalytics lineage).
- Add a release workflow (tag → build → publish, trusted publishing) and a versioning policy;
  state the GPL-2+ inheritance from the R port explicitly in the README so industry adopters
  can make the call quickly.
- `py.typed` already ships — keep it.

### 8.4 Scale envelope (decide before WS-6 Databento work)

Everything is in-memory pandas; the comfortable envelope today is session-scale
(≤ ~5M events). A full NASDAQ day of MBO is 10–100M events and will not fit the current
design — which collides with the Databento plan in WS-6. Action: (a) write the honest
envelope statement into README/docs now; (b) when WS-6 starts, decide chunked loading
(per-time-slice pipeline runs + concatenated outputs) vs documented pre-slicing. Don't build
streaming infrastructure speculatively.

### 8.5 Tutorial notebooks

Zero `.ipynb` in the repo today, for a Jupyter-native audience. After WS-5's `plot_result()`
lands, add two narrated notebooks under `docs/notebooks/` (rendered into the docs site):
1. *Crypto session*: capture (or bundled sample) → pipeline → gallery → flow-toxicity panels.
2. *LOBSTER session*: load AAPL sample → microstructure metrics → L2/L3 comparison faces.
They double as living integration tests for the public API — run them in CI (`nbclient`,
sample data cached per §8.3).

### 8.6 CI & gallery polish (approved residue of review item 7)

- CI is ubuntu-only — add a macOS smoke job (one Python version, no coverage).
- Add a perf-regression guard after WS-2 lands (`--durations=15` in CI plus a hard wall-time
  assertion on the regression-snapshot module).
- `comparison` view silently **drops** non-comparable concepts (`depth_heatmap`,
  `order_outcome`, `liquidity_at_touch` simply vanish) — render them as single-column cards
  labeled "L2-only / L3-only" instead.
- Long captures never rotate `orders.csv` / `raw.jsonl` — add size/time-based rotation to
  `FileCaptureSink`.
- **Deferred by user (2026-06-10): seaborn removal.** Revisit only after the theme redesign
  has settled; do not bundle it into other dependency work.

---

## Sequencing

```
Phase 0  WS-7 bloat sweep + housekeeping + cheap suite wins      (zero-risk: drop --cov from
         addopts, shared sample fixture, tobytes fingerprints — §2.6)
Phase 1  WS-2.1 Tier-1 + WS-2.2 + WS-2.3 pipeline perf; WS-2.4/2.5 gallery perf
         (unblocks fast iteration on everything; demos drop ~2.5 min → ~25s)
Phase 2  WS-1.1 canonical semantics + lifecycle table   (prereq for all L3 work)
         WS-1.2 … 1.6 correctness batch
         WS-8.1 hypothesis property tests               (gate: they protect the refactor)
         WS-8.2 time-semantics policy (+ pre-existing order class from 8.1)
Phase 3  WS-3.0 primitives (CB-safe palettes, seaborn stays) → WS-3.1 … 3.9 face fixes
         (visual-verify each against bundle)
Phase 4  WS-5 API/UX + docs repair; WS-8.5 notebooks    (lands with public API; docs once)
         WS-8.6 CI breadth + comparison-view cards + perf guard
Phase 5  WS-4.1 queue engine + flagship faces; 4.2/4.3  (+ WS-2.1 Tier-2 if engine still hot)
         WS-8.2 latency analytics panel
Phase 6  WS-8.3 publish readiness (sample out of wheel, CITATION.cff, release workflow)
         WS-8.4 scale-envelope decision → WS-6 ecosystem (Databento first)
```

Gate each phase on: full suite green, ruff/ty clean, and (Phases 3+5) a fresh
`bitstamp-demo`/`lobster-demo --view both` render visually compared against the bundle images.

---

## Development workflow (user-approved 2026-06-10)

### Branching & PR granularity

- **One PR per roadmap item**, not per phase. Phases are dependency groups; items are the
  PR-sized units. Branch names follow the repo's existing convention, prefixed by the roadmap
  ref: `chore/ws7-bloat-sweep`, `perf/ws2.2-loader-lexsort`, `perf/ws2.1-depth-binning`,
  `fix/ws1.2-config-format-merge`, `feat/ws3.1-book-snapshot-ladder`, `feat/ws4.1-queue-engine`.
- Conventional commit messages as in history (`perf(depth): …`, `fix(visualization): …`).
  Merge-commit PRs (repo's existing style). Keep every PR independently revertible.
- **Exception — WS-1.1 lands as three serial PRs**, each green on its own:
  (1) schema contract + loader remaining-volume derivation, (2) `order_lifecycles()` table +
  consumer migration, (3) `order_book()` active-set fix. It is the riskiest change in the
  plan; never as one diff.

### Worktrees

- Use `.worktrees/<branch>` (existing convention), `uv sync --group dev --extra interactive`
  per worktree. **Remove the worktree and delete the branch as part of merging its PR** — the
  stale taxonomy worktree was the failure mode.
- Parallelism follows the phase graph: **Phases 0–2 run serial** (one active worktree;
  parallel sessions there just conflict in `depth.py`/`bitstamp.py`). **Phase 3 fans out**
  only after the WS-3.0 primitives PR lands (it touches the shared files); then face fixes
  parallelize cleanly, one worktree each, conflicts limited to the registration tables.
  Phase 4+ items mix freely.
- This roadmap should be **committed in the Phase 0 PR** so it exists inside every worktree;
  until then, agents in worktrees must read it from the main checkout's absolute path
  (`/home/mcz/Documents/GitHub/ob-analytics/docs/plans/…`). Keep it updated as phases land
  (tick items, correct drifted line numbers) in the same PR that implements them.

### PR contents — every PR description carries three things

1. **Roadmap reference** ("Roadmap WS-2.1 Tier 1") and a one-line scope statement.
2. **Gate evidence**: pytest / ruff / ty output (and demo render for visual work).
3. **Claim type**, which sets the evidence bar:
   - *Behavior-preserving* (Phases 0–1): states "no output change"; the regression-snapshot
     tests are the enforcement. This is why perf lands before semantics.
   - *Output-changing* (Phases 2–3): snapshot updates go in a **separate, clearly-labeled
     commit** with before/after rationale — never mixed into the refactor commit.
   - *Visual* (Phases 3, 5): before/after PNGs from a fresh demo render in the PR body, next
     to the matching bundle reference image. No rendered evidence ⇒ not reviewable.
   - *Perf*: measured before/after timings against the WS-2 baselines.
- For the two highest-risk PRs — WS-1.1 (semantics) and WS-2.1 (engine rewrite) — run
  `/code-review ultra <PR#>` before merge.

### Commit/push policy

Commits, pushes, and PR creation are **user-triggered, always** (standing rule). Agents do
the work in the worktree, run the gates, and stop at "ready to commit" — unless the user
pre-authorizes per-branch at session start (e.g. "you may commit to `perf/ws2.1-*` this
session"). Approval for one branch/session never carries over to the next.

### Cadence expectation

Phase 0 + each Phase-1 item land as quick small PRs; Phase 2 is ~5 serial PRs; Phase 3 fans
out in parallel after the primitives PR. After Phases 0–1 land, CI per PR drops from ~10 min
to ~2 min, which is what makes the small-PR cadence cheap.
