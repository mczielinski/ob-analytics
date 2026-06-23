#!/usr/bin/env python3
"""Regenerate the README hero image (the price-levels depth heatmap).

Renders the bundled Bitstamp sample's busiest window as the depth heatmap shown
in README.md. The depth_heatmap face already provides the gray background, white
midprice line, and white y-gridlines; this recipe adds the hero-specific bits:

  * a focused time window (the busiest N minutes) + price band around the action,
  * a biased viridis palette (``col_bias`` < 1) so resting liquidity walls glow,
  * regular HH:MM time ticks with the date moved into the axis label.

Run via:

    uv run python scripts/make_hero.py
    uv run python scripts/make_hero.py --output assets/ob-analytics-price-levels.png
    uv run python scripts/make_hero.py --window-min 10 --col-bias 0.45
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import pandas as pd

from ob_analytics import Pipeline, sample_csv_path
from ob_analytics.visualization.gallery import plot_result

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "ob-analytics-price-levels.png"


def main() -> None:
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Regenerate the README hero image")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--window-min",
        type=int,
        default=10,
        help="length of the busiest window to focus on, in minutes",
    )
    parser.add_argument(
        "--col-bias",
        type=float,
        default=0.45,
        help="viridis palette bias; < 1 brightens the resting-liquidity levels",
    )
    args = parser.parse_args()

    result = Pipeline().run(sample_csv_path())
    tr = result.trades

    # Focus on the busiest window (the price spike) and the price band there.
    counts = (
        tr.set_index("timestamp").sort_index().assign(n=1)["n"].resample("1min").sum()
    )
    peak = counts.rolling(args.window_min).sum().idxmax()
    start = peak - pd.Timedelta(minutes=args.window_min - 1)
    end = peak + pd.Timedelta(minutes=1)
    win = tr[(tr["timestamp"] >= start) & (tr["timestamp"] <= end)]
    lo, hi = float(win["price"].min()), float(win["price"].max())
    pad = (hi - lo) * 0.12

    fig = plot_result(
        result,
        "depth_heatmap",
        backend="matplotlib",
        col_bias=args.col_bias,
        start_time=start,
        end_time=end,
        price_from=lo - pad,
        price_to=hi + pad,
    )

    # The gray background, white midprice, and white y-grid are defaults of the
    # depth_heatmap face now. The hero adds only a clearer time axis: regular
    # HH:MM ticks, unrotated, with the date in the label.
    ax = next(
        (a for a in fig.axes if a.get_title() == "Price Levels Over Time"), fig.axes[0]
    )
    step = max(1, round(args.window_min / 5))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", labelrotation=0)
    ax.set_xlabel(f"Time ({tr['timestamp'].min():%Y-%m-%d} UTC)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(
        f"wrote {args.output}  ({start:%H:%M}..{end:%H:%M}, col_bias={args.col_bias})"
    )


if __name__ == "__main__":
    main()
