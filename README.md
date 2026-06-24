# ob-analytics

[![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html)
[![CI](https://github.com/mczielinski/ob-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/mczielinski/ob-analytics/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mczielinski/ob-analytics/branch/main/graph/badge.svg)](https://codecov.io/gh/mczielinski/ob-analytics)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)

**Limit order book analytics and visualization for Python.**

Reconstruct trades from raw exchange events, classify order types, compute
depth metrics, and visualize market microstructure — from Bitstamp-style CSVs
or [LOBSTER](https://lobsterdata.com/) message and orderbook files.

> Ported from the R
> [obAnalytics](https://cran.r-project.org/package=obAnalytics) CRAN package.
> Now a standalone Python package with a pipeline API, pluggable formats,
> flow-toxicity metrics, and Matplotlib/Plotly backends.

<p align="center">
  <img src="./assets/ob-analytics-price-levels.png" alt="Price levels depth heatmap (Bitstamp sample)" width="700">
</p>

---

## Installation

The package is not yet published on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/mczielinski/ob-analytics.git
```

With [uv](https://github.com/astral-sh/uv):

```bash
uv add git+https://github.com/mczielinski/ob-analytics.git
```

From a local clone:

```bash
git clone https://github.com/mczielinski/ob-analytics.git
cd ob-analytics
pip install -e .
```

Optional extras (from a local clone):

```bash
pip install -e ".[interactive]"   # Plotly figures
pip install -e ".[live]"          # live exchange capture (websockets)
```

**Requires** Python 3.11+. Core dependencies: NumPy, pandas, matplotlib,
seaborn, pydantic, pyarrow, loguru.

---

## Quickstart

### Bitstamp (bundled sample)

A 30-minute Bitstamp BTC/USD live capture (~314k events, ~280 trades) is
bundled with the package as paired `orders.csv.gz` + `trades.csv`. Point the
pipeline at the orders path; the companion trades file is auto-resolved:

```python
from ob_analytics import Pipeline, sample_csv_path

result = Pipeline().run(sample_csv_path())

result.events         # enriched events with order types and aggressiveness
result.trades         # trades from the live feed, with maker/taker attribution
result.depth          # price-level volume time series
result.depth_summary  # best bid/ask, BPS bins, spread
```

### LOBSTER

`trading_date` lives on `RunContext`, not on `LobsterFormat`:

```python
from ob_analytics import Pipeline, LobsterFormat, RunContext

result = Pipeline(
    format=LobsterFormat(),
    ctx=RunContext(trading_date="2012-06-21"),
).run("/path/to/lobster_data")
```

### Flow-toxicity metrics

Flow-toxicity metrics are plain functions you call on `result.trades` after
the run — no pipeline configuration, no plugin registry:

```python
from ob_analytics import Pipeline, compute_vpin, compute_kyle_lambda, sample_csv_path

result = Pipeline().run(sample_csv_path())
vpin = compute_vpin(result.trades, bucket_volume=10.0)
kyle = compute_kyle_lambda(result.trades)
```

Add your own by writing a function over a trades DataFrame — see the
[Extending guide](docs/extending.md).

### Live capture (Bitstamp)

With the `[live]` extra installed, capture a live order book straight to
`orders.csv` / `trades.csv` that the pipeline understands:

```bash
ob-analytics capture bitstamp --pair btcusd --out ./capture --minutes 30
```

`ob-analytics capture --list` shows registered venues. Add new ones by
implementing the `LiveCapturer` protocol in `ob_analytics.live`.

See the [full quickstart](https://mczielinski.github.io/ob-analytics/quickstart/)
for step-by-step usage, configuration, custom loaders, flow-toxicity metrics,
CLI commands, and demo scripts.

---

## Development

The project uses [uv](https://github.com/astral-sh/uv) for dependency
management. A `uv.lock` lockfile is committed for reproducible installs.

```bash
git clone https://github.com/mczielinski/ob-analytics.git
cd ob-analytics
uv sync --group dev --extra interactive
```

Pre-commit hooks (Ruff lint/format, `ty` type check) are configured in
`.pre-commit-config.yaml`:

```bash
uv run pre-commit install
```

### Testing & CI

```bash
uv run pytest tests/ -v
uv run ruff check ob_analytics/ tests/
uv run ty check ob_analytics/
```

CI runs automatically on push/PR via GitHub Actions — lint, type check,
pytest on Python 3.11/3.12/3.13, and Codecov coverage upload.

See [ARCHITECTURE.md](ARCHITECTURE.md) (or the rendered
[Architecture page](https://mczielinski.github.io/ob-analytics/architecture/))
for design decisions, the module map, and class diagrams.
See [CHANGELOG.md](CHANGELOG.md) for the release history.

---

## Documentation

API reference generated from docstrings using
[Zensical](https://github.com/zensicalHQ/zensical) (installed as a dev
dependency):

```bash
uv run zensical serve      # local preview at http://localhost:8000
uv run zensical build      # static site in site/
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style,
and conventions for adding new formats.

## License

ob-analytics is licensed under the **GNU General Public License v2.0 or later
(GPL-2.0-or-later)** — see [LICENSE.md](LICENSE.md).

It is a Python port of the R
[obAnalytics](https://cran.r-project.org/package=obAnalytics) package
(© 2015–2016 Philip Stubbings), which is GPL-2.0-or-later; that copyleft is
inherited here.

## Citation

If you use ob-analytics in published work, please cite it — GitHub's **Cite this
repository** button reads [CITATION.cff](CITATION.cff), which also credits the
original R obAnalytics lineage.
