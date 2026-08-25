"""Tests for the L2 (price-level) depth-native ingestion path.

Covers issue #98: price-level feeds (no order IDs) flow to a valid depth
frame, depth metrics / spread run, trade signs are classified, the per-order
(L3-only) stages no-op with a clear reason, and every validator passes — plus
the graceful degradation of data-quality and the gallery on an L2 result.
"""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from ob_analytics import (
    DepthCsvFormat,
    Level,
    Pipeline,
    data_quality_summary,
    list_formats,
)
from ob_analytics.datasets import toy_l2_depth, toy_l2_trades
from ob_analytics.depth import depth_metrics, get_spread
from ob_analytics.depth_l2 import DepthCsvWriter, L2DepthLoader, L2TradeReader
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import DepthSource, FeedType
from ob_analytics.schemas import (
    validate_depth_df,
    validate_events_df,
    validate_trades_df,
)

_BASE_MS = 1_700_000_000_000  # arbitrary epoch-ms anchor


def _write_l2_dir(
    directory,
    depth_rows: list[tuple],
    trades_rows: list[dict] | None = None,
) -> None:
    """Write a ``depth.csv`` (and optional ``trades.csv``) into *directory*.

    *depth_rows* are ``(epoch_ms, side, price, volume)`` tuples.
    """
    pd.DataFrame(depth_rows, columns=["timestamp", "side", "price", "volume"]).to_csv(
        directory / "depth.csv", index=False
    )
    if trades_rows is not None:
        pd.DataFrame(trades_rows).to_csv(directory / "trades.csv", index=False)


@pytest.fixture
def toy_l2_dir(tmp_path):
    """The bundled toy L2 fixture written to the canonical CSV schema."""
    DepthCsvWriter().write(
        {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, tmp_path
    )
    return tmp_path


@pytest.fixture
def toy_l2_result(toy_l2_dir):
    """A price-level ``PipelineResult`` from the toy fixture."""
    return Pipeline.from_format("depth_csv").run(toy_l2_dir)


# ---------------------------------------------------------------------------
# Format descriptor + registration
# ---------------------------------------------------------------------------


class TestDepthCsvFormat:
    def test_declares_l2_resolution(self):
        fmt = DepthCsvFormat()
        assert fmt.resolution is Level.L2
        assert fmt.name == "depth_csv"
        # A price-level feed is the venue's own aggregated (matched) view.
        assert fmt.feed_type is FeedType.MATCHED_BOOK

    def test_registered(self):
        assert "depth_csv" in list_formats()

    def test_self_describing_no_required_context(self):
        assert DepthCsvFormat().required_context() == []

    def test_loader_is_a_depth_source(self):
        from ob_analytics import PipelineConfig
        from ob_analytics.protocols import RunContext

        loader = DepthCsvFormat().create_loader(PipelineConfig(), RunContext())
        assert isinstance(loader, DepthSource)


# ---------------------------------------------------------------------------
# The L2 loader / schema
# ---------------------------------------------------------------------------


class TestL2DepthLoader:
    def test_reads_canonical_depth_frame(self, tmp_path):
        _write_l2_dir(
            tmp_path,
            [
                (_BASE_MS, "bid", 99.0, 2.0),
                (_BASE_MS, "ask", 101.0, 3.0),
                (_BASE_MS + 1000, "bid", 100.0, 1.0),
            ],
        )
        depth = L2DepthLoader().load(tmp_path)
        validate_depth_df(depth)
        assert list(depth["direction"].cat.categories) == ["bid", "ask"]
        assert len(depth) == 3
        # volumes are the absolute level sizes, not signed deltas
        assert depth["volume"].tolist() == [2.0, 3.0, 1.0]

    def test_accepts_flexible_column_spellings(self, tmp_path):
        # `direction` instead of `side`, `size` instead of `volume`, `time`
        pd.DataFrame(
            {
                "time": [_BASE_MS, _BASE_MS],
                "direction": ["buy", "sell"],  # buy->bid, sell->ask
                "price": [99.0, 101.0],
                "size": [2.0, 3.0],
            }
        ).to_csv(tmp_path / "depth.csv", index=False)
        depth = L2DepthLoader().load(tmp_path)
        assert depth["direction"].tolist() == ["bid", "ask"]

    def test_zero_volume_is_kept_as_removal(self, tmp_path):
        _write_l2_dir(
            tmp_path,
            [(_BASE_MS, "bid", 99.0, 2.0), (_BASE_MS + 1, "bid", 99.0, 0.0)],
        )
        depth = L2DepthLoader().load(tmp_path)
        assert depth["volume"].tolist() == [2.0, 0.0]

    def test_negative_volume_dropped(self, tmp_path):
        _write_l2_dir(
            tmp_path,
            [(_BASE_MS, "bid", 99.0, 2.0), (_BASE_MS + 1, "bid", 98.0, -1.0)],
        )
        depth = L2DepthLoader().load(tmp_path)
        assert len(depth) == 1

    def test_unknown_side_raises(self, tmp_path):
        _write_l2_dir(tmp_path, [(_BASE_MS, "sideways", 99.0, 2.0)])
        with pytest.raises(ConfigError, match="unrecognised side"):
            L2DepthLoader().load(tmp_path)

    def test_missing_columns_raises(self, tmp_path):
        pd.DataFrame({"timestamp": [_BASE_MS], "price": [99.0]}).to_csv(
            tmp_path / "depth.csv", index=False
        )
        with pytest.raises(ConfigError, match="missing required columns"):
            L2DepthLoader().load(tmp_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            L2DepthLoader().load(tmp_path)


# ---------------------------------------------------------------------------
# The toy fixture
# ---------------------------------------------------------------------------


class TestToyL2Fixture:
    def test_depth_is_schema_valid(self):
        validate_depth_df(toy_l2_depth())

    def test_trades_is_schema_valid(self):
        validate_trades_df(toy_l2_trades())

    def test_documented_final_book(self):
        summary = depth_metrics(toy_l2_depth())
        last = summary.iloc[-1]
        assert last["best_bid_price"] == 99.0
        assert last["best_ask_price"] == 100.0

    def test_trades_have_no_order_identity(self):
        trades = toy_l2_trades()
        for col in ("maker", "taker", "maker_event_id", "taker_event_id"):
            assert trades[col].isna().all()

    def test_lee_ready_recovers_true_signs(self):
        """Nulling the direction and reclassifying round-trips the taker side."""
        from ob_analytics.trade_sign import classify_trade_sign

        trades = toy_l2_trades()
        summary = depth_metrics(toy_l2_depth())
        unlabelled = trades.assign(
            direction=pd.Categorical([None] * len(trades), categories=["buy", "sell"])
        )
        recovered = classify_trade_sign(unlabelled, "lee_ready", quotes=summary)
        assert list(recovered) == list(trades["direction"])


# ---------------------------------------------------------------------------
# The pipeline L2 path (acceptance criteria)
# ---------------------------------------------------------------------------


class TestL2Pipeline:
    def test_run_produces_l2_result(self, toy_l2_result):
        r = toy_l2_result
        assert r.resolution is Level.L2

    def test_depth_and_summary_valid(self, toy_l2_result):
        r = toy_l2_result
        validate_depth_df(r.depth)
        assert not r.depth.empty
        # depth_metrics ran → best bid/ask columns exist; get_spread works
        assert "best_bid_price" in r.depth_summary.columns
        spread = get_spread(r.depth_summary)
        assert not spread.empty

    def test_events_empty_but_schema_valid(self, toy_l2_result):
        """The L3-only stages no-op: events comes back empty yet valid."""
        r = toy_l2_result
        assert r.events.empty
        validate_events_df(r.events)
        # order classification / aggressiveness never ran, so their columns
        # carry no data.
        assert r.events["type"].isna().all()

    def test_trades_valid(self, toy_l2_result):
        r = toy_l2_result
        validate_trades_df(r.trades)
        assert len(r.trades) == len(toy_l2_trades())

    def test_unsigned_trades_are_classified(self, tmp_path):
        """A trades.csv with no side column → pipeline fills it (Lee–Ready)."""
        _write_l2_dir(
            tmp_path,
            [
                (_BASE_MS, "bid", 99.0, 5.0),
                (_BASE_MS, "ask", 101.0, 5.0),
                (_BASE_MS + 5000, "bid", 100.0, 5.0),  # mid 100.5
            ],
            trades_rows=[
                {"timestamp": _BASE_MS + 6000, "price": 101.0, "amount": 1.0},  # >mid
                {"timestamp": _BASE_MS + 7000, "price": 100.0, "amount": 1.0},  # <mid
            ],
        )
        r = Pipeline.from_format("depth_csv").run(tmp_path)
        assert list(r.trades["direction"]) == ["buy", "sell"]

    def test_native_side_is_respected(self, tmp_path):
        """A trades.csv that labels the taker side is not reclassified."""
        _write_l2_dir(
            tmp_path,
            [(_BASE_MS, "bid", 99.0, 5.0), (_BASE_MS, "ask", 101.0, 5.0)],
            trades_rows=[
                # price at the bid but explicitly labelled buy → kept as buy
                {
                    "timestamp": _BASE_MS + 1000,
                    "price": 99.0,
                    "amount": 1.0,
                    "side": "buy",
                }
            ],
        )
        r = Pipeline.from_format("depth_csv").run(tmp_path)
        assert list(r.trades["direction"]) == ["buy"]

    def test_depth_only_feed_runs_without_trades(self, tmp_path):
        """A missing trades.csv is fine — depth analytics still run."""
        _write_l2_dir(
            tmp_path,
            [(_BASE_MS, "bid", 99.0, 5.0), (_BASE_MS, "ask", 101.0, 5.0)],
        )
        r = Pipeline.from_format("depth_csv").run(tmp_path)
        assert r.resolution is Level.L2
        assert r.trades.empty
        validate_depth_df(r.depth)

    def test_l3_default_unchanged(self, tiny_bitstamp_orders_csv):
        """The default (Bitstamp/L3) path still reports L3 with populated events."""
        r = Pipeline().run(tiny_bitstamp_orders_csv)
        assert r.resolution is Level.L3
        assert not r.events.empty


# ---------------------------------------------------------------------------
# Round-trip through the writer
# ---------------------------------------------------------------------------


class TestDepthCsvWriter:
    def test_round_trip(self, tmp_path):
        depth_in = toy_l2_depth()
        DepthCsvWriter().write({"depth": depth_in}, tmp_path)
        assert (tmp_path / "depth.csv").exists()
        depth_out = L2DepthLoader().load(tmp_path)
        # Both the toy fixture and the loader output carry the canonical
        # tz-aware UTC nanosecond clock (issue #154), so the instants compare
        # directly.
        expected = depth_in.copy()
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True),
            depth_out.reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
        )

    def test_writes_companion_trades(self, tmp_path):
        DepthCsvWriter().write(
            {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, tmp_path
        )
        assert (tmp_path / "trades.csv").exists()
        trades = L2TradeReader().load(pd.DataFrame(), tmp_path)
        assert len(trades) == len(toy_l2_trades())


# ---------------------------------------------------------------------------
# Graceful degradation: data quality + gallery
# ---------------------------------------------------------------------------


class TestL2DataQuality:
    def test_summary_on_l2_result(self, toy_l2_result):
        r = toy_l2_result
        summary = data_quality_summary(
            r.events, r.trades, feed_type=FeedType.MATCHED_BOOK, depth=r.depth
        )
        # Per-order metrics degrade to zero; the report still renders.
        assert summary.n_events == 0
        assert summary.n_orders == 0
        assert summary.pre_existing_orders == 0
        assert summary.n_trades == len(r.trades)
        # A matched aggregated book is not crossed, and unlabelled attribution
        # is not counted as a failure.
        assert summary.crossed_pct == pytest.approx(0.0)
        assert summary.unmatched_trades_pct == pytest.approx(0.0)
        assert "matched_book" in summary.render()


class TestL2Gallery:
    def test_available_concepts_excludes_l3_only(self, toy_l2_result):
        from ob_analytics.visualization import available_concepts

        concepts = set(available_concepts(toy_l2_result))
        # depth/trades faces are present …
        assert "depth_heatmap" in concepts
        # … and every per-order (L3) face is skipped, not errored.
        for l3_only in (
            "queue_position",
            "order_outcome",
            "order_activity",
            "cancellations",
            "book_snapshot",
            "events_histogram",
        ):
            assert l3_only not in concepts

    def test_depth_heatmap_renders(self, toy_l2_result):
        import matplotlib.pyplot as plt

        from ob_analytics.visualization import plot_result

        fig = plot_result(toy_l2_result, "depth_heatmap", backend="matplotlib")
        plt.close(fig)

    def test_generate_gallery_builds(self, toy_l2_result, tmp_path):
        from ob_analytics.visualization.gallery import generate_gallery

        out = generate_gallery(
            toy_l2_result, tmp_path / "g", view="l2", backends=["matplotlib"]
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestL2CLI:
    def _write_cli_fixture(self, directory):
        DepthCsvWriter().write(
            {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, directory
        )

    def test_formats_lists_depth_csv(self, cli_runner):
        r = cli_runner("formats")
        assert r.returncode == 0
        assert "depth_csv" in r.stdout

    def test_process_writes_parquet(self, cli_runner, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        self._write_cli_fixture(src)
        out = tmp_path / "out"
        r = cli_runner(
            "process", str(src), "--format", "depth_csv", "--output", str(out)
        )
        assert r.returncode == 0, r.stderr
        assert (out / "depth.parquet").exists()
        assert (out / "depth_summary.parquet").exists()
        assert (out / "events.parquet").exists()  # empty but written

    def test_validate_runs(self, cli_runner, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        self._write_cli_fixture(src)
        r = cli_runner("validate", str(src), "--format", "depth_csv")
        assert r.returncode == 0, r.stderr
        assert "matched_book" in r.stdout


# ---------------------------------------------------------------------------
# The Level enum is one shared identity across ingestion + viz
# ---------------------------------------------------------------------------


class TestLevel:
    def test_shared_identity(self):
        from ob_analytics.protocols import Level as CoreLevel
        from ob_analytics.visualization import Level as VizLevel

        assert CoreLevel is VizLevel
        assert list(CoreLevel) == [CoreLevel.L2, CoreLevel.L3]

    def test_l3_formats_default_resolution(self):
        """Bitstamp / LOBSTER declare (or structurally default to) L3."""
        from ob_analytics import BitstampFormat, LobsterFormat

        assert getattr(BitstampFormat(), "resolution", Level.L3) is Level.L3
        assert getattr(LobsterFormat(), "resolution", Level.L3) is Level.L3
