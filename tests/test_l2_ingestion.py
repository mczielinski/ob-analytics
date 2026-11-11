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
    DepthCsvSource,
    Level,
    Pipeline,
    data_quality_summary,
    list_sources,
)
from ob_analytics.analytics import order_aggressiveness, set_order_types
from ob_analytics.bitstamp import BitstampWriter
from ob_analytics.book import order_book, order_lifecycles
from ob_analytics.capture_files import DEPTH_COLUMNS
from ob_analytics.config import PipelineConfig
from ob_analytics.databento import DatabentoWriter
from ob_analytics.datasets import toy_l2_depth, toy_l2_trades
from ob_analytics.depth import depth_metrics, get_spread, price_level_volume
from ob_analytics.depth_csv import DepthCsvLoader, DepthCsvTradeReader, DepthCsvWriter
from ob_analytics.exceptions import ConfigError, UnknownNameError
from ob_analytics.hidden_liquidity import hidden_trades
from ob_analytics.interop import HftbacktestWriter, NautilusWriter
from ob_analytics.lobster import LobsterWriter
from ob_analytics.protocols import Clocks, DepthLoader, FeedType
from ob_analytics.queue import queue_age_grid, queue_positions
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
    return Pipeline.from_source("depth_csv").run(toy_l2_dir)


@pytest.fixture
def sparse_l2_result(tmp_path):
    """A short, sparse price-level capture: a snapshot, then 6 updates in 70 s.

    The gallery's summary faces drop the first minute of the depth summary,
    so they get only the rows at 68-70 s, while the zoom window from the
    depth clock ends at 52.5 s.
    """
    snapshot = [
        row
        for i in range(80)
        for row in (
            (_BASE_MS, "bid", 50.0 - i * 0.01, 10.0 + i),
            (_BASE_MS, "ask", 50.01 + i * 0.01, 10.0 + i),
        )
    ]
    updates = [
        (_BASE_MS + 10_000, "bid", 50.0, 25.0),
        (_BASE_MS + 25_000, "ask", 50.01, 5.0),
        (_BASE_MS + 40_000, "bid", 49.99, 0.0),
        (_BASE_MS + 68_000, "ask", 50.02, 30.0),
        (_BASE_MS + 69_000, "bid", 50.0, 12.0),
        (_BASE_MS + 70_000, "ask", 50.01, 8.0),
    ]
    _write_l2_dir(tmp_path, snapshot + updates)
    return Pipeline.from_source("depth_csv").run(tmp_path)


# ---------------------------------------------------------------------------
# Source descriptor + registration
# ---------------------------------------------------------------------------


class TestDepthCsvSource:
    def test_declares_l2_level(self):
        source = DepthCsvSource()
        assert source.level is Level.L2
        assert source.name == "depth_csv"
        assert source.feed_type is FeedType.PRICE_LEVELS

    def test_registered(self):
        assert "depth_csv" in list_sources()

    def test_self_describing_no_required_context(self):
        from ob_analytics.protocols import required_context_of

        assert required_context_of(DepthCsvSource()) == []

    def test_loader_is_a_depth_source(self):
        from ob_analytics import PipelineConfig
        from ob_analytics.protocols import RunContext

        loader = DepthCsvSource().create_loader(PipelineConfig(), RunContext())
        assert isinstance(loader, DepthLoader)


# ---------------------------------------------------------------------------
# The L2 loader / schema
# ---------------------------------------------------------------------------


class TestDepthCsvLoader:
    def test_reads_canonical_depth_frame(self, tmp_path):
        _write_l2_dir(
            tmp_path,
            [
                (_BASE_MS, "bid", 99.0, 2.0),
                (_BASE_MS, "ask", 101.0, 3.0),
                (_BASE_MS + 1000, "bid", 100.0, 1.0),
            ],
        )
        depth = DepthCsvLoader().load(tmp_path)
        validate_depth_df(depth)
        assert list(depth["direction"].cat.categories) == ["bid", "ask"]
        assert len(depth) == 3
        # volumes are the absolute level sizes, not signed deltas, and are
        # stored as integer lots on the default 1e-8 grid (issue #226)
        assert depth["volume"].tolist() == [200_000_000, 300_000_000, 100_000_000]

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
        depth = DepthCsvLoader().load(tmp_path)
        assert depth["direction"].tolist() == ["bid", "ask"]

    def test_zero_volume_is_kept_as_removal(self, tmp_path):
        _write_l2_dir(
            tmp_path,
            [(_BASE_MS, "bid", 99.0, 2.0), (_BASE_MS + 1, "bid", 99.0, 0.0)],
        )
        depth = DepthCsvLoader().load(tmp_path)
        assert depth["volume"].tolist() == [200_000_000, 0]

    def test_negative_volume_dropped(self, tmp_path):
        _write_l2_dir(
            tmp_path,
            [(_BASE_MS, "bid", 99.0, 2.0), (_BASE_MS + 1, "bid", 98.0, -1.0)],
        )
        depth = DepthCsvLoader().load(tmp_path)
        assert len(depth) == 1

    def test_unknown_side_raises(self, tmp_path):
        _write_l2_dir(tmp_path, [(_BASE_MS, "sideways", 99.0, 2.0)])
        with pytest.raises(ConfigError, match="unrecognised side"):
            DepthCsvLoader().load(tmp_path)

    def test_missing_columns_raises(self, tmp_path):
        pd.DataFrame({"timestamp": [_BASE_MS], "price": [99.0]}).to_csv(
            tmp_path / "depth.csv", index=False
        )
        with pytest.raises(ConfigError, match="missing required columns"):
            DepthCsvLoader().load(tmp_path)

    def test_off_grid_price_raises(self, tmp_path):
        # 0.036 is not a whole number of the default 0.01 ticks; rounding it
        # to 0.04 would move the level without a word.
        _write_l2_dir(tmp_path, [(_BASE_MS, "bid", 0.036, 1.0)])
        with pytest.raises(ConfigError, match="not whole multiples of tick_size"):
            DepthCsvLoader().load(tmp_path)

    def test_finer_tick_keeps_the_price(self, tmp_path):
        from ob_analytics.config import PipelineConfig

        _write_l2_dir(tmp_path, [(_BASE_MS, "bid", 0.036, 1.0)])
        depth = DepthCsvLoader(PipelineConfig(tick_size=0.001)).load(tmp_path)
        assert depth["price"].tolist() == [36]

    def test_off_grid_trade_price_raises(self, tmp_path):
        _write_l2_dir(
            tmp_path,
            [(_BASE_MS, "bid", 0.03, 1.0)],
            [{"timestamp": _BASE_MS, "price": 0.037, "volume": 1.0}],
        )
        with pytest.raises(ConfigError, match="DepthCsvTradeReader"):
            DepthCsvTradeReader().load(pd.DataFrame(), tmp_path)

    @pytest.mark.parametrize("column", ["price", "volume"])
    @pytest.mark.parametrize("bad", [None, "abc"])
    def test_a_depth_value_that_is_not_a_number_raises(self, tmp_path, column, bad):
        # #397: an empty cell is read as NaN, a word as text.
        rows = [(_BASE_MS, "bid", 99.0, 2.0), (_BASE_MS + 1, "bid", 98.0, 1.0)]
        frame = pd.DataFrame(rows, columns=["timestamp", "side", "price", "volume"])
        frame[column] = frame[column].astype(object)
        frame.loc[1, column] = bad
        frame.to_csv(tmp_path / "depth.csv", index=False)
        with pytest.raises(
            ConfigError,
            match=r"DepthCsvLoader\.load\(.*depth\.csv\): 1 of 2 row.*row 2",
        ):
            DepthCsvLoader().load(tmp_path)

    @pytest.mark.parametrize("column", ["price", "volume"])
    def test_a_trade_value_that_is_not_a_number_raises(self, tmp_path, column):
        trade = {"timestamp": _BASE_MS, "price": 0.03, "volume": 1.0, column: ""}
        _write_l2_dir(tmp_path, [(_BASE_MS, "bid", 0.03, 1.0)], [trade])
        with pytest.raises(
            ConfigError, match=r"DepthCsvTradeReader\.load\(.*trades\.csv\): 1 of 1 row"
        ):
            DepthCsvTradeReader().load(pd.DataFrame(), tmp_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DepthCsvLoader().load(tmp_path)


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
        assert r.level is Level.L2

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
        r = Pipeline.from_source("depth_csv").run(tmp_path)
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
        r = Pipeline.from_source("depth_csv").run(tmp_path)
        assert list(r.trades["direction"]) == ["buy"]

    def test_depth_only_feed_runs_without_trades(self, tmp_path):
        """A missing trades.csv is fine — depth analytics still run."""
        _write_l2_dir(
            tmp_path,
            [(_BASE_MS, "bid", 99.0, 5.0), (_BASE_MS, "ask", 101.0, 5.0)],
        )
        r = Pipeline.from_source("depth_csv").run(tmp_path)
        assert r.level is Level.L2
        assert r.trades.empty
        validate_depth_df(r.depth)

    def test_l3_default_unchanged(self, tiny_bitstamp_orders_csv):
        """The default (Bitstamp/L3) path still reports L3 with populated events."""
        r = Pipeline().run(tiny_bitstamp_orders_csv)
        assert r.level is Level.L3
        assert not r.events.empty


# ---------------------------------------------------------------------------
# Round-trip through the writer
# ---------------------------------------------------------------------------


class TestDepthCsvWriter:
    def test_round_trip(self, tmp_path):
        depth_in = toy_l2_depth()
        DepthCsvWriter().write({"depth": depth_in}, tmp_path)
        assert (tmp_path / "depth.csv").exists()
        depth_out = DepthCsvLoader().load(tmp_path)
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

    def test_keeps_the_capture_columns(self, tmp_path):
        # A capture's depth.csv: the venue sequence, and which part of the
        # capture wrote each row.
        pd.DataFrame(
            {
                "timestamp": [_BASE_MS, _BASE_MS + 1_000],
                "exchange_timestamp": [_BASE_MS, _BASE_MS + 990],
                "side": ["bid", "ask"],
                "price": [100.0, 101.0],
                "volume": [1.0, 1.0],
                "sequence": [None, 12],
                "origin": ["snapshot", "stream"],
            }
        ).to_csv(tmp_path / "depth.csv", index=False)
        loader = DepthCsvLoader(PipelineConfig(track_sequence=True))
        depth_in = loader.load(tmp_path)
        out = tmp_path / "out"
        DepthCsvWriter().write({"depth": depth_in}, out)

        assert pd.read_csv(out / "depth.csv").columns.tolist() == list(DEPTH_COLUMNS)
        depth_out = loader.load(out)
        for column in ("sequence", "origin"):
            pd.testing.assert_series_equal(depth_in[column], depth_out[column])

    def test_writes_companion_trades(self, tmp_path):
        DepthCsvWriter().write(
            {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, tmp_path
        )
        assert (tmp_path / "trades.csv").exists()
        trades = DepthCsvTradeReader().load(pd.DataFrame(), tmp_path)
        assert len(trades) == len(toy_l2_trades())


class TestVenueClock:
    """The L2 depth frame keeps the venue's time, so audit can check it (#310)."""

    @staticmethod
    def _write(directory, venue_ms: list[int | None]) -> None:
        """Two depth rows received 1 s apart, with the given venue times."""
        pd.DataFrame(
            {
                "timestamp": [_BASE_MS, _BASE_MS + 1_000],
                "exchange_timestamp": venue_ms,
                "side": ["bid", "ask"],
                "price": [100.0, 101.0],
                "volume": [1.0, 1.0],
            }
        ).to_csv(directory / "depth.csv", index=False)

    def test_the_loader_keeps_exchange_timestamp(self, tmp_path):
        self._write(tmp_path, [_BASE_MS - 5, _BASE_MS + 990])
        depth = DepthCsvLoader().load(tmp_path)
        validate_depth_df(depth)
        assert list(depth.columns[:2]) == ["timestamp", "exchange_timestamp"]
        assert str(depth["exchange_timestamp"].dtype) == "datetime64[ns, UTC]"
        assert depth["exchange_timestamp"].iloc[0] == pd.Timestamp(
            _BASE_MS - 5, unit="ms", tz="UTC"
        )

    def test_a_file_with_one_clock_has_no_exchange_timestamp(self, tmp_path):
        _write_l2_dir(tmp_path, [(_BASE_MS, "bid", 100.0, 1.0)])
        assert "exchange_timestamp" not in DepthCsvLoader().load(tmp_path).columns

    def test_the_writer_round_trips_it(self, tmp_path):
        self._write(tmp_path, [_BASE_MS - 5, _BASE_MS + 990])
        depth_in = DepthCsvLoader().load(tmp_path)
        out = tmp_path / "out"
        DepthCsvWriter().write({"depth": depth_in}, out)
        depth_out = DepthCsvLoader().load(out)
        pd.testing.assert_series_equal(
            depth_in["exchange_timestamp"], depth_out["exchange_timestamp"]
        )

    def test_audit_checks_the_clocks_on_an_l2_capture(self, tmp_path):
        # The second row's venue time is after the time it was received.
        self._write(tmp_path, [_BASE_MS - 5, _BASE_MS + 1_500])
        r = Pipeline.from_source("depth_csv").run(tmp_path)
        summary = data_quality_summary(
            r.events, r.trades, feed_type=FeedType.PRICE_LEVELS, depth=r.depth
        )
        assert summary.exchange_time_after_receive == 1
        assert "exchange_time_after_receive" in {c.name for c in summary.errors}

    def test_audit_says_why_a_file_with_one_clock_is_not_checked(self, tmp_path):
        _write_l2_dir(tmp_path, [(_BASE_MS, "bid", 100.0, 1.0)])
        r = Pipeline.from_source("depth_csv").run(tmp_path)
        summary = data_quality_summary(
            r.events, r.trades, feed_type=FeedType.PRICE_LEVELS, depth=r.depth
        )
        assert summary.clocks is Clocks.RECEIVE_ONLY
        assert "clock order           : not checked" in summary.render()


# ---------------------------------------------------------------------------
# Graceful degradation: data quality + gallery
# ---------------------------------------------------------------------------


class TestL2DataQuality:
    def test_summary_on_l2_result(self, toy_l2_result):
        r = toy_l2_result
        summary = data_quality_summary(
            r.events, r.trades, feed_type=FeedType.PRICE_LEVELS, depth=r.depth
        )
        # Per-order metrics degrade to zero; the report still renders.
        assert summary.n_events == 0
        assert summary.n_orders == 0
        assert summary.pre_existing_orders == 0
        assert summary.n_trades == len(r.trades)
        # The toy price-level book is not crossed, and unlabelled attribution
        # is not counted as a failure.
        assert summary.crossed_pct == pytest.approx(0.0)
        assert summary.unmatched_trades_pct == pytest.approx(0.0)
        assert "price_levels" in summary.render()


class TestPerOrderFunctionsRefuseL2:
    """A function that needs per-order events says so on an L2 run, naming
    L3, instead of answering with an empty frame or writing an empty file."""

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda r: set_order_types(r.events, r.trades), id="types"),
            pytest.param(lambda r: order_lifecycles(r.events), id="lifecycles"),
            pytest.param(lambda r: order_book(r.events), id="order_book"),
            pytest.param(
                lambda r: order_aggressiveness(r.events, r.depth_summary),
                id="aggressiveness",
            ),
            pytest.param(lambda r: price_level_volume(r.events), id="depth"),
            pytest.param(lambda r: queue_positions(r.events), id="queue"),
            pytest.param(lambda r: queue_age_grid(r.events), id="queue_age"),
            pytest.param(
                lambda r: hidden_trades(r.events, r.trades, r.depth_summary),
                id="hidden_trades",
            ),
        ],
    )
    def test_an_analysis_names_l3(self, toy_l2_result, call):
        with pytest.raises(ConfigError, match=r"needs per-order \(L3\) events"):
            call(toy_l2_result)

    @pytest.mark.parametrize(
        "fmt", ["hftbacktest", "nautilus", "databento", "lobster", "bitstamp"]
    )
    def test_a_writer_refuses_instead_of_writing_an_empty_file(
        self, toy_l2_result, tmp_path, fmt
    ):
        config = toy_l2_result.config
        writer = {
            "hftbacktest": lambda: HftbacktestWriter(config),
            "nautilus": lambda: NautilusWriter(config),
            "databento": lambda: DatabentoWriter(config),
            "lobster": lambda: LobsterWriter(config, trading_date="2024-01-02"),
            "bitstamp": lambda: BitstampWriter(config),
        }[fmt]()
        dest = tmp_path / "out" / "feed"
        tables = {
            "events": toy_l2_result.events,
            "trades": toy_l2_result.trades,
            "depth": toy_l2_result.depth,
        }

        with pytest.raises(ConfigError, match=r"needs per-order \(L3\) events"):
            writer.write(tables, dest)
        assert not dest.exists()


#: The built-in metrics that read the trades.
_TRADE_METRICS = (
    "vpin",
    "kyle_lambda",
    "order_flow_imbalance",
    "ofi_horizon",
    "transaction_costs",
    "amihud",
    "roll_spread",
)


@pytest.fixture
def no_trade_result(tmp_path):
    """A price-level run whose trades file has no rows."""
    DepthCsvWriter().write(
        {"depth": toy_l2_depth(), "trades": toy_l2_trades().iloc[:0]}, tmp_path
    )
    return Pipeline.from_source("depth_csv").run(tmp_path)


class TestNoTrades:
    """A run with no trades offers no trade metric, and computing every metric
    that applies does not raise."""

    def test_available_concepts_lists_no_trade_metric(self, no_trade_result):
        from ob_analytics.visualization import available_concepts

        assert no_trade_result.trades.empty
        concepts = available_concepts(no_trade_result)
        assert not set(_TRADE_METRICS) & set(concepts)
        assert "book_signals" in concepts

    def test_result_metrics_does_not_raise(self, no_trade_result):
        computed = no_trade_result.metrics()
        assert not set(_TRADE_METRICS) & set(computed)
        assert "book_signals" in computed

    def test_plotting_a_trade_metric_says_why_not(self, no_trade_result):
        with pytest.raises(UnknownNameError, match="needs trades"):
            no_trade_result.plot("vpin")


class TestL2Gallery:
    def test_available_concepts_excludes_l3_only(self, toy_l2_result):
        from ob_analytics.visualization import available_concepts

        concepts = available_concepts(toy_l2_result)
        # depth/trades faces are present …
        assert "depth_heatmap" in concepts
        # … and every per-order (L3) face is skipped, not errored.  The book
        # snapshot and depth chart used to be skipped too, when both their
        # faces read the per-order book; their L2 faces now read the depth
        # table, so an L2 run has them (and only them).
        for l3_only in (
            "queue_position",
            "order_outcome",
            "order_activity",
            "cancellations",
            "events_histogram",
        ):
            assert l3_only not in concepts
        assert concepts["book_snapshot"] == ["L2"]
        assert concepts["depth_chart"] == ["L2"]

    @pytest.mark.parametrize("key", ["book_snapshot", "depth_chart"])
    def test_the_book_faces_draw_the_price_level_book(self, toy_l2_result, key):
        from ob_analytics.visualization.gallery import build_gallery_model

        model = build_gallery_model(toy_l2_result)
        spec = next(c for c in model.concepts if c.key == key).at(Level.L2)
        assert spec is not None
        data = spec.prepare(**spec.prep_kwargs)

        assert not data["bids"].empty and not data["asks"].empty
        assert data["bids"]["price"].max() < data["asks"]["price"].min()

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

    @pytest.mark.parametrize("concept", ["price_view", "volume_percentiles"])
    def test_summary_faces_render_on_sparse_capture(self, sparse_l2_result, concept):
        """A short, sparse capture still draws the depth-summary faces.

        Before the fix both faces got zero rows (the zoom window ended before
        their data started) and raised, and the gallery dropped them.
        """
        import matplotlib.pyplot as plt

        from ob_analytics.visualization import plot_result

        fig = plot_result(sparse_l2_result, concept, backend="matplotlib")
        plt.close(fig)

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
    @pytest.mark.parametrize("concept", ["price_view", "volume_percentiles"])
    def test_summary_faces_draw_no_data_on_an_empty_window(
        self, sparse_l2_result, concept, backend
    ):
        """A window the caller passes that holds no rows gives a "no data" figure.

        Before the fix, price_view raised a TypeError (matplotlib) and
        volume_percentiles raised a KeyError (both backends).
        """
        import matplotlib.pyplot as plt

        from ob_analytics.visualization import plot_result

        if backend == "plotly":
            pytest.importorskip("plotly")
        t0 = sparse_l2_result.depth["timestamp"].min()
        fig = plot_result(
            sparse_l2_result,
            concept,
            backend=backend,
            start_time=t0 - pd.Timedelta(hours=2),
            end_time=t0 - pd.Timedelta(hours=1),
        )
        if backend == "matplotlib":
            # The theme places titles on the left; read every slot.
            ax = fig.axes[0]
            title = " ".join(ax.get_title(loc=s) for s in ("left", "center", "right"))
            plt.close(fig)
        else:
            title = fig.layout.title.text
        assert "no data" in title


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestL2CLI:
    def _write_cli_fixture(self, directory):
        DepthCsvWriter().write(
            {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, directory
        )

    def test_sources_lists_depth_csv(self, cli_runner):
        r = cli_runner("sources")
        assert r.returncode == 0
        assert "depth_csv" in r.stdout

    def test_process_writes_parquet(self, cli_runner, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        self._write_cli_fixture(src)
        out = tmp_path / "out"
        r = cli_runner(
            "process", str(src), "--source", "depth_csv", "--output", str(out)
        )
        assert r.returncode == 0, r.stderr
        assert (out / "depth.parquet").exists()
        assert (out / "depth_summary.parquet").exists()
        assert (out / "events.parquet").exists()  # empty but written

    def test_process_uses_the_recorded_tick_size(self, cli_runner, tmp_path):
        import json

        from ob_analytics.data import load_data

        src = tmp_path / "src"
        src.mkdir()
        _write_l2_dir(
            src, [(_BASE_MS, "bid", 0.036, 1.0), (_BASE_MS, "ask", 0.039, 2.0)]
        )
        (src / "meta.json").write_text(json.dumps({"tick_size": 0.001}))
        out = tmp_path / "out"
        r = cli_runner(
            "process", str(src), "--source", "depth_csv", "--output", str(out)
        )
        assert r.returncode == 0, r.stderr
        depth = load_data(out)["depth"]
        assert depth.attrs["tick_size"] == 0.001
        assert sorted(depth["price"].tolist()) == [36, 39]

    def test_validate_runs(self, cli_runner, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        self._write_cli_fixture(src)
        r = cli_runner("validate", str(src), "--source", "depth_csv")
        assert r.returncode == 0, r.stderr
        assert "price_levels" in r.stdout


# ---------------------------------------------------------------------------
# The Level enum is one shared identity across ingestion + viz
# ---------------------------------------------------------------------------


class TestLevel:
    def test_shared_identity(self):
        from ob_analytics.protocols import Level as CoreLevel
        from ob_analytics.visualization import Level as VizLevel

        assert CoreLevel is VizLevel
        assert list(CoreLevel) == [CoreLevel.L2, CoreLevel.L3]

    def test_l3_sources_declare_l3_level(self):
        """Bitstamp / LOBSTER declare L3."""
        from ob_analytics import BitstampSource, LobsterSource

        assert BitstampSource().level is Level.L3
        assert LobsterSource().level is Level.L3
