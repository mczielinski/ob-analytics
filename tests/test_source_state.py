"""A source keeps nothing from a run, and each run reads its own context.

The pipeline builds each run's loader and trade source from that run's config
and context, and a loader knows its own feed's price encoding.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ob_analytics.config import InstrumentGrid, PipelineConfig
from ob_analytics.lobster import LobsterLoader, LobsterSource, LobsterTradeReader
from ob_analytics.pipeline import Pipeline
from ob_analytics.protocols import RunContext, grid_of

#: The trading date of the ``lobster_day_dir`` fixture.
LOBSTER_DAY_DATE = "2024-01-02"

#: The day after it, for a second session.
_OTHER_DATE = "2024-01-03"


def _lobster_day(day: Path, directory: Path, date: str, *, shift: int = 0) -> Path:
    """Copy the LOBSTER *day* to *directory* under *date*, prices raised by *shift*.

    *shift* is in LOBSTER's raw units, ten-thousandths of a dollar.
    """
    directory.mkdir()
    for file in day.iterdir():
        text = file.read_text()
        for raw in ("1000000", "1010000"):
            text = text.replace(raw, str(int(raw) + shift))
        (directory / file.name.replace(LOBSTER_DAY_DATE, date)).write_text(text)
    return directory


def _first_best_bid(result) -> float:
    """The first best bid of a run, in dollars."""
    bids = result.depth_summary["best_bid_price"].dropna()
    return float(bids.iloc[0]) * result.config.tick_size


class TestASharedSourceKeepsRunsApart:
    def test_two_pipelines_on_one_source_each_read_their_own_depth(
        self, lobster_day_dir, tmp_path
    ):
        near_100 = lobster_day_dir
        near_150 = _lobster_day(
            lobster_day_dir, tmp_path / "b", _OTHER_DATE, shift=500_000
        )
        source = LobsterSource()
        first = Pipeline(source=source, ctx=RunContext(trading_date=LOBSTER_DAY_DATE))
        second = Pipeline(source=source, ctx=RunContext(trading_date=_OTHER_DATE))

        # The second pipeline runs first, so a loader kept on the source from
        # that run would hand its orderbook file to the first.
        later = second.run(near_150)
        earlier = first.run(near_100)

        assert _first_best_bid(earlier) == pytest.approx(100.0)
        assert _first_best_bid(later) == pytest.approx(150.0)

    def test_the_source_holds_no_loader(self):
        assert not hasattr(LobsterSource(), "_loader")


class TestRunContextReachesTheLoader:
    def test_the_lobster_how_to_example_runs(self, lobster_day_dir):
        ctx = RunContext(trading_date=LOBSTER_DAY_DATE)
        result = Pipeline(source=LobsterSource()).run(lobster_day_dir, ctx=ctx)

        assert not result.events.empty
        stamps = result.events["timestamp"].dt.tz_convert("America/New_York")
        assert (stamps.dt.date == pd.Timestamp(LOBSTER_DAY_DATE).date()).all()

    def test_the_run_context_overrides_the_pipelines(self, lobster_day_dir, tmp_path):
        folder = _lobster_day(lobster_day_dir, tmp_path / "day", _OTHER_DATE)
        pipeline = Pipeline(
            source=LobsterSource(), ctx=RunContext(trading_date=LOBSTER_DAY_DATE)
        )
        result = pipeline.run(
            folder, ctx=RunContext(trading_date=_OTHER_DATE, symbol="TEST")
        )

        stamps = result.events["timestamp"].dt.tz_convert("America/New_York")
        assert (stamps.dt.date == pd.Timestamp(_OTHER_DATE).date()).all()
        assert (result.events["symbol"] == "TEST").all()

    def test_the_pipelines_context_serves_a_run_without_one(self, lobster_day_dir):
        pipeline = Pipeline(
            source=LobsterSource(), ctx=RunContext(trading_date=LOBSTER_DAY_DATE)
        )
        result = pipeline.run(lobster_day_dir)
        assert "symbol" not in result.events.columns

    def test_a_windowed_run_takes_its_own_context(self, lobster_day_dir, tmp_path):
        from ob_analytics.data import load_data

        out = Pipeline(source=LobsterSource()).run_windows(
            lobster_day_dir,
            [pd.Timestamp(f"{LOBSTER_DAY_DATE} 09:30:02", tz="America/New_York")],
            tmp_path / "out",
            ctx=RunContext(trading_date=LOBSTER_DAY_DATE),
        )
        assert len(load_data(out)["events"]) == 5


class TestALoaderKnowsItsFormat:
    def test_a_lobster_loader_built_by_hand_reads_the_pipelines_prices(
        self, lobster_day_dir, lobster_day_pipeline
    ):
        by_hand = LobsterLoader(trading_date=LOBSTER_DAY_DATE).load(lobster_day_dir)
        run = lobster_day_pipeline.run(lobster_day_dir)

        assert by_hand["price"].iloc[0] == 10_000  # $100.00 on a cent tick
        by_id = run.events.set_index("event_id")
        for column in ("price", "volume"):
            assert by_hand.set_index("event_id")[column].equals(
                by_id.loc[by_hand["event_id"], column]
            )

    def test_a_lobster_loader_built_by_hand_is_on_whole_shares(self, lobster_day_dir):
        events = LobsterLoader(trading_date=LOBSTER_DAY_DATE).load(lobster_day_dir)
        assert events["volume"].iloc[0] == 100  # 100 shares are 100 lots

    def test_a_lobster_trade_reader_built_by_hand_reads_the_pipelines_trades(
        self, lobster_day_dir, lobster_day_pipeline
    ):
        events = LobsterLoader(trading_date=LOBSTER_DAY_DATE).load(lobster_day_dir)
        by_hand = LobsterTradeReader().load(events, lobster_day_dir)
        run = lobster_day_pipeline.run(lobster_day_dir)

        assert len(by_hand) == 2
        for column in ("price", "volume"):
            assert by_hand[column].tolist() == run.trades[column].tolist()

    def test_a_partial_config_keeps_the_lobster_grid(self, lobster_day_dir):
        # Only the fields a config sets replace the grid's, as in a run.
        config = PipelineConfig(depth_bps=50)
        events = LobsterLoader(config, trading_date=LOBSTER_DAY_DATE).load(
            lobster_day_dir
        )
        assert events["volume"].iloc[0] == 100


class TestTheGrid:
    def test_a_config_is_laid_over_the_grid(self):
        config = InstrumentGrid(tick_size=0.25, lot_size=1.0).config(
            PipelineConfig(tick_size=0.5, price_decimals=1)
        )
        assert (config.tick_size, config.price_decimals) == (0.5, 1)
        assert (config.lot_size, config.volume_decimals) == (1.0, 0)

    def test_a_source_without_a_grid_has_the_default_one(self):
        assert grid_of(object()) == InstrumentGrid()

    def test_a_grid_of_the_wrong_type_is_refused(self):
        class Source:
            grid = (0.25, 1.0)

        with pytest.raises(TypeError, match="InstrumentGrid"):
            grid_of(Source())


class TestDepthCsvSettings:
    def test_the_empty_base_settings_give_the_defaults(self):
        from ob_analytics.config import SourceSettings
        from ob_analytics.depth_l2 import DepthCsvSource, L2DepthLoader

        source = DepthCsvSource(settings=SourceSettings())
        loader = source.create_loader(PipelineConfig(), RunContext())
        assert isinstance(loader, L2DepthLoader)

    def test_another_sources_settings_are_refused(self):
        from ob_analytics.databento import DatabentoSettings
        from ob_analytics.depth_l2 import DepthCsvSource
        from ob_analytics.exceptions import ConfigError

        source = DepthCsvSource(settings=DatabentoSettings())
        with pytest.raises(ConfigError, match="DepthCsvSettings"):
            source.create_loader(PipelineConfig(), RunContext())


class TestAWriterKeepsTheRunsConfig:
    def test_a_partial_config_is_the_runs_own(self):
        # A writer's config describes the frames it is given, so a field the
        # config leaves out keeps the config's default, not LOBSTER's grid.
        from ob_analytics.lobster import LobsterWriter

        writer = LobsterWriter(
            PipelineConfig(tick_size=0.01), trading_date="2024-01-02"
        )
        assert writer._config.lot_size == PipelineConfig().lot_size
