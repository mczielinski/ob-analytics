"""Tests for ob_analytics.config and ob_analytics.exceptions."""

import pytest
from pydantic import ValidationError

from ob_analytics.config import InstrumentGrid, PipelineConfig
from ob_analytics.exceptions import ConfigError, ObAnalyticsError


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.price_decimals == 2
        assert cfg.volume_decimals == 8
        assert cfg.depth_bps == 25
        assert cfg.depth_bins == 20

    def test_price_multiplier(self):
        # price_multiplier is the integer inverse of tick_size (issue #155),
        # no longer tied to price_decimals.
        assert PipelineConfig(tick_size=0.01).price_multiplier == 100
        assert PipelineConfig(tick_size=1e-8).price_multiplier == 100_000_000
        assert PipelineConfig(tick_size=0.25).price_multiplier == 4

    def test_bps_labels(self):
        cfg = PipelineConfig(depth_bps=25, depth_bins=3)
        assert cfg.bps_labels == ["25bps", "50bps", "75bps"]

    def test_frozen(self):
        cfg = PipelineConfig()
        with pytest.raises(ValidationError):
            cfg.price_decimals = 5  # ty: ignore[invalid-assignment]

    def test_custom_values(self):
        cfg = PipelineConfig(
            price_decimals=4,
            depth_bps=50,
        )
        assert cfg.price_decimals == 4
        assert cfg.depth_bps == 50

    def test_validation_rejects_zero_depth_bins(self):
        with pytest.raises(ValidationError):
            PipelineConfig(depth_bins=0)

    def test_a_lot_finer_than_the_shown_decimals_is_refused(self):
        # Sizes would be shown rounded to 8 places, so a size of one lot of
        # 1e-9 would be shown as 0.
        with pytest.raises(ConfigError, match="volume_decimals=9"):
            PipelineConfig(lot_size=1e-9)

    @pytest.mark.parametrize(
        ("lot_size", "volume_decimals"), [(1e-9, 9), (0.25, 2), (1.0, 0), (1e-8, 8)]
    )
    def test_a_lot_the_shown_decimals_can_show_is_accepted(
        self, lot_size, volume_decimals
    ):
        config = PipelineConfig(lot_size=lot_size, volume_decimals=volume_decimals)
        assert config.lot_size == lot_size

    def test_a_step_set_over_a_grid_brings_its_own_decimals(self):
        # Whole shares show no decimals; a caller's finer lot must not keep
        # the grid's 0, which the config would refuse.
        grid = InstrumentGrid(tick_size=0.01, lot_size=1.0)
        config = grid.config(PipelineConfig(lot_size=0.01))
        assert (config.lot_size, config.volume_decimals) == (0.01, 2)
        config = grid.config(PipelineConfig(lot_size=0.01, volume_decimals=4))
        assert config.volume_decimals == 4

    @pytest.mark.parametrize("name", ["price_divisor", "timestamp_unit"])
    def test_a_feed_encoding_field_is_refused(self, name):
        # Each loader reads its own format now; a setting the config no longer
        # has is refused rather than dropped.
        with pytest.raises(ValidationError, match=name):
            PipelineConfig(**{name: 1})


class TestExceptionHierarchy:
    def test_config_error_inherits_from_base(self):
        assert issubclass(ConfigError, ObAnalyticsError)

    def test_base_inherits_from_exception(self):
        assert issubclass(ObAnalyticsError, Exception)

    def test_catchable_by_base(self):
        with pytest.raises(ObAnalyticsError):
            raise ConfigError("test")

    def test_message_preserved(self):
        err = ObAnalyticsError("something went wrong")
        assert str(err) == "something went wrong"
