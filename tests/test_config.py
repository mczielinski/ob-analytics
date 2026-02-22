"""Tests for ob_analytics.config and ob_analytics.exceptions."""


import pytest
from pydantic import ValidationError

from ob_analytics.config import PipelineConfig
from ob_analytics.exceptions import (
    ConfigurationError,
    InsufficientDataError,
    InvalidDataError,
    MatchingError,
    ObAnalyticsError,
)


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.price_decimals == 2
        assert cfg.volume_decimals == 8
        assert cfg.match_cutoff_ms == 5000
        assert cfg.price_jump_threshold == 10.0
        assert cfg.depth_bps == 25
        assert cfg.depth_bins == 20
        assert cfg.zombie_offset_seconds == 60

    def test_price_multiplier(self):
        assert PipelineConfig(price_decimals=2).price_multiplier == 100
        assert PipelineConfig(price_decimals=8).price_multiplier == 100_000_000

    def test_bps_labels(self):
        cfg = PipelineConfig(depth_bps=25, depth_bins=3)
        assert cfg.bps_labels == ["25bps", "50bps", "75bps"]

    def test_frozen(self):
        cfg = PipelineConfig()
        with pytest.raises(ValidationError):
            cfg.price_decimals = 5

    def test_custom_values(self):
        cfg = PipelineConfig(
            price_decimals=4,
            match_cutoff_ms=100,
            price_jump_threshold=5.0,
        )
        assert cfg.price_decimals == 4
        assert cfg.match_cutoff_ms == 100
        assert cfg.price_jump_threshold == 5.0

    def test_validation_rejects_negative_match_cutoff(self):
        with pytest.raises(ValidationError):
            PipelineConfig(match_cutoff_ms=-1)

    def test_validation_rejects_zero_depth_bins(self):
        with pytest.raises(ValidationError):
            PipelineConfig(depth_bins=0)

    def test_timestamp_unit_default_ms(self):
        assert PipelineConfig().timestamp_unit == "ms"

    def test_timestamp_unit_accepts_valid(self):
        assert PipelineConfig(timestamp_unit="us").timestamp_unit == "us"
        assert PipelineConfig(timestamp_unit="ns").timestamp_unit == "ns"

    def test_timestamp_unit_rejects_invalid(self):
        with pytest.raises(ValidationError):
            PipelineConfig(timestamp_unit="s")


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        assert issubclass(InvalidDataError, ObAnalyticsError)
        assert issubclass(MatchingError, ObAnalyticsError)
        assert issubclass(InsufficientDataError, ObAnalyticsError)
        assert issubclass(ConfigurationError, ObAnalyticsError)

    def test_base_inherits_from_exception(self):
        assert issubclass(ObAnalyticsError, Exception)

    def test_catchable_by_base(self):
        with pytest.raises(ObAnalyticsError):
            raise InvalidDataError("test")

    def test_message_preserved(self):
        err = MatchingError("something went wrong")
        assert str(err) == "something went wrong"
