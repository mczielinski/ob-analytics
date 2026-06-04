"""Tests for the ob-analytics exception hierarchy."""

import pickle

import pytest

from ob_analytics.exceptions import ConfigError, ObAnalyticsError


def test_config_error_inherits_from_base():
    """ConfigError is catchable via the base class."""
    e = ConfigError("test message")
    assert isinstance(e, ObAnalyticsError)
    assert isinstance(e, Exception)
    assert str(e) == "test message"


def test_hierarchy():
    assert issubclass(ObAnalyticsError, Exception)
    assert issubclass(ConfigError, ObAnalyticsError)


def test_base_is_catchable_alone():
    """``except ObAnalyticsError`` catches ConfigError and the base alike."""
    with pytest.raises(ObAnalyticsError):
        raise ConfigError("x")
    with pytest.raises(ObAnalyticsError):
        raise ObAnalyticsError("x")


def test_config_error_is_distinguishable():
    """A bare ObAnalyticsError is not a ConfigError."""
    with pytest.raises(ConfigError):
        raise ConfigError("x")
    e = ObAnalyticsError("x")
    assert not isinstance(e, ConfigError)


@pytest.mark.parametrize("cls", [ObAnalyticsError, ConfigError])
def test_exceptions_are_picklable(cls):
    """Required for multiprocessing / cross-thread propagation."""
    e = cls("hello")
    restored = pickle.loads(pickle.dumps(e))
    assert type(restored) is cls
    assert str(restored) == "hello"
