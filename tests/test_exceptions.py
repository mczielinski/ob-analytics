"""Tests for the ob-analytics exception hierarchy."""

import pickle

import pytest

from ob_analytics.exceptions import (
    ConfigurationError,
    InsufficientDataError,
    InvalidDataError,
    MatchingError,
    ObAnalyticsError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [InvalidDataError, MatchingError, InsufficientDataError, ConfigurationError],
)
def test_all_inherit_from_base(exc_cls):
    """Every package-specific exception is catchable via the base class."""
    e = exc_cls("test message")
    assert isinstance(e, ObAnalyticsError)
    assert isinstance(e, Exception)
    assert str(e) == "test message"


def test_base_is_catchable_alone():
    """``except ObAnalyticsError`` catches everything in the hierarchy."""
    for cls in (
        InvalidDataError,
        MatchingError,
        InsufficientDataError,
        ConfigurationError,
    ):
        with pytest.raises(ObAnalyticsError):
            raise cls("x")


def test_specific_exceptions_distinguishable():
    """Each subclass is *not* caught by its siblings."""
    with pytest.raises(InvalidDataError):
        raise InvalidDataError("x")
    with pytest.raises(MatchingError):
        raise MatchingError("x")

    # InvalidDataError is not a MatchingError
    e = InvalidDataError("x")
    assert not isinstance(e, MatchingError)


def test_exceptions_are_picklable():
    """Required for multiprocessing / cross-thread propagation."""
    for cls in (
        InvalidDataError,
        MatchingError,
        InsufficientDataError,
        ConfigurationError,
    ):
        e = cls("hello")
        restored = pickle.loads(pickle.dumps(e))
        assert type(restored) is cls
        assert str(restored) == "hello"
