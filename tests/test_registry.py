"""Tests for the generic Registry."""

import pytest

from ob_analytics._registry import Registry
from ob_analytics.exceptions import ObAnalyticsError, UnknownNameError


def test_register_get_case_insensitive_for_str_keys():
    r: Registry[str, int] = Registry("widget")
    r.register("Foo", 1)
    assert r.get("foo") == 1
    assert "FOO" in r
    assert r.list() == ["foo"]


def test_get_unknown_lists_registered():
    r: Registry[str, int] = Registry("widget")
    r.register("a", 1)
    with pytest.raises(UnknownNameError, match="Unknown widget 'b'.*Known: 'a'"):
        r.get("b")


def test_get_unknown_names_close_matches_not_every_key():
    r: Registry[str, int] = Registry("widget")
    for i in range(40):
        r.register(f"name_{i:02d}", i)
    r.register("bitstamp", 0)
    with pytest.raises(UnknownNameError) as info:
        r.get("bitstmp")
    message = str(info.value)
    assert message == "Unknown widget 'bitstmp'. Did you mean 'bitstamp'?"
    assert "name_00" not in message


def test_unknown_name_error_is_a_key_error_and_a_package_error():
    r: Registry[str, int] = Registry("widget")
    with pytest.raises(KeyError):
        r.get("b")
    with pytest.raises(ObAnalyticsError):
        r.get("b")


def test_tuple_keys_pass_through():
    r: Registry[tuple[str, str], int] = Registry("renderer")
    r.register(("trades", "matplotlib"), 1)
    assert r.get(("trades", "matplotlib")) == 1
    assert ("trades", "matplotlib") in r
