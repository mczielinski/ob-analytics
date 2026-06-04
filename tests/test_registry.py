"""Tests for the generic Registry."""

import pytest

from ob_analytics._registry import Registry


def test_register_get_case_insensitive_for_str_keys():
    r: Registry[str, int] = Registry("widget")
    r.register("Foo", 1)
    assert r.get("foo") == 1
    assert "FOO" in r
    assert r.list() == ["foo"]


def test_get_unknown_lists_registered():
    r: Registry[str, int] = Registry("widget")
    r.register("a", 1)
    with pytest.raises(KeyError, match="Unknown widget 'b'.*Registered.*a"):
        r.get("b")


def test_tuple_keys_pass_through():
    r: Registry[tuple[str, str], int] = Registry("renderer")
    r.register(("trades", "matplotlib"), 1)
    assert r.get(("trades", "matplotlib")) == 1
    assert ("trades", "matplotlib") in r
