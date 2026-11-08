"""Tests for the docs example runner, ``scripts/check_doc_examples.py``.

The runner itself runs in the docs workflow, because each page takes a
pipeline run. These tests cover how it reads a page, which is cheap.
"""

from __future__ import annotations

import pytest

from scripts.check_doc_examples import (
    PAGE_GLOBS,
    ROOT,
    PageError,
    extract_blocks,
)


def test_python_blocks_run_in_order_and_other_languages_do_not():
    page = (
        "# Title\n"
        "```python\na = 1\n```\n"
        "```bash\nob-analytics process orders.csv\n```\n"
        '```python title="b.py"\nb = a + 1\n```\n'
    )
    blocks = extract_blocks(page)
    assert [(b.line, b.code, b.kind) for b in blocks] == [
        (3, "a = 1\n", "run"),
        (9, "b = a + 1\n", "run"),
    ]


def test_a_block_inside_an_admonition_is_dedented():
    page = '!!! note "x"\n    ```python\n    if True:\n        a = 1\n    ```\n'
    (block,) = extract_blocks(page)
    assert block.code == "if True:\n    a = 1\n"


def test_skip_needs_a_reason_and_keeps_it():
    page = "<!-- docs-test: skip needs an API key -->\n\n```python\nx\n```\n"
    (block,) = extract_blocks(page)
    assert block.skip == "needs an API key"
    with pytest.raises(PageError, match="needs a reason"):
        extract_blocks("<!-- docs-test: skip -->\n```python\nx\n```\n")


def test_a_marker_must_reach_a_block():
    with pytest.raises(PageError, match="has no code block"):
        extract_blocks("<!-- docs-test: skip key -->\nSome prose.\n```python\nx\n```\n")
    with pytest.raises(PageError, match="has no code block"):
        extract_blocks("<!-- docs-test: output -->\n")
    with pytest.raises(PageError, match="non-Python"):
        extract_blocks("<!-- docs-test: skip key -->\n```bash\nls\n```\n")


def test_hidden_code_runs_at_its_place_with_its_own_line_numbers():
    page = "```python\na = 1\n```\n<!-- docs-test\n    assert a == 1\n-->\n"
    _, hidden = extract_blocks(page)
    assert (hidden.line, hidden.code, hidden.kind) == (5, "assert a == 1\n", "hidden")
    with pytest.raises(PageError, match="not closed"):
        extract_blocks("<!-- docs-test\nassert True\n")


def test_an_output_marker_takes_the_next_block_of_any_language():
    page = (
        "```python\nprint(1)\n```\n"
        "<!-- docs-test: output -->\n```text\n1\n```\n"
        "<!-- docs-test: output report.render() -->\n```\nok\n```\n"
    )
    _, printed, expression = extract_blocks(page)
    assert (printed.kind, printed.code, printed.expected) == ("output", "", "1\n")
    assert (expression.code, expression.expected) == ("report.render()", "ok\n")


def test_an_output_check_needs_a_block_before_it_that_runs():
    with pytest.raises(PageError, match="no Python block before it"):
        extract_blocks("<!-- docs-test: output -->\n```text\n1\n```\n")
    skipped = "<!-- docs-test: skip key -->\n```python\nprint(1)\n```\n"
    with pytest.raises(PageError, match="no Python block before it"):
        extract_blocks(skipped + "<!-- docs-test: output -->\n```text\n1\n```\n")
    (_, check) = extract_blocks(skipped + "<!-- docs-test: output x -->\n```\n1\n```\n")
    assert check.code == "x"


def test_every_docs_page_reads():
    """A malformed marker on any page fails here, before the docs workflow."""
    pages = sorted({p for g in PAGE_GLOBS for p in ROOT.glob(g)})
    assert pages
    steps = [b for p in pages for b in extract_blocks(p.read_text(encoding="utf-8"))]
    assert any(b.kind == "output" for b in steps)
    assert all(b.skip for b in steps if b.skip is not None)
