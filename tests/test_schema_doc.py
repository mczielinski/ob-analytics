"""``docs/schema.md`` describes the same tables as the contracts (issue #499).

Each table on the page has a ``<!-- contract: NAME -->`` marker before it.
The test reads the column, Arrow type and unit of every row and checks them
against :data:`ob_analytics.schemas.TABLES`: the same columns, in the same
order, with the Arrow type of each pandas dtype, and with a price shown in
ticks and a size in lots on an integer column.
"""

from __future__ import annotations

import pathlib
import re

import pandas as pd
import pytest

from ob_analytics.schemas import PRICE, SIZE, TABLES, UTC_NS_DTYPE, Column, Table

_DOC = pathlib.Path(__file__).parents[1] / "docs" / "schema.md"
_MARKER = re.compile(r"<!-- contract: (\w+) -->")

# The Arrow type the page names for each pandas dtype.  An ``object`` column
# has no one type, so the page lists the types it can hold.
_ARROW: dict[str, str] = {
    "int64": "int64",
    "Int64": "int64",
    "float64": "double",
    "str": "string",
    UTC_NS_DTYPE: "timestamp[ns, tz=UTC]",
}


def _doc_tables() -> dict[str, list[tuple[str, str, str]]]:
    """``(column, arrow type, unit)`` for each row, by the marker's table name."""
    lines = _DOC.read_text().splitlines()
    tables: dict[str, list[tuple[str, str, str]]] = {}
    for i, line in enumerate(lines):
        marker = _MARKER.fullmatch(line.strip())
        if marker is None:
            continue
        name = marker.group(1)
        assert name not in tables, f"two tables on the page are marked {name!r}"
        rows = lines[i + 1 :]
        while rows and not rows[0].startswith("|"):
            rows = rows[1:]
        body = []
        for row in rows[2:]:  # skip the header and the |---| line
            if not row.startswith("|"):
                break
            cells = [
                cell.strip().replace("`", "") for cell in row.strip("|").split("|")
            ]
            body.append((cells[0], cells[1], cells[2]))
        tables[name] = body
    return tables


def _contract_rows(table: Table) -> list[Column]:
    """The columns the page lists: the fixed ones, then the bin rules."""
    return [*table.columns, *table.bin_columns]


def _arrow(column: Column) -> str | None:
    if isinstance(column.dtype, pd.CategoricalDtype):
        return "dictionary<string>"
    return _ARROW.get(str(column.dtype))


DOC_TABLES = _doc_tables()


def test_every_marker_names_a_contract() -> None:
    assert set(DOC_TABLES) == set(TABLES)


@pytest.mark.parametrize("name", sorted(TABLES))
def test_doc_lists_the_contract_columns(name: str) -> None:
    table = TABLES[name]
    documented = [row[0] for row in DOC_TABLES[name]]
    assert sorted(documented) == sorted(c.name for c in _contract_rows(table))
    fixed = [column for column in documented if column in table.names]
    assert fixed == list(table.names), "the fixed columns are in another order"


@pytest.mark.parametrize("name", sorted(TABLES))
def test_doc_types_match_the_contract_dtypes(name: str) -> None:
    columns = {c.name: c for c in _contract_rows(TABLES[name])}
    for column_name, arrow, _ in DOC_TABLES[name]:
        column = columns[column_name]
        expected = _arrow(column)
        if column.dtype == "object":
            assert " / " in arrow, f"{name}.{column_name}: list the types it holds"
        else:
            assert expected is not None, f"no Arrow type known for {column.dtype!r}"
            assert arrow == expected, f"{name}.{column_name}: {arrow} != {expected}"


@pytest.mark.parametrize("name", sorted(TABLES))
def test_doc_units_match_the_price_and_size_tags(name: str) -> None:
    columns = {c.name: c for c in _contract_rows(TABLES[name])}
    for column_name, _, unit in DOC_TABLES[name]:
        column = columns[column_name]
        integer = column.dtype in ("int64", "Int64")
        where = f"{name}.{column_name}"
        # An integer price is in ticks and an integer size in lots.  A float
        # can be either too (a bar's ``vwap`` is in ticks), so only the tag is
        # checked for it.
        if integer and column.kind is not None:
            assert unit == {PRICE: "ticks", SIZE: "lots"}[column.kind], where
        if unit == "ticks":
            assert column.kind == PRICE, where
        if unit == "lots":
            assert column.kind == SIZE, where
