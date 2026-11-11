"""The module map in ARCHITECTURE.md lists every file in the package.

The map is the first thing a contributor reads, and it goes out of date when a
module is added or removed without an edit to it.  This test reads the tree drawn in the
"Module map" section and compares it with the files on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ob_analytics"
ARCHITECTURE = ROOT / "ARCHITECTURE.md"

# One tree line: the indent (one "│   " or "    " per level), the branch, then
# the name.  A name that ends in "/" is a folder.
_TREE_LINE = re.compile(r"^((?:│   |    )*)(?:├── |└── )(\S+)")


def _map_block() -> list[str]:
    """Return the lines of the code block under the "Module map" heading."""
    text = ARCHITECTURE.read_text(encoding="utf-8")
    section = text.split("\n## Module map\n", 1)[1]
    block = section.split("```", 2)[1]
    lines = block.splitlines()
    # The first line is the fence's language tag (empty here), the second the root.
    assert lines[1] == "ob_analytics/", lines[1]
    return lines[2:]


def _mapped_files() -> set[str]:
    """Return the file paths the map lists, relative to the package."""
    files: set[str] = set()
    folders: list[str] = []
    for line in _map_block():
        match = _TREE_LINE.match(line)
        if match is None:
            continue  # a blank "│" spacer line
        depth = len(match.group(1)) // 4
        name = match.group(2)
        del folders[depth:]
        if name.endswith("/"):
            folders.append(name.rstrip("/"))
        else:
            files.add("/".join([*folders, name]))
    return files


def _package_files() -> set[str]:
    """Return every file in the package on disk, relative to the package."""
    return {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def test_module_map_parser_reads_nested_folders() -> None:
    files = _mapped_files()
    assert "pipeline.py" in files
    assert "engine/_queue.py" in files
    assert "live/_supervisor.py" in files
    assert "visualization/_bokeh.py" in files


def test_module_map_lists_every_package_file() -> None:
    missing = sorted(_package_files() - _mapped_files())
    assert not missing, (
        f"add these files to the module map in ARCHITECTURE.md: {missing}"
    )


def test_module_map_names_only_existing_files() -> None:
    gone = sorted(_mapped_files() - _package_files())
    assert not gone, (
        f"remove these files from the module map in ARCHITECTURE.md: {gone}"
    )
