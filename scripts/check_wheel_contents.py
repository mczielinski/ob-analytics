"""Check that a wheel holds only the ob_analytics package and its metadata.

    python scripts/check_wheel_contents.py dist/ob_analytics-0.1.0-py3-none-any.whl

It lists every file in each wheel given and exits non-zero if any file lies
outside ``ob_analytics/`` and the wheel's ``*.dist-info/`` folder, or is a
hidden file or a cache inside the package (a path part that starts with ``.``,
or ``__pycache__``).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def stray_files(wheel: Path) -> list[str]:
    """Return the files in *wheel* that are not the package's own files or metadata."""
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    return [name for name in names if not _belongs(name)]


def _belongs(name: str) -> bool:
    top, *rest = name.split("/")
    if top.startswith("ob_analytics-") and top.endswith(".dist-info"):
        return True
    if top != "ob_analytics":
        return False
    return not any(part.startswith(".") or part == "__pycache__" for part in rest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheels", nargs="+", type=Path, help="wheel files to check")
    args = parser.parse_args(argv)

    failed = False
    for wheel in args.wheels:
        stray = stray_files(wheel)
        if stray:
            failed = True
            print(f"FAIL: {wheel} has files that do not belong in it ({len(stray)}):")
            for name in stray[:20]:
                print(f"  {name}")
            if len(stray) > 20:
                print(f"  ... and {len(stray) - 20} more")
        else:
            print(f"ok: {wheel} holds only the package")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
