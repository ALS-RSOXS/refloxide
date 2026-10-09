#!/usr/bin/env python3
"""Set the PyPI project name in pyproject.toml for a wheel variant build.

CPU wheels publish as ``refloxide``. GPU wheels publish as ``refloxide-gpu``
(same import path ``refloxide``; install only one of the two distributions).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_NAME_RE = re.compile(r'^name = "(refloxide(?:-gpu)?)"$', re.MULTILINE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name",
        choices=("refloxide", "refloxide-gpu"),
        help="PyPI project name to write into [project]",
    )
    args = parser.parse_args()
    text = _PYPROJECT.read_text(encoding="utf-8")
    if _NAME_RE.search(text) is None:
        print("FAIL: could not find project name in pyproject.toml", file=sys.stderr)
        return 1
    updated = _NAME_RE.sub(f'name = "{args.name}"', text, count=1)
    if updated == text and f'name = "{args.name}"' not in text:
        print("FAIL: name substitution did not apply", file=sys.stderr)
        return 1
    _PYPROJECT.write_text(updated, encoding="utf-8")
    print(f"OK: project name -> {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
