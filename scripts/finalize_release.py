#!/usr/bin/env python3
"""Validate version lockstep and emit the tag name for Finalize Release."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CARGO = ROOT / "Cargo.toml"
INIT = ROOT / "src" / "refloxide" / "__init__.py"

TOML_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
INIT_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def _read_toml_version(path: Path) -> str:
    match = TOML_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"missing version in {path}")
    return match.group(1)


def _read_init_version(path: Path) -> str:
    match = INIT_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"missing __version__ in {path}")
    return match.group(1)


def _tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default="",
        help="Expected version (default: pyproject.toml)",
    )
    parser.add_argument(
        "--confirm",
        required=True,
        help='Safety latch; must be exactly "publish"',
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Append GITHUB_OUTPUT-compatible key/value lines",
    )
    args = parser.parse_args()

    if args.confirm != "publish":
        raise SystemExit(
            'refusing to finalize: --confirm must be exactly "publish" '
            f"(got {args.confirm!r})"
        )

    py_version = _read_toml_version(PYPROJECT)
    cargo_version = _read_toml_version(CARGO)
    init_version = _read_init_version(INIT)
    if not (py_version == cargo_version == init_version):
        raise SystemExit(
            "version mismatch across package files: "
            f"pyproject={py_version} cargo={cargo_version} init={init_version}"
        )

    expected = args.version.strip() or py_version
    expected = expected.removeprefix("v")
    if expected != py_version:
        raise SystemExit(
            f"requested version {expected} does not match tree version {py_version}"
        )

    tag = f"v{py_version}"
    if _tag_exists(tag):
        raise SystemExit(f"tag {tag} already exists")

    print(f"VERSION={py_version}")
    print(f"TAG={tag}")
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"version={py_version}\n")
            handle.write(f"tag={tag}\n")


if __name__ == "__main__":
    main()
