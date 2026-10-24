#!/usr/bin/env python3
"""Decide whether the version on this checkout still needs a release.

Reads the workspace version from ``Cargo.toml``, the ``pyproject.toml`` version, and
``__version__`` in ``python/refloxide/__init__.py``. Exits non-zero when they differ.
Reports ``release=true`` when the git tag ``vX.Y.Z`` does not exist on ``origin``.
With ``--github-output FILE`` the ``version`` and ``release`` keys are appended in
GitHub Actions output format.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARGO = ROOT / "Cargo.toml"
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "python" / "refloxide" / "__init__.py"

TOML_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
INIT_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def read_version(path: Path, pattern: re.Pattern[str]) -> str:
    """First version string in ``path`` matching ``pattern``."""
    match = pattern.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"missing version in {path}")
    return match.group(1)


def tag_on_origin(tag: str) -> bool:
    """True when ``refs/tags/<tag>`` exists on the ``origin`` remote."""
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{tag}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def main() -> None:
    """Validate version lockstep and print the release decision."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--github-output", type=Path, default=None)
    args = parser.parse_args()

    versions = {
        "Cargo.toml": read_version(CARGO, TOML_RE),
        "pyproject.toml": read_version(PYPROJECT, TOML_RE),
        "__init__.py": read_version(INIT, INIT_RE),
    }
    if len(set(versions.values())) != 1:
        raise SystemExit(f"version mismatch across package files: {versions}")
    version = versions["Cargo.toml"]
    release = not tag_on_origin(f"v{version}")
    print(f"version={version}")
    print(f"release={str(release).lower()}")
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"version={version}\nrelease={str(release).lower()}\n")


if __name__ == "__main__":
    main()
