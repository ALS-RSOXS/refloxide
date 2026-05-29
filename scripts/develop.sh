#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv python install 3.13
uv sync --group dev

PYREF_ROOT="${PYREF_ROOT:-$ROOT/../pyref}"
if [[ -f "$PYREF_ROOT/pyproject.toml" ]]; then
  echo "Installing editable pyref from $PYREF_ROOT"
  UV_NO_CONFIG=1 uv pip install -e "$PYREF_ROOT" "hvplot>=0.12.2"
fi

UV_NO_CONFIG=1 uv run maturin develop --release

export PYREF_ROOT
uv run python - <<'PY'
import os
import sys

import refloxide
import refloxide.rust as rust

print(f"refloxide {refloxide.__version__}")
print(f"rust extension: {rust.uniaxial_reflectivity.__name__}")

try:
    import importlib

    importlib.import_module("pyref.fitting")
except ImportError:
    print(
        "pyref not installed; Rust extension OK. "
        "Set PYREF_ROOT or place pyref beside refloxide, then re-run make develop.",
        file=sys.stderr,
    )
    sys.exit(0)

from refloxide.integrations.pyref import patch_pyref, pyref_patched

patch_pyref(use_rust=True, parallel=False)
assert pyref_patched(), "pyref patch did not apply"
print("pyref patch OK")
PY

echo "develop OK"
