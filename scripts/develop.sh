#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv python install 3.13
# dev: tooling + refnx for tests/benches; plugin: polarized modeling extras
uv sync --group dev --group plugin

UV_NO_CONFIG=1 uv run maturin develop --release

uv run python - <<'PY'
import refloxide
import refloxide.rust as rust

print(f"refloxide {refloxide.__version__}")
print(f"rust extension: {rust.uniaxial_reflectivity.__name__}")
assert hasattr(refloxide, "uniaxial_reflectivity")
print("develop OK")
PY

echo "develop OK"
