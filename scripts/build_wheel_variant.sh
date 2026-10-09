#!/usr/bin/env bash
# Build a CPU or GPU wheel variant.
#   CPU  -> PyPI name refloxide,      cargo features: python
#   GPU  -> PyPI name refloxide-gpu,  cargo features: python,gpu
#
# Invokes maturin from the project venv directly so a temporary rename to
# ``refloxide-gpu`` does not make ``uv run`` re-resolve/install the project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VARIANT="${1:-}"
if [[ "$VARIANT" != "cpu" && "$VARIANT" != "gpu" ]]; then
  echo "usage: $0 cpu|gpu [extra maturin args...]" >&2
  exit 2
fi
shift || true

OUT="${OUT:-dist}"
mkdir -p "$OUT"

if [[ ! -x "$ROOT/.venv/bin/maturin" ]]; then
  echo "==> Ensuring maturin is available in .venv..."
  uv sync --group dev --extra plugin
fi
MATURIN="$ROOT/.venv/bin/maturin"
PYTHON="${PYTHON:-python3}"

restore_name() {
  "$PYTHON" "$ROOT/scripts/set_pypi_package_name.py" refloxide >/dev/null
}
trap restore_name EXIT

if [[ "$VARIANT" == "gpu" ]]; then
  "$PYTHON" "$ROOT/scripts/set_pypi_package_name.py" refloxide-gpu
  FEATURES=(--no-default-features --features python,gpu)
else
  "$PYTHON" "$ROOT/scripts/set_pypi_package_name.py" refloxide
  FEATURES=(--no-default-features --features python)
fi

echo "==> Building ${VARIANT} wheel (features: ${FEATURES[*]})"
"$MATURIN" build --release --out "$OUT" --compatibility pypi "${FEATURES[@]}" "$@"

echo "OK: ${VARIANT} wheel(s) in ${OUT}/"
ls -lh "$OUT"/*.whl 2>/dev/null || true
