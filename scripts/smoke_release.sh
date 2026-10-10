#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

rm -rf dist .smoke-venv

uv python install 3.13
PY="$(uv python find 3.13)"

echo "==> Building sdist..."
uv build --sdist -o dist --clear

build_wheel_host() {
  uv run maturin build --release --out dist --compatibility pypi --find-interpreter
}

build_wheel_docker() {
  docker run --rm \
    -v "$ROOT:/io" \
    -w /io \
    ghcr.io/pyo3/maturin:v1.13.3 \
    build --release --out dist --compatibility pypi -i python3.13 --manylinux 2_28
}

if [[ "$(uname -s)" == "Linux" ]]; then
  echo "==> Building PyPI-compatible wheel on Linux host..."
  build_wheel_host
else
  if command -v docker >/dev/null 2>&1; then
    echo "==> Building PyPI-compatible wheel in manylinux Docker..."
    build_wheel_docker
  else
    echo "==> Docker unavailable; building native wheel for import smoke only..."
    uv run maturin build --release --out dist --compatibility pypi -i "$PY"
  fi
fi

whl="$(ls -1 dist/*.whl 2>/dev/null | head -1 || true)"
if [[ -z "$whl" ]]; then
  echo "FAIL: no wheel produced under dist/"
  exit 1
fi

echo "Wheel: $(basename "$whl")"
case "$whl" in
  *linux_x86_64.whl)
    echo "FAIL: bare linux_x86_64 tag is rejected by PyPI"
    exit 1
    ;;
  *manylinux*|*musllinux*)
    echo "OK: portable Linux platform tag"
    ;;
  *macosx*|*win*)
    echo "OK: native platform wheel (local smoke)"
    ;;
  *)
    echo "WARN: unexpected wheel platform in $(basename "$whl")"
    ;;
esac

install_and_import() {
  uv venv .smoke-venv --python 3.13 --seed
  .smoke-venv/bin/pip install --no-deps "$ROOT/$whl"
  .smoke-venv/bin/python -c "import refloxide; import refloxide.rust; print(refloxide.__version__)"
}

if [[ "$whl" == *manylinux* || "$whl" == *musllinux* || "$whl" == *macosx* || "$whl" == *win* ]]; then
  echo "==> Installing wheel in clean venv and importing..."
  install_and_import
fi

echo "Release smoke passed."
