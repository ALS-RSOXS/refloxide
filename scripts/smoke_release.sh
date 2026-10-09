#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

rm -rf dist .smoke-venv .smoke-venv-gpu

uv python install 3.13
PY="$(uv python find 3.13)"

echo "==> Building sdist..."
uv build --sdist -o dist --clear

build_variant_host() {
  local variant="$1"
  bash scripts/build_wheel_variant.sh "$variant" --find-interpreter
}

build_variant_docker() {
  local variant="$1"
  local name="refloxide"
  local features="python"
  if [[ "$variant" == "gpu" ]]; then
    name="refloxide-gpu"
    features="python,gpu"
  fi
  # Patch name inside the container's mounted tree; restore afterward.
  uv run python scripts/set_pypi_package_name.py "$name"
  docker run --rm \
    -v "$ROOT:/io" \
    -w /io \
    ghcr.io/pyo3/maturin:v1.13.3 \
    build --release --out dist --compatibility pypi \
    --no-default-features --features "$features" \
    -i python3.13 --manylinux 2_28
  uv run python scripts/set_pypi_package_name.py refloxide
}

if [[ "$(uname -s)" == "Linux" ]]; then
  echo "==> Building CPU + GPU PyPI wheels on Linux host..."
  build_variant_host cpu
  build_variant_host gpu
else
  if command -v docker >/dev/null 2>&1; then
    echo "==> Building CPU + GPU manylinux wheels in Docker..."
    build_variant_docker cpu
    build_variant_docker gpu
  else
    echo "==> Docker unavailable; building native CPU + GPU wheels for import smoke..."
    bash scripts/build_wheel_variant.sh cpu -i "$PY"
    bash scripts/build_wheel_variant.sh gpu -i "$PY"
  fi
fi

cpu_whl="$(ls -1 dist/refloxide-*.whl 2>/dev/null | head -1 || true)"
gpu_whl="$(ls -1 dist/refloxide_gpu-*.whl dist/refloxide-gpu-*.whl 2>/dev/null | head -1 || true)"
if [[ -z "$cpu_whl" ]]; then
  echo "FAIL: no CPU wheel (refloxide-*.whl) under dist/"
  exit 1
fi
if [[ -z "$gpu_whl" ]]; then
  echo "FAIL: no GPU wheel (refloxide_gpu-*.whl) under dist/"
  exit 1
fi

echo "CPU wheel: $(basename "$cpu_whl")"
echo "GPU wheel: $(basename "$gpu_whl")"

assert_platform_ok() {
  local whl="$1"
  case "$whl" in
    *linux_x86_64.whl|*linux_aarch64.whl)
      echo "FAIL: bare linux_* tag is rejected by PyPI: $(basename "$whl")"
      exit 1
      ;;
    *manylinux*|*musllinux*|*macosx*|*win*)
      echo "OK: platform tag in $(basename "$whl")"
      ;;
    *)
      echo "WARN: unexpected wheel platform in $(basename "$whl")"
      ;;
  esac
}

assert_platform_ok "$cpu_whl"
assert_platform_ok "$gpu_whl"

smoke_import() {
  local whl="$1"
  local venv="$2"
  local expect_gpu="$3"
  uv venv "$venv" --python 3.13 --seed
  "$venv/bin/pip" install --no-deps "$ROOT/$whl"
  "$venv/bin/python" - <<PY
import refloxide
import refloxide.rust as rust
print(refloxide.__version__, "$whl")
assert hasattr(rust, "uniaxial_reflectivity")
expect_gpu = "$expect_gpu" == "1"
# device=\"gpu\" must not report a missing cargo feature when this is a GPU wheel.
import numpy as np
q = np.linspace(0.01, 0.1, 4)
layers = np.array([[0.0, 0.0, 0.0, 0.0], [50.0, 1e-3, 2e-4, 1.0], [0.0, 7.5e-4, 1.2e-4, 2.0]])
tensor = np.zeros((3, 3, 3), dtype=np.complex128)
for i, n in enumerate([0j, 1e-3 + 2e-4j, 7.5e-4 + 1.2e-4j]):
    tensor[i] = np.diag([n, n, n])
try:
    rust.uniaxial_reflectivity(q, layers, tensor, 284.4, False, "gpu")
    gpu_msg = ""
except Exception as exc:
    gpu_msg = str(exc)
marker = "built without the"
if expect_gpu:
    assert marker not in gpu_msg, gpu_msg
else:
    assert marker in gpu_msg, gpu_msg
print("smoke OK expect_gpu=", expect_gpu)
PY
}


echo "==> Smoke-import CPU wheel..."
smoke_import "$cpu_whl" .smoke-venv 0
echo "==> Smoke-import GPU wheel..."
smoke_import "$gpu_whl" .smoke-venv-gpu 1

echo "Release smoke passed (CPU + GPU)."
