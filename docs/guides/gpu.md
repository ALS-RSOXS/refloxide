# GPU backend

Portable GPU evaluation of the uniaxial-z recursion through [wgpu](https://wgpu.rs/)
(Metal, Vulkan, DX12). The public Python path is single-precision recursion;
double-single (`df64`) and general-tensor shaders exist behind the Rust
`gpu` feature for experiments and examples.

## Enablement

GPU support is a **separate wheel** from the default CPU package (PyPI cannot
publish two different binaries under the same name and tag):

```bash
pip install refloxide-gpu          # GPU wheel (import path still `refloxide`)
# modeling on top of a GPU wheel: install plugin deps into the same env, e.g.
pip install 'refloxide-gpu' refnx periodictable polars pyarrow scipy arviz
```

Do **not** install both `refloxide` and `refloxide-gpu` in one environment;
they both provide the `refloxide` package.

Default `pip install refloxide` wheels are CPU-only (`features = ["python"]`).
GPU wheels are built with:

```bash
bash scripts/build_wheel_variant.sh gpu
# cargo: --no-default-features --features python,gpu
# PyPI name temporarily set to refloxide-gpu for that build
```

Local editable installs (for contributors) enable GPU:

```bash
make develop   # maturin develop --features python,gpu
```

For Rust-only work:

```bash
cargo check --features gpu
cargo run --profile perf --no-default-features --features gpu --example gpu_recursive
```

If the crate was built without `gpu` (the CPU wheel), `device="gpu"` raises a
clear error that the feature is missing. If the feature is present but no
adapter is available, the same path fails at device acquisition.

## Usage

Kernel entry points accept `device="cpu"` (default) or `device="gpu"`:

```python
from refloxide.tmm import uniaxial_reflectivity

refl, tran = uniaxial_reflectivity(
    q, layers, tensor, energy_ev, parallel=False, device="gpu"
)
```

`ReflectModel` forwards the same flag on every evaluation:

```python
from refloxide.model import ReflectModel

model = ReflectModel(structure, device="gpu", parallel=False)
```

For differential evolution or MCMC, keep `device="gpu"` and pass
`refloxide.objective.gpu_batch` as the worker/pool so each generation or
walker ensemble is one dispatch. Fitting helpers that wrap refnx need the
plugin stack (`refnx`, …) installed alongside `refloxide-gpu`:

```bash
pip install refloxide-gpu
pip install refnx periodictable polars pyarrow scipy arviz
```

```python
from refloxide.objective import gpu_batch

model.device = "gpu"
# from refnx.analysis import CurveFitter
# fitter = CurveFitter(objective)
# fitter.fit("differential_evolution", workers=gpu_batch(objective))
```

`parallel` is ignored on the GPU. Prefer `parallel=False` on the CPU when
the surrounding fitter already parallelizes walkers or energies.

## Contracts

| Behavior | GPU (`device="gpu"`) |
| --- | --- |
| Arithmetic | `f32` uniaxial-z recursion |
| Reflectance vs CPU `f64` | about `1e-4` relative (often better on soft-X-ray stacks) |
| Cross-polarized reflectance | exactly zero |
| Transmission | not computed; every entry is `nan + nanj` |
| Stack shape | every stack in a batch must share the same layer count |
| Fused bookended kernel | CPU-only; GPU always materializes the stack |

Finite-difference gradient methods (L-BFGS-B, least squares) should stay on
`device="cpu"`. Derivative-free searches and samplers (DE, MCMC) are the
intended GPU workload.

### Platform notes

Native `f64` shaders (`wgpu` `SHADER_F64`) are available on typical Vulkan
and DX12 adapters, not on Metal. The public Python path does not need them.
Rust `GpuContext::reflectivity_points_general` with `GeneralPrecision::F64`
errors on Metal; use `supports_f64()` or the `f32` build for cross-checks
only (general `f32` is not accurate enough for fitting on tilted or biaxial
stacks).

## Examples and benchmarks

| Artifact | Role |
| --- | --- |
| `examples/gpu_recursive.rs` | CPU `f32` / CPU `f64` / GPU reflectance and throughput |
| `examples/gpu_df64.rs`, `examples/gpu_df64_bench.rs` | Double-single recursion experiments |
| `examples/gpu_general.rs`, `examples/gpu_general_check.rs` | General-tensor shaders (runtime and naga validate) |
| `examples/kernel_precision.rs` | Precision study across kernel formulations |
| `examples/gpu_backend_bench.py` | Cross-backend wall time and peak Python heap |

Regenerate the uniaxial backend benchmark (1 slab and **10,000** film
slabs). Backends: Rust CPU serial, Rust CPU parallel, GPU, and the in-tree
**PyPXR / refnx plugin** uniaxial path (`refloxide.pxr.plugin`). Core CPU
rows need `refloxide`; GPU rows need a `refloxide-gpu` (or `make develop`)
build; the plugin row needs the plugin deps:

```bash
uv sync --group dev --extra plugin
make develop
uv run python examples/gpu_backend_bench.py --plot
```

Pre-merge CI runs the same script on Ubuntu and macOS, enforces ranking
invariants (`GPU < parallel < serial < PyPXR` at 10k slabs), and feeds
`benchmark-action.json` into
[`github-action-benchmark`](https://github.com/benchmark-action/github-action-benchmark)
(`customSmallerIsBetter`). History is restored from Actions cache (updated
only on `main` / merge queue); PRs compare against that baseline, comment on
alert, and fail above a 2× wall-time/RSS regression. CSV / PNG artifacts are
still uploaded. Automated precision checks live in
`tests/test_gpu_precision.py` and skip when no GPU adapter is present; those
tests do not import `refnx`.
