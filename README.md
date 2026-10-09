# refloxide

An extremely fast 4×4 transfer-matrix engine for **polarized** reflectivity
from stratified media — Rust core, optional wgpu GPU, thin Python bindings.

<p align="center">
  <img src="docs/assets/performance/bench_hero_light.png#gh-light-mode-only" alt="10k-slab uniaxial wall time" width="520">
  <img src="docs/assets/performance/bench_hero_dark.png#gh-dark-mode-only" alt="10k-slab uniaxial wall time" width="520">
</p>

<p align="center">
  <sub>10 000 uniaxial film slabs · 256 q · Apple Silicon · vs PyPXR plugin</sub>
</p>

<details>
<summary>More charts</summary>

<br>

<p align="center">
  <img src="docs/assets/performance/bench_time_1slab_light.png#gh-light-mode-only" width="380">
  <img src="docs/assets/performance/bench_time_1slab_dark.png#gh-dark-mode-only" width="380">
</p>
<p align="center"><sub>1 slab · wall time</sub></p>

<p align="center">
  <img src="docs/assets/performance/bench_rss_1slab_light.png#gh-light-mode-only" width="380">
  <img src="docs/assets/performance/bench_rss_1slab_dark.png#gh-dark-mode-only" width="380">
</p>
<p align="center"><sub>1 slab · peak RSS</sub></p>

<p align="center">
  <img src="docs/assets/performance/bench_time_10k_light.png#gh-light-mode-only" width="380">
  <img src="docs/assets/performance/bench_time_10k_dark.png#gh-dark-mode-only" width="380">
</p>
<p align="center"><sub>10k slabs · wall time</sub></p>

<p align="center">
  <img src="docs/assets/performance/bench_rss_10k_light.png#gh-light-mode-only" width="380">
  <img src="docs/assets/performance/bench_rss_10k_dark.png#gh-dark-mode-only" width="380">
</p>
<p align="center"><sub>10k slabs · peak RSS</sub></p>

```bash
uv sync --group dev --extra plugin
uv run python examples/gpu_backend_bench.py --plot
```
</details>

## Installation

CPU and GPU ship as **separate wheels** (same `import refloxide`; install one):

| Install | What you get |
| --- | --- |
| `refloxide` / `refloxide[core]` | Fast Rust TMM on CPU (lightweight wheel) |
| `refloxide[plugin]` | CPU core plus the refnx modeling / fitting stack |
| `refloxide-gpu` | Same API with the wgpu GPU backend (`device="gpu"`) |
| `refloxide[all]` | Alias for `refloxide[plugin]` |

```bash
pip install 'refloxide[core]'       # CPU
pip install 'refloxide[plugin]'     # CPU + modeling
pip install refloxide-gpu           # GPU wheel (do not also install refloxide)
```

With uv:

```bash
uv add 'refloxide[core]'
uv add 'refloxide[plugin]'
uv add refloxide-gpu
```

Bare `pip install refloxide` is the same as `[core]` (numpy + the CPU extension).

## Quick start

```python
from refloxide.tmm import uniaxial_reflectivity

# CPU (default)
refl, tran = uniaxial_reflectivity(q, layers, tensor, energy_ev, parallel=False)

# GPU (install the refloxide-gpu wheel, not refloxide)
refl, tran = uniaxial_reflectivity(
    q, layers, tensor, energy_ev, parallel=False, device="gpu"
)
```

## Development

```bash
git clone https://github.com/ALS-RSOXS/refloxide.git
cd refloxide
make develop
make test && make verify
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
