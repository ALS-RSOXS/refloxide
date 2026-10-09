# refloxide

An extremely fast 4x4 transfer-matrix engine for **polarized** reflectivity
from stratified media — written in Rust, with optional wgpu GPU acceleration
and thin Python bindings.

## Performance

**10,000 uniaxial film slabs** · 256 q-points · Apple Silicon

| Backend | Time | vs PyPXR | RSS |
| --- | ---: | ---: | ---: |
| **GPU** | **4.1 ms** | **2,117×** | 44 MiB |
| **CPU parallel** | **139 ms** | **62×** | 32 MiB |
| **CPU serial** | **784 ms** | **11×** | 32 MiB |
| PyPXR | 8,615 ms | — | 2,147 MiB |

<p align="center">
  <img src="docs/assets/performance/bench_hero_light.png#gh-light-mode-only" alt="10k-slab wall time" width="560">
  <img src="docs/assets/performance/bench_hero_dark.png#gh-dark-mode-only" alt="10k-slab wall time" width="560">
</p>

<details>
<summary>More charts</summary>

<br>

**1 slab — time / RSS**

<p align="center">
  <img src="docs/assets/performance/bench_time_1slab_light.png#gh-light-mode-only" width="340">
  <img src="docs/assets/performance/bench_time_1slab_dark.png#gh-dark-mode-only" width="340">
  <img src="docs/assets/performance/bench_rss_1slab_light.png#gh-light-mode-only" width="340">
  <img src="docs/assets/performance/bench_rss_1slab_dark.png#gh-dark-mode-only" width="340">
</p>

**10k slabs — time / RSS**

<p align="center">
  <img src="docs/assets/performance/bench_time_10k_light.png#gh-light-mode-only" width="340">
  <img src="docs/assets/performance/bench_time_10k_dark.png#gh-dark-mode-only" width="340">
  <img src="docs/assets/performance/bench_rss_10k_light.png#gh-light-mode-only" width="340">
  <img src="docs/assets/performance/bench_rss_10k_dark.png#gh-dark-mode-only" width="340">
</p>

```bash
uv sync --group dev --group plugin
uv run python examples/gpu_backend_bench.py --plot
```

Uniaxial only (not scalar Abeles). RSS = peak process RSS in a fresh
subprocess. `refnx` is a dev/plugin extra for the PyPXR plugin path.
Guide: [docs/guides/gpu.md](docs/guides/gpu.md)

</details>

## Installation

```bash
pip install refloxide
# or
uv add refloxide
```

## Quick start

```python
from refloxide.tmm import uniaxial_reflectivity

refl, tran = uniaxial_reflectivity(q, layers, tensor, energy_ev, device="gpu")
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
