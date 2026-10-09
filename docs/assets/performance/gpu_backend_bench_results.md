# Uniaxial backend bench

- regenerated plots: `2026-10-09T06:43:41Z`
- source: `docs/assets/performance/gpu_backend_bench_results.csv`
- machine: `Darwin arm64 / arm`

| backend | n_film | n_q | n_layers | median_ms | heap_MiB | rss_MiB | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| refloxide CPU serial | 1 | 256 | 3 | 0.230 | 0.000 | 28.0 |  |
| refloxide CPU parallel | 1 | 256 | 3 | 0.116 | 0.000 | 28.6 |  |
| refloxide GPU | 1 | 256 | 3 | 0.744 | 0.000 | 39.8 | wgpu adapter present |
| PyPXR / refnx plugin | 1 | 256 | 3 | 2.268 | 1.529 | 150.3 | pxr.plugin uniaxial (pure-Python TMM) |
| refloxide CPU serial | 10000 | 256 | 10002 | 784.299 | 0.000 | 31.7 |  |
| refloxide CPU parallel | 10000 | 256 | 10002 | 138.803 | 0.000 | 32.3 |  |
| refloxide GPU | 10000 | 256 | 10002 | 4.070 | 0.000 | 43.9 | wgpu adapter present |
| PyPXR / refnx plugin | 10000 | 256 | 10002 | 8614.550 | 4220.977 | 2146.7 | pxr.plugin uniaxial (pure-Python TMM) |

## Best results (10,000 slabs)

| Backend | Time | vs PyPXR | RSS |
| --- | ---: | ---: | ---: |
| **GPU** | **4.1 ms** | 2,117× | 44 MiB |
| **CPU parallel** | **138.8 ms** | 62× | 32 MiB |
| **CPU serial** | **784.3 ms** | 11× | 32 MiB |
| **PyPXR** | **8614.5 ms** | — | 2147 MiB |
