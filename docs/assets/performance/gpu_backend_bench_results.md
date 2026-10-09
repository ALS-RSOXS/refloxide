# Uniaxial backend bench

- generated: `2026-10-09T06:37:33Z`
- machine: `Darwin arm64 / arm`
- energy_ev: `250.0`
- n_q: `256`
- repeats/warmup: `3` / `1`
- gpu: `wgpu adapter present`

| backend | n_film | n_q | n_layers | median_ms | heap_MiB | rss_MiB | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| refloxide CPU serial | 1 | 256 | 3 | 0.189 | 0.000 | 28.2 |  |
| refloxide CPU parallel | 1 | 256 | 3 | 0.094 | 0.000 | 28.6 |  |
| refloxide GPU | 1 | 256 | 3 | 0.352 | 0.000 | 39.8 | wgpu adapter present |
| PyPXR / refnx plugin | 1 | 256 | 3 | 2.216 | 1.529 | 150.2 | pxr.plugin uniaxial (pure-Python TMM) |
| refloxide CPU serial | 10000 | 256 | 10002 | 817.328 | 0.000 | 31.8 |  |
| refloxide CPU parallel | 10000 | 256 | 10002 | 144.998 | 0.000 | 32.3 |  |
| refloxide GPU | 10000 | 256 | 10002 | 7.406 | 0.000 | 43.9 | wgpu adapter present |
| PyPXR / refnx plugin | 10000 | 256 | 10002 | 9835.808 | 4220.977 | 3521.1 | pxr.plugin uniaxial (pure-Python TMM) |
