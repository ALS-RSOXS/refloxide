# GPU backend bench results

- generated: `2026-10-09T06:33:02Z`
- machine: `Darwin arm64 / arm`
- energy_ev: `250.0`
- headline n_film: `10000`
- repeats/warmup: `3` / `1`
- gpu: `wgpu adapter present`

| backend | n_film | n_q | n_layers | median_ms | heap_KiB | rss_delta_MiB | points/s | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| refnx Abeles (isotropic) | 100 | 256 | 102 | 0.868 | 7.8 | 0.00 | 294,902 | scalar Abeles 2x2 on isotropic twin; not polarized TMM |
| refloxide.python.tmm | 100 | 256 | 102 | 73.073 | 44087.7 | 0.11 | 3,503 | pure-Python polarized TMM |
| refloxide CPU parallel=False | 100 | 256 | 102 | 7.929 | 0.3 | 0.00 | 32,287 |  |
| refloxide CPU parallel=True | 100 | 256 | 102 | 1.337 | 0.3 | 0.03 | 191,414 |  |
| refloxide GPU | 100 | 256 | 102 | 0.330 | 0.3 | 0.00 | 774,877 | wgpu adapter present |
| refnx Abeles (isotropic) | 1000 | 256 | 1002 | 8.566 | 36.0 | 0.00 | 29,886 | scalar Abeles 2x2 on isotropic twin; not polarized TMM |
| refloxide.python.tmm | 1000 | 256 | 1002 | n/a | n/a | n/a | n/a | skipped: n_film>256 (pass --include-python-tmm) |
| refloxide CPU parallel=False | 1000 | 256 | 1002 | 75.826 | 0.3 | 0.00 | 3,376 |  |
| refloxide CPU parallel=True | 1000 | 256 | 1002 | 12.752 | 0.3 | 0.00 | 20,075 |  |
| refloxide GPU | 1000 | 256 | 1002 | 2.852 | 0.3 | 0.00 | 89,762 | wgpu adapter present |
| refnx Abeles (isotropic) | 10000 | 256 | 10002 | 83.914 | 317.2 | 0.00 | 3,051 | scalar Abeles 2x2 on isotropic twin; not polarized TMM |
| refloxide.python.tmm | 10000 | 256 | 10002 | n/a | n/a | n/a | n/a | skipped: n_film>256 (pass --include-python-tmm) |
| refloxide CPU parallel=False | 10000 | 256 | 10002 | 769.390 | 0.3 | 0.00 | 333 |  |
| refloxide CPU parallel=True | 10000 | 256 | 10002 | 128.525 | 0.3 | 0.00 | 1,992 |  |
| refloxide GPU | 10000 | 256 | 10002 | 7.062 | 0.3 | 0.00 | 36,249 | wgpu adapter present |
