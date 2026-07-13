# API reference

Python API pages are generated from source with mkdocstrings. The PyO3 extension is documented with rustdoc and copied into the published site next to these pages when `cargo` is available during `mkdocs build`.

## Python

- [refloxide](python/refloxide.md) — top-level package (Rust TMM default)
- [refloxide.python.tmm](python/python/tmm.md) — opt-in pure-Python TMM
- [refloxide.pxr](python/pxr/index.md) — legacy path (will relocate; prefer top-level modules)

## Rust

- [Rust extension `rust`](rust.md) — rustdoc for the PyO3 crate
