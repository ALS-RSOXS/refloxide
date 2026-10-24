# Rust extension (`rust`)

The native extension is built with PyO3 and is imported from Python as
`refloxide.rust`. Its public Rust API is documented with **rustdoc**.

During `mkdocs build`, rustdoc output is written under `docs/rustdoc/` (when `cargo` is available) and published with the rest of the site. Browse the crate root here:

- [refloxide-core (rustdoc)](../rustdoc/refloxide_core/index.html): shared types, scalar trait, 4x4 helpers, errors
- [refloxide-optics (rustdoc)](../rustdoc/refloxide_optics/index.html): optical-constant interpolation and lab-frame tensors
- [refloxide-tmm (rustdoc)](../rustdoc/refloxide_tmm/index.html): transfer-matrix kernels and CPU drivers
- [refloxide-gpu (rustdoc)](../rustdoc/refloxide_gpu/index.html): wgpu backend
- [PyO3 bindings (rustdoc)](../rustdoc/rust/index.html)

To view docs without MkDocs, run `cargo doc --no-deps --workspace --open` from the repository root; HTML is written under `target/doc/`.
