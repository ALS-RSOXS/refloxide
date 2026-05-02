# Rust extension (`_core`)

The native extension is built with PyO3. Its public Rust API is documented with **rustdoc**.

During `mkdocs build`, rustdoc output is written under `docs/rustdoc/` (when `cargo` is available) and published with the rest of the site. Browse the crate root here:

- [_core crate (rustdoc)](../rustdoc/_core/index.html)

To view docs without MkDocs, run `cargo doc --no-deps --open` from the repository root; HTML is written under `target/doc/_core/`.
