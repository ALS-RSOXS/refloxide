# Rust extension (`rust`)

The native extension is built with PyO3 and is imported from Python as
`refloxide.rust`. Its public Rust API is documented with **rustdoc**.

During `mkdocs build`, rustdoc output is written under `docs/rustdoc/` (when `cargo` is available) and published with the rest of the site. Browse the crate root here:

- [refloxide crate (rustdoc)](../rustdoc/refloxide/index.html)

To view docs without MkDocs, run `cargo doc --no-deps --open` from the repository root; HTML is written under `target/doc/refloxide/`.
