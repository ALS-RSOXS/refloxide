# refloxide-py

The only crate that depends on Python. It builds the `refloxide.rust` extension module with PyO3 and maturin.

## Contents

Twelve functions registered on the module: `uniaxial_reflectivity`, `uniaxial_reflectivity_batch`, `uniaxial_reflectivity_points`, `uniaxial_reflectivity_points_jvp`, `general_reflectivity`, `bookended_uniaxial_reflectivity`, `interp_ooc_linear`, `lab_tensor_diagonals_batch`, `isotropic_lab_tensor`, `molecular_index_at_ooc`, `uniaxial_lab_tensor`, `tensor_to_slab_row`.

The layer marshals numpy arrays into owned Rust buffers, releases the GIL around solver calls, and maps `RefloxideError` to `ValueError` (shape, layer count, energy) or `RuntimeError` (singular matrix, GPU). It contains no solver logic.

## Build

From the repository root:

```bash
make develop
```

`pyproject.toml` points maturin at this crate (`manifest-path`) and at `python/` for the package. Type stubs are in `python/refloxide/rust.pyi`.
