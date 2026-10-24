# refloxide-core

Shared foundation for the refloxide transfer-matrix crates.

## Contents

- `Real`: scalar trait implemented for `f32` and `f64`, so one kernel body serves both precisions.
- `Layer`, `LayerCoeffs<T>`: slab description and the flat per-slab record (`chi = eps - 1` for the ordinary and extraordinary axes) consumed by the kernels.
- `layer_coeffs`, `wavenumber`, `HC_EV_ANGSTROM`: tensor packing under `epsilon = conj(I - 2 * tensor)` and photon energy to wavenumber conversion.
- `validate_batch`, `validate_points`, `jvp_coeffs`: shape, energy, and tangent checks shared by the CPU and GPU backends.
- `UniaxialOutput`, `UniaxialBatchOutput`, `PointsJvpOutput`, `PointsTangents`: result and tangent containers.
- `c4x4`, `math`: fixed-size 4x4 complex helpers and the exact 4x4 inverse.
- `RefloxideError`, `Result<T>`: typed errors for the whole workspace.

## Boundaries

No solver loops, no threading, no Python or GPU dependencies. Solvers live in `refloxide-tmm`; optical constants live in `refloxide-optics`.

## Conventions

The tensor diagonal carries `delta + i*beta` per principal axis. Reflectance layout is `[:, 0, 0] = R_pp` and `[:, 1, 1] = R_ss`.
