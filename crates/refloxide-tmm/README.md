# refloxide-tmm

4x4 transfer-matrix reflectivity kernels and CPU drivers for stratified media.

## Contents

- `kernel`: backend-agnostic per-point solvers generic over `Real`: uniaxial-z recursion, R-matrix, full general 4x4 with cross-polarization, and forward-mode tangents.
- `uniaxial`: validated drivers over q-points, energies, and `(q, stack)` points, plus the JVP entry point.
- `general_stack`: driver for general 3x3-tensor stacks.
- `bookended`: graded book-ended film stacks (mesh, orientation and density profiles) solved in one fused call.

## Threading

Per-point solvers do no threading. The drivers take `parallel: bool` and use rayon's global pool; pass `false` from fitting loops that already parallelize to avoid oversubscription. `RAYON_NUM_THREADS` caps the pool.

## Examples

`cargo run -p refloxide-tmm --release --example kernel_precision` compares kernel formulations in `f32` and `f64`; `general_reference` checks the general kernel against a reference.

## Boundaries

Shared types come from `refloxide-core` and optical constants from `refloxide-optics`. No Python and no GPU code here.
