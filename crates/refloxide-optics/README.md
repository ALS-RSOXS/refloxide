# refloxide-optics

Optical constants and laboratory-frame tensors for uniaxial stacks.

## Contents

- `optics`: piecewise-linear interpolation of tabulated optical constants on a sorted energy grid (`interpolate_ooc_linear`, clamped at the ends), and assembly of diagonal uniaxial lab-frame tensors from molecular-frame indices and orientation angles (`lab_diagonal_uniaxial`, `lab_diagonal_uniaxial_batch`, `pack_diagonal_tensors`, `isotropic_tensor`).
- `sld`: single-point density scaling and slab-row packing (`molecular_index`, `molecular_index_at_ooc`, `uniaxial_lab_tensor`, `isotropic_lab_tensor`, `tensor_to_slab_row`).

## Boundaries

Depends only on `num-complex`. It does not solve reflectivity, validate stacks, cache per-energy results, or touch Python or the GPU. Tensor entries are `delta + i*beta` per principal axis, the convention the kernels in `refloxide-tmm` expect.
