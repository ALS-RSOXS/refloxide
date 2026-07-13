"""Lab-frame optical-tensor primitives, backed by the Rust `refloxide.rust` extension.

Owns OOC (tabulated optical constant) interpolation, density scaling to
molecular indices, uniaxial/isotropic laboratory-frame tensor construction,
and refnx slab-row packing. These are the building blocks a `Scatterer`
subclass calls directly to compute its own energy-dependent tensor — see
`refloxide.data.OpticalConstants` for the cached table wrapper that feeds
`molecular_index_at_ooc`.

This module does not implement any of this math itself; every function here
is a thin, type-annotated re-export of the compiled extension so callers get
real docstrings and static types without importing `refloxide.rust` directly.
"""

from __future__ import annotations

from refloxide.rust import (
    interp_ooc_linear,
    isotropic_lab_tensor,
    lab_tensor_diagonals_batch,
    molecular_index_at_ooc,
    tensor_to_slab_row,
    uniaxial_lab_tensor,
)

__all__ = [
    "interp_ooc_linear",
    "isotropic_lab_tensor",
    "lab_tensor_diagonals_batch",
    "molecular_index_at_ooc",
    "tensor_to_slab_row",
    "uniaxial_lab_tensor",
]
