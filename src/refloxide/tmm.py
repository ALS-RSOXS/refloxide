"""Transfer-matrix-method reflectivity kernels, backed by the Rust extension.

Deliverables 1 and 2: the uniaxial tensor reflectivity kernel through a
layer stack, and the fused book-ended graded-film kernel. `refloxide.model`
composes these with `Structure`/`ReflectModel`; call them directly here for
benchmarking or a custom fitting loop that bypasses the `Structure` layer
entirely.

Every function here is a thin, type-annotated re-export of the compiled
extension so callers get real docstrings and static types without importing
`refloxide.rust` directly.
"""

from __future__ import annotations

from refloxide.rust import (
    bookended_uniaxial_reflectivity,
    uniaxial_reflectivity,
    uniaxial_reflectivity_batch,
)

__all__ = [
    "bookended_uniaxial_reflectivity",
    "uniaxial_reflectivity",
    "uniaxial_reflectivity_batch",
]
