"""Public Python interface for refloxide.

Default surface is the Rust-backed transfer-matrix method. Import modeling
and other surfaces by submodule name; they are not re-exported here.

* :mod:`refloxide.tmm` — uniaxial and book-ended reflectivity kernels (Rust).
* :mod:`refloxide.optics` — OOC interpolation and laboratory-frame tensors (Rust).
* :mod:`refloxide.data` — cached optical constants and reflectivity datasets.
* :mod:`refloxide.model` — energy-deferred ``Structure`` / ``ReflectModel``.
* :mod:`refloxide.objective` — multi-energy ``Objective`` for refnx fitters.
* :mod:`refloxide.python.tmm` — opt-in pure-Python TMM (import by submodule).
* :mod:`refloxide.python.model` — opt-in pure-Python modeling (pyref.fitting-shaped).

Legacy helpers remain under :mod:`refloxide.pxr` and emit a deprecation warning
until they relocate into the top-level modules above.
"""

from __future__ import annotations

from refloxide.tmm import (
    bookended_uniaxial_reflectivity,
    uniaxial_reflectivity,
    uniaxial_reflectivity_batch,
)

__all__ = [
    "__version__",
    "bookended_uniaxial_reflectivity",
    "uniaxial_reflectivity",
    "uniaxial_reflectivity_batch",
]
__version__ = "0.1.5"
