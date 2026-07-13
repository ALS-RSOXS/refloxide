"""Public Python interface for refloxide.

The native extension lives in :mod:`refloxide.rust` (Rust / PyO3). Prefer the
flat package surface for new work:

* :mod:`refloxide.tmm` — uniaxial and book-ended reflectivity kernels.
* :mod:`refloxide.optics` — OOC interpolation and laboratory-frame tensors.
* :mod:`refloxide.data` — cached optical constants and reflectivity datasets.
* :mod:`refloxide.model` — energy-deferred ``Structure`` / ``ReflectModel``.
* :mod:`refloxide.objective` — multi-energy ``Objective`` for refnx fitters.

Legacy deferred-energy stacks and fitting glue remain under :mod:`refloxide.pxr`
(``pxr.energy``, ``pxr.plugin``, ``pxr.objective``) for existing notebooks until
that surface is retired.
"""

from __future__ import annotations

from refloxide import pxr

__all__ = [
    "__version__",
    "pxr",
]
__version__ = "0.1.4"
