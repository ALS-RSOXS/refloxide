"""Public Python interface for refloxide.

The native extension lives in :mod:`refloxide._core` (Rust / PyO3). This
package re-exports the small surface intended to replace ad hoc Python
pipelines such as :func:`refloxide.pxr.uniaxial_reflectivity` once the
stack representation and Passler pipeline are wired through.

* :func:`compute_amplitudes` — one-shot amplitude solve (stack, frequency,
  incidence).
* :func:`compute_field` — field reconstruction at a depth inside a layer.
* :mod:`refloxide.batch` — NumPy broadcasting over frequency and incidence (and
  optional legacy *q* grids).

Legacy EMpy-style helpers remain under :mod:`refloxide.pxr`.
"""

from __future__ import annotations

from refloxide import batch, pxr
from refloxide._core import (
    compute_amplitudes_py,
    compute_field_py,
)

compute_amplitudes = compute_amplitudes_py
compute_field = compute_field_py

__all__ = [
    "__version__",
    "batch",
    "compute_amplitudes",
    "compute_amplitudes_py",
    "compute_field",
    "compute_field_py",
    "pxr",
]
__version__ = "0.1.0"
