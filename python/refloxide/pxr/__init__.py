"""Legacy polarization / energy / plugin helpers under the ``pxr`` label.

This package path is deprecated. Prefer the top-level modules
(:mod:`refloxide.tmm`, :mod:`refloxide.optics`, :mod:`refloxide.data`,
:mod:`refloxide.model`, :mod:`refloxide.objective`) and, for the pure-Python
TMM, :mod:`refloxide.python.tmm`. Contents under ``pxr`` will relocate out of
this namespace in a future release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "refloxide.pxr is deprecated and will be relocated into top-level "
    "modules (tmm, optics, data, model, objective) and "
    "refloxide.python.*; import those surfaces directly.",
    DeprecationWarning,
    stacklevel=2,
)
