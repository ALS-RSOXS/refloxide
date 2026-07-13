"""Optional adapters that connect refloxide kernels to external fitting stacks.

Each submodule targets one host ecosystem (for example ``pyref``). Import
adapters explicitly when wiring a workflow; they are not loaded from the
top-level :mod:`refloxide` package.
"""

from refloxide.integrations import pyref
from refloxide.integrations.pyref import (
    PyrefPatchReport,
    patch_pyref,
    patch_pyref_if_needed,
    pyref_patch_report,
    pyref_patched,
    reflectivity,
    require_pyref_patched,
    uniaxial_reflectivity,
)

__all__ = [
    "PyrefPatchReport",
    "patch_pyref",
    "patch_pyref_if_needed",
    "pyref",
    "pyref_patch_report",
    "pyref_patched",
    "reflectivity",
    "require_pyref_patched",
    "uniaxial_reflectivity",
]
