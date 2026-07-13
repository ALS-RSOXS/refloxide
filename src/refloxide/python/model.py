"""Pure-Python modeling surface (pyref.fitting-shaped API).

Import this submodule by name; :mod:`refloxide.python` does not re-export
these symbols::

    import refloxide.python.model as py

    vacuum = py.MaterialSLD("", density=0.0, energy=250.0, name="vacuum")
    model = py.ReflectModel(vacuum(0, 0) | ..., energy=250.0, pol="sp")

Scatterers and :class:`ReflectModel` here evaluate reflectivity through
:mod:`refloxide.python.tmm` (pure Python), not the Rust kernels. Prefer
:mod:`refloxide.model` for the energy-deferred Rust-backed path.
"""

from __future__ import annotations

from refnx.analysis import CurveFitter, GlobalObjective, Objective, Transform
from refnx.dataset import ReflectDataset

from refloxide.pxr.plugin.fitters import (
    AnisotropyObjective,
    Fitter,
    LogpExtra,
)
from refloxide.pxr.plugin.io import XrayReflectDataset
from refloxide.pxr.plugin.model import ReflectModel, reflectivity
from refloxide.pxr.plugin.structure import (
    SLD,
    MaterialSLD,
    MixedMaterialSlab,
    Slab,
    Structure,
    UniTensorSLD,
)

__all__ = [
    "SLD",
    "AnisotropyObjective",
    "CurveFitter",
    "Fitter",
    "GlobalObjective",
    "LogpExtra",
    "MaterialSLD",
    "MixedMaterialSlab",
    "Objective",
    "ReflectDataset",
    "ReflectModel",
    "Slab",
    "Structure",
    "Transform",
    "UniTensorSLD",
    "XrayReflectDataset",
    "reflectivity",
]
