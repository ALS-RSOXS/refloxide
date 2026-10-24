"""Deferred optical models for refloxide stacks.

Structure and scatterer definitions are built once; slabs and tensors resolve at
evaluation energy through :class:`~refloxide.pxr.energy.probe.Probe` without
cloning the stack. Structure-level and per-scatterer ``energy_offset``
parameters shift tabulated OOC lookup; hot interpolation and batch tensor
assembly call into :mod:`refloxide.rust`.
"""

from refloxide.pxr.energy.bookended import (
    BookendedOrientationProfile,
    EnergyBookendedOrientationDensityProfile,
    bookended_from_three_slabs,
)
from refloxide.pxr.energy.compile import SlabEnergyPlan, compile_structure
from refloxide.pxr.energy.fused import evaluate_fused_bookended_reflectivity
from refloxide.pxr.energy.migrate import upgrade_scatterer, upgrade_structure
from refloxide.pxr.energy.model import CompiledReflectivityModel, compile_model
from refloxide.pxr.energy.ooc import OocAnchor
from refloxide.pxr.energy.probe import EnergyProbe, Probe
from refloxide.pxr.energy.scatterer import (
    DeferredScatterer,
    OocUniTensorScatterer,
    RefloxideScatterer,
    TabulatedUniTensorSLD,
    bind_scatterer_energy_offset,
)
from refloxide.pxr.energy.scatterers import (
    DispersiveMaterialSLD,
    EnergyDependentMaterialSLD,
    EnergyDependentScatterer,
    EnergyDependentUniTensorSLD,
    FixedTensorScatterer,
    FreeTensorScatterer,
    FunctionScatterer,
)
from refloxide.pxr.energy.structure import (
    DispersiveStructure,
    EnergyDependentStructure,
    EnergyOrientationSlab,
    OrientationSlab,
    StackSnapshot,
)

__all__ = [
    "BookendedOrientationProfile",
    "CompiledReflectivityModel",
    "DeferredScatterer",
    "DispersiveMaterialSLD",
    "DispersiveStructure",
    "EnergyBookendedOrientationDensityProfile",
    "EnergyDependentMaterialSLD",
    "EnergyDependentScatterer",
    "EnergyDependentStructure",
    "EnergyDependentUniTensorSLD",
    "EnergyOrientationSlab",
    "EnergyProbe",
    "FixedTensorScatterer",
    "FreeTensorScatterer",
    "FunctionScatterer",
    "OocAnchor",
    "OocUniTensorScatterer",
    "OrientationSlab",
    "Probe",
    "RefloxideScatterer",
    "SlabEnergyPlan",
    "StackSnapshot",
    "TabulatedUniTensorSLD",
    "bind_scatterer_energy_offset",
    "bookended_from_three_slabs",
    "compile_model",
    "compile_structure",
    "evaluate_fused_bookended_reflectivity",
    "upgrade_scatterer",
    "upgrade_structure",
]
