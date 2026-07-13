"""Compiled reflectivity model parity tests."""

from __future__ import annotations

import numpy as np

from refloxide.pxr.energy.compile import compile_structure
from refloxide.pxr.energy.model import compile_model
from refloxide.pxr.plugin.dispersive_model import DispersiveReflectModel
from refloxide.pxr.plugin.structure import SLD, MaterialSLD


def _structure():
    vac = SLD(1.0, symmetry="iso", name="vac")
    film = SLD(
        np.array([1.0 - 0.01 - 0.001j, 1.0 - 0.01 - 0.001j, 1.0 - 0.02 - 0.002j]),
        symmetry="uni",
        name="film",
    )
    sub = MaterialSLD("Si", density=2.33, name="Si")
    return compile_structure(vac(0, 0) | film(50, 2) | sub(0, 0))


def test_compiled_model_caches_parameters_tree() -> None:
    structure = _structure()
    compiled = compile_model(structure, [250.0, 284.4], parallel=False)
    first = compiled.parameters
    second = compiled.parameters
    assert first is second


def test_compiled_model_parity_vs_dispersive_reflect_model() -> None:
    structure = _structure()
    energies = [250.0, 284.4]
    q = np.linspace(0.03, 0.15, 20)
    legacy = DispersiveReflectModel(structure, energies, pol="s")
    compiled = compile_model(structure, energies, parallel=False)
    for energy in energies:
        legacy_curve = legacy.model(q, energy=energy, pol="s")
        new_curve = compiled.reflectivity(q, energy, pol="s")
        np.testing.assert_allclose(new_curve, legacy_curve, rtol=1e-10, atol=1e-12)
