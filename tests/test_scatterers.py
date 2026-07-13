"""Tests for deferred scatterer primitives."""

from __future__ import annotations

import numpy as np
import pytest

from refloxide.pxr.energy.probe import Probe
from refloxide.pxr.energy.scatterers import FreeTensorScatterer, FunctionScatterer
from refloxide.pxr.plugin.structure import SLD


def test_free_tensor_shared_matches_sld_tensor() -> None:
    sld = SLD(
        np.array([1.0 - 0.01 - 0.001j, 1.0 - 0.01 - 0.001j, 1.0 - 0.02 - 0.002j]),
        symmetry="uni",
        name="film",
    )
    scatterer = FreeTensorScatterer.from_sld(sld)
    probe = Probe(base_energy_ev=250.0)
    np.testing.assert_allclose(scatterer.tensor_at(probe), sld.tensor)


def test_free_tensor_per_energy_selects_group() -> None:
    sld = SLD(1.0 - 0.01 - 0.001j, symmetry="iso", name="film")
    energies = [250.0, 284.4]
    scatterer = FreeTensorScatterer.from_sld(sld, energies=energies)
    scatterer._energy_groups[284.4].xx.value = 0.5
    probe = Probe(base_energy_ev=284.4)
    tensor = scatterer.tensor_at(probe)
    assert tensor[0, 0].real == pytest.approx(0.5)


def test_free_tensor_per_energy_raises_on_unknown_energy() -> None:
    scatterer = FreeTensorScatterer(symmetry="iso", name="film", energies=[250.0])
    with pytest.raises(ValueError, match="No FreeTensorScatterer group"):
        scatterer.tensor_at(Probe(base_energy_ev=300.0))


def test_function_scatterer_callable() -> None:
    def fn(energy_ev: float, scale: float) -> np.ndarray:
        n = complex(1.0 - scale * 1e-3, -scale * 1e-4)
        n *= 1.0 + 0.01 * energy_ev / 250.0
        return np.diag([n, n, n]).astype(np.complex128)

    scatterer = FunctionScatterer(fn, hyperparams={"scale": 1.0}, name="fn")
    tensor = scatterer.tensor_at(Probe(base_energy_ev=250.0))
    assert tensor.shape == (3, 3)
    assert np.isfinite(tensor).all()
    assert tensor[0, 0] == tensor[1, 1] == tensor[2, 2]
