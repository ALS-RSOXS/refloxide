"""Fused book-ended reflectivity: correctness and benchmark vs Python assembly."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from refloxide.pxr.energy.bookended import EnergyBookendedOrientationDensityProfile
from refloxide.pxr.energy.fused import (
    evaluate_fused_bookended_reflectivity,
)
from refloxide.pxr.energy.ooc import OocAnchor
from refloxide.pxr.plugin.structure import MaterialSLD, Slab, Structure


def _ooc_frame() -> pd.DataFrame:
    e = np.linspace(250.0, 300.0, 40)
    return pd.DataFrame(
        {
            "energy": e,
            "n_xx": 1.5 + 0.01 * (e - 275.0),
            "n_ixx": 0.02,
            "n_zz": 1.55 + 0.008 * (e - 275.0),
            "n_izz": 0.03,
        }
    )


def _bookended_stack(num_slabs: int = 100) -> tuple:
    energy = 283.7
    vac_sld = MaterialSLD("", density=0.0, energy=energy, name="vac")
    sio2_sld = MaterialSLD("SiO2", density=2.15, energy=energy, name="oxide")
    si_sld = MaterialSLD("Si", density=2.33, energy=energy, name="si")
    profile = EnergyBookendedOrientationDensityProfile(
        OocAnchor.from_dataframe(_ooc_frame()),
        total_thick=120.0,
        surface_roughness=5.0,
        density_bulk=1.2,
        density_si=1.0,
        density_vac=0.85,
        tau_si=12.0,
        tau_vac=8.0,
        alpha_bulk=0.35,
        alpha_si=0.55,
        alpha_vac=0.15,
        energy=energy,
        num_slabs=num_slabs,
        mesh_constant=0.1,
    )
    structure = Structure(
        Slab(0.0, vac_sld, 0.0, name="vac"),
        profile,
        Slab(8.0, sio2_sld, 6.0, name="oxide"),
        Slab(0.0, si_sld, 3.0, name="si"),
    )
    return structure, profile, energy


def test_fused_matches_assembled_kernel() -> None:
    structure, _profile, energy = _bookended_stack(num_slabs=24)
    q = np.linspace(0.01, 0.25, 32)
    fused = evaluate_fused_bookended_reflectivity(q, structure, energy, parallel=False)
    assert fused is not None
    refl_fused, _ = fused

    from refloxide.integrations.pyref import uniaxial_reflectivity

    slabs = structure.slabs()
    tensor = structure.tensor(energy=energy)
    refl_slow, _ = uniaxial_reflectivity(q, slabs, tensor, energy, parallel=False)
    np.testing.assert_allclose(refl_fused, refl_slow, rtol=1e-10, atol=1e-12)


def test_slabs_calls_tensor_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _, profile, _energy = _bookended_stack(num_slabs=12)
    calls = 0
    original = profile.tensor

    def counted_tensor(energy_arg=None):
        nonlocal calls
        calls += 1
        return original(energy_arg)

    monkeypatch.setattr(profile, "tensor", counted_tensor)
    profile.slabs()
    assert calls == 1


def test_legacy_triple_tensor_slabs_slower_than_fused() -> None:
    """Profile ``slabs()`` that re-entered ``tensor()`` via delta/beta was ~3x work."""
    _, profile, energy = _bookended_stack(num_slabs=50)

    def legacy_slabs() -> np.ndarray:
        thicknesses = profile.slab_thick
        tens = profile.tensor(energy)
        iso = np.trace(tens, axis1=1, axis2=2)
        out = np.zeros((profile.num_slabs, 4))
        out[:, 0] = thicknesses
        out[:, 1] = np.real(iso)
        out[:, 2] = np.imag(iso)
        out[0, 3] = float(profile.surface_roughness.value or 0.0)
        return out

    t0 = time.perf_counter()
    for _ in range(30):
        legacy_slabs()
        profile.tensor(energy)
    legacy_ms = (time.perf_counter() - t0) / 30 * 1000

    t1 = time.perf_counter()
    for _ in range(30):
        profile.slabs()
    fixed_ms = (time.perf_counter() - t1) / 30 * 1000

    assert fixed_ms < legacy_ms * 0.6


@pytest.mark.slow
def test_fused_benchmark_100_slabs() -> None:
    structure, _, energy = _bookended_stack(num_slabs=100)
    q = np.linspace(0.01, 0.28, 200)
    from refloxide.integrations.pyref import uniaxial_reflectivity

    t0 = time.perf_counter()
    for _ in range(5):
        slabs = structure.slabs()
        tensor = structure.tensor(energy=energy)
        uniaxial_reflectivity(q, slabs, tensor, energy, parallel=False)
    slow_ms = (time.perf_counter() - t0) / 5 * 1000

    t1 = time.perf_counter()
    for _ in range(20):
        out = evaluate_fused_bookended_reflectivity(
            q, structure, energy, parallel=False
        )
        assert out is not None
    fast_ms = (time.perf_counter() - t1) / 20 * 1000

    assert fast_ms < 100.0
    print(
        f"assembled={slow_ms:.2f}ms fused={fast_ms:.2f}ms "
        f"ratio={slow_ms / fast_ms:.2f}x"
    )
