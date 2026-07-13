"""Fused book-ended reflectivity vs the assembled (Python-materialized) kernel.

Testing Policy exception #2 (parity between two independently-computed code
paths): floating-point agreement between the fused Rust path
(`evaluate_fused_bookended_reflectivity`) and the assembled path
(`Structure.slabs()`/`Structure.tensor()` fed through the plain uniaxial
kernel) cannot be verified by reading the code, only by running both and
comparing — so this is a committed test, not a manual check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from refloxide.pxr.energy.bookended import (
    BookendedOrientationProfile,
    bookended_from_three_slabs,
)
from refloxide.pxr.energy.fused import evaluate_fused_bookended_reflectivity
from refloxide.pxr.energy.ooc import OocAnchor
from refloxide.pxr.plugin.structure import MaterialSLD, Slab, Structure, UniTensorSLD
from refloxide.tmm import uniaxial_reflectivity


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


def _bookended_stack(
    num_slabs: int = 24,
) -> tuple[Structure, BookendedOrientationProfile, float]:
    energy = 283.7
    vac_sld = MaterialSLD("", density=0.0, energy=energy, name="vac")
    sio2_sld = MaterialSLD("SiO2", density=2.15, energy=energy, name="oxide")
    si_sld = MaterialSLD("Si", density=2.33, energy=energy, name="si")
    profile = BookendedOrientationProfile(
        ooc=OocAnchor.from_dataframe(_ooc_frame()),
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
    refl_fused, _tran_fused = fused

    slabs = structure.slabs()
    tensor = structure.tensor(energy=energy)
    refl_assembled, _tran_assembled = uniaxial_reflectivity(
        q, slabs, tensor, energy, parallel=False
    )

    np.testing.assert_allclose(refl_fused, refl_assembled, rtol=1e-10, atol=1e-12)


def test_profile_without_ooc_or_energy_raises_a_clear_error_not_silently_wrong():
    profile = BookendedOrientationProfile(
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
        num_slabs=8,
    )
    with pytest.raises(RuntimeError, match="optical constants"):
        _ = profile.anchor
    with pytest.raises(RuntimeError, match="no energy bound"):
        profile.probe_at()
    with pytest.raises(RuntimeError, match="no energy bound"):
        profile.tensor()


def test_bind_ooc_matches_eager_construction():
    energy = 283.7
    ooc = OocAnchor.from_dataframe(_ooc_frame())
    kwargs = {
        "total_thick": 120.0,
        "surface_roughness": 5.0,
        "density_bulk": 1.2,
        "density_si": 1.0,
        "density_vac": 0.85,
        "tau_si": 12.0,
        "tau_vac": 8.0,
        "alpha_bulk": 0.35,
        "alpha_si": 0.55,
        "alpha_vac": 0.15,
        "num_slabs": 8,
    }

    eager = BookendedOrientationProfile(ooc=ooc, energy=energy, **kwargs)
    deferred = BookendedOrientationProfile(**kwargs)
    deferred.bind_ooc(ooc, energy=energy)

    np.testing.assert_allclose(
        deferred.tensor(), eager.tensor(), rtol=1e-12, atol=1e-15
    )


def test_bookended_from_three_slabs_takes_exactly_three_slabs():
    energy = 283.7
    ooc = _ooc_frame()
    surface = UniTensorSLD(
        ooc, rotation=0.15, density=0.85, energy=energy, name="surf"
    )(20.0, 3.0)
    bulk = UniTensorSLD(ooc, rotation=0.35, density=1.2, energy=energy, name="bulk")(
        80.0, 0.0
    )
    interface = UniTensorSLD(
        ooc, rotation=0.55, density=1.0, energy=energy, name="interface"
    )(20.0, 0.0)

    profile = bookended_from_three_slabs(
        surface, bulk, interface, ooc, energy=energy, num_slabs=8
    )

    assert isinstance(profile, BookendedOrientationProfile)
    np.testing.assert_allclose(float(profile.total_thick.value or 0.0), 120.0)


def test_bookended_from_three_slabs_defers_ooc_and_energy_by_default():
    ooc = _ooc_frame()
    surface = UniTensorSLD(ooc, rotation=0.15, density=0.85, name="surf")(20.0, 3.0)
    bulk = UniTensorSLD(ooc, rotation=0.35, density=1.2, name="bulk")(80.0, 0.0)
    interface = UniTensorSLD(ooc, rotation=0.55, density=1.0, name="interface")(
        20.0, 0.0
    )

    profile = bookended_from_three_slabs(surface, bulk, interface, num_slabs=8)

    with pytest.raises(RuntimeError, match="no energy bound"):
        profile.tensor()
    profile.bind_ooc(OocAnchor.from_dataframe(ooc), energy=283.7)
    assert profile.tensor().shape == (8, 3, 3)
