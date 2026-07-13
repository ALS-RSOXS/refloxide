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

from refloxide.integrations.pyref import uniaxial_reflectivity
from refloxide.pxr.energy.bookended import BookendedOrientationProfile
from refloxide.pxr.energy.fused import evaluate_fused_bookended_reflectivity
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


def _bookended_stack(
    num_slabs: int = 24,
) -> tuple[Structure, BookendedOrientationProfile, float]:
    energy = 283.7
    vac_sld = MaterialSLD("", density=0.0, energy=energy, name="vac")
    sio2_sld = MaterialSLD("SiO2", density=2.15, energy=energy, name="oxide")
    si_sld = MaterialSLD("Si", density=2.33, energy=energy, name="si")
    profile = BookendedOrientationProfile(
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
    refl_fused, _tran_fused = fused

    slabs = structure.slabs()
    tensor = structure.tensor(energy=energy)
    refl_assembled, _tran_assembled = uniaxial_reflectivity(
        q, slabs, tensor, energy, parallel=False
    )

    np.testing.assert_allclose(refl_fused, refl_assembled, rtol=1e-10, atol=1e-12)
