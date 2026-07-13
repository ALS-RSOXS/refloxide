"""BookendedComponent: wiring the bookended profile into refloxide.model's Structure.

Testing Policy exception #2 (parity between independently-computed code
paths): `refloxide.model.Structure.materialize_at`'s multi-row handling for
`BookendedComponent` and the legacy `pxr.plugin.structure.Structure.slabs()`/
`.tensor()` assembly are two separately-written code paths over the SAME
`BookendedOrientationProfile` instance -- agreement can only be shown by
running both, not by reading either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import refloxide.model as refloxide_model_module
from refloxide.model import BookendedComponent, MaterialSLD, ReflectModel, UniTensorSLD
from refloxide.pxr.energy.bookended import BookendedOrientationProfile
from refloxide.pxr.energy.ooc import OocAnchor
from refloxide.pxr.plugin.model import ReflectModel as LegacyReflectModel
from refloxide.pxr.plugin.structure import MaterialSLD as LegacyMaterialSLD
from refloxide.pxr.plugin.structure import Slab as LegacySlab
from refloxide.pxr.plugin.structure import Structure as LegacyStructure

_ENERGY = 283.7
_PROFILE_PARAMS = {
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
}


def _ooc_frame() -> pd.DataFrame:
    e = np.linspace(250.0, 320.0, 60)
    return pd.DataFrame(
        {
            "energy": e,
            "n_xx": 1.5 + 0.01 * (e - 275.0),
            "n_ixx": 0.02,
            "n_zz": 1.55 + 0.008 * (e - 275.0),
            "n_izz": 0.03,
        }
    )


def _shared_profile() -> BookendedOrientationProfile:
    anchor = OocAnchor.from_dataframe(_ooc_frame())
    return BookendedOrientationProfile(
        ooc=anchor,
        energy=_ENERGY,
        num_slabs=24,
        mesh_constant=0.1,
        name="ZnPc",
        **_PROFILE_PARAMS,
    )


def test_bookended_component_matches_legacy_stack_exactly():
    """Same profile instance, two independent structure/model assemblies."""
    profile = _shared_profile()
    q = np.linspace(0.01, 0.25, 32)

    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    new_structure = vacuum | BookendedComponent(profile) | substrate
    new_model = ReflectModel(new_structure)
    new_r = new_model(q, _ENERGY)

    vacuum_sld = LegacyMaterialSLD("", density=0.0, energy=_ENERGY, name="vacuum")
    si_sld = LegacyMaterialSLD("Si", density=2.33, energy=_ENERGY, name="substrate")
    legacy_structure = LegacyStructure(
        LegacySlab(0.0, vacuum_sld, 0.0, name="vacuum"),
        profile,
        LegacySlab(0.0, si_sld, 3.0, name="substrate"),
    )
    legacy_model = LegacyReflectModel(legacy_structure, energy=_ENERGY, pol="s")
    legacy_s = legacy_model.model(q)  # native kernel [:, 1, 1]
    legacy_model.pol = "p"
    legacy_p = legacy_model.model(q)  # native kernel [:, 0, 0]

    # label swap (test_legacy_parity.py convention):
    # new.s <-> legacy pol='p', new.p <-> legacy pol='s'
    np.testing.assert_allclose(new_r.s, legacy_p, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(new_r.p, legacy_s, rtol=1e-10, atol=1e-12)


def test_bookended_component_parameters_share_the_wrapped_profile():
    """Fitting through the new-model Structure mutates the SAME profile Parameters."""
    profile = _shared_profile()
    wrapped = BookendedComponent(profile)

    assert wrapped.parameters is profile.parameters
    profile.alpha_bulk.value = 1.234
    assert float(wrapped.parameters["alpha_bulk"].value) == 1.234


def test_multi_row_component_rejects_single_row_protocol_calls():
    profile = _shared_profile()
    wrapped = BookendedComponent(profile)

    try:
        wrapped.slab_row_at(_ENERGY)
    except TypeError as exc:
        assert "slab_rows_at" in str(exc)
    else:
        raise AssertionError("expected TypeError")

    try:
        wrapped.tensor_at(_ENERGY)
    except TypeError as exc:
        assert "tensor_rows_at" in str(exc)
    else:
        raise AssertionError("expected TypeError")


def test_fused_bookended_path_matches_assembled_path_exactly(monkeypatch):
    """`_plan_fused_bookended` engaging vs. not must give identical reflectivity.

    Testing Policy exception #2: the fused Rust path
    (`refloxide.tmm.bookended_uniaxial_reflectivity`, mesh + profile + tensor
    construction entirely in Rust) and the assembled path
    (`Structure.materialize_at` -> `BookendedComponent.rows_and_tensors_at`
    -> `refloxide.tmm.uniaxial_reflectivity`) are two independently-written
    code paths over the same profile -- forcing one off via monkeypatch and
    diffing against the other is the only way to show they agree.
    """
    profile = _shared_profile()
    q = np.linspace(0.01, 0.25, 32)
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    oxide = MaterialSLD("SiO2", density=2.2, name="oxide")(8, 3)
    si = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | BookendedComponent(profile) | oxide | si
    model = ReflectModel(structure)

    plan = refloxide_model_module._plan_fused_bookended(model.structure, _ENERGY)
    assert plan is not None, "expected this vacuum/profile/oxide/Si stack to qualify"

    r_fused = model(q, _ENERGY)
    monkeypatch.setattr(
        refloxide_model_module, "_plan_fused_bookended", lambda *_args, **_kwargs: None
    )
    r_assembled = model(q, _ENERGY)

    np.testing.assert_allclose(r_fused.s, r_assembled.s, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(r_fused.p, r_assembled.p, rtol=1e-10, atol=1e-12)


def test_fused_bookended_path_falls_back_for_anisotropic_backing():
    """An anisotropic substrate must disqualify the fused path, not silently misfit it.

    `bookended_uniaxial_reflectivity`'s fronting/backing rows are isotropic
    `[thickness, delta, beta, roughness]` only, unlike the assembled path's
    per-component tensor -- using it for a genuinely anisotropic backing
    would flatten that anisotropy away. `_plan_fused_bookended` must refuse
    and let `ReflectModel` fall back to the assembled path instead, which
    handles anisotropic backing correctly.
    """
    profile = _shared_profile()
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    anisotropic_substrate = UniTensorSLD(
        _ooc_frame(), density=1.61, rotation=0.5, name="aniso"
    )(0, 3)
    structure = vacuum | BookendedComponent(profile) | anisotropic_substrate

    plan = refloxide_model_module._plan_fused_bookended(structure, _ENERGY)
    assert plan is None


def test_fused_bookended_path_matches_assembled_for_array_energy(monkeypatch):
    """The array-energy batch path must also agree, fused vs. assembled."""
    profile = _shared_profile()
    q = np.linspace(0.01, 0.25, 16)
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    si = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | BookendedComponent(profile) | si
    model = ReflectModel(structure)
    energies = np.array([260.0, 280.0, 300.0])

    r_fused = model(q, energies)
    monkeypatch.setattr(
        refloxide_model_module, "_plan_fused_bookended", lambda *_args, **_kwargs: None
    )
    r_assembled = model(q, energies)

    np.testing.assert_allclose(r_fused.s, r_assembled.s, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(r_fused.p, r_assembled.p, rtol=1e-10, atol=1e-12)
