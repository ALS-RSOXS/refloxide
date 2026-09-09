"""Fresnel / Brewster locks for physical s/p channel naming."""

from __future__ import annotations

import numpy as np
from periodictable.xsf import index_of_refraction

from refloxide.model import MaterialSLD, ReflectModel

_HC_EV_A = 12398.42
_ENERGY_EV = 250.0


def _fresnel_power(n: complex, theta_grazing_deg: float) -> tuple[float, float]:
    """Vacuum -> ``n`` Fresnel power reflectance at a grazing angle."""
    theta_normal = np.deg2rad(90.0 - theta_grazing_deg)
    cos_i = np.cos(theta_normal)
    sin_i = np.sin(theta_normal)
    cos_t = np.sqrt(1.0 - (sin_i / n) ** 2)
    r_s = (cos_i - n * cos_t) / (cos_i + n * cos_t)
    r_p = (n * cos_i - cos_t) / (n * cos_i + cos_t)
    return float(abs(r_s) ** 2), float(abs(r_p) ** 2)


def test_reflectmodel_channels_match_fresnel_at_brewster():
    """``Reflectivity.p`` is physical R_pp (Brewster); ``.s`` is physical R_ss."""
    n_si = complex(index_of_refraction("Si", density=2.33, energy=_ENERGY_EV * 1e-3))
    theta_b = 90.0 - float(np.rad2deg(np.arctan(n_si.real)))
    wavelength_a = _HC_EV_A / _ENERGY_EV
    q = np.array(
        [(4.0 * np.pi / wavelength_a) * np.sin(np.deg2rad(theta_b))],
        dtype=np.float64,
    )

    structure = MaterialSLD("", 0.0, name="vacuum")(0.0, 0.0) | MaterialSLD(
        "Si", density=2.33, name="silicon"
    )(0.0, 0.0)
    r = ReflectModel(structure)(q, _ENERGY_EV)
    fresnel_s, fresnel_p = _fresnel_power(n_si, theta_b)

    assert fresnel_p < 0.05 * fresnel_s
    assert float(r.p[0]) < 0.05 * float(r.s[0])
    np.testing.assert_allclose(r.s[0], fresnel_s, rtol=5e-2, atol=1e-8)
    np.testing.assert_allclose(r.p[0], fresnel_p, rtol=5e-2, atol=1e-8)


def test_fused_bookended_matches_assembled_with_energy_offset(monkeypatch):
    """Fused path must use nominal energy for wavelength when OC is shifted."""
    import pandas as pd

    import refloxide.model as model_module
    from refloxide.model import BookendedComponent
    from refloxide.pxr.energy.bookended import BookendedOrientationProfile
    from refloxide.pxr.energy.ooc import OocAnchor

    e = np.linspace(250.0, 320.0, 60)
    frame = pd.DataFrame(
        {
            "energy": e,
            "n_xx": 1.5 + 0.01 * (e - 275.0),
            "n_ixx": 0.02,
            "n_zz": 1.55 + 0.008 * (e - 275.0),
            "n_izz": 0.03,
        }
    )
    profile = BookendedOrientationProfile(
        ooc=OocAnchor.from_dataframe(frame),
        energy=283.7,
        num_slabs=24,
        mesh_constant=0.1,
        name="ZnPc",
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
    )
    structure = (
        MaterialSLD("", 0)(0, 0)
        | BookendedComponent(profile)
        | MaterialSLD("Si", 2.33)(0, 3)
    )
    model = ReflectModel(structure)
    model.energy_offset.setp(0.5)
    q = np.linspace(0.01, 0.25, 64)

    r_fused = model(q, 283.7)
    monkeypatch.setattr(
        model_module, "_plan_fused_bookended", lambda *_args, **_kwargs: None
    )
    r_assembled = model(q, 283.7)

    np.testing.assert_allclose(r_fused.s, r_assembled.s, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(r_fused.p, r_assembled.p, rtol=1e-10, atol=1e-12)
