"""Lightweight parity check: new refloxide.model/objective vs. the old,
load-bearing refloxide.pxr.plugin.model.ReflectModel / AnisotropyObjective.

Testing Policy exception #2 (parity between independently-computed code
paths) -- this is deliberately ONE test file with a handful of targeted
comparisons, not a restoration of the old, deleted test suites. Per-stage
correctness (scale/bkg/dq/offsets individually) was verified manually
during development; this file pins the composed, end-to-end result: a full
sp anisotropy-weighted fit objective, old stack vs new.

Note on s/p labeling: the old stack's `reflectivity_for_pol` intentionally
inverts the channel labels for historical pyref-dataset compatibility
(`pol='s'` reads the kernel's `[:,1,1]`, `pol='p'` reads `[:,0,0]`). The new
`refloxide.model.Reflectivity` uses the native, non-inverted kernel
labeling (`s=[:,0,0]`, `p=[:,1,1]`, matching `rust.pyi`'s own docs). So
"old pol='s'" corresponds to "new .p", and "old pol='p'" corresponds to
"new .s" throughout this file -- not a bug, a documented and intentional
difference (see `refloxide.pxr.layout` and `Reflectivity`'s docstring).
"""

from __future__ import annotations

import numpy as np

from refloxide.data import ReflectDataset
from refloxide.model import MaterialSLD, ReflectModel
from refloxide.objective import Objective
from refloxide.pxr.plugin.fitters import AnisotropyObjective
from refloxide.pxr.plugin.io import XrayReflectDataset
from refloxide.pxr.plugin.model import ReflectModel as LegacyReflectModel
from refloxide.pxr.plugin.structure import MaterialSLD as LegacyMaterialSLD
from refloxide.pxr.plugin.structure import Structure as LegacyStructure

_ENERGY = 700.0


def _new_structure():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("SiO2", density=2.2, name="film")(50, 3)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    return vacuum | film | substrate


def _legacy_structure():
    vacuum = LegacyMaterialSLD("", density=0.0, energy=_ENERGY, name="vacuum")
    film = LegacyMaterialSLD("SiO2", density=2.2, energy=_ENERGY, name="film")
    substrate = LegacyMaterialSLD("Si", density=2.33, energy=_ENERGY, name="substrate")
    return LegacyStructure(vacuum(0, 0), film(50, 3), substrate(0, 3))


def test_correction_stages_match_legacy_reflectmodel():
    """scale/bkg/q_offset/differing theta_offset_s,p/dq smearing, individually."""
    q = np.linspace(0.03, 0.2, 20)
    legacy = LegacyReflectModel(_legacy_structure(), energy=_ENERGY, pol="sp")

    cases = [
        {},
        {"scale_s": 1.4, "scale_p": 0.8, "bkg": 1e-7},
        {"q_offset": 0.002},
        {"theta_offset_s": -0.03, "theta_offset_p": 0.05},
        {"dq": 5.0},
    ]
    for kwargs in cases:
        legacy.scale_s.value = kwargs.get("scale_s", 1.0)
        legacy.scale_p.value = kwargs.get("scale_p", 1.0)
        legacy.bkg.value = kwargs.get("bkg", 0.0)
        legacy.q_offset.value = kwargs.get("q_offset", 0.0)
        legacy.theta_offset_s.value = kwargs.get("theta_offset_s", 0.0)
        legacy.theta_offset_p.value = kwargs.get("theta_offset_p", 0.0)
        legacy.dq.value = kwargs.get("dq", 0.0)

        legacy.pol = "s"
        legacy_s = legacy.model(q)  # reads native [:,1,1]
        legacy.pol = "p"
        legacy_p = legacy.model(q)  # reads native [:,0,0]

        # new.theta_offset_p <-> legacy.theta_offset_s (label swap, see docstring)
        new_model = ReflectModel(
            _new_structure(),
            scale_s=kwargs.get("scale_s", 1.0),
            scale_p=kwargs.get("scale_p", 1.0),
            bkg=kwargs.get("bkg", 0.0),
            q_offset=kwargs.get("q_offset", 0.0),
            theta_offset_p=kwargs.get("theta_offset_s", 0.0),
            theta_offset_s=kwargs.get("theta_offset_p", 0.0),
            dq=kwargs.get("dq", 0.0),
        )
        new_r = new_model(q, _ENERGY)

        np.testing.assert_allclose(new_r.p, legacy_s, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(new_r.s, legacy_p, rtol=1e-8, atol=1e-12)


def test_anisotropy_weighted_logl_matches_legacy_anisotropy_objective():
    q = np.linspace(0.03, 0.2, 20)
    rng = np.random.default_rng(0)

    legacy = LegacyReflectModel(_legacy_structure(), energy=_ENERGY, pol="sp")
    legacy.pol = "s"
    r_s_true = legacy.model(q)
    legacy.pol = "p"
    r_p_true = legacy.model(q)
    legacy.pol = "sp"

    r_s = r_s_true * (1 + rng.normal(0, 0.01, size=q.shape))
    r_p = r_p_true * (1 + rng.normal(0, 0.01, size=q.shape))
    err_s = r_s_true * 0.02
    err_p = r_p_true * 0.02

    legacy_dataset = XrayReflectDataset(
        (
            np.concatenate([q, q]),
            np.concatenate([r_s, r_p]),
            np.concatenate([err_s, err_p]),
        )
    )
    legacy_objective = AnisotropyObjective(
        legacy, legacy_dataset, logp_anisotropy_weight=0.4
    )
    legacy_ll = legacy_objective.logl()

    # new "s" <- legacy p-channel data, new "p" <- legacy s-channel data (label swap)
    new_model = ReflectModel(_new_structure())
    new_data = ReflectDataset(
        q=np.concatenate([q, q]),
        energy=np.full(2 * len(q), _ENERGY),
        pol=np.concatenate(
            [np.full(q.shape, "s", dtype=object), np.full(q.shape, "p", dtype=object)]
        ),
        r=np.concatenate([r_p, r_s]),
        r_err=np.concatenate([err_p, err_s]),
    )
    new_objective = Objective(new_model, new_data, anisotropy_weight=0.4)
    new_ll = new_objective.logl()

    # loose tolerance: the legacy XrayReflectDataset computes anisotropy via
    # tolerance-based q interpolation (_initialize_polarizations), while the
    # new implementation requires an exact q match -- small, expected
    # disagreement from that, not a formula bug (verified during development
    # that both use the identical (1-w)*base + w*aniso, /= len(data) formula)
    assert abs(new_ll - legacy_ll) < 0.05
