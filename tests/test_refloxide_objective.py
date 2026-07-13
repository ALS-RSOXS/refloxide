"""refloxide.data.ReflectDataset and refloxide.objective.Objective.

Testing Policy exception #2 (parity between independently-computed paths):
Objective.logl() grouping/batching by (energy, pol) must sum to exactly what
computing each group's Gaussian log-likelihood by hand gives -- this can't
be verified by reading the grouping code, only by running both and
comparing.
"""

from __future__ import annotations

import numpy as np
import pytest
from refnx.analysis import CurveFitter, Transform

from refloxide.data import ReflectDataset
from refloxide.model import MaterialSLD, ReflectModel
from refloxide.objective import Objective, gaussian_logl


def _si_on_si_structure():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("SiO2", density=2.2, name="film")(50, 2)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    return vacuum | film | substrate


def test_reflectdataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        ReflectDataset(
            q=[0.1, 0.2, 0.3], energy=[700.0, 700.0], pol=["s", "s", "s"],
            r=[0.5, 0.4, 0.3], r_err=[0.01, 0.01, 0.01],
        )


def test_reflectdataset_rejects_invalid_pol():
    with pytest.raises(ValueError, match="'s' or 'p'"):
        ReflectDataset(
            q=[0.1], energy=[700.0], pol=["sp"], r=[0.5], r_err=[0.01]
        )


def test_reflectdataset_from_arrays_broadcasts_energy_and_pol():
    q = np.linspace(0.01, 0.2, 10)
    data = ReflectDataset.from_arrays(
        q, r=np.full(10, 0.5), r_err=np.full(10, 0.01), energy=700.0, pol="s"
    )
    assert len(data) == 10
    assert np.all(data.energy == 700.0)
    assert np.all(data.pol == "s")


def test_reflectdataset_groups_by_energy_and_pol():
    data = ReflectDataset(
        q=[0.1, 0.2, 0.1, 0.2, 0.1],
        energy=[700.0, 700.0, 705.0, 705.0, 700.0],
        pol=["s", "s", "s", "s", "p"],
        r=[1, 2, 3, 4, 5],
        r_err=[1, 1, 1, 1, 1],
    )
    groups = {(energy, pol): tuple(idx) for energy, pol, idx in data.groups()}
    assert set(groups) == {(700.0, "s"), (705.0, "s"), (700.0, "p")}
    assert groups[(700.0, "s")] == (0, 1)
    assert groups[(705.0, "s")] == (2, 3)
    assert groups[(700.0, "p")] == (4,)


def test_objective_logl_matches_hand_computed_gaussian_logl_single_group():
    model = ReflectModel(_si_on_si_structure())
    q = np.linspace(0.03, 0.15, 20)
    energy = 700.0

    truth = model(q, energy).s
    rng = np.random.default_rng(0)
    r = truth * (1.0 + rng.normal(0, 0.01, size=truth.shape))
    r_err = np.full_like(truth, 0.02) * truth

    data = ReflectDataset.from_arrays(q, r=r, r_err=r_err, energy=energy, pol="s")
    objective = Objective(model, data)

    expected = gaussian_logl(r, r_err, model(q, energy).s, weighted=True)
    assert objective.logl() == pytest.approx(expected)


def test_objective_logl_sums_across_energy_and_pol_groups():
    model = ReflectModel(_si_on_si_structure())
    q1 = np.linspace(0.03, 0.15, 15)
    q2 = np.linspace(0.02, 0.12, 10)
    rng = np.random.default_rng(1)

    r1_s = model(q1, 700.0).s * (1 + rng.normal(0, 0.01, size=q1.shape))
    r2_p = model(q2, 705.0).p * (1 + rng.normal(0, 0.01, size=q2.shape))
    err1 = np.full_like(q1, 0.01)
    err2 = np.full_like(q2, 0.01)

    q = np.concatenate([q1, q2])
    energy = np.concatenate([np.full_like(q1, 700.0), np.full_like(q2, 705.0)])
    pol = np.concatenate(
        [np.full(q1.shape, "s", dtype=object), np.full(q2.shape, "p", dtype=object)]
    )
    r = np.concatenate([r1_s, r2_p])
    r_err = np.concatenate([err1, err2])

    data = ReflectDataset(q=q, energy=energy, pol=pol, r=r, r_err=r_err)
    objective = Objective(model, data)

    expected_s = gaussian_logl(r1_s, err1, model(q1, 700.0).s, weighted=True)
    expected_p = gaussian_logl(r2_p, err2, model(q2, 705.0).p, weighted=True)
    expected = expected_s + expected_p
    assert objective.logl() == pytest.approx(expected)


def test_objective_transform_applies_to_data_and_model():
    model = ReflectModel(_si_on_si_structure())
    q = np.linspace(0.03, 0.15, 20)
    truth = model(q, 700.0).s
    data = ReflectDataset.from_arrays(
        q, r=truth, r_err=truth * 0.02, energy=700.0, pol="s"
    )

    plain = Objective(model, data)
    transformed = Objective(model, data, transform=Transform("logY"))

    # both should be near-zero logl for noiseless synthetic data, but the
    # transformed likelihood is not literally equal to the plain one --
    # confirm the transform is actually being applied, not silently ignored
    assert plain.logl() != pytest.approx(transformed.logl())


def test_objective_plugs_into_refnx_curvefitter():
    model = ReflectModel(_si_on_si_structure())
    q = np.linspace(0.03, 0.15, 20)
    truth = model(q, 700.0).s
    rng = np.random.default_rng(2)
    r = truth * (1 + rng.normal(0, 0.01, size=truth.shape))
    r_err = truth * 0.02

    # every other parameter defaults to vary=True too (Scatterer.__call__'s
    # own convention) -- fix everything else so the fit has one free
    # parameter to recover, not a 12-dimensional problem in a 5-iteration
    # smoke test
    structure = model.structure
    for component in structure.parameters.flattened():
        component.vary = False
    film = structure.components[1]
    film.thick.setp(vary=True, bounds=(20, 100))

    data = ReflectDataset.from_arrays(q, r=r, r_err=r_err, energy=700.0, pol="s")
    objective = Objective(model, data)
    fitter = CurveFitter(objective)

    assert len(objective.varying_parameters()) == 1
    assert objective.varying_parameters()[0] is film.thick
    result = fitter.fit(
        method="differential_evolution", maxiter=5, polish=False, seed=0
    )
    assert result is not None
    assert np.isfinite(objective.logl())
