"""Tests for experiment corrections, NC prior, and Structure indexing."""

from __future__ import annotations

import numpy as np
import pytest

from refloxide.data import ReflectDataset
from refloxide.model import MaterialSLD, ReflectModel, Slab, Structure
from refloxide.objective import LogpExtra, Objective


def _si_structure() -> Structure:
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("SiO2", density=2.2, name="film")(100, 3)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    return vacuum | film | substrate


def _dataset(energies: list[float], *, n_q: int = 20) -> ReflectDataset:
    q = np.linspace(0.03, 0.15, n_q)
    q_rows, e_rows, pol_rows, r_rows, err_rows = [], [], [], [], []
    model = ReflectModel(_si_structure(), energies=energies)
    for e in energies:
        r = model(q, e)
        for pol, col in (("s", r.s), ("p", r.p)):
            q_rows.append(q)
            e_rows.append(np.full(q.shape, e))
            pol_rows.append(np.full(q.shape, pol, dtype=object))
            r_rows.append(col)
            err_rows.append(np.maximum(col * 0.05, 1e-8))
    return ReflectDataset(
        q=np.concatenate(q_rows),
        energy=np.concatenate(e_rows),
        pol=np.concatenate(pol_rows),
        r=np.concatenate(r_rows),
        r_err=np.concatenate(err_rows),
    )


def test_reflectmodel_creates_channels_for_energies():
    energies = [250.0, 283.7, 285.1]
    model = ReflectModel(_si_structure(), energies=energies)
    assert model.corrections.energies == energies
    assert str(model.scale_s.at(285.1).name).endswith("@285p1eV")


def test_objective_syncs_missing_channels_from_dataset():
    energies = [250.0, 280.0, 285.1]
    data = _dataset(energies)
    model = ReflectModel(_si_structure())  # no energies at construct
    assert model.corrections.energies == []
    Objective(model, data)
    assert model.corrections.energies == energies


def test_per_energy_scale_affects_only_that_energy():
    energies = [250.0, 285.1]
    model = ReflectModel(_si_structure(), energies=energies)
    q = np.linspace(0.03, 0.15, 30)
    before = model(q, np.asarray(energies))
    model.scale_s.at(285.1).value = 2.0
    after = model(q, np.asarray(energies))
    np.testing.assert_allclose(after.s[:, 0], before.s[:, 0], rtol=1e-12)
    np.testing.assert_allclose(after.s[:, 1], 2.0 * before.s[:, 1], rtol=1e-12)
    np.testing.assert_allclose(after.p[:, 1], before.p[:, 1], rtol=1e-12)


def test_shared_energy_offset_shifts_oc_lookup():
    model = ReflectModel(_si_structure(), energies=[700.0])
    q = np.linspace(0.03, 0.15, 30)
    baseline = model(q, 700.0).s.copy()
    model.energy_offset.value = 5.0
    shifted = model(q, 700.0).s
    assert not np.allclose(baseline, shifted)
    model.energy_offset.value = 0.0
    np.testing.assert_allclose(model(q, 700.0).s, baseline, rtol=1e-12)


def test_nc_constraint_rejects_thin_rough_slab():
    structure = _si_structure()
    film = structure.slab("film")
    film.thick.setp(vary=True, bounds=(0.0, 200.0), value=1.0)
    film.rough.setp(vary=True, bounds=(0.0, 20.0), value=10.0)
    data = _dataset([250.0])
    model = ReflectModel(structure, energies=[250.0])
    objective = Objective(model, data, nc_constraint=True)
    assert objective.logp() == float(-np.inf)
    objective.nc_constraint = False
    assert np.isfinite(objective.logp())


def test_nc_constraint_skips_substrate_and_accepts_valid_film():
    structure = _si_structure()
    assert not structure.slab("vacuum").enforce_nevot_croce
    assert not structure.slab("substrate").enforce_nevot_croce
    assert structure.slab("film").enforce_nevot_croce
    structure.slab("substrate").rough.setp(vary=True, value=10.0, bounds=(0.0, 20.0))
    structure.slab("film").thick.setp(vary=True, value=100.0)
    structure.slab("film").rough.setp(vary=True, value=3.0)
    data = _dataset([250.0])
    objective = Objective(ReflectModel(structure, energies=[250.0]), data)
    assert np.isfinite(objective.logp())


def test_logp_extra_callable_matches_nc_toggle():
    structure = _si_structure()
    film = structure.slab("film")
    film.thick.setp(vary=True, bounds=(0.0, 200.0), value=1.0)
    film.rough.setp(vary=True, bounds=(0.0, 20.0), value=10.0)
    data = _dataset([250.0])
    model = ReflectModel(structure, energies=[250.0])
    objective = Objective(model, data, nc_constraint=True)
    assert isinstance(objective.logp_extra, LogpExtra)
    assert objective.logp_extra(model, data) == float(-np.inf)


def test_nll_rejects_nc_violation_without_nlpost():
    """NC is enforced in ``nll``, so default CurveFitter target cannot bypass it."""
    structure = _si_structure()
    film = structure.slab("film")
    film.thick.setp(vary=True, bounds=(0.0, 200.0), value=1.0)
    film.rough.setp(vary=True, bounds=(0.0, 20.0), value=10.0)
    data = _dataset([250.0])
    objective = Objective(ReflectModel(structure, energies=[250.0]), data)
    assert objective.nevot_croce_logp() == float(-np.inf)
    assert objective.logp() == float(-np.inf)
    assert np.isfinite(objective.logl())
    assert objective.nll() == float(np.inf)
    assert not np.isfinite(objective.logpost())
    objective.nc_constraint = False
    assert objective.nll() < float(np.inf)
    assert np.isfinite(objective.nll())


def test_residuals_reject_nc_for_least_squares():
    structure = _si_structure()
    film = structure.slab("film")
    film.thick.setp(vary=True, bounds=(0.0, 200.0), value=1.0)
    film.rough.setp(vary=True, bounds=(0.0, 20.0), value=10.0)
    data = _dataset([250.0])
    objective = Objective(ReflectModel(structure, energies=[250.0]), data)
    resid = objective.residuals()
    assert resid.shape == (len(data),)
    assert np.all(resid >= 1.0e12)


def test_structure_getitem_by_name_and_index():
    structure = _si_structure()
    assert structure[1] is structure["film"]
    film = structure.slab("film")
    assert isinstance(film, Slab)
    assert isinstance(film.sld, MaterialSLD)
    assert film.sld.density.value == pytest.approx(2.2)
    with pytest.raises(KeyError, match="no component"):
        _ = structure["missing"]


def test_reflectivity_channels_one_kernel_per_pol_when_thetas_differ():
    """Shared q with unequal theta still uses one materialize; both channels match."""
    structure = _si_structure()
    model = ReflectModel(structure, energies=[250.0], parallel=False)
    model.theta_offset_s.at(250.0).value = 0.02
    model.theta_offset_p.at(250.0).value = -0.01
    q = np.linspace(0.03, 0.15, 40)
    full = model(q, 250.0)
    s, p = model.reflectivity_channels_at_energy(250.0, q_s=q, q_p=q)
    assert s is not None and p is not None
    np.testing.assert_allclose(s, full.s, rtol=1e-12)
    np.testing.assert_allclose(p, full.p, rtol=1e-12)


def test_objective_predicted_handles_split_thetas_without_doubling():
    energies = [250.0, 285.1]
    data = _dataset(energies)
    model = ReflectModel(_si_structure(), energies=energies, parallel=False)
    for e in energies:
        model.theta_offset_s.at(e).value = 0.02
        model.theta_offset_p.at(e).value = -0.015
    objective = Objective(model, data)
    pred = objective._predicted()
    assert pred.shape == (len(data),)
    assert np.all(np.isfinite(pred))
    assert pred.min() >= 0.0
