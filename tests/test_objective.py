"""Reflectivity objective tests."""

from __future__ import annotations

import numpy as np
import pytest

from refloxide.pxr.energy.compile import compile_structure
from refloxide.pxr.energy.model import compile_model
from refloxide.pxr.objective import ReflectivityObjective, ReflectivityTerm
from refloxide.pxr.plugin.batched_global import (
    BatchedGlobalObjective,
    ReflectivityBatchTerm,
)
from refloxide.pxr.plugin.dispersive_model import DispersiveReflectModel
from refloxide.pxr.plugin.structure import SLD, MaterialSLD, Slab


def _setup():
    vac = SLD(1.0, symmetry="iso", name="vac")
    film = SLD(
        np.array([1.0 - 0.01 - 0.001j, 1.0 - 0.01 - 0.001j, 1.0 - 0.02 - 0.002j]),
        symmetry="uni",
        name="film",
    )
    sub = MaterialSLD("Si", density=2.33, name="Si")
    structure = compile_structure(vac(0, 0) | film(50, 2) | sub(0, 0))
    energies = [250.0, 284.4]
    q = np.linspace(0.03, 0.12, 15)
    y = np.full_like(q, 1e-4)
    y_err = np.full_like(q, 1e-5)
    return structure, energies, q, y, y_err


def test_reflectivity_objective_batched_eval_mixed_q() -> None:
    structure, energies, q, y, y_err = _setup()
    compiled = compile_model(structure, energies, parallel=False)
    q_alt = q * 1.01
    terms = [
        ReflectivityTerm(q=q, y=y, y_err=y_err, energy=energies[0], pol="s"),
        ReflectivityTerm(q=q_alt, y=y, y_err=y_err, energy=energies[1], pol="s"),
    ]
    obj = ReflectivityObjective(compiled, terms, use_weights=True)
    assert len(obj._eval_batches) == 2
    assert obj.logl() == pytest.approx(
        sum(
            ReflectivityObjective(
                compiled,
                [term],
                use_weights=True,
            ).logl()
            for term in terms
        ),
        rel=1e-9,
        abs=1e-8,
    )


def test_reflectivity_objective_logpost_parity_vs_batched_global() -> None:
    structure, energies, q, y, y_err = _setup()
    dispersive = DispersiveReflectModel(structure, energies, pol="s")
    compiled = compile_model(structure, energies, parallel=False)
    batch_terms = [
        ReflectivityBatchTerm(x=q, y=y, y_err=y_err, pol="s", energy=e, lambda_=1.0)
        for e in energies
    ]
    legacy_obj = BatchedGlobalObjective(dispersive, batch_terms, use_weights=True)
    new_terms = [
        ReflectivityTerm(q=q, y=y, y_err=y_err, energy=e, pol="s", lambda_=1.0)
        for e in energies
    ]
    new_obj = ReflectivityObjective(compiled, new_terms, use_weights=True)
    assert new_obj.logpost() == pytest.approx(legacy_obj.logl(), rel=1e-9, abs=1e-8)


def test_nevot_croce_flag_returns_neg_inf() -> None:
    vac = SLD(1.0, symmetry="iso", name="vac")
    film = SLD(1.0 - 0.01 - 0.001j, symmetry="iso", name="film")
    sub = MaterialSLD("Si", density=2.33, name="Si")
    film_slab = Slab(5.0, film, 10.0, name="film", enforce_nevot_croce=True)
    structure = compile_structure(
        vac(0, 0) | film_slab | sub(0, 0),
        default_nevot_croce=False,
    )
    assert structure.logp_nevot_croce() == -np.inf
