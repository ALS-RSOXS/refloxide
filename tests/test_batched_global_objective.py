"""Tests for batched global reflectivity objectives."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from refnx.analysis import Parameter, Parameters

from refloxide.pxr.plugin.batched_global import (
    BatchedGlobalObjective,
    ReflectivityBatchTerm,
    _gaussian_logl,
    evaluate_reflectivity_batch,
)

pytest.importorskip("refloxide.rust")


def test_gaussian_logl_weighted() -> None:
    y = np.array([1.0, 2.0])
    model = np.array([1.1, 1.9])
    err = np.array([0.1, 0.1])
    ll = _gaussian_logl(y, err, model, weighted=True, lnsigma=None)
    assert np.isfinite(ll)


def test_batched_logl_uses_batch_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    q = np.linspace(0.02, 0.1, 8)
    terms = [
        ReflectivityBatchTerm.from_dataset(
            x=q, y=np.ones_like(q), y_err=np.ones_like(q) * 0.1, pol="s", energy=250.0
        ),
        ReflectivityBatchTerm.from_dataset(
            x=q,
            y=np.ones_like(q) * 2,
            y_err=np.ones_like(q) * 0.1,
            pol="p",
            energy=283.7,
        ),
    ]
    model = MagicMock()
    model.parameters = Parameters([Parameter(1.0, name="dummy", vary=False)])
    model.logp.return_value = 0.0
    model.scale_s.value = 1.0
    model.scale_p.value = 1.0
    model.bkg.value = 0.0
    model.dq = 0.0
    model.q_offset.value = 0.0
    model.theta_offset_s.value = 0.0
    model.theta_offset_p.value = 0.0
    model.structure.slabs.return_value = np.zeros((3, 4))
    model.structure.tensor.return_value = np.zeros((3, 3, 3), dtype=complex)
    obj = BatchedGlobalObjective(model, terms, use_weights=True)

    curves = {0: np.ones_like(q) * 1.01, 1: np.ones_like(q) * 2.02}

    def _fake_batch(*_args, **_kwargs):
        return curves

    monkeypatch.setattr(
        "refloxide.pxr.plugin.batched_global.evaluate_reflectivity_batch",
        _fake_batch,
    )
    ll = obj.logl(obj.varying_parameters().pvals)
    assert np.isfinite(ll)


def test_parallel_terms_matches_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    q = np.linspace(0.02, 0.08, 6)
    model = MagicMock()
    model.logp.return_value = 0.0
    call_count = {"n": 0}

    def _fake_eval(_model, terms, **kwargs):
        call_count["n"] += 1
        return {i: np.full_like(t.x, float(i + 1)) for i, t in enumerate(terms)}

    def _fake_term(_model, term, parallel_kernels=False):
        return np.full_like(term.x, term.energy)

    monkeypatch.setattr(
        "refloxide.pxr.plugin.batched_global._evaluate_reflectivity_term",
        _fake_term,
    )
    terms = [
        ReflectivityBatchTerm.from_dataset(
            x=q, y=np.ones_like(q), y_err=None, pol="s", energy=250.0
        ),
        ReflectivityBatchTerm.from_dataset(
            x=q, y=np.ones_like(q), y_err=None, pol="p", energy=283.7
        ),
    ]
    serial = evaluate_reflectivity_batch(model, terms, parallel_terms=False)
    parallel = evaluate_reflectivity_batch(model, terms, parallel_terms=True)
    assert serial.keys() == parallel.keys()
    for key in serial:
        assert np.allclose(serial[key], parallel[key])
