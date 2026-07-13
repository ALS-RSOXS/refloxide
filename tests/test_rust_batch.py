"""Rust batch kernel parity tests."""

from __future__ import annotations

import numpy as np

from refloxide.rust import uniaxial_reflectivity, uniaxial_reflectivity_batch


def _stack_arrays() -> tuple[np.ndarray, np.ndarray]:
    layers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [100.0, 1e-6, 1e-8, 2.0],
            [0.0, 2e-6, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    n_o = complex(1.0 - 2e-6, -2e-8)
    n_e = complex(1.0 - 3e-6, -3e-8)
    n_b = complex(1.0 - 2e-6, 0.0)
    tensor = np.array(
        [
            np.diag([1.0, 1.0, 1.0]),
            np.diag([n_o, n_o, n_e]),
            np.diag([n_b, n_b, n_b]),
        ],
        dtype=np.complex128,
    )
    return layers, tensor


def test_batch_matches_sequential() -> None:
    q = np.linspace(0.02, 0.2, 12)
    layers, tensor = _stack_arrays()
    energies = np.array([250.0, 284.4], dtype=np.float64)
    sequential = [
        uniaxial_reflectivity(q, layers, tensor, float(e), parallel=False)[0]
        for e in energies
    ]
    batch_refl, _batch_tran = uniaxial_reflectivity_batch(
        q,
        np.stack([layers, layers], axis=0),
        np.stack([tensor, tensor], axis=0),
        energies,
        parallel=False,
    )
    assert batch_refl.shape == (2, q.size, 2, 2)
    for idx, one in enumerate(sequential):
        np.testing.assert_allclose(batch_refl[idx], one, rtol=0.0, atol=0.0)
