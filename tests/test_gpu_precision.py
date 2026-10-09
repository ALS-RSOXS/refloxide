"""GPU vs CPU reflectance precision for the uniaxial recursion path.

Hardware-dependent cases skip when the extension lacks the ``gpu`` feature
or when no wgpu adapter can be acquired. Timing is never asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from refloxide.rust import uniaxial_reflectivity


def _stack_arrays() -> tuple[np.ndarray, np.ndarray]:
    layers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [100.0, 1e-6, 1e-8, 2.0],
            [50.0, 1.5e-6, 1.2e-8, 1.0],
            [0.0, 2e-6, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    vac = complex(0.0, 0.0)
    n_o = complex(1e-6, 1e-8)
    n_e = complex(1.5e-6, 1.2e-8)
    n_mid_o = complex(1.2e-6, 0.9e-8)
    n_mid_e = complex(1.8e-6, 1.4e-8)
    n_b = complex(2e-6, 0.0)
    tensor = np.array(
        [
            np.diag([vac, vac, vac]),
            np.diag([n_o, n_o, n_e]),
            np.diag([n_mid_o, n_mid_o, n_mid_e]),
            np.diag([n_b, n_b, n_b]),
        ],
        dtype=np.complex128,
    )
    return layers, tensor


def _gpu_available() -> bool:
    layers, tensor = _stack_arrays()
    q = np.array([0.05], dtype=np.float64)
    try:
        uniaxial_reflectivity(q, layers, tensor, 284.0, parallel=False, device="gpu")
    except RuntimeError as exc:
        message = str(exc).lower()
        if "gpu" in message or "adapter" in message or "device" in message:
            return False
        raise
    return True


@pytest.fixture(scope="module")
def require_gpu() -> None:
    if not _gpu_available():
        pytest.skip("GPU feature or adapter unavailable")


def test_gpu_reflectance_matches_cpu(require_gpu: None) -> None:
    layers, tensor = _stack_arrays()
    q = np.linspace(0.02, 0.25, 64)
    energy_ev = 284.4
    cpu_refl, _cpu_tran = uniaxial_reflectivity(
        q, layers, tensor, energy_ev, parallel=False, device="cpu"
    )
    gpu_refl, gpu_tran = uniaxial_reflectivity(
        q, layers, tensor, energy_ev, parallel=False, device="gpu"
    )

    assert gpu_refl.shape == cpu_refl.shape == (q.size, 2, 2)
    np.testing.assert_allclose(
        gpu_refl[:, 0, 0], cpu_refl[:, 0, 0], rtol=1e-3, atol=1e-12
    )
    np.testing.assert_allclose(
        gpu_refl[:, 1, 1], cpu_refl[:, 1, 1], rtol=1e-3, atol=1e-12
    )
    np.testing.assert_allclose(gpu_refl[:, 0, 1], 0.0, atol=0.0)
    np.testing.assert_allclose(gpu_refl[:, 1, 0], 0.0, atol=0.0)
    assert np.all(np.isnan(gpu_tran.real)) and np.all(np.isnan(gpu_tran.imag))


def test_gpu_batch_matches_cpu_single(require_gpu: None) -> None:
    from refloxide.rust import uniaxial_reflectivity_batch

    layers, tensor = _stack_arrays()
    q = np.linspace(0.03, 0.2, 32)
    energies = np.array([250.0, 284.4], dtype=np.float64)
    cpu_batch, _ = uniaxial_reflectivity_batch(
        q,
        np.stack([layers, layers], axis=0),
        np.stack([tensor, tensor], axis=0),
        energies,
        parallel=False,
        device="cpu",
    )
    gpu_batch, _ = uniaxial_reflectivity_batch(
        q,
        np.stack([layers, layers], axis=0),
        np.stack([tensor, tensor], axis=0),
        energies,
        parallel=False,
        device="gpu",
    )
    np.testing.assert_allclose(gpu_batch, cpu_batch, rtol=1e-3, atol=1e-12)
