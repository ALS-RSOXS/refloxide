"""Tests for pyref fitting integration adapters."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from refloxide.integrations.pyref import (
    _swap_sp_block,
    reflectivity,
    uniaxial_reflectivity,
)
from refloxide.pxr.layout import (
    apply_laboratory_scales,
    reflectivity_for_pol,
    reflectmodel_layout,
)
from refloxide.pxr.stacks import Layer, Material, stack_slabs, stack_tensor
from refloxide.pxr.tjf4x4 import uniaxial_reflectivity as refloxide_kernel

pytest.importorskip("refloxide.rust")

_PYREF_UNIAXIAL = Path("/Users/hduva/projects/pyref/python/pyref/fitting/uniaxial.py")


def _load_pyref_uniaxial():
    if not _PYREF_UNIAXIAL.is_file():
        pytest.skip("pyref checkout not available at expected path")
    spec = importlib.util.spec_from_file_location("pyref_uniaxial", _PYREF_UNIAXIAL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_stack() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    energy = 250.0
    q = np.linspace(0.01, 0.12, 64)
    vac = Layer(0.0, 0.0, Material.SCALAR, complex(1.0, 0.0))
    film = Layer(120.0, 3.0, Material.SCALAR, complex(1.5, 0.01))
    back = Layer(0.0, 0.5, Material.SCALAR, complex(3.0, 0.0))
    layers = [vac, film, back]
    slabs = np.asarray(stack_slabs(layers, energy=energy), dtype=np.float64)
    tensor = np.asarray(stack_tensor(layers, energy=energy), dtype=np.complex128)
    return q, slabs, energy, tensor


def _anisotropic_stack() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    q, slabs, energy, tensor = _reference_stack()
    tensor = tensor.copy()
    tensor[1, 0, 0] = complex(1.52, 0.012)
    tensor[1, 1, 1] = complex(1.41, 0.008)
    return q, slabs, energy, tensor


def test_adapter_keeps_laboratory_jones_layout() -> None:
    q, slabs, energy, tensor = _anisotropic_stack()
    native, *_ = refloxide_kernel(q, slabs, tensor, energy)
    for use_rust in (True, False):
        adapter_refl, *_ = uniaxial_reflectivity(
            q,
            slabs,
            tensor,
            energy,
            use_rust=use_rust,
            parallel=False,
        )
        assert np.max(np.abs(adapter_refl - native)) < 1e-10


def test_reflectivity_for_pol_sp_and_ps_ordering() -> None:
    q, slabs, energy, tensor = _anisotropic_stack()
    refl, *_ = uniaxial_reflectivity(
        q,
        slabs,
        tensor,
        energy,
        use_rust=True,
        parallel=False,
    )
    q_s = np.linspace(0.02, 0.08, 16)
    q_p = np.linspace(0.09, 0.11, 12)
    qvals = np.linspace(0.02, 0.11, len(q))
    r_ss_on_qs = np.interp(q_s, qvals, refl[:, 0, 0])
    r_pp_on_qs = np.interp(q_s, qvals, refl[:, 1, 1])
    r_ss_on_qp = np.interp(q_p, qvals, refl[:, 0, 0])
    r_pp_on_qp = np.interp(q_p, qvals, refl[:, 1, 1])

    sp_out = reflectivity_for_pol("sp", refl, qvals, q_s, q_p)
    assert sp_out.shape == (len(q_s) + len(q_p),)
    assert np.max(np.abs(sp_out[: len(q_s)] - r_pp_on_qs)) < 1e-12
    assert np.max(np.abs(sp_out[len(q_s) :] - r_ss_on_qp)) < 1e-12

    ps_out = reflectivity_for_pol("ps", refl, qvals, q_p, q_s)
    assert np.max(np.abs(ps_out[: len(q_p)] - r_ss_on_qp)) < 1e-12
    assert np.max(np.abs(ps_out[len(q_p) :] - r_pp_on_qs)) < 1e-12


def test_reflectmodel_layout_legacy_pyref_diagonals() -> None:
    q, slabs, energy, tensor = _anisotropic_stack()
    native, *_ = refloxide_kernel(q, slabs, tensor, energy)
    legacy = reflectmodel_layout(native)
    assert np.max(np.abs(legacy[:, 1, 1] - native[:, 0, 0])) == 0.0
    assert np.max(np.abs(legacy[:, 0, 0] - native[:, 1, 1])) == 0.0


def test_swap_sp_block_matches_reflectmodel_layout() -> None:
    q, slabs, energy, tensor = _reference_stack()
    native, *_ = refloxide_kernel(q, slabs, tensor, energy)
    swapped = _swap_sp_block(native)
    assert np.max(np.abs(swapped - reflectmodel_layout(native))) == 0.0


def test_pyref_channel_extraction_matches_stock_reflectmodel() -> None:
    q, slabs, energy, tensor = _anisotropic_stack()
    native, *_ = refloxide_kernel(q, slabs, tensor, energy)
    adapter_refl, *_ = uniaxial_reflectivity(
        q,
        slabs,
        tensor,
        energy,
        use_rust=True,
        parallel=False,
    )
    r_s = reflectivity_for_pol("s", adapter_refl, q, q, q)
    r_p = reflectivity_for_pol("p", adapter_refl, q, q, q)
    assert np.max(np.abs(r_s - native[:, 1, 1])) < 1e-10
    assert np.max(np.abs(r_p - native[:, 0, 0])) < 1e-10


def test_patch_pyref_replaces_reflect_model_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import types

    qvals = np.linspace(0.01, 0.12, 12)
    q_s = qvals[:6]
    q_p = qvals[6:]
    refl = np.zeros((len(qvals), 2, 2), dtype=np.float64)
    refl[:, 0, 0] = 0.31
    refl[:, 1, 1] = 0.72

    class _ReflectModel:
        pol = "sp"
        __refloxide_model_patched__ = False

        def _model(self, x, p=None, x_err=None):
            del x, p, x_err
            return qvals, q_s, q_p, refl, None, []

    fake_model = types.ModuleType("pyref.fitting.model")
    fake_model.ReflectModel = _ReflectModel
    fake_uni = types.ModuleType("pyref.fitting.uniaxial")
    monkeypatch.setitem(sys.modules, "pyref", types.ModuleType("pyref"))
    monkeypatch.setitem(sys.modules, "pyref.fitting", types.ModuleType("pyref.fitting"))
    monkeypatch.setitem(sys.modules, "pyref.fitting.model", fake_model)
    monkeypatch.setitem(sys.modules, "pyref.fitting.uniaxial", fake_uni)

    import refloxide.integrations.pyref as adapter

    importlib.reload(adapter)
    adapter.patch_pyref(use_rust=True, parallel=False)

    stub = _ReflectModel()
    out = stub.model(np.concatenate([q_s, q_p]))
    assert np.allclose(out[: len(q_s)], 0.72)
    assert np.allclose(out[len(q_s) :], 0.31)


def test_reflectivity_wrapper_applies_laboratory_scales() -> None:
    q, slabs, energy, tensor = _anisotropic_stack()
    base, *_ = uniaxial_reflectivity(
        q,
        slabs,
        tensor,
        energy,
        use_rust=True,
        parallel=False,
    )
    scaled = base.copy()
    apply_laboratory_scales(scaled, scale_s=0.8, scale_p=1.2)
    wrapped = reflectivity(
        q,
        slabs,
        tensor,
        energy,
        scale_s=0.8,
        scale_p=1.2,
        use_rust=True,
        parallel=False,
    )
    assert wrapped is not None
    wrapped_refl, _, _ = wrapped
    assert np.max(np.abs(wrapped_refl - scaled)) < 1e-10


def test_reflectivity_wrapper_zero_smear() -> None:
    q, slabs, energy, tensor = _reference_stack()
    uni_refl, *_ = uniaxial_reflectivity(
        q,
        slabs,
        tensor,
        energy,
        use_rust=True,
        parallel=False,
    )
    wrapped = reflectivity(
        q,
        slabs,
        tensor,
        energy,
        use_rust=True,
        parallel=False,
    )
    assert wrapped is not None
    wrapped_refl, _, _ = wrapped
    assert np.max(np.abs(wrapped_refl - uni_refl)) < 1e-10


def test_reflectivity_rejects_non_uni_backend() -> None:
    q, slabs, energy, tensor = _reference_stack()
    with pytest.raises(ValueError, match="backend='uni'"):
        reflectivity(
            q,
            slabs,
            tensor,
            energy,
            backend="bi",  # ty: ignore[invalid-argument-type]
        )


def test_pyref_patched_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import types

    fake_model = types.ModuleType("pyref.fitting.model")
    fake_model.ReflectModel = type(
        "ReflectModel",
        (),
        {"model": lambda *a, **k: None, "__refloxide_model_patched__": False},
    )
    fake_uni = types.ModuleType("pyref.fitting.uniaxial")
    monkeypatch.setitem(sys.modules, "pyref", types.ModuleType("pyref"))
    monkeypatch.setitem(sys.modules, "pyref.fitting", types.ModuleType("pyref.fitting"))
    monkeypatch.setitem(sys.modules, "pyref.fitting.model", fake_model)
    monkeypatch.setitem(sys.modules, "pyref.fitting.uniaxial", fake_uni)

    import refloxide.integrations.pyref as adapter

    importlib.reload(adapter)
    assert not adapter.pyref_patched()
    adapter.patch_pyref(use_rust=True, parallel=False)
    assert adapter.pyref_patched()


def test_patch_pyref_replaces_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import types

    pyref_mod = _load_pyref_uniaxial()
    fake_model = type("M", (), {"ReflectModel": type("RM", (), {})})()
    fake_uni = type(
        "U",
        (),
        {"uniaxial_reflectivity": pyref_mod.uniaxial_reflectivity},
    )()
    monkeypatch.setitem(sys.modules, "pyref", types.ModuleType("pyref"))
    monkeypatch.setitem(sys.modules, "pyref.fitting", types.ModuleType("pyref.fitting"))
    monkeypatch.setitem(sys.modules, "pyref.fitting.model", fake_model)
    monkeypatch.setitem(sys.modules, "pyref.fitting.uniaxial", fake_uni)

    import refloxide.integrations.pyref as adapter

    importlib.reload(adapter)
    adapter.patch_pyref(use_rust=True, parallel=False)
    assert fake_uni.uniaxial_reflectivity is not pyref_mod.uniaxial_reflectivity
    q, slabs, energy, tensor = _anisotropic_stack()
    patched_refl, *_ = fake_uni.uniaxial_reflectivity(q, slabs, tensor, energy)
    expected, *_ = uniaxial_reflectivity(
        q,
        slabs,
        tensor,
        energy,
        use_rust=True,
        parallel=False,
    )
    assert np.max(np.abs(patched_refl - expected)) < 1e-10
