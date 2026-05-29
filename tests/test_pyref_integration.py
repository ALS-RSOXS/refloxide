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


def test_swap_sp_block_inverts_refloxide_layout() -> None:
    q, slabs, energy, tensor = _reference_stack()
    native, *_ = refloxide_kernel(q, slabs, tensor, energy)
    swapped = _swap_sp_block(native)
    assert np.max(np.abs(swapped[:, 0, 0] - native[:, 1, 1])) == 0.0
    assert np.max(np.abs(swapped[:, 1, 1] - native[:, 0, 0])) == 0.0


def test_adapter_matches_pyref_kernel_layout() -> None:
    pyref_mod = _load_pyref_uniaxial()
    q, slabs, energy, tensor = _reference_stack()
    pyref_refl, *_ = pyref_mod.uniaxial_reflectivity(q, slabs, tensor, energy)
    for use_rust in (True, False):
        adapter_refl, *_ = uniaxial_reflectivity(
            q,
            slabs,
            tensor,
            energy,
            use_rust=use_rust,
            parallel=False,
        )
        assert np.max(np.abs(adapter_refl - pyref_refl)) < 1e-10


def test_reflectivity_wrapper_zero_smear() -> None:
    pyref_uni = _load_pyref_uniaxial()
    q, slabs, energy, tensor = _reference_stack()
    pyref_refl, *_ = pyref_uni.uniaxial_reflectivity(q, slabs, tensor, energy)
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
    assert np.max(np.abs(wrapped_refl - pyref_refl)) < 1e-10


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


def test_patch_pyref_replaces_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import types

    pyref_mod = _load_pyref_uniaxial()
    fake_model = type("M", (), {"reflectivity": object()})()
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
    q, slabs, energy, tensor = _reference_stack()
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
