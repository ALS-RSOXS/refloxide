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
from refloxide.pxr.layout import apply_reflectmodel_scales, reflectmodel_layout
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


def test_reflectmodel_layout_maps_laboratory_channels() -> None:
    q, slabs, energy, tensor = _anisotropic_stack()
    native, *_ = refloxide_kernel(q, slabs, tensor, energy)
    layout = reflectmodel_layout(native)
    assert np.max(np.abs(layout[:, 1, 1] - native[:, 0, 0])) == 0.0
    assert np.max(np.abs(layout[:, 0, 0] - native[:, 1, 1])) == 0.0
    assert np.max(np.abs(layout[:, 1, 0] - native[:, 0, 1])) == 0.0
    assert np.max(np.abs(layout[:, 0, 1] - native[:, 1, 0])) == 0.0


def test_swap_sp_block_inverts_refloxide_layout() -> None:
    q, slabs, energy, tensor = _reference_stack()
    native, *_ = refloxide_kernel(q, slabs, tensor, energy)
    swapped = _swap_sp_block(native)
    assert np.max(np.abs(swapped[:, 0, 0] - native[:, 1, 1])) == 0.0
    assert np.max(np.abs(swapped[:, 1, 1] - native[:, 0, 0])) == 0.0


def test_adapter_assigns_laboratory_ss_to_reflectmodel_s_index() -> None:
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
        assert np.max(np.abs(adapter_refl[:, 1, 1] - native[:, 0, 0])) < 1e-10
        assert np.max(np.abs(adapter_refl[:, 0, 0] - native[:, 1, 1])) < 1e-10


def test_pyref_kernel_layout_mismatches_reflectmodel_s_channel() -> None:
    pyref_mod = _load_pyref_uniaxial()
    q, slabs, energy, tensor = _anisotropic_stack()
    pyref_refl, *_ = pyref_mod.uniaxial_reflectivity(q, slabs, tensor, energy)
    native, *_ = refloxide_kernel(q, slabs, tensor, energy)
    assert np.max(np.abs(pyref_refl[:, 1, 1] - native[:, 0, 0])) > 1e-6
    assert np.max(np.abs(pyref_refl[:, 0, 0] - native[:, 1, 1])) > 1e-6


def test_reflectmodel_pol_reads_adapter_channels() -> None:
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
    r_s = adapter_refl[:, 1, 1]
    r_p = adapter_refl[:, 0, 0]
    assert np.max(np.abs(r_s - native[:, 0, 0])) < 1e-10
    assert np.max(np.abs(r_p - native[:, 1, 1])) < 1e-10
    anisotropy = (r_p - r_s) / (r_p + r_s)
    expected = (native[:, 1, 1] - native[:, 0, 0]) / (native[:, 1, 1] + native[:, 0, 0])
    assert np.max(np.abs(anisotropy - expected)) < 1e-10


def test_reflectivity_wrapper_applies_reflectmodel_scales() -> None:
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
    apply_reflectmodel_scales(scaled, scale_s=0.8, scale_p=1.2)
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
