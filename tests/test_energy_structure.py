"""Tests for deferred-energy structures and Rust OOC helpers."""

from __future__ import annotations

import numpy as np
import pytest

from refloxide.pxr.energy import (
    EnergyDependentMaterialSLD,
    EnergyDependentStructure,
    EnergyDependentUniTensorSLD,
    EnergyProbe,
    OocAnchor,
    upgrade_scatterer,
    upgrade_structure,
)
from refloxide.pxr.energy.orientation import (
    AdaptiveOrientationScatterer,
    bookended_orientation_angles,
)
from refloxide.pxr.energy.structure import EnergyOrientationSlab, StackSnapshot
from refloxide.pxr.plugin.structure import MaterialSLD, Structure, UniTensorSLD


def _sample_anchor() -> OocAnchor:
    return OocAnchor(
        energy_ev=np.array([280.0, 285.0, 290.0]),
        n_xx=np.array([0.01, 0.02, 0.03]),
        n_ixx=np.array([0.001, 0.002, 0.003]),
        n_zz=np.array([0.02, 0.03, 0.04]),
        n_izz=np.array([0.002, 0.003, 0.004]),
    )


def test_ooc_anchor_at_grid_point() -> None:
    anchor = _sample_anchor()
    vals = anchor.values_at(285.0)
    assert vals[0] == pytest.approx(0.02)
    assert vals[2] == pytest.approx(0.03)


def test_energy_probe_effective_ev() -> None:
    probe = EnergyProbe(284.0, structure_offset_ev=0.5, component_offset_ev=-0.1)
    assert probe.effective_ev == pytest.approx(284.4)


def test_material_sld_migration_and_materialize() -> None:
    vac = MaterialSLD("", density=1.0, energy=250.0, name="vac")
    si = MaterialSLD("Si", density=2.33, energy=250.0, name="si")
    stack = Structure(vac(0, 0), si(10, 1.0), name="t")
    ed = upgrade_structure(stack)
    snap = ed.materialize(284.0)
    assert isinstance(snap, StackSnapshot)
    assert snap.layers.shape[0] == snap.tensors.shape[0]
    assert snap.tensors.shape[1:] == (3, 3)


def test_unitensor_migration_changes_with_energy() -> None:
    anchor = _sample_anchor()
    legacy = UniTensorSLD(
        anchor.to_dataframe(),
        density=1.2,
        rotation=0.1,
        energy=280.0,
        name="film",
    )
    upgraded = upgrade_scatterer(legacy)
    assert isinstance(upgraded, EnergyDependentUniTensorSLD)
    t_lo = upgraded.tensor_at(EnergyProbe(280.0))
    t_hi = upgraded.tensor_at(EnergyProbe(290.0))
    assert not np.allclose(t_lo, t_hi)


def test_bookended_profile_tensor_shape() -> None:
    from refloxide.pxr.energy.bookended import EnergyBookendedOrientationDensityProfile

    anchor = _sample_anchor()
    prof = EnergyBookendedOrientationDensityProfile(
        anchor,
        total_thick=180.0,
        surface_roughness=3.0,
        density_bulk=1.4,
        density_si=1.3,
        density_vac=1.2,
        tau_si=10.0,
        tau_vac=8.0,
        alpha_bulk=0.5,
        alpha_si=0.8,
        alpha_vac=1.0,
        energy=285.0,
        num_slabs=12,
    )
    t = prof.tensor(285.0)
    assert t.shape == (12, 3, 3)
    slabs = prof.slabs()
    assert slabs.shape == (12, 4)


def test_orientation_slab_batch_rust() -> None:
    scatterer = AdaptiveOrientationScatterer(
        _sample_anchor(),
        bookended_orientation_angles(4, 0.0, np.pi / 4),
        density=1.0,
        name="oriented",
    )
    slab = scatterer.orientation_slab(40.0, 2.0, name="oriented")
    assert isinstance(slab, EnergyOrientationSlab)
    vac = EnergyDependentMaterialSLD("", density=1.0, name="vac")
    ed = EnergyDependentStructure(vac(0, 0), slab, name="stack")
    snap = ed.materialize(285.0)
    assert snap.tensors.shape[0] >= 5
