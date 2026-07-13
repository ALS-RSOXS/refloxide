"""Tests for structure compilation."""

from __future__ import annotations

import numpy as np

from refloxide.pxr.energy.compile import compile_structure
from refloxide.pxr.energy.migrate import upgrade_structure
from refloxide.pxr.energy.scatterers import FreeTensorScatterer
from refloxide.pxr.plugin.structure import SLD, MaterialSLD


def _plugin_stack() -> object:
    vac = SLD(1.0, symmetry="iso", name="vac")
    film = SLD(
        np.array([1.0 - 0.01 - 0.001j, 1.0 - 0.01 - 0.001j, 1.0 - 0.02 - 0.002j]),
        symmetry="uni",
        name="film",
    )
    sub = MaterialSLD("Si", density=2.33, name="Si")
    return vac(0, 0) | film(50, 2) | sub(0, 0)


def test_compile_structure_matches_upgrade_structure() -> None:
    plugin = _plugin_stack()
    upgraded = upgrade_structure(plugin)
    compiled = compile_structure(plugin)
    snap_up = upgraded.materialize(250.0)
    snap_co = compiled.materialize(250.0)
    np.testing.assert_allclose(snap_up.layers, snap_co.layers)
    np.testing.assert_allclose(snap_up.tensors, snap_co.tensors)


def test_compile_structure_upgrades_sld_to_free_tensor() -> None:
    compiled = compile_structure(_plugin_stack())
    slabs = [c for c in compiled.components if hasattr(c, "sld")]
    film = next(c for c in slabs if c.name == "film")
    assert isinstance(film.sld, FreeTensorScatterer)
