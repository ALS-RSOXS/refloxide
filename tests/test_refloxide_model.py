"""refloxide.model: Structure/Slab/MaterialSLD composition and numeric parity.

Testing Policy exception #2 (parity between two independently-computed code
paths): the new deferred `MaterialSLD.tensor_at` must match the existing,
already-validated `DispersiveMaterialSLD.tensor_at` (via `Probe`) — floating
point agreement here can't be verified by reading the code, only by running
both and comparing.
"""

from __future__ import annotations

import numpy as np

from refloxide.model import MaterialSLD, Slab, Structure
from refloxide.pxr.energy.probe import Probe
from refloxide.pxr.energy.scatterers import DispersiveMaterialSLD


def test_pipe_composition_builds_structure_in_order():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("Si", density=2.33, name="film")(150, 3)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)

    structure = vacuum | film | substrate

    assert isinstance(structure, Structure)
    assert structure.components == [vacuum, film, substrate]
    assert all(isinstance(c, Slab) for c in structure.components)


def test_setp_bounds_and_constraint_use_unmodified_refnx_idiom():
    film = MaterialSLD("Si", density=2.33, name="film")(150, 3)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)

    film.thick.setp(vary=True, bounds=(100, 200))
    assert film.thick.vary is True
    assert (film.thick.bounds.lb, film.thick.bounds.ub) == (100, 200)

    substrate.rough.setp(vary=None, constraint=film.rough)
    assert substrate.rough.constraint is film.rough


def test_slab_rows_at_returns_one_row_per_component_with_sane_shape():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("Si", density=2.33, name="film")(150, 3)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | film | substrate

    rows = structure.slab_rows_at(700.0)

    assert rows.shape == (3, 4)
    # thickness column matches construction values
    np.testing.assert_allclose(rows[:, 0], [0.0, 150.0, 0.0])
    # roughness column matches construction values
    np.testing.assert_allclose(rows[:, 3], [0.0, 3.0, 3.0])
    # vacuum's delta/beta should be exactly zero (density=0)
    assert rows[0, 1] == 0.0
    assert rows[0, 2] == 0.0
    # Si film/substrate delta should be positive and order 1e-6-1e-4
    assert 0.0 < rows[1, 1] < 1e-2


def test_material_sld_tensor_at_matches_dispersive_material_sld():
    energy_ev = 700.0
    new = MaterialSLD("Si", density=2.33, name="si_new")
    legacy = DispersiveMaterialSLD("Si", density=2.33, name="si_legacy")

    new_tensor = new.tensor_at(energy_ev)
    legacy_tensor = legacy.tensor_at(Probe(base_energy_ev=energy_ev))

    np.testing.assert_allclose(new_tensor, legacy_tensor, rtol=1e-12, atol=1e-15)


def test_material_sld_tensor_at_is_energy_dependent_not_cached():
    si = MaterialSLD("Si", density=2.33, name="si")
    tensor_low = si.tensor_at(250.0)
    tensor_high = si.tensor_at(900.0)

    assert not np.allclose(tensor_low, tensor_high)


def test_energy_offset_shifts_the_effective_lookup_energy():
    plain = MaterialSLD("Si", density=2.33, name="plain")
    offset = MaterialSLD("Si", density=2.33, name="offset", energy_offset=50.0)

    np.testing.assert_allclose(
        offset.tensor_at(650.0), plain.tensor_at(700.0), rtol=1e-12, atol=1e-15
    )
