"""refloxide.model: Structure/Slab/MaterialSLD composition and numeric parity.

Testing Policy exception #2 (parity between two independently-computed code
paths): the new deferred `MaterialSLD.tensor_at` must match the existing,
already-validated `DispersiveMaterialSLD.tensor_at` (via `Probe`) — floating
point agreement here can't be verified by reading the code, only by running
both and comparing.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from refloxide import tmm
from refloxide.data import OpticalConstants
from refloxide.model import (
    FreeTensorSLD,
    MaterialSLD,
    MixedUniTensorSLD,
    ReflectModel,
    Slab,
    Structure,
    UniTensorSLD,
)
from refloxide.pxr.energy.compile import compile_structure
from refloxide.pxr.energy.probe import Probe
from refloxide.pxr.energy.scatterer import OocUniTensorScatterer
from refloxide.pxr.energy.scatterers import DispersiveMaterialSLD
from refloxide.pxr.plugin.structure import MaterialSLD as LegacyMaterialSLD


def _ooc_table() -> pl.DataFrame:
    energy = [500.0, 700.0, 900.0]
    return pl.DataFrame(
        {
            "energy": energy,
            "n_xx": [4.0e-6, 5.0e-6, 6.0e-6],
            "n_ixx": [1.0e-7, 1.2e-7, 1.4e-7],
            "n_zz": [8.0e-6, 9.0e-6, 1.0e-5],
            "n_izz": [2.0e-7, 2.2e-7, 2.4e-7],
        }
    )


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


def test_unitensorsld_pipe_composition_and_slab_row():
    ooc = _ooc_table()
    znpc_sld = UniTensorSLD(ooc, density=1.61, rotation=1.35, name="ZnPc")
    znpc = znpc_sld(191, 8.8)
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    structure = vacuum | znpc

    assert structure.components == [vacuum, znpc]
    assert isinstance(znpc, Slab)

    row = znpc.slab_row_at(700.0)
    assert row.shape == (4,)
    np.testing.assert_allclose(row[0], 191.0)
    np.testing.assert_allclose(row[3], 8.8)
    assert row[1] > 0.0  # delta positive, order 1e-6-1e-5


def test_unitensorsld_tensor_at_matches_ooc_uni_tensor_scatterer():
    energy_ev = 700.0
    ooc_polars = _ooc_table()
    ooc_pandas = ooc_polars.to_pandas()

    new = UniTensorSLD(ooc_polars, density=1.61, rotation=1.35, name="new")
    legacy = OocUniTensorScatterer(
        ooc_pandas, density=1.61, rotation=1.35, name="legacy"
    )

    new_tensor = new.tensor_at(energy_ev)
    legacy_tensor = legacy.tensor_at(Probe(base_energy_ev=energy_ev))

    np.testing.assert_allclose(new_tensor, legacy_tensor, rtol=1e-10, atol=1e-15)


def test_unitensorsld_shares_one_cached_opticalconstants_across_instances(tmp_path):
    OpticalConstants._cache.clear()
    ooc_path = tmp_path / "znpc_ooc.csv"
    _ooc_table().write_csv(ooc_path)

    surface = UniTensorSLD(str(ooc_path), density=1.61, rotation=1.35, name="surface")
    bulk = UniTensorSLD(str(ooc_path), density=1.61, rotation=0.0, name="bulk")
    interface = UniTensorSLD(
        str(ooc_path), density=1.55, rotation=0.5, name="interface"
    )

    assert surface.ooc is bulk.ooc is interface.ooc
    assert OpticalConstants.cache_size() == 1
    OpticalConstants._cache.clear()


def test_unitensorsld_accepts_pandas_or_polars_or_path(tmp_path):
    OpticalConstants._cache.clear()
    ooc_path = tmp_path / "ooc.csv"
    _ooc_table().write_csv(ooc_path)

    from_polars = UniTensorSLD(_ooc_table(), density=1.0, name="a")
    from_pandas = UniTensorSLD(_ooc_table().to_pandas(), density=1.0, name="b")
    from_path = UniTensorSLD(str(ooc_path), density=1.0, name="c")
    from_existing = UniTensorSLD(from_path.ooc, density=1.0, name="d")

    for scatterer in (from_polars, from_pandas, from_path, from_existing):
        tensor = scatterer.tensor_at(700.0)
        assert tensor.shape == (3, 3)
    assert from_path.ooc is from_existing.ooc
    OpticalConstants._cache.clear()


def _new_si_structure() -> Structure:
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("Si", density=2.33, name="film")(50, 2)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    return vacuum | film | substrate


def _legacy_si_structure():
    vac = LegacyMaterialSLD("", density=0.0, name="vac")
    film = LegacyMaterialSLD("Si", density=2.33, name="film")
    sub = LegacyMaterialSLD("Si", density=2.33, name="Si")
    return compile_structure(vac(0, 0) | film(50, 2) | sub(0, 3))


def test_reflectmodel_returns_reflectivity_with_s_and_p_attributes():
    model = ReflectModel(_new_si_structure())
    q = np.linspace(0.03, 0.15, 20)

    r = model(q, 700.0)

    assert r.s.shape == (20,)
    assert r.p.shape == (20,)
    # tuple-unpacking still works (Reflectivity is a NamedTuple)
    r_s, r_p = model(q, 700.0)
    np.testing.assert_allclose(r_s, r.s)
    np.testing.assert_allclose(r_p, r.p)


def test_reflectmodel_array_energy_shape_is_q_by_energy():
    model = ReflectModel(_new_si_structure())
    q = np.linspace(0.03, 0.15, 20)
    energies = np.array([250.0, 284.4, 700.0])

    r = model(q, energies)

    assert r.s.shape == (20, 3)
    assert r.p.shape == (20, 3)
    # each energy column matches the scalar-energy call for that energy
    for i, energy in enumerate(energies):
        scalar = model(q, float(energy))
        np.testing.assert_allclose(r.s[:, i], scalar.s, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(r.p[:, i], scalar.p, rtol=1e-10, atol=1e-12)


def test_reflectmodel_matches_raw_kernel_on_legacy_materialized_layers():
    """Parity vs the legacy DispersiveStructure's materialized layers/tensor.

    DispersiveReflectModel itself applies default instrument correction
    stages (resolution smearing, background, scale) that refloxide.model's
    ReflectModel deliberately doesn't own (see tmp/USAGE.md -- correction
    stages are out of scope for this pass), so comparing against
    DispersiveReflectModel.model() directly would be comparing smeared vs
    unsmeared curves, not a real parity check. Instead, bypass instrument
    correction on both sides: materialize the legacy DispersiveStructure's
    raw layers/tensor and feed them into the same raw
    refloxide.tmm.uniaxial_reflectivity kernel ReflectModel calls.
    """
    energy_ev = 700.0
    q = np.linspace(0.03, 0.15, 20)

    new_model = ReflectModel(_new_si_structure())
    new_curve = new_model(q, energy_ev).s

    legacy_structure = _legacy_si_structure()
    snap = legacy_structure.materialize(energy_ev, structure_offset_ev=0.0)
    legacy_refl, _tran = tmm.uniaxial_reflectivity(
        q, snap.layers, snap.tensors, energy_ev, parallel=False
    )

    np.testing.assert_allclose(new_curve, legacy_refl[:, 0, 0], rtol=1e-10, atol=1e-12)


def test_reflectmodel_setp_on_structure_affects_every_energy():
    # film (SiO2) and substrate (Si) are genuinely different materials here --
    # a Si-on-Si structure would make film thickness optically invisible
    # (no index contrast at that interface), which isn't what this test
    # means to exercise.
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("SiO2", density=2.2, name="film")(50, 2)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | film | substrate
    model = ReflectModel(structure)
    q = np.linspace(0.03, 0.15, 20)

    before = model(q, 700.0).s
    film.thick.value = 200.0  # same Parameter used at every energy, no per-energy copy
    after_700 = model(q, 700.0).s
    after_250 = model(q, 250.0).s

    assert not np.allclose(before, after_700)
    # changing thick affected the 250 eV evaluation too -- same object, not per-energy
    film.thick.value = 50.0
    reverted_250 = model(q, 250.0).s
    assert not np.allclose(after_250, reverted_250)


def test_materialize_batch_at_matches_looped_materialize_at():
    """`materialize_batch_at` must exactly match calling `materialize_at` per energy.

    Every dispersive `Scatterer.tensor_at_many` override replicates the
    scalar `tensor_at` formula (OOC interpolation, density scaling, uniaxial
    lab-frame projection) directly in numpy instead of delegating to it —
    real, independent code paths that floating-point agreement can only
    confirm by running both, not by reading either one.
    """
    energies = np.array([250.0, 275.3, 281.0, 283.7, 289.0, 300.0])

    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    ooc = _ooc_table()
    uni_film = UniTensorSLD(ooc, density=1.61, rotation=0.9, name="uni")(150, 3)
    mixed_film = MixedUniTensorSLD(
        [ooc, ooc],
        vf=[0.7, 0.3],
        rotation=[0.2, 1.0],
        density=[1.61, 1.5],
        name="mixed",
    )(80, 3)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | uni_film | mixed_film | substrate

    looped_rows, looped_tensors = zip(
        *(structure.materialize_at(float(e)) for e in energies), strict=True
    )
    looped_rows = np.stack(looped_rows, axis=0)
    looped_tensors = np.stack(looped_tensors, axis=0)

    batch_rows, batch_tensors = structure.materialize_batch_at(energies)

    np.testing.assert_allclose(batch_rows, looped_rows, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(batch_tensors, looped_tensors, rtol=1e-10, atol=1e-12)


def test_free_tensor_sld_batch_lookup_matches_looped_scalar_lookup():
    """`FreeTensorSLD.tensor_at_many`'s vectorized nearest-energy search must
    exactly match calling `tensor_at` (a brute-force `min()` scan) per energy.

    Also pins the "fixed at construction" contract this scatterer exists
    for: `ensure_energies` must not regenerate an already-registered
    channel's `Parameter` objects (which would silently disconnect a
    fitter's in-progress optimization state from what `tensor_at`/
    `tensor_at_many` actually read).
    """
    registered = [250.0, 275.0, 283.7, 285.1, 289.0]
    sld = FreeTensorSLD(registered, name="free")
    for i, e in enumerate(registered):
        channel = sld.channel_at(e)
        channel.delta_o.value = float(i + 1)
        channel.beta_o.value = 0.01 * (i + 1)
        channel.delta_e.value = 2.0 * (i + 1)
        channel.beta_e.value = 0.02 * (i + 1)

    # off-grid queries, including one past each end of the registered range
    queries = np.array([250.0001, 260.0, 279.0, 284.4, 287.0, 300.0, 100.0])
    looped = np.stack([sld.tensor_at(float(q)) for q in queries], axis=0)
    batched = sld.tensor_at_many(queries)
    np.testing.assert_array_equal(batched, looped)

    # re-registering the same energies must not recreate the Parameters --
    # a fitter's optimization state must stay attached to the same objects
    before = sld._channels[283.7]
    sld.ensure_energies(registered)
    assert sld._channels[283.7] is before
