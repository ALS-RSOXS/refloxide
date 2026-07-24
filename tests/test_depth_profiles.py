"""Structure.sld_profile_at / density_profile_at / orientation_profile_at.

Testing Policy exception #1 (numeric correctness that can't be eyeballed):
the erf-broadened SLD profile and the depth-walk that assigns density/
orientation to each component's own depth range are new, non-trivial
numerical constructions -- wrong indexing (off-by-one interface boundaries,
wrong roughness-to-interface pairing) would be easy to get subtly wrong and
hard to notice by inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from refloxide.model import (
    BookendedComponent,
    MaterialSLD,
    MixedUniTensorSLD,
    ReflectModel,
    UniTensorSLD,
)
from refloxide.pxr.energy.bookended import BookendedOrientationProfile
from refloxide.pxr.energy.ooc import OocAnchor

_ENERGY = 283.7


def _ooc_columns(scale: float = 1.0) -> dict[str, np.ndarray]:
    e = np.linspace(250.0, 320.0, 60)
    return {
        "energy": e,
        "n_xx": scale * (1.5 + 0.01 * (e - 275.0)),
        "n_ixx": np.full_like(e, scale * 0.02),
        "n_zz": scale * (1.55 + 0.008 * (e - 275.0)),
        "n_izz": np.full_like(e, scale * 0.03),
    }


def _ooc_frame(scale: float = 1.0) -> pd.DataFrame:
    """Pandas OOC table, for `OocAnchor.from_dataframe`."""
    return pd.DataFrame(_ooc_columns(scale))


def _ooc_polars_frame(scale: float = 1.0) -> pl.DataFrame:
    """Polars OOC table, for `UniTensorSLD`/`MixedUniTensorSLD`."""
    return pl.DataFrame(_ooc_columns(scale))


def test_density_profile_isotropic_structure_steps_between_layers():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("SiO2", density=2.2, name="film")(100, 5)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | film | substrate

    z, density = structure.density_profile_at(num_points=400)

    assert density[np.argmin(np.abs(z - (-15)))] == pytest.approx(0.0)
    assert density[np.argmin(np.abs(z - 50))] == pytest.approx(2.2)
    assert density[np.argmin(np.abs(z - 130))] == pytest.approx(2.33)


def test_orientation_profile_is_nan_for_isotropic_materials():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("SiO2", density=2.2, name="film")(100, 5)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | film | substrate

    _z, orientation = structure.orientation_profile_at(num_points=200)

    assert np.all(np.isnan(orientation))


def test_density_and_orientation_profiles_for_uni_tensor_sld():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    znpc = UniTensorSLD(_ooc_polars_frame(), density=1.61, rotation=0.9, name="znpc")
    film = znpc(150, 3)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | film | substrate

    z, density = structure.density_profile_at(num_points=600)
    _z2, orientation = structure.orientation_profile_at(num_points=600)

    mid = np.argmin(np.abs(z - 75))
    assert density[mid] == pytest.approx(1.61)
    assert orientation[mid] == pytest.approx(0.9)

    vac = np.argmin(np.abs(z - (-15)))
    assert density[vac] == pytest.approx(0.0)  # MaterialSLD("", 0) has density 0
    assert np.isnan(orientation[vac])  # vacuum has no orientation concept at all


def test_mixed_uni_tensor_sld_reports_volume_fraction_weighted_average():
    mixed = MixedUniTensorSLD(
        [_ooc_polars_frame(), _ooc_polars_frame(1.2)],
        vf=[0.7, 0.3],
        rotation=[0.2, 1.0],
        density=[1.61, 1.5],
        name="mixed",
    )
    expected_density = (0.7 * 1.61 + 0.3 * 1.5) / (0.7 + 0.3)
    expected_rotation = (0.7 * 0.2 + 0.3 * 1.0) / (0.7 + 0.3)
    assert mixed.effective_density() == pytest.approx(expected_density)
    assert mixed.effective_rotation() == pytest.approx(expected_rotation)

    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    slab = mixed(80, 3)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | slab | substrate

    z, density = structure.density_profile_at(num_points=400)
    mid = np.argmin(np.abs(z - 40))
    assert density[mid] == pytest.approx(expected_density)


def test_density_and_orientation_profiles_for_bookended_component():
    anchor = OocAnchor.from_dataframe(_ooc_frame())
    profile = BookendedOrientationProfile(
        ooc=anchor,
        energy=_ENERGY,
        num_slabs=24,
        mesh_constant=0.1,
        name="ZnPc",
        total_thick=120.0,
        surface_roughness=5.0,
        density_bulk=1.2,
        density_si=1.0,
        density_vac=0.85,
        tau_si=12.0,
        tau_vac=8.0,
        alpha_bulk=0.35,
        alpha_si=0.55,
        alpha_vac=0.15,
    )
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    oxide = MaterialSLD("SiO2", density=2.2, name="oxide")(8, 3)
    si = MaterialSLD("Si", density=2.33, name="si")(0, 3)
    structure = vacuum | BookendedComponent(profile) | oxide | si

    z, density = structure.density_profile_at(num_points=2000)
    _z2, orientation = structure.orientation_profile_at(num_points=2000)

    # near the vacuum-side book-end (depth ~0) the profile should closely
    # match the profile's own continuous function evaluated at that depth
    near_vac = np.argmin(np.abs(z - 1.0))
    np.testing.assert_allclose(
        density[near_vac], profile.local_density(z[near_vac]), rtol=1e-3
    )
    np.testing.assert_allclose(
        orientation[near_vac], profile.orientation(z[near_vac]), rtol=1e-3
    )

    # the isotropic oxide/Si layers past the film have no orientation
    past_film = np.argmin(np.abs(z - 125))
    assert density[past_film] == pytest.approx(2.2)
    assert np.isnan(orientation[past_film])

    # and vacuum, before the film, contributes neither
    before_film = np.argmin(np.abs(z - (-15)))
    assert density[before_film] == pytest.approx(0.0)
    assert np.isnan(orientation[before_film])


def test_sld_profile_matches_layer_values_far_from_any_interface():
    """Far from every interface, the erf-broadened profile equals the layer value."""
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("SiO2", density=2.2, name="film")(200, 5)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | film | substrate
    model = ReflectModel(structure)

    profile = structure.sld_profile_at(_ENERGY, num_points=2000)
    film_tensor = film.sld.tensor_at(_ENERGY)
    expected_delta = float(np.real(np.trace(film_tensor)) / 3.0)
    expected_beta = float(np.imag(np.trace(film_tensor)) / 3.0)

    mid_film = np.argmin(np.abs(profile.z - 100))  # deep inside the 200 A film
    assert profile.delta[mid_film] == pytest.approx(expected_delta, rel=1e-6)
    assert profile.beta[mid_film] == pytest.approx(expected_beta, rel=1e-6)

    # sanity: the model this profile describes still builds and evaluates
    assert model(np.linspace(0.01, 0.2, 5), _ENERGY).s.shape == (5,)


def test_sld_profile_roughness_broadens_the_interface_transition():
    """A rougher interface must produce a slower (less step-like) transition."""
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)

    film_sld = MaterialSLD("SiO2", density=2.2, name="film")
    sharp = vacuum | film_sld(100, 0.5) | substrate
    rough = vacuum | film_sld(100, 15.0) | substrate

    z = np.linspace(-30, 30, 2000)
    sharp_profile = sharp.sld_profile_at(_ENERGY, z=z)
    rough_profile = rough.sld_profile_at(_ENERGY, z=z)

    # at a fixed small offset from the vacuum/film interface (boundary at
    # z=0), the sharp (sigma=0.5) interface has already all but completed
    # its erf transition, while the rough (sigma=15) one has barely started
    idx = np.argmin(np.abs(z - 3.0))
    sharp_delta_at_3 = sharp_profile.delta[idx]
    rough_delta_at_3 = rough_profile.delta[idx]
    full_step = sharp_profile.delta[-1] - sharp_profile.delta[0]
    assert abs(sharp_delta_at_3 - sharp_profile.delta[0]) > abs(
        rough_delta_at_3 - rough_profile.delta[0]
    )
    assert full_step != 0.0  # SiO2 and vacuum really do have different delta


def test_named_profiles_at_gives_each_mixed_layer_its_own_vf_trace():
    """Two MixedUniTensorSLD layers with different vf compose into one vf trace each."""
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    surface = MixedUniTensorSLD(
        [_ooc_polars_frame(), _ooc_polars_frame(1.2)],
        vf=[0.85, 0.15],
        rotation=[0.0, 0.2],
        density=[1.61, 1.55],
        name="surface",
    )(15, 3)
    bulk = MixedUniTensorSLD(
        [_ooc_polars_frame(), _ooc_polars_frame(1.2)],
        vf=[0.25, 0.75],
        rotation=[0.9, 1.1],
        density=[1.55, 1.61],
        name="bulk",
    )(50, 0)
    si = MaterialSLD("Si", density=2.33, name="si")(0, 3)
    structure = vacuum | surface | bulk | si

    profiles = structure.named_profiles_at()
    assert set(profiles) == {"density", "orientation", "vf_0", "vf_1"}

    z = structure.depth_grid()
    in_surface = np.argmin(np.abs(z - 7))
    in_bulk = np.argmin(np.abs(z - 40))
    assert profiles["vf_0"][in_surface] == pytest.approx(0.85)
    assert profiles["vf_1"][in_surface] == pytest.approx(0.15)
    assert profiles["vf_0"][in_bulk] == pytest.approx(0.25)
    assert profiles["vf_1"][in_bulk] == pytest.approx(0.75)

    # neither vacuum nor the isotropic Si substrate defines a vf
    before = np.argmin(np.abs(z - (-15)))
    beyond = np.argmin(np.abs(z - 90))
    assert np.isnan(profiles["vf_0"][before])
    assert np.isnan(profiles["vf_0"][beyond])


def test_named_profiles_at_broadens_bookended_component_edges():
    """The two edges touching a BookendedComponent must broaden, not stay sharp.

    Regression test: `_named_depth_walk`'s Slab-to-Slab run broadening
    always treated a `BookendedComponent` as a hard break on both sides
    (it's already continuous, so nothing "runs" through it), which silently
    dropped `surface_roughness`/the neighboring `Slab.rough` from the
    vacuum/film and film/oxide edges in every `roughness=True` plot,
    regardless of their fitted value.
    """
    anchor = OocAnchor.from_dataframe(_ooc_frame())

    def build(surface_roughness: float, oxide_rough: float):
        profile = BookendedOrientationProfile(
            ooc=anchor,
            energy=_ENERGY,
            num_slabs=24,
            mesh_constant=0.1,
            name="ZnPc",
            total_thick=120.0,
            surface_roughness=surface_roughness,
            density_bulk=1.2,
            density_si=1.0,
            density_vac=0.85,
            tau_si=12.0,
            tau_vac=8.0,
            alpha_bulk=0.35,
            alpha_si=0.55,
            alpha_vac=0.15,
        )
        vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
        oxide = MaterialSLD("SiO2", density=2.2, name="oxide")(8, oxide_rough)
        si = MaterialSLD("Si", density=2.33, name="si")(0, 3)
        return vacuum | BookendedComponent(profile) | oxide | si

    z = np.linspace(-30, 140, 4000)
    sharp_structure = build(surface_roughness=16.0, oxide_rough=8.0)
    sharp = sharp_structure.named_profiles_at(z, roughness=False)["density"]
    rough = sharp_structure.named_profiles_at(z, roughness=True)["density"]

    # sharp: vacuum's density is exactly 0 right up to z=0
    before_edge = np.argmin(np.abs(z - (-2.0)))
    assert sharp[before_edge] == pytest.approx(0.0)
    # broadened: the same point already shows the film bleeding in
    assert rough[before_edge] > 0.0

    # a rougher surface broadens the vacuum/film edge further out
    rougher_structure = build(surface_roughness=30.0, oxide_rough=8.0)
    rougher = rougher_structure.named_profiles_at(z, roughness=True)["density"]
    assert rougher[before_edge] > rough[before_edge]

    # sharp: the film/oxide step is a hard jump exactly at total_thick=120
    just_past_film = np.argmin(np.abs(z - 121.0))
    assert sharp[just_past_film] == pytest.approx(2.2, rel=1e-3)
    # broadened: same point hasn't fully reached the oxide value yet
    assert rough[just_past_film] < 2.2

    # a rougher oxide interface broadens the film/oxide edge further
    rougher_oxide_structure = build(surface_roughness=16.0, oxide_rough=20.0)
    rougher_oxide = rougher_oxide_structure.named_profiles_at(z, roughness=True)[
        "density"
    ]
    assert rougher_oxide[just_past_film] < rough[just_past_film]

    # far from either edge, broadening changes essentially nothing -- "far"
    # is relative to sigma=16 here (z=60 is ~3.75 sigma from the leading
    # edge), so the residual is small but not 1e-6-small
    mid_film = np.argmin(np.abs(z - 60.0))
    assert rough[mid_film] == pytest.approx(sharp[mid_film], abs=1e-3)

    # regression guard for the exact original bug: blending against a
    # frozen edge scalar (instead of the profile's own continuous curve)
    # made the erf's midpoint-at-boundary value disagree with the
    # untouched continuous side's true value, producing a real jump in
    # `rough` exactly at the boundary grid point. No adjacent-point step
    # anywhere should come close to that -- the largest real step is at
    # the sharp step's own location, so broadened must be far smaller.
    step_size = np.max(np.abs(np.diff(rough)))
    sharp_step_size = np.max(np.abs(np.diff(sharp)))
    assert step_size < 0.1 * sharp_step_size


def test_structure_plot_oc_and_param_smoke():
    """`structure.plot.oc`/`.param` build real figures without raising."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    mixed = MixedUniTensorSLD(
        [_ooc_polars_frame(), _ooc_polars_frame(1.2)],
        vf=[0.7, 0.3],
        rotation=[0.2, 1.0],
        density=[1.61, 1.5],
        name="mixed",
    )(80, 3)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | mixed | substrate

    fig, (ax, ax_diff) = structure.plot.oc(  # ty: ignore[not-iterable]
        _ENERGY, difference=True, inset=True
    )
    assert fig is not None
    assert len(ax.get_lines()) > 0
    assert len(ax_diff.get_lines()) > 0

    fig2, (ax0, ax1) = structure.plot.oc(  # ty: ignore[not-iterable]
        _ENERGY, difference=True, inset=False
    )
    assert fig2 is not None
    assert ax0 is not ax1

    _fig3, ax3 = structure.plot.param("vf_")
    assert len(ax3.get_lines()) == 2

    with pytest.raises(ValueError, match="no depth-profile quantity matches"):
        structure.plot.param("no_such_key")


def test_depth_grid_is_shared_across_all_three_profile_methods():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("SiO2", density=2.2, name="film")(100, 5)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    structure = vacuum | film | substrate

    z = structure.depth_grid(num_points=128, pad=10.0)
    sld = structure.sld_profile_at(_ENERGY, z=z)
    _z_density, density = structure.density_profile_at(z=z)
    _z_orientation, orientation = structure.orientation_profile_at(z=z)

    np.testing.assert_array_equal(sld.z, z)
    assert density.shape == z.shape
    assert orientation.shape == z.shape
