"""Composable optical-constant tooling: fields, Mix, DepthProfile, sources."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from refloxide.data import OpticalConstants
from refloxide.mixing import (
    mix_bruggeman,
    mix_channels,
    mix_linear,
    mix_maxwell_garnett,
)
from refloxide.model import FreeUniTensor, MaterialSLD, Mix, ReflectModel, UniTensorSLD
from refloxide.profiles import (
    CallableDepthProfile,
    CallableField,
    Constant,
    DepthProfile,
    Diffusion,
    Polynomial,
    SecondOrderTransition,
    Spline,
    lab_tensor_from_order_parameter,
)
from refloxide.sources import (
    FormulaOpticalSource,
    OpticalSource,
    lab_diagonal_from_cos2,
)


def _free(name: str = "film") -> FreeUniTensor:
    mat = FreeUniTensor(energies=[284.4], name=name)
    ch = mat.channel_at(284.4)
    ch.delta_o.value = 1.2e-3
    ch.beta_o.value = 4e-4
    ch.delta_e.value = 1.8e-3
    ch.beta_e.value = 6e-4
    return mat


def test_second_order_transition_matches_closed_form():
    t = 100.0
    field = SecondOrderTransition(
        thickness=t, bulk=0.4, top=0.15, bottom=0.9, tau_top=12.0, tau_bottom=25.0
    )
    z = np.linspace(0.0, t, 50)
    expected = (
        0.4 + (0.15 - 0.4) * np.exp(-z / 12.0) + (0.9 - 0.4) * np.exp(-(t - z) / 25.0)
    )
    np.testing.assert_allclose(field(z), expected, rtol=1e-12)


def test_diffusion_couple_and_exponential():
    t = 40.0
    couple = Diffusion(thickness=t, kind="couple", left=1.0, right=0.0, length=6.0)
    z = np.array([0.0, 20.0, 40.0])
    vals = np.asarray(couple(z), dtype=np.float64)
    assert vals[0] > vals[1] > vals[2]

    expo = Diffusion(thickness=t, kind="exponential", edge=1.0, base=0.0, length=10.0)
    assert float(expo(0.0)) == pytest.approx(1.0)
    assert float(expo(10.0)) == pytest.approx(np.exp(-1.0))

    finite = Diffusion(thickness=t, kind="finite")
    with pytest.raises(NotImplementedError):
        finite(0.0)


def test_polynomial_and_spline():
    poly = Polynomial(coeffs=[1.0, -0.02, 0.0001])
    assert float(poly(0.0)) == pytest.approx(1.0)
    assert float(poly(10.0)) == pytest.approx(1.0 - 0.2 + 0.01)

    spline = Spline(knots=[0.0, 20.0, 40.0], values=[1.0, 0.5, 0.0])
    assert float(spline(0.0)) == pytest.approx(1.0)
    assert float(spline(40.0)) == pytest.approx(0.0)


def test_lab_frame_ema_and_order_parameter_maps():
    mixed = mix_linear([1.0 + 0.1j, 2.0 + 0.2j], [0.3, 0.7])
    assert complex(mixed) == pytest.approx(0.3 * (1 + 0.1j) + 0.7 * (2 + 0.2j))

    n_o, n_e = lab_diagonal_from_cos2(1.0 + 0.1j, 2.0 + 0.2j, 0.25)
    tensor = lab_tensor_from_order_parameter(
        1.0 + 0.1j, 2.0 + 0.2j, 0.25, kind="cos2_gamma"
    )
    assert complex(tensor[0, 0]) == pytest.approx(complex(n_o))
    assert complex(tensor[2, 2]) == pytest.approx(complex(n_e))

    gamma_tensor = lab_tensor_from_order_parameter(
        1.0 + 0.1j, 2.0 + 0.2j, 0.5, kind="gamma"
    )
    assert gamma_tensor.shape == (3, 3)


def test_maxwell_garnett_and_bruggeman_limits_and_formula():
    """EMA limits and spherical MG closed form on XR delta+i*beta channels."""
    host = 0.6e-3 + 2.0e-4j
    incl = 1.1e-3 + 3.5e-4j

    assert complex(mix_maxwell_garnett(host, incl, 0.0)) == pytest.approx(complex(host))
    assert complex(mix_maxwell_garnett(host, incl, 1.0)) == pytest.approx(complex(incl))

    assert complex(mix_bruggeman(incl, host, 0.0)) == pytest.approx(complex(host))
    assert complex(mix_bruggeman(incl, host, 1.0)) == pytest.approx(complex(incl))

    phi = 0.3
    n_h = 1.0 - host.real + 1j * host.imag
    n_i = 1.0 - incl.real + 1j * incl.imag
    eps_h, eps_i = n_h * n_h, n_i * n_i
    alpha = (eps_i - eps_h) / (eps_i + 2.0 * eps_h)
    eps_mg = eps_h * (1.0 + 2.0 * phi * alpha) / (1.0 - phi * alpha)
    n_mg = np.sqrt(eps_mg)
    if n_mg.real < 0.0:
        n_mg = -n_mg
    expected_mg = (1.0 - n_mg.real) + 1j * n_mg.imag
    assert complex(mix_maxwell_garnett(host, incl, phi)) == pytest.approx(
        complex(expected_mg), rel=1e-12
    )

    gamma = (3.0 * phi - 1.0) * eps_i + (2.0 - 3.0 * phi) * eps_h
    eps_br = 0.25 * (gamma + np.sqrt(gamma * gamma + 8.0 * eps_i * eps_h))
    n_br = np.sqrt(eps_br)
    if n_br.real < 0.0:
        n_br = -n_br
    expected_br = (1.0 - n_br.real) + 1j * n_br.imag
    assert complex(mix_bruggeman(incl, host, phi)) == pytest.approx(
        complex(expected_br), rel=1e-12
    )

    # Soft XR contrast: EMA on eps stays near the linear channel mix.
    linear = complex(mix_linear([incl, host], [phi, 1.0 - phi]))
    mg = complex(
        mix_channels(
            [incl, host], [phi, 1.0 - phi], rule="maxwell_garnett", host_index=1
        )
    )
    br = complex(mix_channels([incl, host], [phi, 1.0 - phi], rule="bruggeman"))
    assert mg.real == pytest.approx(linear.real, rel=2e-4)
    assert br.real == pytest.approx(linear.real, rel=2e-4)
    assert mg.real > 0.0 and br.real > 0.0


def test_mix_homogeneous_and_depth_phi():
    a = _free("a")
    b = _free("b")
    b.channel_at(284.4).delta_o.value = 0.6e-3
    mixed = Mix([a, b], fractions=[0.3, 0.7], rule="linear", name="mix")
    tensor = mixed.tensor_at(284.4)
    expected_o = 0.3 * complex(1.2e-3, 4e-4) + 0.7 * complex(0.6e-3, 4e-4)
    assert complex(tensor[0, 0]) == pytest.approx(expected_o)

    phi = Diffusion(thickness=40.0, left=1.0, right=0.0, length=6.0, kind="couple")
    layer = DepthProfile(
        material=Mix([a, b], rule="linear"),
        thickness=40.0,
        roughness=1.0,
        phi=phi,
        n_slabs=20,
        mesh="uniform",
    )
    rows, tensors = layer.rows_and_tensors_at(284.4)
    assert rows.shape == (20, 4)
    assert tensors.shape == (20, 3, 3)
    assert complex(tensors[0, 0, 0]).real > complex(tensors[-1, 0, 0]).real


def _flat_ooc(n_xx: float, n_ixx: float, n_zz: float, n_izz: float):
    import polars as pl

    energy = np.linspace(250.0, 320.0, 12)
    return OpticalConstants.from_source(
        pl.DataFrame(
            {
                "energy": energy,
                "n_xx": np.full(energy.shape, n_xx),
                "n_ixx": np.full(energy.shape, n_ixx),
                "n_zz": np.full(energy.shape, n_zz),
                "n_izz": np.full(energy.shape, n_izz),
            }
        )
    )


def test_phi_profile_keeps_component_density_and_rotation():
    """A volume-fraction profile still evaluates each component's density and tilt."""
    energy = 284.4
    a = UniTensorSLD(
        _flat_ooc(1.2e-3, 2e-4, 1.8e-3, 3e-4),
        density=1.25,
        rotation=0.3,
        name="a",
    )
    b = UniTensorSLD(
        _flat_ooc(0.7e-3, 1e-4, 0.9e-3, 1.5e-4),
        density=1.55,
        rotation=1.1,
        name="b",
    )
    phi = SecondOrderTransition(
        thickness=40.0,
        bulk=0.5,
        top=0.8,
        bottom=0.2,
        tau_top=8.0,
        tau_bottom=12.0,
        name="phi",
    )
    layer = DepthProfile(
        material=Mix([a, b], rule="linear", name="mix"),
        thickness=40.0,
        phi=phi,
        n_slabs=6,
        mesh="uniform",
    )
    _, baseline = layer.rows_and_tensors_at(energy)
    a.density.value = 1.8
    _, scaled = layer.rows_and_tensors_at(energy)
    assert complex(scaled[0, 0, 0]).real != pytest.approx(
        complex(baseline[0, 0, 0]).real
    )
    a.density.value = 1.25
    a.rotation.value = 1.2
    _, tilted = layer.rows_and_tensors_at(energy)
    biref = tilted[:, 2, 2] - tilted[:, 0, 0]
    base_biref = baseline[:, 2, 2] - baseline[:, 0, 0]
    assert not np.allclose(biref, base_biref)


def test_mix_profile_composes_per_component_density_and_orientation():
    """Any depth law on phi, density, and gamma is applied per mix component."""
    from refloxide.optics import uniaxial_lab_tensor

    energy = 284.4
    thickness = 80.0
    rho_a = Spline(
        knots=[0.0, 40.0, 80.0],
        values=[1.1, 1.4, 1.7],
        name="rho_a",
    )
    rho_b = SecondOrderTransition(
        thickness=thickness,
        bulk=1.2,
        top=0.9,
        bottom=1.5,
        tau_top=10.0,
        tau_bottom=16.0,
        name="rho_b",
    )
    gamma_a = SecondOrderTransition(
        thickness=thickness,
        bulk=0.4,
        top=0.15,
        bottom=1.1,
        tau_top=14.0,
        tau_bottom=20.0,
        name="gamma_a",
    )
    gamma_b = Constant(0.8, name="gamma_b")
    a = UniTensorSLD(
        _flat_ooc(1.2e-3, 2.0e-4, 1.8e-3, 3.0e-4),
        density=rho_a,
        rotation=gamma_a,
        name="a",
    )
    b = UniTensorSLD(
        _flat_ooc(0.7e-3, 1.0e-4, 0.9e-3, 1.5e-4),
        density=rho_b,
        rotation=gamma_b,
        name="b",
    )
    phi = SecondOrderTransition(
        thickness=thickness,
        bulk=0.45,
        top=0.8,
        bottom=0.2,
        tau_top=12.0,
        tau_bottom=18.0,
        name="phi",
    )
    layer = DepthProfile(
        material=Mix([a, b], rule="linear", name="mix"),
        thickness=thickness,
        phi=phi,
        n_slabs=8,
        mesh="uniform",
        name="film",
    )
    names = {p.name for p in layer.parameters.flattened()}
    assert {
        "phi_top",
        "rho_a_v0",
        "rho_b_bulk",
        "gamma_a_bottom",
        "gamma_b_value",
    } <= names
    assert "a_density" not in names
    assert "b_rotation" not in names
    assert all(not vf.vary for vf in layer.material.vf)

    rows, tensors = layer.rows_and_tensors_at(energy)
    z = np.cumsum(rows[:, 0]) - 0.5 * rows[:, 0]
    for zi, tensor in zip(z, tensors, strict=True):
        fraction = float(phi.evaluate(zi, thickness=thickness))
        rho_ai = float(rho_a.evaluate(zi, thickness=thickness))
        rho_bi = float(rho_b.evaluate(zi, thickness=thickness))
        gamma_ai = float(gamma_a.evaluate(zi, thickness=thickness))
        gamma_bi = float(gamma_b.evaluate(zi, thickness=thickness))
        n_xx_a, n_zz_a = a.ooc.molecular_index_at(energy, rho_ai)
        n_xx_b, n_zz_b = b.ooc.molecular_index_at(energy, rho_bi)
        ta = np.asarray(uniaxial_lab_tensor(n_xx_a, n_zz_a, gamma_ai))
        tb = np.asarray(uniaxial_lab_tensor(n_xx_b, n_zz_b, gamma_bi))
        expected_o = fraction * ta[0, 0] + (1.0 - fraction) * tb[0, 0]
        expected_e = fraction * ta[2, 2] + (1.0 - fraction) * tb[2, 2]
        assert complex(tensor[0, 0]) == pytest.approx(complex(expected_o))
        assert complex(tensor[2, 2]) == pytest.approx(complex(expected_e))


def test_mix_profile_normalizes_a_phi_field_per_component():
    """Three or more components take one weight profile each, renormalized."""
    energy = 284.4
    thickness = 30.0
    materials = [
        UniTensorSLD(_flat_ooc(1.0e-3, 1e-4, 2.0e-3, 2e-4), density=1.0, name="a"),
        UniTensorSLD(_flat_ooc(3.0e-3, 1e-4, 3.5e-3, 2e-4), density=1.2, name="b"),
        UniTensorSLD(_flat_ooc(0.4e-3, 1e-4, 0.6e-3, 2e-4), density=0.8, name="c"),
    ]
    weights = [
        Constant(0.2, name="w0"),
        Constant(0.2, name="w1"),
        Constant(0.6, name="w2"),
    ]
    layer = DepthProfile(
        material=Mix(materials, rule="linear", name="mix"),
        thickness=thickness,
        phi=tuple(weights),
        n_slabs=2,
        mesh="uniform",
    )
    _, tensors = layer.rows_and_tensors_at(energy)
    channels = []
    for mat, weight in zip(materials, (0.2, 0.2, 0.6), strict=True):
        tensor = mat.tensor_at(energy)
        channels.append(weight * complex(tensor[0, 0]))
    assert complex(tensors[0, 0, 0]) == pytest.approx(sum(channels))


def test_depth_profile_second_order_gamma_stacks():
    mat = _free()
    eta = SecondOrderTransition(
        thickness=100.0,
        bulk=0.40,
        top=0.15,
        bottom=0.90,
        tau_top=12.0,
        tau_bottom=25.0,
    )
    layer = DepthProfile(
        material=mat,
        thickness=100.0,
        roughness=2.0,
        gamma=eta,
        n_slabs=16,
        mesh="uniform",
    )
    vac = MaterialSLD("", 0, name="vacuum")(0, 0)
    si = MaterialSLD("Si", 2.33, name="si")(0, 3.0)
    stack = vac | layer | si
    model = ReflectModel(stack, energies=[284.4], parallel=False)
    q = np.linspace(0.02, 0.15, 40)
    refl = model(q, energy=284.4)
    assert refl.s.shape == (40,)
    assert np.all(np.isfinite(refl.s))
    assert np.all(np.isfinite(refl.p))


def test_callable_field_and_depth_profile():
    mat = _free()

    def my_gamma(
        z: np.ndarray, *, g0: float, lam: float, thickness: float
    ) -> np.ndarray:
        del thickness
        return g0 * np.exp(-z / lam)

    gamma = CallableField(my_gamma, params={"g0": 0.5, "lam": 20.0})
    layer = DepthProfile(
        material=mat, thickness=120.0, gamma=gamma, n_slabs=12, mesh="uniform"
    )
    z, values = layer.channel_profile("gamma", n=50)
    assert z.shape == (50,)
    assert values[0] == pytest.approx(0.5)

    def my_tensor(
        z: np.ndarray,
        *,
        material: FreeUniTensor,
        energy_ev: float,
        g0: float,
        lam: float,
        thickness: float,
    ) -> np.ndarray:
        del thickness
        n_o, n_e = material.lab_diagonal_at(energy_ev)
        cos2 = np.cos(g0 * np.exp(-z / lam)) ** 2
        return lab_tensor_from_order_parameter(n_o, n_e, cos2, kind="cos2_gamma")

    cdp = CallableDepthProfile(
        my_tensor,
        material=mat,
        params={"g0": 0.5, "lam": 20.0},
        thickness=120.0,
        n_slabs=10,
        mesh="uniform",
    )
    rows, tensors = cdp.rows_and_tensors_at(284.4)
    assert rows.shape[0] == 10
    assert tensors.shape == (10, 3, 3)


def test_optical_source_protocol():
    formula = FormulaOpticalSource("Si")
    assert isinstance(formula, OpticalSource)
    n_xx, n_zz = formula.molecular_index_at(284.4, 2.33)
    assert n_xx == n_zz
    assert np.isfinite(n_xx.real)

    frame = {
        "energy": np.linspace(250.0, 320.0, 20),
        "n_xx": np.full(20, 1e-3),
        "n_ixx": np.full(20, 1e-4),
        "n_zz": np.full(20, 2e-3),
        "n_izz": np.full(20, 2e-4),
    }
    import polars as pl

    ooc = OpticalConstants.from_source(pl.DataFrame(frame))
    assert isinstance(ooc, OpticalSource)
    mol = ooc.molecular_index_at(284.4, 1.5)
    assert mol[0].real == pytest.approx(1.5e-3)


def test_bookended_component_emits_deprecation():
    from refloxide.model import BookendedComponent
    from refloxide.pxr.energy.bookended import BookendedOrientationProfile

    profile = BookendedOrientationProfile(
        name="film",
        total_thick=50.0,
        surface_roughness=2.0,
        density_bulk=1.5,
        density_si=1.2,
        density_vac=1.0,
        tau_si=10.0,
        tau_vac=8.0,
        alpha_bulk=0.5,
        alpha_si=0.8,
        alpha_vac=0.2,
        num_slabs=8,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BookendedComponent(profile)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_constant_field_and_mix_channels_dispatch():
    c = Constant(0.7)
    assert float(c(3.0)) == pytest.approx(0.7)
    out = mix_channels([1.0, 2.0], [0.25, 0.75], rule="linear")
    assert complex(out) == pytest.approx(1.75)


def test_free_uni_tensor_alias():
    assert FreeUniTensor is not None
    mat = FreeUniTensor(energies=[284.4], name="x")
    n_o, n_e = mat.lab_diagonal_at(284.4)
    assert n_o == 0j
    assert n_e == 0j


def test_named_depth_walk_includes_depth_profile_phi():
    a = _free("a")
    b = _free("b")
    phi = SecondOrderTransition(
        thickness=60.0, bulk=0.5, top=0.9, bottom=0.1, tau_top=8.0, tau_bottom=15.0
    )
    layer = DepthProfile(
        material=Mix([a, b], rule="linear"),
        thickness=60.0,
        phi=phi,
        n_slabs=10,
        mesh="uniform",
    )
    vac = MaterialSLD("", 0, name="vacuum")(0, 0)
    si = MaterialSLD("Si", 2.33, name="si")(0, 3.0)
    stack = vac | layer | si
    profiles = stack.named_profiles_at(num_points=200)
    assert "phi" in profiles
    mid = np.argmin(np.abs(stack.depth_grid(num_points=200) - 30.0))
    assert np.isfinite(profiles["phi"][mid])
