"""Regression tests against the uniaxial single-interface closed form.

The uniaxial substrate with optic axis along ``z_hat`` is the simplest
birefringent test that stresses the Xu piecewise eigenvector extraction
and the ``o/e`` relabeling discussed in ``docs/theory/interface_matrices.md``
and ``docs/theory/reflection_transmission.md``. At this geometry the
ordinary ray remains pure ``s`` and the extraordinary ray remains pure
``p``, so the expected cross-polarization amplitudes vanish.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.regression._adapter import (
    KernelNotAvailableError,
    compute_amplitudes,
)
from tests.regression._analytic import fresnel_uniaxial_optic_axis_z
from tests.regression._stacks import uniaxial_substrate_stack

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@pytest.mark.regression
@pytest.mark.parametrize(
    ("eps_ordinary", "eps_extraordinary"),
    [
        (2.25 + 0.0j, 2.89 + 0.0j),
        (4.0 + 0.0j, 3.0 + 0.0j),
        (6.5 + 0.1j, 8.2 + 0.15j),
    ],
    ids=["positive_uniaxial", "negative_uniaxial", "absorbing_uniaxial"],
)
def test_uniaxial_o_and_e_channels_match_closed_form(
    eps_ordinary: complex,
    eps_extraordinary: complex,
    incidence_angle_grid_rad: NDArray[np.float64],
    tol_analytical: float,
    xfail_if_kernel_missing: Callable[[Exception], None],
) -> None:
    """Ordinary and extraordinary channels satisfy scalar Fresnel relations.

    With optic axis along ``z_hat`` the ordinary branch reduces to a
    scalar Fresnel problem with substrate index ``n_o`` and the
    extraordinary branch reduces to a scalar Fresnel problem with
    effective index that depends on angle through the uniaxial
    dispersion relation. Both reductions are standard results in Born
    and Wolf, and any deviation at the double-precision tolerance
    floor is a kernel bug.
    """
    stack = uniaxial_substrate_stack(eps_ordinary, eps_extraordinary)
    wavenumber_cm = 1000.0
    for theta in incidence_angle_grid_rad:
        try:
            got = compute_amplitudes(stack, float(theta), wavenumber_cm)
        except KernelNotAvailableError as exc:
            xfail_if_kernel_missing(exc)
            return  # pragma: no cover - defensive return after xfail
        expected = fresnel_uniaxial_optic_axis_z(
            eps_incident=1.0 + 0.0j,
            eps_ordinary=eps_ordinary,
            eps_extraordinary=eps_extraordinary,
            theta_incident_rad=float(theta),
        )
        # For optic axis along z, r_ss corresponds to the ordinary branch
        # and r_pp corresponds to the extraordinary branch. The Passler
        # relabeling ss -> se and pp -> po applies here.
        assert np.isclose(got.r_ss, expected.r_s, atol=tol_analytical), (
            f"r_se (ordinary) mismatch at theta={np.rad2deg(theta):.2f} deg"
        )
        assert np.isclose(got.r_pp, expected.r_p, atol=tol_analytical), (
            f"r_po (extraordinary) mismatch at theta={np.rad2deg(theta):.2f} deg"
        )


@pytest.mark.regression
def test_uniaxial_cross_polarization_vanishes_at_axial_incidence(
    tol_analytical: float,
    xfail_if_kernel_missing: Callable[[Exception], None],
) -> None:
    """At optic axis along ``z_hat`` there is no p-s mode mixing.

    This is the uniaxial analogue of the isotropic cross-polarization
    check. Nonzero ``r_ps`` or ``r_sp`` in this geometry signals that
    either the Xu eigenvectors are not being used or the ``Lambda_{1324}``
    permutation is swapping modes incorrectly.
    """
    stack = uniaxial_substrate_stack()
    wavenumber_cm = 1000.0
    for theta_deg in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0):
        theta = float(np.deg2rad(theta_deg))
        try:
            got = compute_amplitudes(stack, theta, wavenumber_cm)
        except KernelNotAvailableError as exc:
            xfail_if_kernel_missing(exc)
            return  # pragma: no cover - defensive return after xfail
        assert abs(got.r_ps) < tol_analytical, (
            f"r_pe should vanish for optic axis along z; "
            f"got {abs(got.r_ps):.2e} at theta={theta_deg:.1f} deg"
        )
        assert abs(got.r_sp) < tol_analytical, (
            f"r_so should vanish for optic axis along z; "
            f"got {abs(got.r_sp):.2e} at theta={theta_deg:.1f} deg"
        )
