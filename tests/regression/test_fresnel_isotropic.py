"""Regression tests against the closed-form isotropic Fresnel limit.

The single-interface isotropic stack is the most stringent correctness
check available without external dependencies, because any deviation
from the closed-form Fresnel amplitudes points to a bug in one of
``core::delta``, ``core::interface``, ``core::transfer``, or
``core::coefficients``. The tolerance on this class of test is set near
double precision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.regression._adapter import (
    KernelNotAvailableError,
    compute_amplitudes,
)
from tests.regression._analytic import fresnel_isotropic
from tests.regression._stacks import fresnel_isotropic_stack

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@pytest.mark.regression
@pytest.mark.parametrize(
    ("eps_incident", "eps_substrate"),
    [
        (1.0 + 0.0j, 2.25 + 0.0j),
        (1.0 + 0.0j, 12.0 + 0.01j),
        (2.25 + 0.0j, 1.0 + 0.0j),
    ],
    ids=["vacuum_to_glass", "vacuum_to_absorber", "glass_to_vacuum"],
)
def test_fresnel_reflection_matches_closed_form(
    eps_incident: complex,
    eps_substrate: complex,
    incidence_angle_grid_rad: NDArray[np.float64],
    tol_analytical: float,
    xfail_if_kernel_missing: Callable[[Exception], None],
) -> None:
    """Verify that ``r_pp`` and ``r_ss`` match the Fresnel expressions.

    The isotropic single-interface limit reduces the ``N = 0`` transfer
    matrix to ``A_0^{-1} A_1``. The eight amplitude coefficients in that
    limit decouple into the two scalar Fresnel amplitudes in each
    polarization and vanish in the cross-polarization channel.
    """
    stack = fresnel_isotropic_stack(eps_incident, eps_substrate)
    wavenumber_cm = 1000.0
    for theta in incidence_angle_grid_rad:
        try:
            got = compute_amplitudes(stack, float(theta), wavenumber_cm)
        except KernelNotAvailableError as exc:
            xfail_if_kernel_missing(exc)
            return  # pragma: no cover - defensive return after xfail
        expected = fresnel_isotropic(eps_incident, eps_substrate, float(theta))
        assert np.isclose(got.r_pp, expected.r_p, atol=tol_analytical), (
            f"r_pp mismatch at theta={np.rad2deg(theta):.2f} deg: "
            f"got {got.r_pp}, expected {expected.r_p}"
        )
        assert np.isclose(got.r_ss, expected.r_s, atol=tol_analytical), (
            f"r_ss mismatch at theta={np.rad2deg(theta):.2f} deg: "
            f"got {got.r_ss}, expected {expected.r_s}"
        )


@pytest.mark.regression
def test_fresnel_cross_polarization_vanishes(
    incidence_angle_grid_rad: NDArray[np.float64],
    tol_analytical: float,
    xfail_if_kernel_missing: Callable[[Exception], None],
) -> None:
    """An isotropic interface produces no cross-polarization coupling.

    At an isotropic interface the Jones reflection matrix is strictly
    diagonal. A nonzero ``r_ps`` or ``r_sp`` in this limit indicates a
    bug in the eigenvector sorting or in the ``Lambda_{1324}``
    permutation.
    """
    stack = fresnel_isotropic_stack()
    wavenumber_cm = 1000.0
    for theta in incidence_angle_grid_rad:
        try:
            got = compute_amplitudes(stack, float(theta), wavenumber_cm)
        except KernelNotAvailableError as exc:
            xfail_if_kernel_missing(exc)
            return  # pragma: no cover - defensive return after xfail
        assert abs(got.r_ps) < tol_analytical, (
            f"r_ps should vanish in isotropic limit; "
            f"got {abs(got.r_ps):.2e} at theta={np.rad2deg(theta):.2f} deg"
        )
        assert abs(got.r_sp) < tol_analytical, (
            f"r_sp should vanish in isotropic limit; "
            f"got {abs(got.r_sp):.2e} at theta={np.rad2deg(theta):.2f} deg"
        )
