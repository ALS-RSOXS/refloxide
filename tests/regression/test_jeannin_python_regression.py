"""Regression tests against the Jeannin Python reference implementation.

The Jeannin Python port at Zenodo 3417751 is an independent
reimplementation of the Passler-Paarmann 4x4 formalism that includes the
2019 erratum corrections from the outset. Agreement with Jeannin gives a
second independent correctness anchor beyond the Passler MATLAB code,
and is particularly valuable for the birefringent-substrate regime that
the 2019 erratum was written to fix.

Tests in this module are skipped when the golden data file is not
present on disk. To regenerate the golden data see
``tests/regression/references/README.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.regression._adapter import (
    KernelNotAvailableError,
    compute_amplitudes,
)
from tests.regression._stacks import (
    sic_gan_sic_otto_stack,
    uniaxial_substrate_stack,
)
from tests.regression.conftest import REFERENCES_DIR

if TYPE_CHECKING:
    from collections.abc import Callable


JEANNIN_DIR = REFERENCES_DIR / "jeannin_python"


def _golden_present(name: str) -> bool:
    """Return True when a named Jeannin golden file exists on disk."""
    return (JEANNIN_DIR / name).is_file()


@pytest.mark.regression
@pytest.mark.reference_jeannin
@pytest.mark.skipif(
    not _golden_present("sic_gan_sic_otto.npz"),
    reason=(
        "Jeannin Python golden data not present. "
        "See tests/regression/references/README.md to generate."
    ),
)
def test_sic_gan_sic_otto_amplitudes_match_jeannin(
    tol_reference_code: float,
    xfail_if_kernel_missing: Callable[[Exception], None],
) -> None:
    """The eight amplitudes match Jeannin Python across the Otto sweep."""
    golden = np.load(JEANNIN_DIR / "sic_gan_sic_otto.npz")
    thetas = golden["theta_rad"]
    wavenumbers = golden["wavenumber_cm"]
    stack = sic_gan_sic_otto_stack(
        thickness_gan_nm=float(golden["thickness_gan_nm"])
    )

    for theta_idx, theta in enumerate(thetas):
        for k_idx, k in enumerate(wavenumbers):
            try:
                got = compute_amplitudes(stack, float(theta), float(k))
            except KernelNotAvailableError as exc:
                xfail_if_kernel_missing(exc)
                return  # pragma: no cover
            for name in ("r_pp", "r_ss", "r_ps", "r_sp", "t_pp", "t_ss", "t_ps", "t_sp"):
                expected = golden[name][theta_idx, k_idx]
                actual = getattr(got, name)
                assert np.isclose(actual, expected, atol=tol_reference_code), (
                    f"{name} mismatch at theta={np.rad2deg(theta):.2f} deg, "
                    f"k={k:.1f} cm^-1"
                )


@pytest.mark.regression
@pytest.mark.reference_jeannin
@pytest.mark.skipif(
    not _golden_present("uniaxial_substrate.npz"),
    reason=(
        "Jeannin Python uniaxial golden data not present. "
        "See tests/regression/references/README.md to generate."
    ),
)
def test_uniaxial_substrate_matches_jeannin(
    tol_reference_code: float,
    xfail_if_kernel_missing: Callable[[Exception], None],
) -> None:
    """The uniaxial substrate cross-polarization matches Jeannin Python.

    This test exercises the birefringent branch of ``core::interface``
    that the PP2019 erratum corrects. Rotated optic-axis cases must
    agree with Jeannin at the reference-code tolerance, including
    nonzero ``r_ps`` and ``r_sp`` amplitudes.
    """
    golden = np.load(JEANNIN_DIR / "uniaxial_substrate.npz")
    thetas = golden["theta_rad"]
    eps_o = complex(golden["eps_ordinary"])
    eps_e = complex(golden["eps_extraordinary"])
    stack = uniaxial_substrate_stack(eps_o, eps_e)
    wavenumber_cm = float(golden["wavenumber_cm"])

    for theta_idx, theta in enumerate(thetas):
        try:
            got = compute_amplitudes(stack, float(theta), wavenumber_cm)
        except KernelNotAvailableError as exc:
            xfail_if_kernel_missing(exc)
            return  # pragma: no cover
        for name in ("r_pp", "r_ss", "r_ps", "r_sp"):
            expected = golden[name][theta_idx]
            actual = getattr(got, name)
            assert np.isclose(actual, expected, atol=tol_reference_code), (
                f"{name} mismatch at theta={np.rad2deg(theta):.2f} deg"
            )
