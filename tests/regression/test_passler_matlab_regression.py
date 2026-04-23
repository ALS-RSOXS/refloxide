"""Regression tests against the Passler MATLAB reference implementation.

The Passler MATLAB code at Zenodo 601496 is the canonical reference for
the erratum-corrected 4x4 formalism, including the SiC/GaN/SiC
Otto-geometry benchmark of PP2017 Section 3.C. The reference was updated
alongside the 2019 erratum, so agreement with it is the strongest
single-implementation correctness statement available.

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
from tests.regression._stacks import sic_gan_sic_otto_stack
from tests.regression.conftest import REFERENCES_DIR

if TYPE_CHECKING:
    from collections.abc import Callable


GOLDEN_FILE = REFERENCES_DIR / "passler_matlab" / "sic_gan_sic_otto.npz"


def _golden_data_present() -> bool:
    """Return True when the Passler MATLAB golden file is on disk."""
    return GOLDEN_FILE.is_file()


@pytest.mark.regression
@pytest.mark.reference_matlab
@pytest.mark.skipif(
    not _golden_data_present(),
    reason=(
        "Passler MATLAB golden data not present. "
        "See tests/regression/references/README.md to generate."
    ),
)
def test_sic_gan_sic_otto_amplitudes_match_passler_matlab(
    tol_reference_code: float,
    xfail_if_kernel_missing: Callable[[Exception], None],
) -> None:
    """The eight amplitudes match Passler MATLAB across the Otto sweep.

    The golden file carries a two-dimensional grid in
    ``(theta, wavenumber)``, and the test loops over every grid point
    comparing all eight amplitudes. The tolerance is set by the
    expected conditioning of the Passler reference matrix inversions
    rather than by double precision arithmetic.
    """
    golden = np.load(GOLDEN_FILE)
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
                    f"k={k:.1f} cm^-1: got {actual}, expected {expected}"
                )
