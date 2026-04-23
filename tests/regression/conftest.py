"""Shared fixtures and configuration for the regression test suite.

The conftest exposes tolerance fixtures, standard incidence-angle grids,
and an ``xfail_if_kernel_missing`` helper that converts a
``KernelNotAvailableError`` raised by the adapter into
``pytest.xfail`` without marking the test as failed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.regression._adapter import KernelNotAvailableError

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


REFERENCES_DIR: Path = Path(__file__).parent / "references"
"""Filesystem root holding golden reference data from external codes."""


@pytest.fixture(scope="session")
def tol_analytical() -> float:
    """Tolerance for closed-form analytical benchmarks.

    Analytical benchmarks have no cancellation and should be reproduced
    to double precision modulo floating-point noise in the kernel's
    internal linear algebra.
    """
    return 1e-10


@pytest.fixture(scope="session")
def tol_self_consistent() -> float:
    """Tolerance for self-consistency checks of the kernel.

    Tangential continuity at interfaces should hold to near-double
    precision unless the stack is thick enough to incur conditioning
    loss in the ``A_i`` inversions.
    """
    return 1e-9


@pytest.fixture(scope="session")
def tol_reference_code() -> float:
    """Tolerance for agreement with external reference implementations.

    The Passler MATLAB reference and the Jeannin Python port both use
    double precision linear algebra but with slightly different
    matrix-inversion strategies, so the achievable tolerance is looser
    than the analytical benchmarks.
    """
    return 1e-8


@pytest.fixture(scope="session")
def incidence_angle_grid_rad() -> NDArray[np.float64]:
    """Dense grid of incident angles used across parameterized tests.

    The grid spans the full propagating regime and extends two degrees
    past the critical angle of the densest glass-to-vacuum interface
    used in the stacks, so that evanescent coupling is exercised.
    """
    return np.deg2rad(np.linspace(0.0, 89.0, 90))


@pytest.fixture(scope="session")
def wavenumber_grid_cm() -> NDArray[np.float64]:
    """Representative probe wavenumbers in cm^-1.

    The values are chosen inside the reststrahlen band of SiC for
    parity with the PP2017 Section 3.C Otto-geometry benchmark.
    """
    return np.array([780.0, 800.0, 820.0, 840.0, 860.0], dtype=np.float64)


@pytest.fixture
def xfail_if_kernel_missing() -> Callable[[Exception], None]:
    """Return a helper that converts adapter misses into pytest.xfail.

    Test bodies use the helper as::

        try:
            result = compute_amplitudes(stack, theta, k)
        except KernelNotAvailableError as exc:
            xfail_if_kernel_missing(exc)

    which keeps CI green while the kernel is under construction.
    """

    def _xfail(exc: Exception) -> None:
        if isinstance(exc, KernelNotAvailableError):
            pytest.xfail(str(exc))
        raise exc

    return _xfail


def _reference_file_present(relative_path: str) -> bool:
    """Return True if a reference data file is present on disk.

    Args:
        relative_path: Path relative to ``REFERENCES_DIR``.

    Returns:
        True when the file exists.
    """
    return (REFERENCES_DIR / relative_path).is_file()


@pytest.fixture(scope="session")
def passler_matlab_reference_available() -> bool:
    """Return True when Passler MATLAB golden data are present."""
    return _reference_file_present("passler_matlab/sic_gan_sic_otto.npz")


@pytest.fixture(scope="session")
def jeannin_python_reference_available() -> bool:
    """Return True when Jeannin Python golden data are present."""
    return _reference_file_present("jeannin_python/sic_gan_sic_otto.npz")
