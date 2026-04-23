"""Self-consistency tests for tangential field continuity at interfaces.

A direct algebraic consequence of the interface matrix construction
``L_i = A_{i-1}^{-1} A_i`` is that the four tangential field components
``(E_x, H_y, E_y, H_x)`` match across every interface. This test does
not require any external reference, so it runs as a self-consistent
check on the kernel's interface matrix assembly and its amplitude
propagation. Failures indicate a bug in ``core::interface``,
``core::transfer``, or ``core::field``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.regression._adapter import (
    KernelNotAvailableError,
    compute_field,
)
from tests.regression._stacks import sic_gan_sic_otto_stack

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.regression
def test_e_tangential_continuity_across_gan_sic_interfaces(
    tol_self_consistent: float,
    xfail_if_kernel_missing: Callable[[Exception], None],
) -> None:
    """E_x and E_y match across every interface in the SiC/GaN/SiC stack.

    The test samples the field just above and just below each finite
    interface of the stack. A discontinuity beyond the self-consistency
    tolerance indicates that the interface matrix factorization has
    lost precision or that the amplitude-propagation recursion has been
    assembled in the wrong order.
    """
    stack = sic_gan_sic_otto_stack(thickness_gan_nm=500.0)
    wavenumber_cm = 820.0
    theta = float(np.deg2rad(30.0))
    eps_sample = 1e-3  # nm below and above each interface

    for layer_idx in range(len(stack.layers) - 1):
        top_layer = stack.layers[layer_idx]
        thickness_top = top_layer.thickness_nm
        if not np.isfinite(thickness_top):
            # Incident half space has no finite "just above" sample.
            z_above = np.array([-eps_sample])
        else:
            z_above = np.array([thickness_top - eps_sample])
        z_below = np.array([eps_sample])

        try:
            field_above = compute_field(
                stack,
                theta,
                wavenumber_cm,
                layer_index=layer_idx,
                z_nm=z_above,
                polarization="p",
            )
            field_below = compute_field(
                stack,
                theta,
                wavenumber_cm,
                layer_index=layer_idx + 1,
                z_nm=z_below,
                polarization="p",
            )
        except KernelNotAvailableError as exc:
            xfail_if_kernel_missing(exc)
            return  # pragma: no cover - defensive return after xfail

        # Rows 0 and 1 of the field array are E_x and E_y.
        dex = abs(field_above.field[0, 0] - field_below.field[0, 0])
        dey = abs(field_above.field[1, 0] - field_below.field[1, 0])
        assert dex < tol_self_consistent, (
            f"E_x discontinuity {dex:.2e} at interface {layer_idx}"
        )
        assert dey < tol_self_consistent, (
            f"E_y discontinuity {dey:.2e} at interface {layer_idx}"
        )
