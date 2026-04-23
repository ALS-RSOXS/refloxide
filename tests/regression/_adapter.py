"""Thin adapter between the regression suite and the refloxide kernel.

The purpose of this module is to isolate the regression tests from the
specific Rust API surface that the kernel will expose. The tests call
``compute_amplitudes`` and ``compute_field`` and the adapter forwards to
whatever kernel entry point is currently wired. When the kernel is not
yet implemented the adapter raises ``KernelNotAvailableError``, and the
test fixtures translate that into ``pytest.xfail`` so that continuous
integration stays green while the kernel lands.

The kernel is expected to expose the following modules from
``refloxide._core``, mirroring the pipeline structure laid out in
``docs/theory/pipeline.md``.

    - ``core::delta`` emits the 4x4 Delta matrix per layer.
    - ``core::interface`` emits the per-layer eigenvalues and normalized
      eigenvectors, along with the interface matrix ``A_i``.
    - ``core::transfer`` emits the Yeh-layout transfer matrix and the
      partial propagation callable ``P_i(z)``.
    - ``core::coefficients`` emits the eight amplitude coefficients.
    - ``core::field`` emits the three-component electric field at
      requested ``z`` inside any layer.

Until those modules exist the adapter functions return placeholders
wrapped in a sentinel exception so that tests using them mark xfail
cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._stacks import Stack


class KernelNotAvailableError(NotImplementedError):
    """Raised when the refloxide Rust kernel lacks the required entry point.

    Tests catch this and convert it into ``pytest.xfail`` so that the
    regression suite continues to run against the analytical benchmarks
    even while the Rust kernel is under development.
    """


@dataclass(frozen=True)
class AmplitudeResult:
    """Eight complex amplitude coefficients returned by the kernel.

    The naming follows Passler and Paarmann (2019) Section 2.B. The first
    subscript labels the outgoing polarization and the second labels the
    incident polarization, so ``r_ps`` is the amplitude of an outgoing
    ``p`` wave per unit incident ``s`` wave. For birefringent substrates
    the caller is expected to rename ``pp -> po``, ``ss -> se``,
    ``ps -> pe``, ``sp -> so`` at the semantic layer.

    Attributes:
        r_pp: Reflection amplitude, p in, p out.
        r_ss: Reflection amplitude, s in, s out.
        r_ps: Reflection amplitude, s in, p out.
        r_sp: Reflection amplitude, p in, s out.
        t_pp: Transmission amplitude, p in, p out.
        t_ss: Transmission amplitude, s in, s out.
        t_ps: Transmission amplitude, s in, p out.
        t_sp: Transmission amplitude, p in, s out.
    """

    r_pp: complex
    r_ss: complex
    r_ps: complex
    r_sp: complex
    t_pp: complex
    t_ss: complex
    t_ps: complex
    t_sp: complex


@dataclass(frozen=True)
class FieldProfile:
    """Three-component electric field sampled on a z-grid within one layer.

    Attributes:
        z_nm: Relative z coordinate inside the layer, in nanometers.
        field: Complex electric field as a 3-by-N array with rows
            ``(E_x, E_y, E_z)`` sampled at each grid point. The ``x``
            phase factor ``exp(i xi (omega/c) x)`` is factored out and
            is uniform across the stack.
        polarization: Either ``"p"`` or ``"s"`` to indicate which incident
            polarization the profile was computed for.
    """

    z_nm: NDArray[np.float64]
    field: NDArray[np.complex128]
    polarization: str


def compute_amplitudes(
    stack: Stack,
    theta_incident_rad: float,
    wavenumber_cm: float,
) -> AmplitudeResult:
    """Evaluate the eight amplitude coefficients for a stack.

    Args:
        stack: Stratified stack including cladding half spaces.
        theta_incident_rad: Angle of incidence in radians.
        wavenumber_cm: Probe wavenumber in cm^-1. The kernel converts
            this to angular frequency internally.

    Returns:
        Eight complex amplitude coefficients in the ``AmplitudeResult``
        container.

    Raises:
        KernelNotAvailableError: If the refloxide kernel does not yet
            expose a ``compute_amplitudes`` entry point.
    """
    try:
        from refloxide._core import compute_amplitudes as _kernel_amplitudes  # type: ignore[attr-defined]
    except ImportError as exc:
        raise KernelNotAvailableError(
            "refloxide._core.compute_amplitudes is not yet implemented. "
            "This regression test will xfail until core::coefficients "
            "lands. See docs/theory/pipeline.md stage 5."
        ) from exc

    # The kernel signature is TBD. This placeholder will be wired to the
    # actual call once the Rust side lands. Keep the adapter narrow so
    # that only this module needs updating when the API stabilizes.
    raw = _kernel_amplitudes(  # pragma: no cover - kernel not landed
        stack,
        theta_incident_rad,
        wavenumber_cm,
    )
    return AmplitudeResult(**raw)  # pragma: no cover - kernel not landed


def compute_field(
    stack: Stack,
    theta_incident_rad: float,
    wavenumber_cm: float,
    layer_index: int,
    z_nm: NDArray[np.float64],
    polarization: str,
) -> FieldProfile:
    """Evaluate the three-component electric field inside a layer.

    Args:
        stack: Stratified stack including cladding half spaces.
        theta_incident_rad: Angle of incidence in radians.
        wavenumber_cm: Probe wavenumber in cm^-1.
        layer_index: Zero-based index of the layer in which the field is
            sampled. ``0`` is the incident half space, ``len(layers) - 1``
            is the substrate.
        z_nm: One-dimensional array of relative z coordinates inside the
            layer, in nanometers.
        polarization: Either ``"p"`` or ``"s"``.

    Returns:
        A ``FieldProfile`` carrying the sampled field.

    Raises:
        KernelNotAvailableError: If the refloxide kernel does not yet
            expose a ``compute_field`` entry point.
        ValueError: If ``polarization`` is not ``"p"`` or ``"s"``.
    """
    if polarization not in {"p", "s"}:
        raise ValueError(
            f"polarization must be 'p' or 's', got {polarization!r}"
        )

    try:
        from refloxide._core import compute_field as _kernel_field  # type: ignore[attr-defined]
    except ImportError as exc:
        raise KernelNotAvailableError(
            "refloxide._core.compute_field is not yet implemented. "
            "This regression test will xfail until core::field lands. "
            "See docs/theory/pipeline.md stage 6."
        ) from exc

    raw = _kernel_field(  # pragma: no cover - kernel not landed
        stack,
        theta_incident_rad,
        wavenumber_cm,
        layer_index,
        z_nm,
        polarization,
    )
    return FieldProfile(  # pragma: no cover - kernel not landed
        z_nm=np.asarray(raw["z_nm"], dtype=np.float64),
        field=np.asarray(raw["field"], dtype=np.complex128),
        polarization=polarization,
    )
