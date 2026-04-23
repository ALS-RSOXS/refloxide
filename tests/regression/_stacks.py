"""Canonical test stacks used across the regression suite.

The module exposes a small set of stacks that exercise the transfer matrix
kernel in distinct regimes and that have either a closed-form analytical
answer or a published numerical reference. Each stack is a frozen
dataclass so that fixtures can pass them around without defensive copying.

The stacks in this module are the minimum needed to satisfy the
verification plan laid out in ``docs/theory/``. They are not intended as
end-to-end physics demonstrations; the API examples directory carries
those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Layer:
    """A single homogeneous layer in a stratified stack.

    The dielectric tensor is stored in the principal frame of the material
    as a diagonal three-vector ``(eps_x, eps_y, eps_z)`` along with three
    Euler angles ``(theta, phi, psi)`` that rotate the principal frame
    into the lab frame following the convention of Passler and Paarmann
    (2017), Eq. (2).

    Attributes:
        thickness_nm: Layer thickness in nanometers. Use ``numpy.inf`` for
            a semi-infinite cladding (incident medium or substrate).
        eps_principal: Complex principal-frame dielectric constants as a
            length-3 array ``(eps_x, eps_y, eps_z)``.
        euler_deg: Euler angles in degrees as ``(theta, phi, psi)``. The
            angles are interpreted in the convention used by PP2017 Eq.
            (2), referenced in the module docstring of core::delta.
        mu: Scalar magnetic permeability. Defaults to 1.
    """

    thickness_nm: float
    eps_principal: NDArray[np.complex128] = field(
        default_factory=lambda: np.ones(3, dtype=np.complex128)
    )
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mu: complex = 1.0 + 0.0j


@dataclass(frozen=True)
class Stack:
    """An ordered sequence of layers with semi-infinite cladding on each end.

    The layer list includes both cladding half spaces as the first and
    last entries. Intermediate layers have finite thickness. A single
    isotropic interface is represented by a two-element list with both
    entries semi-infinite.

    Attributes:
        name: Human-readable identifier used in test reports.
        layers: Ordered layer sequence from incident side to substrate
            side. The first and last entries must have infinite thickness.
    """

    name: str
    layers: tuple[Layer, ...]

    @property
    def incident(self) -> Layer:
        """Return the incident half space."""
        return self.layers[0]

    @property
    def substrate(self) -> Layer:
        """Return the substrate half space."""
        return self.layers[-1]

    @property
    def intermediate(self) -> tuple[Layer, ...]:
        """Return the finite-thickness intermediate layers."""
        return self.layers[1:-1]


def isotropic_layer(eps: complex, thickness_nm: float = np.inf) -> Layer:
    """Construct a scalar-isotropic layer.

    Args:
        eps: Complex scalar permittivity.
        thickness_nm: Thickness in nanometers. Defaults to infinity for
            use as a cladding.

    Returns:
        A ``Layer`` with all three principal dielectric constants set to
        ``eps`` and zero Euler angles.
    """
    return Layer(
        thickness_nm=thickness_nm,
        eps_principal=np.array([eps, eps, eps], dtype=np.complex128),
    )


def uniaxial_layer(
    eps_ordinary: complex,
    eps_extraordinary: complex,
    thickness_nm: float = np.inf,
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Layer:
    """Construct a uniaxial layer with optic axis along the principal z.

    Args:
        eps_ordinary: Ordinary ray permittivity ``eps_o = eps_x = eps_y``.
        eps_extraordinary: Extraordinary ray permittivity ``eps_e = eps_z``.
        thickness_nm: Thickness in nanometers.
        euler_deg: Euler angles rotating the optic axis into the lab frame.
            Default leaves the optic axis along ``z_hat``.

    Returns:
        A uniaxial ``Layer`` in the requested orientation.
    """
    return Layer(
        thickness_nm=thickness_nm,
        eps_principal=np.array(
            [eps_ordinary, eps_ordinary, eps_extraordinary],
            dtype=np.complex128,
        ),
        euler_deg=euler_deg,
    )


def fresnel_isotropic_stack(
    eps_incident: complex = 1.0 + 0.0j,
    eps_substrate: complex = 2.25 + 0.0j,
) -> Stack:
    """Return a single-interface isotropic stack for the Fresnel benchmark.

    Args:
        eps_incident: Permittivity of the incident half space. Default is
            vacuum.
        eps_substrate: Permittivity of the substrate half space. Default
            is ``n = 1.5`` glass.

    Returns:
        A two-layer ``Stack`` with both halves semi-infinite and isotropic.
    """
    return Stack(
        name=f"fresnel_iso_eps{eps_incident.real}_{eps_substrate.real}",
        layers=(
            isotropic_layer(eps_incident),
            isotropic_layer(eps_substrate),
        ),
    )


def uniaxial_substrate_stack(
    eps_ordinary: complex = 2.25 + 0.0j,
    eps_extraordinary: complex = 2.89 + 0.0j,
) -> Stack:
    """Return a single-interface uniaxial-substrate stack.

    The incident medium is vacuum. The substrate is uniaxial with optic
    axis along the lab ``z_hat``, for which the closed-form ordinary and
    extraordinary Fresnel coefficients are tabulated in Born and Wolf,
    Principles of Optics, 7th ed., Chapter XIV.

    Args:
        eps_ordinary: Ordinary permittivity of the substrate.
        eps_extraordinary: Extraordinary permittivity of the substrate.

    Returns:
        A two-layer ``Stack`` with an isotropic incident half space and a
        uniaxial substrate.
    """
    return Stack(
        name=f"uniaxial_substrate_eo{eps_ordinary.real}_{eps_extraordinary.real}",
        layers=(
            isotropic_layer(1.0 + 0.0j),
            uniaxial_layer(eps_ordinary, eps_extraordinary),
        ),
    )


def sic_gan_sic_otto_stack(thickness_gan_nm: float = 1000.0) -> Stack:
    """Return the SiC/GaN/SiC Otto-geometry stack of PP2017 Section 3.C.

    The stack is used as the published reference benchmark for the
    evanescent-regime field reconstruction. The Passler MATLAB reference
    implementation and the Jeannin Python port both target this stack.
    Dielectric functions at a specific frequency must be supplied by the
    caller; the stack here sets placeholder scalars that the material
    parameterization layer will replace at evaluation time.

    Args:
        thickness_gan_nm: Thickness of the intermediate GaN layer in
            nanometers. Default matches the PP2017 example.

    Returns:
        A three-layer ``Stack`` with SiC cladding on both sides and a GaN
        middle layer.

    Notes:
        The dielectric functions of SiC and GaN are strongly dispersive in
        the reststrahlen band. Regression tests against the Passler
        reference must evaluate at the same set of wavenumbers that the
        reference was generated for; see
        ``tests/regression/references/README.md``.
    """
    sic_placeholder = 6.5 + 0.0j
    gan_placeholder = 5.35 + 0.0j
    return Stack(
        name=f"sic_gan_sic_otto_gan{int(thickness_gan_nm)}nm",
        layers=(
            isotropic_layer(sic_placeholder),
            isotropic_layer(gan_placeholder, thickness_nm=thickness_gan_nm),
            isotropic_layer(sic_placeholder),
        ),
    )


ALL_STACKS: tuple[Stack, ...] = (
    fresnel_isotropic_stack(),
    fresnel_isotropic_stack(
        eps_incident=2.25 + 0.0j,
        eps_substrate=1.0 + 0.0j,
    ),
    uniaxial_substrate_stack(),
    sic_gan_sic_otto_stack(),
)
"""Default collection used by parameterized regression tests."""
