"""Closed-form analytical benchmarks for regression tests.

The module collects the closed-form amplitude reflection and transmission
coefficients that the 4x4 transfer matrix kernel must reproduce in known
limiting cases. Two families are supplied.

The isotropic Fresnel family targets the ``N = 0`` single-interface limit
of the pipeline. It must be reproduced to full double precision because
no numerical cancellation is expected.

The uniaxial Fresnel family targets a single birefringent interface with
optic axis along the lab ``z_hat``, for which the ordinary and
extraordinary rays decouple and each satisfies its own scalar Fresnel
relation. The expressions are those in Born and Wolf (2002), Chapter XIV,
reproduced here in their amplitude form rather than the intensity form.

All functions return complex amplitudes rather than intensities, so that
phase information is available for checks that compare to the kernel's
per-mode reflection and transmission amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FresnelAmplitudes:
    """Complex amplitude reflection and transmission coefficients.

    Attributes:
        r_s: Amplitude reflection coefficient for ``s``-polarized light.
        r_p: Amplitude reflection coefficient for ``p``-polarized light.
        t_s: Amplitude transmission coefficient for ``s``-polarized light.
        t_p: Amplitude transmission coefficient for ``p``-polarized light.
    """

    r_s: complex
    r_p: complex
    t_s: complex
    t_p: complex


def fresnel_isotropic(
    eps_incident: complex,
    eps_substrate: complex,
    theta_incident_rad: float,
) -> FresnelAmplitudes:
    """Amplitude Fresnel coefficients for a single isotropic interface.

    The conventions follow Hecht, Optics, 4th ed., Eqs. (4.34) to (4.38),
    with the positive ``r_p`` convention where ``r_p = +1`` at grazing
    incidence on a metallic substrate.

    Args:
        eps_incident: Complex permittivity of the incident half space.
        eps_substrate: Complex permittivity of the substrate half space.
        theta_incident_rad: Angle of incidence relative to the surface
            normal, in radians.

    Returns:
        Amplitude reflection and transmission coefficients in both
        polarizations.
    """
    n1 = np.sqrt(eps_incident)
    n2 = np.sqrt(eps_substrate)
    cos1 = np.cos(theta_incident_rad)
    sin1 = np.sin(theta_incident_rad)

    # Snell's law for the cosine in medium 2, preserving the complex branch
    # cut consistent with absorption in the substrate.
    cos2 = np.sqrt(1.0 - (n1 / n2) ** 2 * sin1**2)

    r_s = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
    r_p = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
    t_s = 2.0 * n1 * cos1 / (n1 * cos1 + n2 * cos2)
    t_p = 2.0 * n1 * cos1 / (n2 * cos1 + n1 * cos2)

    return FresnelAmplitudes(r_s=r_s, r_p=r_p, t_s=t_s, t_p=t_p)


def fresnel_uniaxial_optic_axis_z(
    eps_incident: complex,
    eps_ordinary: complex,
    eps_extraordinary: complex,
    theta_incident_rad: float,
) -> FresnelAmplitudes:
    """Amplitude Fresnel coefficients at a uniaxial interface.

    The substrate is uniaxial with optic axis along the lab ``z_hat``, so
    the ordinary ray ``o`` is purely ``s``-polarized and the
    extraordinary ray ``e`` is purely ``p``-polarized. The standard
    uniaxial relations in Born and Wolf (2002), Eqs. (14.4.10) through
    (14.4.14), reduce in this geometry to scalar Fresnel expressions with
    modified effective indices.

    The effective index for the extraordinary ray at angle ``theta_e`` in
    the substrate satisfies

    .. math::

        \\frac{\\cos^2\\theta_e}{\\varepsilon_o}
        + \\frac{\\sin^2\\theta_e}{\\varepsilon_e}
        = \\frac{1}{n_e^2(\\theta_e)}

    and Snell's law for the extraordinary ray couples ``theta_e`` to the
    incident angle through ``n_e(\\theta_e)\\sin\\theta_e = n_1
    \\sin\\theta_1``.

    Args:
        eps_incident: Complex permittivity of the incident half space.
        eps_ordinary: Ordinary permittivity of the substrate.
        eps_extraordinary: Extraordinary permittivity of the substrate.
        theta_incident_rad: Angle of incidence in radians.

    Returns:
        Amplitude reflection and transmission coefficients for the
        ordinary (``s``-polarized) and extraordinary (``p``-polarized)
        substrate rays. ``r_s`` and ``t_s`` refer to the ordinary branch,
        ``r_p`` and ``t_p`` to the extraordinary branch.
    """
    n1 = np.sqrt(eps_incident)
    n_o = np.sqrt(eps_ordinary)
    cos1 = np.cos(theta_incident_rad)
    sin1 = np.sin(theta_incident_rad)

    # Ordinary ray is a scalar Fresnel problem with substrate index n_o.
    cos_o = np.sqrt(1.0 - (n1 / n_o) ** 2 * sin1**2)
    r_s = (n1 * cos1 - n_o * cos_o) / (n1 * cos1 + n_o * cos_o)
    t_s = 2.0 * n1 * cos1 / (n1 * cos1 + n_o * cos_o)

    # Extraordinary ray. The in-plane wavevector component is conserved,
    # so k_x = (omega/c) n_1 sin(theta_1). The dispersion relation for
    # the extraordinary ray in a uniaxial medium with optic axis along z
    # is k_x^2 / eps_e + k_z^2 / eps_o = (omega/c)^2, where eps_e is the
    # extraordinary permittivity and eps_o is the ordinary. Solving for
    # k_z gives the effective cosine below.
    kx2 = eps_incident * sin1**2
    kz2_e = eps_ordinary * (1.0 - kx2 / eps_extraordinary)
    kz_e = np.sqrt(kz2_e)
    # Effective refractive index that participates in the Fresnel ratio.
    # For p polarization at a uniaxial interface with optic axis normal to
    # the surface the boundary conditions yield the canonical form of
    # Yeh, Optical Waves in Layered Media (1988), Eq. (4.4-4).
    n_e_eff = np.sqrt(kz2_e + kx2)
    cos_e_eff = kz_e / n_e_eff

    r_p = (n_e_eff * cos1 - n1 * cos_e_eff) / (n_e_eff * cos1 + n1 * cos_e_eff)
    t_p = 2.0 * n1 * cos1 / (n_e_eff * cos1 + n1 * cos_e_eff)

    return FresnelAmplitudes(r_s=r_s, r_p=r_p, t_s=t_s, t_p=t_p)
