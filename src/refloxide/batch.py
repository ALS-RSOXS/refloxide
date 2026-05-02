"""Scalar-kernel batching: frequency grids, incidence, and legacy *q* maps.

The Rust kernel takes one ``omega_rad_per_s`` and one ``theta_rad`` per solve
(see ``docs/theory/pipeline.md``). This module vectorizes in Python with
NumPy broadcasting; it does not change numerics inside ``refloxide._core``.

Photon energy to angular frequency uses CODATA 2018 constants consistent with
``src/types/scalar.rs`` (speed of light and :math:`\\hbar`).

The legacy *q* mapping matches :func:`refloxide.pxr.tjf4x4.uniaxial_reflectivity`
(``hc`` in eV·Å, :math:`\\theta = \\pi/2 - \\arcsin(q/(2k_0))`` with
:math:`k_0=2\\pi/\\lambda`). That convention is for the historical script only;
tangential :math:`\\xi` in the Berreman formalism is documented in
``docs/theory/foundations.md`` and is not equal to ``q`` without the explicit
``\\omega/c`` scaling used inside the kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from refloxide._core import compute_amplitudes_py

if TYPE_CHECKING:
    from numpy.typing import NDArray

_HBAR_J_S = 1.054571817e-34
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_HC_EV_ANGSTROM = 12398.4193


def omega_rad_per_s_from_photon_energy_ev(
    energy_ev: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Angular frequency (rad/s) from photon energy in electron-volts.

    Uses :math:`\\omega = E_{\\mathrm{J}} / \\hbar` with
    :math:`E_{\\mathrm{J}} = E_{\\mathrm{eV}} \\, e`.

    Parameters
    ----------
    energy_ev:
        Photon energy or broadcastable array, eV.

    Returns
    -------
    numpy.ndarray
        Same shape as ``energy_ev``, dtype float64.
    """
    e = np.asarray(energy_ev, dtype=np.float64)
    ej = e * _ELEMENTARY_CHARGE_C
    return ej / _HBAR_J_S


def theta_rad_from_legacy_tjf4x4_q(
    q_inv_angstrom: float | NDArray[np.float64],
    photon_energy_ev: float,
) -> NDArray[np.float64]:
    """Incidence angle (rad) from legacy *q* grid used in ``tjf4x4``.

    Mirrors ``wl = hc / energy``, ``k0 = 2*pi/wl`` (Å⁻¹),
    ``theta = pi/2 - arcsin(clip(q/(2*k0), -1, 1))`` with ``hc = 12398.4193``
    eV·Å as in :mod:`refloxide.pxr.tjf4x4`.

    Parameters
    ----------
    q_inv_angstrom:
        Tangential wavevector magnitude in inverse ångströms (same units as the
        legacy ``uniaxial_reflectivity`` *q* argument).
    photon_energy_ev:
        Single photon energy (eV) defining vacuum wavelength for that row.

    Returns
    -------
    numpy.ndarray
        Shape broadcast with ``q_inv_angstrom``, dtype float64.
    """
    wl_a = _HC_EV_ANGSTROM / photon_energy_ev
    k0 = 2.0 * np.pi / wl_a
    q = np.asarray(q_inv_angstrom, dtype=np.float64)
    return np.asarray(
        np.pi / 2.0 - np.arcsin(np.clip(q / (2.0 * k0), -1.0, 1.0)),
        dtype=np.float64,
    )


def compute_amplitudes_broadcast(
    stack_repr: dict[str, object],
    omega_rad_per_s: float | NDArray[np.float64],
    theta_rad: float | NDArray[np.float64],
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """Evaluate :func:`compute_amplitudes` on broadcast-compatible grids.

    Parameters
    ----------
    stack_repr:
        Stack mapping accepted by ``refloxide._core.compute_amplitudes_py``.
    omega_rad_per_s, theta_rad:
        Scalars or arrays broadcast together to a common shape.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        Eight complex arrays ``r_pp, r_ss, r_ps, r_sp, t_pp, t_ss, t_ps, t_sp``,
        shape matching the broadcast shape of the frequency and angle inputs.
    """
    o = np.asarray(omega_rad_per_s, dtype=np.float64)
    t = np.asarray(theta_rad, dtype=np.float64)
    b = np.broadcast_arrays(o, t)
    flat_o = b[0].ravel()
    flat_t = b[1].ravel()
    n = flat_o.shape[0]
    names_order = (
        "r_pp",
        "r_ss",
        "r_ps",
        "r_sp",
        "t_pp",
        "t_ss",
        "t_ps",
        "t_sp",
    )
    out = {k: np.empty(n, dtype=np.complex128) for k in names_order}
    for i in range(n):
        row = compute_amplitudes_py(stack_repr, float(flat_o[i]), float(flat_t[i]))
        for k, name in enumerate(names_order):
            out[name][i] = row[k]
    shape = b[0].shape
    ordered = tuple(out[name].reshape(shape) for name in names_order)
    return ordered


def compute_amplitudes_ev_theta_broadcast(
    stack_repr: dict[str, object],
    photon_energy_ev: float | NDArray[np.float64],
    theta_rad: float | NDArray[np.float64],
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """Like :func:`compute_amplitudes_broadcast` with energies in eV.

    Converts ``photon_energy_ev`` to ``omega_rad_per_s`` via
    :func:`omega_rad_per_s_from_photon_energy_ev` then calls the scalar kernel.
    """
    w = omega_rad_per_s_from_photon_energy_ev(photon_energy_ev)
    return compute_amplitudes_broadcast(stack_repr, w, theta_rad)


def compute_amplitudes_legacy_q_broadcast(
    stack_repr: dict[str, object],
    photon_energy_ev: float,
    q_inv_angstrom: float | NDArray[np.float64],
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """Broadcast solve over a legacy *q* grid at fixed photon energy (eV).

    Maps *q* to ``theta_rad`` with :func:`theta_rad_from_legacy_tjf4x4_q`, then
    runs :func:`compute_amplitudes_broadcast`. Use only when matching
    ``tjf4x4`` conventions; for general lab :math:`q_x` mappings see the theory
    docs and align with ``geometry::tangential_xi`` in Rust.
    """
    th = theta_rad_from_legacy_tjf4x4_q(q_inv_angstrom, photon_energy_ev)
    w = np.asarray(
        omega_rad_per_s_from_photon_energy_ev(photon_energy_ev),
        dtype=np.float64,
    )
    return compute_amplitudes_broadcast(stack_repr, w, th)
