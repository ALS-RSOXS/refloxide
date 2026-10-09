"""Volume-fraction mixing rules on laboratory uniaxial diagonals.

Mixing happens **after** each component is projected into the laboratory frame
(matching `MixedUniTensorSLD`). Channels are packed as ``delta + 1j*beta``
with refractive index ``n = 1 - delta + 1j*beta``.

Linear mixing averages those channels directly (volume-weighted SLD).
Maxwell-Garnett and Bruggeman apply the spherical EMA to the dielectric
function ``eps = n**2``, then convert back to ``delta + 1j*beta``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

MixRule = Literal["linear", "maxwell_garnett", "bruggeman"]

__all__ = [
    "MixRule",
    "mix_bruggeman",
    "mix_channels",
    "mix_linear",
    "mix_maxwell_garnett",
    "pack_lab_diagonal",
]


def _as_c(x: complex | float | NDArray[np.complex128]) -> NDArray[np.complex128]:
    return np.asarray(x, dtype=np.complex128)


def _channel_to_eps(
    channel: complex | float | NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Map ``delta + 1j*beta`` to ``eps = n**2`` with ``n = 1 - delta + 1j*beta``."""
    rho = _as_c(channel)
    n = 1.0 - np.real(rho) + 1j * np.imag(rho)
    return n * n


def _eps_to_channel(
    eps: complex | float | NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Map dielectric ``eps`` back to ``delta + 1j*beta`` via ``n = sqrt(eps)``."""
    n = np.sqrt(_as_c(eps))
    n = np.where(np.real(n) >= 0.0, n, -n)
    return (1.0 - np.real(n)) + 1j * np.imag(n)


def _ema_maxwell_garnett(
    eps_h: NDArray[np.complex128],
    eps_i: NDArray[np.complex128],
    phi: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Spherical Maxwell-Garnett EMA on dielectric values."""
    denom = eps_i + 2.0 * eps_h
    alpha = np.where(np.abs(denom) > 0.0, (eps_i - eps_h) / denom, 0.0)
    return eps_h * (1.0 + 2.0 * phi * alpha) / (1.0 - phi * alpha)


def _ema_bruggeman(
    eps_a: NDArray[np.complex128],
    eps_b: NDArray[np.complex128],
    phi_a: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Symmetric two-component Bruggeman EMA on dielectric values.

    Solves
    ``phi (eps_a - eps) / (eps_a + 2 eps)
    + (1-phi) (eps_b - eps) / (eps_b + 2 eps) = 0``
    via ``eps = [gamma + sqrt(gamma^2 + 8 eps_a eps_b)] / 4`` with
    ``gamma = (3 phi - 1) eps_a + (2 - 3 phi) eps_b``.
    """
    gamma = (3.0 * phi_a - 1.0) * eps_a + (2.0 - 3.0 * phi_a) * eps_b
    disc = gamma * gamma + 8.0 * eps_a * eps_b
    sqrt_disc = np.sqrt(disc)
    eps = 0.25 * (gamma + sqrt_disc)
    # Prefer the passive branch when both constituents absorb (Im eps >= 0).
    alt = 0.25 * (gamma - sqrt_disc)
    prefer_alt = (
        (np.imag(eps) < 0.0) & (np.imag(eps_a) >= 0.0) & (np.imag(eps_b) >= 0.0)
    )
    return np.where(prefer_alt, alt, eps)


def mix_linear(
    values: Sequence[complex | float | NDArray[np.complex128]],
    fractions: Sequence[float | NDArray[np.float64]],
) -> NDArray[np.complex128]:
    """Volume-fraction-weighted sum of complex channels (lab diagonals)."""
    if len(values) != len(fractions):
        msg = "values and fractions must have the same length"
        raise ValueError(msg)
    out = np.zeros_like(_as_c(values[0]), dtype=np.complex128)
    for value, frac in zip(values, fractions, strict=True):
        out = out + _as_c(value) * np.asarray(frac, dtype=np.float64)
    return out


def mix_maxwell_garnett(
    host: complex | float | NDArray[np.complex128],
    inclusion: complex | float | NDArray[np.complex128],
    phi_inclusion: float | NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Maxwell-Garnett EMA with `host` as the continuous matrix.

    Inputs are ``delta + 1j*beta`` channels. The spherical MG formula is
    applied to ``eps = n**2`` with ``n = 1 - delta + 1j*beta``, then mapped
    back to ``delta + 1j*beta``.

    Parameters
    ----------
    host, inclusion
        Host (matrix) and inclusion channels.
    phi_inclusion
        Inclusion volume fraction in ``[0, 1]``.
    """
    phi = np.asarray(phi_inclusion, dtype=np.float64)
    eps = _ema_maxwell_garnett(_channel_to_eps(host), _channel_to_eps(inclusion), phi)
    return _eps_to_channel(eps)


def mix_bruggeman(
    a: complex | float | NDArray[np.complex128],
    b: complex | float | NDArray[np.complex128],
    phi_a: float | NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Symmetric Bruggeman EMA for two ``delta + 1j*beta`` channels.

    Applies the two-component Bruggeman EMA to ``eps = n**2`` and converts
    back. Limits: ``phi_a -> 0`` recovers `b`; ``phi_a -> 1`` recovers `a`.

    Parameters
    ----------
    a, b
        Component channels.
    phi_a
        Volume fraction of `a` (``1 - phi_a`` is the fraction of `b`).
    """
    phi = np.asarray(phi_a, dtype=np.float64)
    eps = _ema_bruggeman(_channel_to_eps(a), _channel_to_eps(b), phi)
    return _eps_to_channel(eps)


def mix_channels(
    values: Sequence[complex | float | NDArray[np.complex128]],
    fractions: Sequence[float | NDArray[np.float64]],
    *,
    rule: MixRule = "linear",
    host_index: int = 0,
) -> NDArray[np.complex128]:
    """Mix complex channels with `rule`.

    Parameters
    ----------
    values, fractions
        Parallel sequences of component channels and volume fractions.
    rule
        ``linear``, ``maxwell_garnett``, or ``bruggeman``.
    host_index
        For Maxwell-Garnett, index of the host component in `values`.
    """
    match rule:
        case "linear":
            return mix_linear(values, fractions)
        case "maxwell_garnett":
            if len(values) != 2:
                msg = "maxwell_garnett currently supports exactly two components"
                raise ValueError(msg)
            host = values[host_index]
            incl_index = 1 - host_index
            return mix_maxwell_garnett(host, values[incl_index], fractions[incl_index])
        case "bruggeman":
            if len(values) != 2:
                msg = "bruggeman currently supports exactly two components"
                raise ValueError(msg)
            return mix_bruggeman(values[0], values[1], fractions[0])
        case _:
            msg = f"unknown mix rule {rule!r}"
            raise ValueError(msg)


def pack_lab_diagonal(
    n_o: complex | NDArray[np.complex128],
    n_e: complex | NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Pack ordinary/extraordinary channels into ``(3,3)`` or ``(n,3,3)``."""
    n_o_arr = _as_c(n_o)
    n_e_arr = _as_c(n_e)
    if n_o_arr.ndim == 0:
        tensor = np.zeros((3, 3), dtype=np.complex128)
        tensor[0, 0] = n_o_arr
        tensor[1, 1] = n_o_arr
        tensor[2, 2] = n_e_arr
        return tensor
    tensor = np.zeros((*n_o_arr.shape, 3, 3), dtype=np.complex128)
    tensor[..., 0, 0] = n_o_arr
    tensor[..., 1, 1] = n_o_arr
    tensor[..., 2, 2] = n_e_arr
    return tensor
