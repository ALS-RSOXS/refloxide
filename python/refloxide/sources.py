"""Optical-constant source protocol for tabular, free-form, formula, and future SOS.

This module owns the narrow `OpticalSource` boundary so sum-of-states and other
external crates can plug in without embedding their physics in refloxide core.
It does not implement TMM, depth profiles, or mixing rules.

Built-in adapters:

* ``refloxide.data.OpticalConstants`` — tabular OOC (already satisfies the
  protocol via ``molecular_index_at`` / ``cache_at``).
* ``FormulaOpticalSource`` — Henke / periodictable chemistry for isotropic
  molecules (Python until a Rust bottleneck appears).
* Sum-of-states — future external crate only; import against this protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import periodictable as pt
import periodictable.xsf as xsf

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "FormulaOpticalSource",
    "OpticalSource",
    "as_complex_pair",
    "lab_diagonal_from_cos2",
]


@runtime_checkable
class OpticalSource(Protocol):
    """Energy-resolved optical constants consumed by scatterers and profiles.

    Implementations answer molecular or laboratory indices at photon energy
    `energy_ev` (eV). They must not know slab thickness or stack position.
    """

    def molecular_index_at(
        self, energy_ev: float, density: float
    ) -> tuple[complex, complex]:
        """Molecular-frame ``(n_xx, n_zz)`` as ``delta + i*beta`` at `energy_ev`.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV.
        density : float
            Mass density (g/cm^3) used to scale tabulated molecular constants.

        Returns
        -------
        tuple[complex, complex]
            Ordinary-plane and extraordinary molecular indices.
        """
        ...

    def cache_at(self, energy_ev: float) -> object:
        """Warm any process-local memo so later lookups at `energy_ev` are cheap.

        Return value is implementation-defined (often ``None`` or cached
        components); callers must not rely on a specific return type.
        """
        ...


class FormulaOpticalSource:
    """Henke / periodictable formula source returning isotropic molecular indices.

    Both axes of the returned pair are identical (``n_xx == n_zz``). Uniaxial
    anisotropy belongs on a scatterer / depth field, not in this source.
    """

    def __init__(self, formula: str) -> None:
        self.formula = formula
        self._cache: dict[tuple[float, float], tuple[complex, complex]] = {}

    def cache_at(self, energy_ev: float) -> None:
        """No table to warm; formula evaluation is keyed in `molecular_index_at`."""
        del energy_ev

    def molecular_index_at(
        self, energy_ev: float, density: float
    ) -> tuple[complex, complex]:
        """Compute isotropic ``(n, n)`` via ``periodictable.xsf.index_of_refraction``.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV (converted to keV for periodictable).
        density : float
            Mass density in g/cm^3.
        """
        key = (float(energy_ev), float(density))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        compound = pt.formula(self.formula)
        energy_kev = float(energy_ev) / 1000.0
        # periodictable: n = 1 - (delta + i*beta); invert for kernel packing
        n = complex(
            xsf.index_of_refraction(compound, density=float(density), energy=energy_kev)
        )
        sld = 1.0 - n
        pair = (sld, sld)
        if len(self._cache) >= 256:
            self._cache.clear()
        self._cache[key] = pair
        return pair


def as_complex_pair(
    n_xx: complex | float, n_zz: complex | float
) -> tuple[complex, complex]:
    """Normalize a molecular index pair to built-in ``complex`` values."""
    return complex(n_xx), complex(n_zz)


def lab_diagonal_from_cos2(
    n_xx: complex,
    n_zz: complex,
    cos2_gamma: float | NDArray[np.float64],
) -> tuple[complex | NDArray[np.complex128], complex | NDArray[np.complex128]]:
    """Lab ordinary/extraordinary indices from ``<cos^2 gamma>`` (no arccos).

    Uses the same uniaxial projection as ``uniaxial_lab_tensor`` with
    ``cos^2`` supplied directly:

    ``n_o = (n_xx * (1 + cos2) + n_zz * sin2) / 2``,
    ``n_e = n_xx * sin2 + n_zz * cos2``.
    """
    cos2 = np.asarray(cos2_gamma, dtype=np.float64)
    sin2 = 1.0 - cos2
    n_o = (n_xx * (1.0 + cos2) + n_zz * sin2) / 2.0
    n_e = n_xx * sin2 + n_zz * cos2
    if cos2.ndim == 0:
        return complex(n_o), complex(n_e)
    return n_o.astype(np.complex128), n_e.astype(np.complex128)
