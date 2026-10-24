"""Refloxide first-class energy-deferred scatterer primitives.

Scatterers defined here resolve optical tensors only at evaluation time through
:class:`~refloxide.pxr.energy.probe.Probe`. Numerics (OOC lookup, laboratory
rotation, slab-row packing) delegate to :mod:`refloxide.rust` so new material
types compose the same parameter surface without duplicating rotation math.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from refnx.analysis import Parameter, Parameters, possibly_create_parameter

from refloxide.pxr.energy.ooc import OocAnchor
from refloxide.pxr.energy.probe import Probe
from refloxide.pxr.plugin.structure import Scatterer

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


def bind_scatterer_energy_offset(
    scatterer: Scatterer,
    structure_offset: Parameter,
) -> None:
    """Constrain ``scatterer.energy_offset`` to follow ``structure_offset``.

    Parameters
    ----------
    scatterer
        Object carrying an ``energy_offset`` :class:`~refnx.analysis.Parameter`.
    structure_offset
        Structure-level offset applied to every slab in the stack.
    """
    if not hasattr(scatterer, "energy_offset"):
        return
    scatterer.energy_offset.setp(
        value=structure_offset.value,
        vary=None,
        bounds=structure_offset.bounds,
        constraint=structure_offset,
    )


class DeferredScatterer(Scatterer):
    """Deferred scatterer resolved through :class:`Probe`.

    Subclasses implement :meth:`tensor_at` only; :meth:`slab_row_at` packs
    refnx slab rows via the Rust :func:`refloxide.rust.tensor_to_slab_row`
    helper using the same convention as
    :class:`~refloxide.pxr.energy.structure.DispersiveStructure`.
    """

    def component_probe(self, probe: Probe) -> Probe:
        """Return ``probe`` with this scatterer's ``energy_offset`` applied."""
        offset = 0.0
        if hasattr(self, "energy_offset"):
            offset = float(self.energy_offset.value or 0.0)
        return Probe(
            base_energy_ev=probe.base_energy_ev,
            structure_offset_ev=probe.structure_offset_ev,
            component_offset_ev=offset,
        )

    def tensor_at(
        self,
        probe: Probe,
    ) -> NDArray[np.complex128]:
        """Return the ``(3, 3)`` laboratory tensor at ``probe.effective_ev``."""
        raise NotImplementedError

    def slab_row_at(
        self,
        probe: Probe,
        thickness: float,
        roughness: float,
    ) -> NDArray[np.float64]:
        """Pack ``[d, delta, beta, sigma]`` for this scatterer at ``probe``."""
        from refloxide.rust import tensor_to_slab_row

        tensor = self.tensor_at(probe)
        row = tensor_to_slab_row(float(thickness), float(roughness), tensor)
        return np.asarray(row, dtype=np.float64)


class OocUniTensorScatterer(DeferredScatterer):
    """Uniaxial scatterer backed by a tabulated OOC curve.

    Accepts a pandas table, CSV path, or :class:`~refloxide.pxr.energy.ooc.OocAnchor`.
    Density, in-plane rotation, and per-material energy offset are refnx parameters.
    Hot loops call Rust for linear OOC lookup and laboratory tensor assembly.

    Parameters
    ----------
    ooc
        Optical-constant table or anchor with columns
        ``energy``, ``n_xx``, ``n_ixx``, ``n_zz``, ``n_izz`` (eV and unitless
        :math:`\\delta`, :math:`\\beta` components).
    rotation
        Polar rotation of the molecular frame in radians.
    density
        Mass-density scale applied to tabulated indices (g/cm**3).
    energy_offset
        Per-material energy offset in eV added on top of the structure offset.
    name
        Scatterer label for refnx parameters and slabs.
    interp
        ``'linear'`` uses the Rust interpolator; ``'pchip'`` defers to SciPy.
    """

    def __init__(
        self,
        ooc: pd.DataFrame | OocAnchor | str | Path,
        *,
        rotation: float = 0.0,
        density: float = 1.0,
        energy_offset: float = 0.0,
        name: str = "",
        interp: Literal["linear", "pchip"] = "linear",
    ) -> None:
        super().__init__(name=name)
        anchor = self._coerce_anchor(ooc, interp=interp)
        self._anchor = anchor
        self.density = possibly_create_parameter(  # type: ignore[assignment]
            density,
            name=f"{name}_density",
            vary=True,
            bounds=(0.0, 5.0 * float(density)),
        )
        self.rotation = possibly_create_parameter(  # type: ignore[assignment]
            rotation, name=f"{name}_rotation", vary=True, bounds=(-np.pi, np.pi)
        )
        self.energy_offset = possibly_create_parameter(  # type: ignore[assignment]
            energy_offset, name=f"{name}_energy_offset", vary=True, bounds=(-0.01, 0.01)
        )
        self._parameters = Parameters(name=name)
        self._parameters.extend([self.density, self.rotation, self.energy_offset])

    @staticmethod
    def _coerce_anchor(
        ooc: pd.DataFrame | OocAnchor | str | Path,
        *,
        interp: Literal["linear", "pchip"],
    ) -> OocAnchor:
        if isinstance(ooc, OocAnchor):
            return ooc
        if isinstance(ooc, (str, Path)):
            return OocAnchor.from_file(ooc, interp=interp)
        return OocAnchor.from_dataframe(ooc, interp=interp)

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        rotation: float = 0.0,
        density: float = 1.0,
        energy_offset: float = 0.0,
        name: str = "",
        interp: Literal["linear", "pchip"] = "linear",
    ) -> OocUniTensorScatterer:
        """Construct from a pandas OOC table with standard column names."""
        return cls(
            frame,
            rotation=rotation,
            density=density,
            energy_offset=energy_offset,
            name=name,
            interp=interp,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        rotation: float = 0.0,
        density: float = 1.0,
        energy_offset: float = 0.0,
        name: str = "",
        interp: Literal["linear", "pchip"] = "linear",
    ) -> OocUniTensorScatterer:
        """Construct from a CSV file containing standard OOC columns."""
        return cls(
            path,
            rotation=rotation,
            density=density,
            energy_offset=energy_offset,
            name=name,
            interp=interp,
        )

    @property
    def anchor(self) -> OocAnchor:
        """Tabulated optical constants backing this scatterer."""
        return self._anchor

    def molecular_index_at(self, probe: Probe) -> tuple[complex, complex]:
        """Density-scaled molecular ``(n_xx, n_zz)`` at ``probe.effective_ev``."""
        eff = self.component_probe(probe)
        rho = float(self.density.value or 1.0)
        if self._anchor.interp == "linear":
            from refloxide.rust import molecular_index_at_ooc

            n_mol_xx, n_mol_zz = molecular_index_at_ooc(
                self._anchor.energy_ev,
                self._anchor.n_xx,
                self._anchor.n_ixx,
                self._anchor.n_zz,
                self._anchor.n_izz,
                eff.effective_ev,
                rho,
            )
            return complex(n_mol_xx), complex(n_mol_zz)
        return self._anchor.molecular_index(eff.effective_ev, rho)

    def tensor_at(self, probe: Probe) -> NDArray[np.complex128]:
        n_mol_xx, n_mol_zz = self.molecular_index_at(probe)
        theta = float(self.rotation.value or 0.0)
        from refloxide.rust import uniaxial_lab_tensor

        tensor = uniaxial_lab_tensor(n_mol_xx, n_mol_zz, theta)
        return np.asarray(tensor, dtype=np.complex128)

    def tensor_batch_at(
        self,
        probe: Probe,
        orientations_rad: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        """Vectorized laboratory tensors for a depth profile at one energy."""
        n_mol_xx, n_mol_zz = self.molecular_index_at(probe)
        from refloxide.rust import lab_tensor_diagonals_batch

        return np.asarray(
            lab_tensor_diagonals_batch(n_mol_xx, n_mol_zz, orientations_rad),
            dtype=np.complex128,
        )

    def __repr__(self) -> str:
        return f"OocUniTensorScatterer(name={self.name!r})"


RefloxideScatterer = DeferredScatterer
TabulatedUniTensorSLD = OocUniTensorScatterer
