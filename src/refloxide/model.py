"""Structure-building primitives and the reflectivity model.

`Structure`, `Component`, `Slab`, `Scatterer` compose via the `|` operator
exactly like refnx/pyref (see tmp/USAGE.md). Energy is late-bound
everywhere: nothing here requires a single energy at construction time — a
`Structure` is built once, and a `Scatterer`'s tensor is resolved fresh at
whatever `energy_ev` it's queried with. Geometry parameters (`thick`,
`rough`) live once per `Slab`, never duplicated per energy channel.

`Scatterer` is meant to be subclassed directly for materials refloxide
doesn't ship a built-in class for — implement `tensor_at`/`parameters`;
`slab_row_at` and `__call__` are provided. See tmp/USAGE.md "Building your
own energy-dependent scatterer in Python" for a full worked example.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import periodictable as pt
import periodictable.xsf as xsf
from refnx.analysis import Parameters, possibly_create_parameter

from refloxide import optics
from refloxide.data import OpticalConstants
from refloxide.pxr.plugin.structure import compound_density

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from numpy.typing import NDArray


class Scatterer(ABC):
    """Base protocol for a material with an energy-dependent tensor index of refraction.

    Subclass this directly to add a material with no built-in class:
    implement `tensor_at` and `parameters`; `slab_row_at` and `__call__`
    are provided and should not normally need overriding.

    Parameters
    ----------
    name : str, optional
        Label used to prefix this scatterer's parameter names.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name

    @abstractmethod
    def tensor_at(self, energy_ev: float) -> NDArray[np.complex128]:
        """Laboratory-frame `(3, 3)` complex tensor index of refraction at `energy_ev`.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV. Implementations resolve this fresh on
            every call — there is no cached, construction-time energy.

        Returns
        -------
        NDArray[np.complex128]
            `(3, 3)` tensor holding delta + i*beta directly (not a full
            refractive index near 1) — the convention every kernel in
            `refloxide.optics`/`refloxide.tmm` expects.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> Parameters:
        """This scatterer's `refnx.analysis.Parameters` (density, rotation, etc.)."""
        raise NotImplementedError

    def slab_row_at(
        self, energy_ev: float, thickness: float, roughness: float
    ) -> NDArray[np.float64]:
        """Pack `[thickness, delta, beta, roughness]` for this scatterer at `energy_ev`.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV.
        thickness, roughness : float
            Slab geometry in Angstrom.

        Returns
        -------
        NDArray[np.float64]
            Default implementation delegates to
            `refloxide.optics.tensor_to_slab_row` on `tensor_at(energy_ev)`
            — override only if a scatterer needs a cheaper or different
            packing than deriving it from the full tensor.
        """
        tensor = self.tensor_at(energy_ev)
        return np.asarray(
            optics.tensor_to_slab_row(float(thickness), float(roughness), tensor),
            dtype=np.float64,
        )

    def __call__(self, thick: float, rough: float) -> Slab:
        """Build a `Slab` pairing this scatterer with slab geometry.

        Parameters
        ----------
        thick, rough : float
            Initial thickness/roughness in Angstrom; each becomes one
            `refnx.analysis.Parameter` shared across every future
            `(q, energy)` evaluation of this slab, with default bounds
            `(0, 2 * value)`.

        Returns
        -------
        Slab
        """
        slab = Slab(thick, self, rough, name=self.name)
        slab.thick.setp(vary=True, bounds=(0, 2 * thick if thick else 1.0))
        slab.rough.setp(vary=True, bounds=(0, 2 * rough if rough else 1.0))
        return slab


class Component(ABC):
    """Base protocol for one member of a `Structure` — a `Slab`, typically."""

    def __init__(self, name: str = "") -> None:
        self.name = name

    @abstractmethod
    def slab_row_at(self, energy_ev: float) -> NDArray[np.float64]:
        """This component's packed `[thickness, delta, beta, roughness]` row."""
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> Parameters:
        """This component's `refnx.analysis.Parameters`."""
        raise NotImplementedError

    def __or__(self, other: Component | Structure) -> Structure:
        if isinstance(other, Structure):
            return Structure(self, *other.components)
        return Structure(self, other)

    def __ror__(self, other: Component) -> Structure:
        return Structure(other, self)


class Slab(Component):
    """One layer: a `Scatterer` with thickness and roughness (Angstrom).

    Parameters
    ----------
    thick, rough : float or refnx.analysis.Parameter
        Slab geometry in Angstrom.
    sld : Scatterer
        The material filling this slab.
    name : str, optional
        Defaults to `sld.name`.
    """

    def __init__(
        self, thick: float, sld: Scatterer, rough: float, name: str = ""
    ) -> None:
        super().__init__(name=name or sld.name)
        self.sld = sld
        self.thick = possibly_create_parameter(thick, name=f"{self.name}_thick")
        self.rough = possibly_create_parameter(rough, name=f"{self.name}_rough")
        self._parameters = Parameters(name=self.name)
        self._parameters.extend([self.thick, self.rough, sld.parameters])

    def slab_row_at(self, energy_ev: float) -> NDArray[np.float64]:
        """Delegate to `sld.slab_row_at` with this slab's current geometry."""
        return self.sld.slab_row_at(
            energy_ev,
            float(self.thick.value or 0.0),
            float(self.rough.value or 0.0),
        )

    @property
    def parameters(self) -> Parameters:
        self._parameters.name = self.name
        return self._parameters

    def __repr__(self) -> str:
        return f"Slab({self.thick!r}, {self.sld!r}, {self.rough!r}, name={self.name!r})"


class Structure:
    """An ordered stack of `Component`s, composed with the `|` operator.

    Parameters
    ----------
    *components : Component
        Initial components, fronting first, substrate last.
    name : str, optional
    """

    def __init__(self, *components: Component, name: str = "") -> None:
        self.components: list[Component] = list(components)
        self.name = name

    def __or__(self, other: Component | Structure) -> Structure:
        if isinstance(other, Structure):
            return Structure(*self.components, *other.components, name=self.name)
        return Structure(*self.components, other, name=self.name)

    def __ror__(self, other: Component) -> Structure:
        return Structure(other, *self.components, name=self.name)

    @property
    def parameters(self) -> Parameters:
        """All components' parameters, in stack order."""
        root = Parameters(name=self.name or "structure")
        root.extend([c.parameters for c in self.components])
        return root

    def slab_rows_at(self, energy_ev: float) -> NDArray[np.float64]:
        """Stacked `(len(components), 4)` packed slab rows at `energy_ev`.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV.

        Returns
        -------
        NDArray[np.float64]
            One `[thickness, delta, beta, roughness]` row per component, in
            stack order — ready for `refloxide.tmm.uniaxial_reflectivity`.
        """
        rows = [component.slab_row_at(energy_ev) for component in self.components]
        return np.asarray(rows, dtype=np.float64)


class MaterialSLD(Scatterer):
    """Isotropic material index of refraction from a chemical formula (periodictable).

    Energy is always resolved at `tensor_at` call time via
    `periodictable.xsf.index_of_refraction` — there is no fixed,
    construction-time energy baked into the tensor.

    Parameters
    ----------
    formula : str
        Chemical formula (e.g. `"Si"`, `"SiO2"`, `""` for vacuum).
    density : float or refnx.analysis.Parameter, optional
        Mass density in g/cm^3. Looked up from `periodictable` if omitted.
    energy : float, optional
        Nominal photon energy (eV) used only by `__complex__`'s quick
        isotropic-SLD summary — not used by `tensor_at`.
    name : str, optional
    energy_offset : float or refnx.analysis.Parameter, optional
        Energy offset (eV) added to whatever `energy_ev` `tensor_at` is
        called with, before the `periodictable` lookup.
    """

    def __init__(
        self,
        formula: str,
        density: float | None = None,
        energy: float = 250.0,
        name: str = "",
        *,
        energy_offset: float = 0.0,
    ) -> None:
        super().__init__(name=name)
        self._formula = pt.formula(formula)
        self._compound = formula
        if density is None:
            density = compound_density(formula)
        self.density = possibly_create_parameter(
            density, name=f"{name}_density", vary=True, bounds=(0.0, 5.0 * density)
        )
        self.energy_offset = possibly_create_parameter(
            energy_offset, name=f"{name}_energy_offset", vary=True, bounds=(-1.0, 1.0)
        )
        self.energy = float(energy)
        self._parameters = Parameters(name=name)
        self._parameters.extend([self.density, self.energy_offset])

    def tensor_at(self, energy_ev: float) -> NDArray[np.complex128]:
        eff_ev = float(energy_ev) + float(self.energy_offset.value or 0.0)
        sldc = xsf.index_of_refraction(
            self._formula,
            density=float(self.density.value or 0.0),
            energy=eff_ev * 1e-3,
        )
        if hasattr(sldc, "item"):
            sldc = sldc.item()
        n = complex(1.0) - complex(sldc)
        return np.asarray(optics.isotropic_lab_tensor(n), dtype=np.complex128)

    @property
    def parameters(self) -> Parameters:
        self._parameters.name = self.name
        return self._parameters

    def __complex__(self) -> complex:
        """Isotropic SLD (`delta + i*beta`) at this scatterer's nominal `energy`."""
        tensor = self.tensor_at(self.energy)
        return complex((2 * tensor[0, 0] + tensor[2, 2]) / 3)

    def __repr__(self) -> str:
        return (
            f"MaterialSLD({self._compound!r}, density={self.density!r}, "
            f"name={self.name!r})"
        )


class UniTensorSLD(Scatterer):
    """Uniaxial material index of refraction from a tabulated optical-constants source.

    Energy is always resolved at `tensor_at` call time via the shared,
    cached `refloxide.data.OpticalConstants` table — N scatterers
    referencing the same source share one loaded table instead of each
    loading and interpolating their own copy (see tmp/USAGE.md "Verifying
    the sharing guarantee").

    Parameters
    ----------
    ooc : OpticalConstants, polars.DataFrame, str, or pathlib.Path
        Optical-constants source, resolved via
        `refloxide.data.OpticalConstants.from_source`.
    density : float or refnx.analysis.Parameter, optional
        Mass-density scale applied to tabulated indices (g/cm^3).
    rotation : float or refnx.analysis.Parameter, optional
        Polar rotation of the molecular frame in radians.
    energy : float, optional
        Nominal photon energy (eV) used only by `__complex__`'s quick
        isotropic-SLD summary — not used by `tensor_at`.
    energy_offset : float or refnx.analysis.Parameter, optional
        Energy offset (eV) added to whatever `energy_ev` `tensor_at` is
        called with, before the table lookup.
    name : str, optional
    """

    def __init__(
        self,
        ooc: OpticalConstants | pl.DataFrame | str | Path,
        *,
        density: float = 1.0,
        rotation: float = 0.0,
        energy: float = 250.0,
        energy_offset: float = 0.0,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.ooc = OpticalConstants.from_source(ooc)
        self.density = possibly_create_parameter(
            density, name=f"{name}_density", vary=True, bounds=(0.0, 5.0 * density)
        )
        self.rotation = possibly_create_parameter(
            rotation, name=f"{name}_rotation", vary=True, bounds=(-np.pi, np.pi)
        )
        self.energy_offset = possibly_create_parameter(
            energy_offset, name=f"{name}_energy_offset", vary=True, bounds=(-0.01, 0.01)
        )
        self.energy = float(energy)
        self._parameters = Parameters(name=name)
        self._parameters.extend([self.density, self.rotation, self.energy_offset])

    def tensor_at(self, energy_ev: float) -> NDArray[np.complex128]:
        eff_ev = float(energy_ev) + float(self.energy_offset.value or 0.0)
        n_mol_xx, n_mol_zz = self.ooc.molecular_index_at(
            eff_ev, float(self.density.value or 0.0)
        )
        tensor = optics.uniaxial_lab_tensor(
            n_mol_xx, n_mol_zz, float(self.rotation.value or 0.0)
        )
        return np.asarray(tensor, dtype=np.complex128)

    @property
    def parameters(self) -> Parameters:
        self._parameters.name = self.name
        return self._parameters

    def __complex__(self) -> complex:
        """Isotropic SLD (`delta + i*beta`) at this scatterer's nominal `energy`."""
        tensor = self.tensor_at(self.energy)
        return complex((2 * tensor[0, 0] + tensor[2, 2]) / 3)

    def __repr__(self) -> str:
        return f"UniTensorSLD(name={self.name!r})"
