"""Deferred scatterers compatible with refloxide plugin structures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import periodictable as pt
import periodictable.xsf as xsf
from refnx.analysis import Parameter, Parameters, possibly_create_parameter

from refloxide.pxr.energy.probe import Probe
from refloxide.pxr.energy.scatterer import (
    DeferredScatterer,
    OocUniTensorScatterer,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from refloxide.pxr.plugin.structure import SLD

SymmetryKind = Literal["iso", "uni", "bi"]
_ENERGY_MATCH_ATOL_EV = 1e-4


def _energy_tag(energy_ev: float) -> str:
    return f"{energy_ev:.6g}".replace(".", "p")


def _diagonal_tensor(
    n_xx: complex,
    n_yy: complex,
    n_zz: complex,
) -> NDArray[np.complex128]:
    return np.array(
        [
            [n_xx, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, n_yy, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, n_zz],
        ],
        dtype=np.complex128,
    )


class _TensorParamGroup:
    """Diagonal tensor parameters with iso, uni, or bi symmetry."""

    def __init__(
        self,
        name: str,
        symmetry: SymmetryKind,
        *,
        seed: SLD | None = None,
    ) -> None:
        self.name = name
        self.symmetry = symmetry
        n_xx = n_yy = n_zz = 1.0 + 0.0j
        if seed is not None:
            tensor = seed.tensor
            n_xx = complex(tensor[0, 0])
            n_yy = complex(tensor[1, 1])
            n_zz = complex(tensor[2, 2])
        self.xx = Parameter(n_xx.real, name=f"{name}_xx")
        self.ixx = Parameter(n_xx.imag, name=f"{name}_ixx")
        self.yy = Parameter(n_yy.real, name=f"{name}_yy")
        self.iyy = Parameter(n_yy.imag, name=f"{name}_iyy")
        self.zz = Parameter(n_zz.real, name=f"{name}_zz")
        self.izz = Parameter(n_zz.imag, name=f"{name}_izz")
        self._parameters = Parameters(name=name)
        self._parameters.extend(
            [self.xx, self.ixx, self.yy, self.iyy, self.zz, self.izz]
        )
        if seed is not None and hasattr(seed, "xx"):
            for dst, src in (
                (self.xx, seed.xx),
                (self.ixx, seed.ixx),
                (self.yy, seed.yy),
                (self.iyy, seed.iyy),
                (self.zz, seed.zz),
                (self.izz, seed.izz),
            ):
                dst.setp(
                    value=src.value,
                    vary=src.vary,
                    bounds=src.bounds,
                    constraint=src.constraint,
                )
        self._apply_symmetry(symmetry)

    def _apply_symmetry(self, symmetry: SymmetryKind) -> None:
        match symmetry:
            case "iso":
                self.yy.setp(self.xx, vary=None, constraint=self.xx)
                self.iyy.setp(self.ixx, vary=None, constraint=self.ixx)
                self.zz.setp(self.xx, vary=None, constraint=self.xx)
                self.izz.setp(self.ixx, vary=None, constraint=self.ixx)
            case "uni":
                self.yy.setp(self.xx, vary=None, constraint=self.xx)
                self.iyy.setp(self.ixx, vary=None, constraint=self.ixx)
            case "bi":
                pass

    def tensor(self) -> NDArray[np.complex128]:
        return _diagonal_tensor(
            complex(self.xx.value or 0.0, self.ixx.value or 0.0),
            complex(self.yy.value or 0.0, self.iyy.value or 0.0),
            complex(self.zz.value or 0.0, self.izz.value or 0.0),
        )

    @property
    def parameters(self) -> Parameters:
        return self._parameters


EnergyDependentScatterer = DeferredScatterer
EnergyDependentUniTensorSLD = OocUniTensorScatterer
TabulatedUniTensorSLD = OocUniTensorScatterer


class DispersiveMaterialSLD(DeferredScatterer):
    """Dispersive isotropic material from a chemical formula (periodictable)."""

    def __init__(
        self,
        formula: str,
        density: float | None = None,
        *,
        energy_offset: float = 0.0,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self._formula = pt.formula(formula)
        self._compound = formula
        if density is None:
            from refloxide.pxr.plugin.structure import compound_density

            density = compound_density(formula)
        self.density = possibly_create_parameter(  # type: ignore[assignment]
            density, name=f"{name}_rho", vary=True, bounds=(0.0, 5.0 * float(density))
        )
        self.energy_offset = possibly_create_parameter(  # type: ignore[assignment]
            energy_offset, name=f"{name}_energy_offset", vary=True, bounds=(-1.0, 1.0)
        )
        self._parameters = Parameters(name=name)
        self._parameters.extend([self.density, self.energy_offset])

    def tensor_at(self, probe: Probe) -> NDArray[np.complex128]:
        eff = self.component_probe(probe)
        energy_kev = eff.effective_ev * 1e-3
        sldc = xsf.index_of_refraction(
            self._formula,
            density=float(self.density.value or 1.0),
            energy=energy_kev,
        )
        if hasattr(sldc, "item"):
            sldc = sldc.item()
        n = complex(1.0) - complex(sldc)
        from refloxide.rust import isotropic_lab_tensor

        return np.asarray(isotropic_lab_tensor(n), dtype=np.complex128)

    def __repr__(self) -> str:
        return f"DispersiveMaterialSLD({self._compound!r}, name={self.name!r})"


class FixedTensorScatterer(DeferredScatterer):
    """Constant ``(3, 3)`` reference tensor independent of energy."""

    def __init__(
        self,
        tensor: NDArray[np.complex128],
        *,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self._fixed_tensor = np.asarray(tensor, dtype=np.complex128)
        if self._fixed_tensor.shape != (3, 3):
            msg = "FixedTensorScatterer requires a (3, 3) tensor"
            raise ValueError(msg)
        self._parameters = Parameters(name=name)

    def tensor_at(self, probe: Probe) -> NDArray[np.complex128]:
        del probe
        return self._fixed_tensor.copy()

    def __repr__(self) -> str:
        return f"FixedTensorScatterer(name={self.name!r})"


class FreeTensorScatterer(DeferredScatterer):
    """Deferred scatterer with user-declared tensor symmetry and free components.

    Parameters
    ----------
    symmetry
        ``'iso'`` (two components), ``'uni'`` (four components), or ``'bi'`` (six).
    name
        Scatterer label; parameter names follow pyref ``SLD`` conventions
        (``{name}_xx``, ``{name}_ixx``, ...).
    energies
        When ``None``, one shared parameter set applies at every photon energy.
        When a sequence is given, one parameter group is registered per energy;
        :meth:`tensor_at` selects the group whose energy matches
        ``probe.base_energy_ev`` within ``1e-4`` eV.
    """

    def __init__(
        self,
        symmetry: SymmetryKind = "uni",
        *,
        name: str = "",
        energies: Sequence[float] | None = None,
    ) -> None:
        super().__init__(name=name)
        self._symmetry = symmetry
        self._shared_group: _TensorParamGroup | None = None
        self._energy_groups: dict[float, _TensorParamGroup] = {}
        self._parameters = Parameters(name=name)
        if energies is None:
            self._shared_group = _TensorParamGroup(name, symmetry)
            self._parameters.extend([self._shared_group.parameters])
        else:
            unique = sorted({float(e) for e in energies})
            if not unique:
                msg = "FreeTensorScatterer energies must be non-empty when provided"
                raise ValueError(msg)
            for energy in unique:
                tag = _energy_tag(energy)
                group_name = f"{name}@{tag}eV" if name else f"@{tag}eV"
                group = _TensorParamGroup(group_name, symmetry)
                self._energy_groups[float(energy)] = group
                self._parameters.extend([group.parameters])

    @property
    def symmetry(self) -> SymmetryKind:
        """Declared tensor symmetry for this scatterer."""
        return self._symmetry

    @property
    def energies(self) -> tuple[float, ...] | None:
        """Registered photon energies for per-energy groups, or ``None`` when shared."""
        if self._shared_group is not None:
            return None
        return tuple(sorted(self._energy_groups))

    @classmethod
    def from_sld(
        cls,
        sld: SLD,
        *,
        energies: Sequence[float] | None = None,
        symmetry: SymmetryKind | None = None,
    ) -> FreeTensorScatterer:
        """Build a deferred free tensor from a plugin :class:`SLD`.

        Parameters
        ----------
        sld
            Source scatterer whose values, bounds, and constraints seed the new
            parameter groups.
        energies
            Forwarded to :class:`FreeTensorScatterer`; ``None`` keeps one shared
            group, a sequence registers one group per energy channel.
        symmetry
            When omitted, uses ``sld.symmetry``.
        """
        sym = symmetry if symmetry is not None else getattr(sld, "symmetry", "uni")
        out = cls(symmetry=sym, name=sld.name, energies=energies)
        if energies is None:
            out._shared_group = _TensorParamGroup(sld.name, sym, seed=sld)
            out._parameters = Parameters(name=sld.name)
            out._parameters.extend([out._shared_group.parameters])
            return out
        seeded: dict[float, _TensorParamGroup] = {}
        for energy in sorted(out._energy_groups):
            tag = _energy_tag(energy)
            group_name = f"{sld.name}@{tag}eV" if sld.name else f"@{tag}eV"
            seeded[energy] = _TensorParamGroup(group_name, sym, seed=sld)
        out._energy_groups = seeded
        out._parameters = Parameters(name=sld.name)
        for energy in sorted(out._energy_groups):
            out._parameters.extend([out._energy_groups[energy].parameters])
        return out

    def _group_for_probe(self, probe: Probe) -> _TensorParamGroup:
        if self._shared_group is not None:
            return self._shared_group
        base = float(probe.base_energy_ev)
        for energy, group in self._energy_groups.items():
            if abs(energy - base) <= _ENERGY_MATCH_ATOL_EV:
                return group
        registered = sorted(self._energy_groups)
        msg = (
            f"No FreeTensorScatterer group for base_energy_ev={base}; "
            f"registered energies: {registered}"
        )
        raise ValueError(msg)

    def group_at(self, energy_ev: float) -> _TensorParamGroup:
        """Return the per-energy tensor parameter group registered at ``energy_ev``."""
        return self._group_for_probe(Probe(base_energy_ev=float(energy_ev)))

    def write_lab_tensor(
        self,
        energy_ev: float,
        tensor: NDArray[np.complex128],
    ) -> None:
        """Load one per-energy parameter group from a laboratory ``(3, 3)`` tensor.

        Parameters
        ----------
        energy_ev
            Photon energy (eV) selecting the registered parameter group.
        tensor
            Complex laboratory-frame index tensor with shape ``(3, 3)``.
        """
        group = self._group_for_probe(Probe(base_energy_ev=float(energy_ev)))
        arr = np.asarray(tensor, dtype=np.complex128)
        if arr.shape != (3, 3):
            msg = f"write_lab_tensor requires (3, 3), got {arr.shape!r}"
            raise ValueError(msg)
        n_xx = complex(arr[0, 0])
        n_zz = complex(arr[2, 2])
        group.xx.setp(value=n_xx.real)
        group.ixx.setp(value=n_xx.imag)
        group.zz.setp(value=n_zz.real)
        group.izz.setp(value=n_zz.imag)

    def tensor_at(self, probe: Probe) -> NDArray[np.complex128]:
        """Return the laboratory ``(3, 3)`` tensor for ``probe.base_energy_ev``."""
        return self._group_for_probe(probe).tensor()

    def __repr__(self) -> str:
        if self._shared_group is not None:
            return (
                f"FreeTensorScatterer(symmetry={self._symmetry!r}, "
                f"name={self.name!r}, shared=True)"
            )
        return (
            f"FreeTensorScatterer(symmetry={self._symmetry!r}, "
            f"name={self.name!r}, energies={self.energies!r})"
        )


class FunctionScatterer(DeferredScatterer):
    """Deferred scatterer from a callable ``fn(energy_ev, **hyperparams) -> (3, 3)``.

    Parameters
    ----------
    fn
        Callable returning a complex ``(3, 3)`` laboratory tensor at the given
        photon energy in eV. Additional keyword arguments are refnx parameters
        on this scatterer.
    hyperparams
        Initial values for fit parameters passed as keyword arguments to ``fn``.
    name
        Scatterer label used to prefix hyperparameter names.
    """

    def __init__(
        self,
        fn: Callable[..., NDArray[np.complex128]],
        *,
        hyperparams: dict[str, float] | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self._fn = fn
        self._hyperparam_names: tuple[str, ...] = tuple((hyperparams or {}).keys())
        self._parameters = Parameters(name=name)
        for key, value in (hyperparams or {}).items():
            param = possibly_create_parameter(
                value,
                name=f"{name}_{key}" if name else key,
            )
            setattr(self, key, param)
            self._parameters.append(param)

    def tensor_at(self, probe: Probe) -> NDArray[np.complex128]:
        """Evaluate ``fn`` at ``probe.effective_ev`` with current hyperparameters."""
        kwargs = {
            key: float(getattr(self, key).value or 0.0)
            for key in self._hyperparam_names
        }
        tensor = self._fn(probe.effective_ev, **kwargs)
        out = np.asarray(tensor, dtype=np.complex128)
        if out.shape != (3, 3):
            msg = f"FunctionScatterer fn must return (3, 3), got {out.shape!r}"
            raise ValueError(msg)
        return out

    def __repr__(self) -> str:
        return f"FunctionScatterer(name={self.name!r})"


EnergyDependentMaterialSLD = DispersiveMaterialSLD
