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
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import periodictable as pt
import periodictable.xsf as xsf
from refnx.analysis import Parameters, possibly_create_parameter

from refloxide import optics, tmm
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

    @abstractmethod
    def tensor_at(self, energy_ev: float) -> NDArray[np.complex128]:
        """This component's `(3, 3)` laboratory tensor at `energy_ev`."""
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

    def tensor_at(self, energy_ev: float) -> NDArray[np.complex128]:
        """Delegate to `sld.tensor_at` — geometry doesn't affect the tensor itself."""
        return self.sld.tensor_at(energy_ev)

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
            stack order — the `layers` argument
            `refloxide.tmm.uniaxial_reflectivity` expects.
        """
        rows = [component.slab_row_at(energy_ev) for component in self.components]
        return np.asarray(rows, dtype=np.float64)

    def tensor_rows_at(self, energy_ev: float) -> NDArray[np.complex128]:
        """Stacked `(len(components), 3, 3)` laboratory tensors at `energy_ev`.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV.

        Returns
        -------
        NDArray[np.complex128]
            One `(3, 3)` tensor per component, in stack order — the
            `tensor` argument `refloxide.tmm.uniaxial_reflectivity` expects
            (the full anisotropic tensor, not `slab_rows_at`'s isotropic
            average).
        """
        tensors = [component.tensor_at(energy_ev) for component in self.components]
        return np.asarray(tensors, dtype=np.complex128)


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


class Reflectivity(NamedTuple):
    """Both polarization channels from one `ReflectModel` evaluation.

    Parameters
    ----------
    s, p : NDArray[np.float64]
        Power reflectance for the s- and p-polarization channels, in the
        native kernel's own labeling (`refloxide.tmm`/`rust.pyi`:
        `refl[:, 0, 0] = R_ss`, `refl[:, 1, 1] = R_pp`) — not the inverted
        `pol='s' -> [:,1,1]` labeling some legacy pyref-compatibility code
        in `refloxide.pxr` uses for historical reasons. Shape `(len(q),)`
        for scalar `energy`; shape `(len(q), len(energy))` for array
        `energy`, one column per energy in `energy`'s order.
    """

    s: NDArray[np.float64]
    p: NDArray[np.float64]


_HC_EV_ANGSTROM = 12398.42
_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _theta_shifted_q(
    q: NDArray[np.float64], energy_ev: float, theta_offset_deg: float
) -> NDArray[np.float64]:
    """Shift `q` by `theta_offset_deg` via a q -> theta -> q round trip.

    Returns `q` unchanged when `theta_offset_deg == 0.0` (the common case),
    skipping the trig entirely.
    """
    if theta_offset_deg == 0.0:
        return q
    wavelength = _HC_EV_ANGSTROM / energy_ev
    arg = np.clip(q * wavelength / (4.0 * np.pi), -1.0, 1.0)
    theta_deg = np.degrees(np.arcsin(arg)) + theta_offset_deg
    return (4.0 * np.pi / wavelength) * np.sin(np.radians(theta_deg))


def _smeared_uniaxial_reflectivity(
    q: NDArray[np.float64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energy_ev: float,
    resolution_percent: float,
    *,
    parallel: bool,
) -> NDArray[np.float64]:
    """Constant-dQ/Q resolution smearing, evaluated through the Rust kernel.

    Same log-grid + Gaussian-convolution + spline-interpolation technique
    `refloxide.pxr.plugin.model._smeared_reflectivity` uses, but calling
    `refloxide.tmm.uniaxial_reflectivity` (Rust) instead of that module's
    pure-Python `tjf4x4.uniaxial_reflectivity` fallback — the actual
    "proper Rust consideration" fix, since the smearing math itself
    (`numpy.convolve` over ~51 points, `scipy` spline evaluation) was
    already cheap and vectorized; the old bottleneck was the kernel call.
    """
    from scipy.interpolate import splev, splrep

    resolution = resolution_percent / 100.0
    gaussnum = 51
    gaussgpoint = (gaussnum - 1) / 2

    lowq = max(np.min(q), 1e-6)
    highq = np.max(q)
    start = np.log10(lowq) - 6 * resolution / _FWHM
    finish = np.log10(highq * (1 + 6 * resolution / _FWHM))
    interpnum = np.round(
        np.abs(np.abs(start - finish) / (1.7 * resolution / _FWHM / gaussgpoint))
    )
    xlin = np.power(10.0, np.linspace(start, finish, int(interpnum)))

    gauss_x = np.linspace(-1.7 * resolution, 1.7 * resolution, gaussnum)
    gauss_y = (
        1.0
        / (resolution / _FWHM)
        / np.sqrt(2 * np.pi)
        * np.exp(-0.5 * gauss_x**2 / (resolution / _FWHM) ** 2)
    )

    refl, _tran = tmm.uniaxial_reflectivity(
        xlin, layers, tensor, energy_ev, parallel=parallel
    )
    refl = np.asarray(refl, dtype=np.float64)
    step = gauss_x[1] - gauss_x[0]
    smeared = np.empty_like(refl)
    for i in range(2):
        for j in range(2):
            smeared[:, i, j] = np.convolve(refl[:, i, j], gauss_y, mode="same") * step

    out = np.empty((len(q), 2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            out[:, i, j] = splev(q, splrep(xlin, smeared[:, i, j]))
    return out


class ReflectModel:
    """Turn a `Structure` into predicted reflectivity for `(q, energy)` pairs.

    No energy or polarization channel is required at construction: energy
    is a call-time argument to `__call__`, and both s and p channels are
    always returned together as a `Reflectivity` — which channel a fit
    actually uses is inferred from the measured dataset an `Objective` is
    built against (see `refloxide.objective`), not chosen here.

    Ports the correction stages of the original, load-bearing
    `refloxide.pxr.plugin.model.ReflectModel` (scale, background,
    resolution smearing, q/theta offsets) onto the Rust-backed kernel —
    that original never used the Rust kernel at all (only
    `refloxide.pxr.tjf4x4`'s pure-Python port, unless patched via
    `patch_pyref`), so this port is also a real speed fix, not just a
    rename.

    Parameters
    ----------
    structure : Structure
    parallel : bool, optional
        Forwarded to the Rust kernel. Keep `False` (the default) when
        calling from inside an already-parallel fitting loop (refnx
        workers, emcee walkers, `multiprocessing.Pool`) to avoid thread
        oversubscription; parallelize at the outer loop instead.
    name : str, optional
    scale_s, scale_p : float or refnx.analysis.Parameter, optional
        Multiplicative scale applied to the s/p channel respectively,
        before `bkg` is added.
    bkg : float or refnx.analysis.Parameter, optional
        Q-independent background added to both channels after scaling.
    dq : float or refnx.analysis.Parameter, optional
        Constant dQ/Q resolution smearing in percent (`dq=5` for 5%).
        `dq == 0` (the default) skips smearing entirely.
    q_offset : float or refnx.analysis.Parameter, optional
        Flat additive shift applied to `q` before the kernel call, shared
        by both channels and (for array `energy`) every energy.
    theta_offset_s, theta_offset_p : float or refnx.analysis.Parameter, optional
        Per-channel scattering-angle offset in degrees, applied via a
        `q -> theta -> q` round trip before the kernel call — models a
        per-polarization instrument miscalibration. When the two differ,
        `__call__` evaluates the kernel once per channel instead of once
        for both, since the two channels are then sampled at genuinely
        different q values.
    """

    def __init__(
        self,
        structure: Structure,
        *,
        parallel: bool = False,
        name: str = "",
        scale_s: float = 1.0,
        scale_p: float = 1.0,
        bkg: float = 0.0,
        dq: float = 0.0,
        q_offset: float = 0.0,
        theta_offset_s: float = 0.0,
        theta_offset_p: float = 0.0,
    ) -> None:
        self.structure = structure
        self.parallel = parallel
        self.name = name
        prefix = f"{name}_" if name else ""
        self.scale_s = possibly_create_parameter(scale_s, name=f"{prefix}scale_s")
        self.scale_p = possibly_create_parameter(scale_p, name=f"{prefix}scale_p")
        self.bkg = possibly_create_parameter(bkg, name=f"{prefix}bkg")
        self.dq = possibly_create_parameter(dq, name=f"{prefix}dq")
        self.q_offset = possibly_create_parameter(q_offset, name=f"{prefix}q_offset")
        self.theta_offset_s = possibly_create_parameter(
            theta_offset_s, name=f"{prefix}theta_offset_s"
        )
        self.theta_offset_p = possibly_create_parameter(
            theta_offset_p, name=f"{prefix}theta_offset_p"
        )

    @property
    def parameters(self) -> Parameters:
        """Instrument correction-stage parameters plus the `Structure`'s own."""
        instrument_name = f"{self.name}_instrument" if self.name else "instrument"
        instrument = Parameters(name=instrument_name)
        instrument.extend(
            [
                self.scale_s,
                self.scale_p,
                self.bkg,
                self.dq,
                self.q_offset,
                self.theta_offset_s,
                self.theta_offset_p,
            ]
        )
        root = Parameters(name=self.name or "reflect_model")
        root.extend([instrument, self.structure.parameters])
        return root

    def _kernel_call(
        self,
        q_eff: NDArray[np.float64],
        layers: NDArray[np.float64],
        tensor: NDArray[np.complex128],
        energy_ev: float,
        dq: float,
    ) -> NDArray[np.float64]:
        if dq < 0.5:
            refl, _tran = tmm.uniaxial_reflectivity(
                q_eff, layers, tensor, energy_ev, parallel=self.parallel
            )
            return np.asarray(refl, dtype=np.float64)
        return _smeared_uniaxial_reflectivity(
            q_eff, layers, tensor, energy_ev, dq, parallel=self.parallel
        )

    def _evaluate_scalar_energy(
        self, q_arr: NDArray[np.float64], energy_ev: float
    ) -> Reflectivity:
        q_off = float(self.q_offset.value or 0.0)
        theta_s = float(self.theta_offset_s.value or 0.0)
        theta_p = float(self.theta_offset_p.value or 0.0)
        dq = float(self.dq.value or 0.0)
        layers = self.structure.slab_rows_at(energy_ev)
        tensor = self.structure.tensor_rows_at(energy_ev)

        if theta_s == theta_p:
            q_eff = _theta_shifted_q(q_arr, energy_ev, theta_s) + q_off
            refl = self._kernel_call(q_eff, layers, tensor, energy_ev, dq)
            s, p = refl[:, 0, 0], refl[:, 1, 1]
        else:
            q_s = _theta_shifted_q(q_arr, energy_ev, theta_s) + q_off
            q_p = _theta_shifted_q(q_arr, energy_ev, theta_p) + q_off
            s = self._kernel_call(q_s, layers, tensor, energy_ev, dq)[:, 0, 0]
            p = self._kernel_call(q_p, layers, tensor, energy_ev, dq)[:, 1, 1]

        scale_s = float(self.scale_s.value or 1.0)
        scale_p = float(self.scale_p.value or 1.0)
        bkg = float(self.bkg.value or 0.0)
        return Reflectivity(s=scale_s * s + bkg, p=scale_p * p + bkg)

    def _evaluate_batch_energy(
        self, q_arr: NDArray[np.float64], energies_arr: NDArray[np.float64]
    ) -> Reflectivity:
        dq = float(self.dq.value or 0.0)
        theta_s = float(self.theta_offset_s.value or 0.0)
        theta_p = float(self.theta_offset_p.value or 0.0)
        if dq >= 0.5 or theta_s != theta_p:
            # Smearing and differing per-channel theta offsets aren't fused
            # into one batched kernel call yet (both make the effective q
            # grid energy- or channel-dependent, which the batch kernel's
            # single shared-q contract doesn't support) -- fall back to
            # looping the scalar path per energy. Still correct, just not
            # fused into a single Rust call; revisit if profiling shows
            # this combination is actually a hot path.
            s_cols = []
            p_cols = []
            for e in energies_arr:
                r = self._evaluate_scalar_energy(q_arr, float(e))
                s_cols.append(r.s)
                p_cols.append(r.p)
            return Reflectivity(s=np.column_stack(s_cols), p=np.column_stack(p_cols))

        q_off = float(self.q_offset.value or 0.0)
        q_eff = q_arr + q_off
        layers = np.stack(
            [self.structure.slab_rows_at(float(e)) for e in energies_arr], axis=0
        )
        tensor = np.stack(
            [self.structure.tensor_rows_at(float(e)) for e in energies_arr], axis=0
        )
        refl, _tran = tmm.uniaxial_reflectivity_batch(
            q_eff, layers, tensor, energies_arr, parallel=self.parallel
        )
        scale_s = float(self.scale_s.value or 1.0)
        scale_p = float(self.scale_p.value or 1.0)
        bkg = float(self.bkg.value or 0.0)
        # refl is (n_E, n_q, 2, 2); Reflectivity wants (n_q, n_E)
        s = scale_s * refl[:, :, 0, 0].T + bkg
        p = scale_p * refl[:, :, 1, 1].T + bkg
        return Reflectivity(s=s, p=p)

    def __call__(
        self, q: NDArray[np.float64] | float, energy: NDArray[np.float64] | float
    ) -> Reflectivity:
        """Evaluate reflectivity for `q` at `energy`.

        Parameters
        ----------
        q : float or NDArray[np.float64]
            Scattering wavevector(s) in inverse angstrom.
        energy : float or NDArray[np.float64]
            Photon energy in eV. Scalar or array — the `Structure` is
            re-materialized (via `Structure.slab_rows_at`/`tensor_rows_at`)
            fresh at every distinct energy, there is no cached,
            construction-time energy anywhere in this call.

        Returns
        -------
        Reflectivity
            Both polarization channels, scale/background/smearing/offsets
            already applied; see `Reflectivity`'s shape rules.
        """
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        if np.ndim(energy) == 0:
            return self._evaluate_scalar_energy(q_arr, float(energy))  # type: ignore[arg-type]
        energies_arr = np.atleast_1d(np.asarray(energy, dtype=np.float64))
        return self._evaluate_batch_energy(q_arr, energies_arr)

    def anisotropy(
        self, q: NDArray[np.float64] | float, energy: float
    ) -> NDArray[np.float64]:
        """`(R_p - R_s) / (R_p + R_s)`, each channel's own offsets already applied.

        Parameters
        ----------
        q : float or NDArray[np.float64]
        energy : float
            Scalar only — anisotropy is a single-energy diagnostic in the
            same sense the original `refloxide.pxr.plugin.model.ReflectModel.anisotropy`
            was.

        Returns
        -------
        NDArray[np.float64]
        """
        r = self._evaluate_scalar_energy(
            np.atleast_1d(np.asarray(q, dtype=np.float64)), float(energy)
        )
        return (r.p - r.s) / (r.p + r.s)
