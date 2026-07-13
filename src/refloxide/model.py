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
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import periodictable as pt
import periodictable.xsf as xsf
from refnx.analysis import Parameters, possibly_create_parameter

from refloxide import optics, tmm
from refloxide.data import OpticalConstants
from refloxide.pxr.plugin.structure import compound_density

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from refnx.analysis import Parameter

    from refloxide.pxr.energy.bookended import BookendedOrientationProfile


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

    def effective_density(self) -> float | None:
        """Representative mass density (g/cm^3), for `Structure.density_profile_at`.

        Returns
        -------
        float or None
            `None` by default — override for a scatterer that has a
            meaningful single density (`MaterialSLD`, `UniTensorSLD`) or a
            representative one (`MixedUniTensorSLD`'s volume-fraction
            average). Left as `None` for anything without one, so a
            depth-density plot shows a visible gap instead of a silently
            wrong flat line.
        """
        return None

    def effective_rotation(self) -> float | None:
        """Representative molecular tilt (radians), for `orientation_profile_at`.

        Returns
        -------
        float or None
            `None` by default (matches an isotropic scatterer like
            `MaterialSLD`, which has no orientation concept at all) —
            override for a scatterer with a meaningful single tilt
            (`UniTensorSLD`) or a representative one (`MixedUniTensorSLD`'s
            volume-fraction average).
        """
        return None

    def named_profile_values(self) -> dict[str, float]:
        """Extra named scalars this scatterer contributes to `Structure.plot.param`.

        Returns
        -------
        dict[str, float]
            Empty by default. `"density"`/`"orientation"` are handled
            separately via `effective_density`/`effective_rotation` — this
            is for anything else worth its own depth-profile trace, e.g.
            `MixedUniTensorSLD`'s per-component `"vf_0"`, `"vf_1"`, ...
            volume fractions.
        """
        return {}


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

    def row_and_tensor_at(
        self, energy_ev: float
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """This component's packed row and raw tensor, computed together.

        Default implementation calls `slab_row_at`/`tensor_at` independently
        — correct but, for a `Scatterer`-backed component, redoes the
        underlying tensor lookup twice. `Slab` overrides this to compute the
        tensor once and derive both outputs from it; override here too if a
        custom `Component` has an expensive `tensor_at`.
        """
        return self.slab_row_at(energy_ev), self.tensor_at(energy_ev)

    @property
    @abstractmethod
    def parameters(self) -> Parameters:
        """This component's `refnx.analysis.Parameters`."""
        raise NotImplementedError

    def geometric_thickness(self) -> float:
        """This component's physical extent along the depth axis (angstrom).

        Energy-independent, unlike `slab_row_at`/`tensor_at` — used to lay
        out `Structure.depth_grid`/`density_profile_at`/
        `orientation_profile_at`. Default assumes a `.thick` attribute
        (matches `Slab`); override for a component without one (see
        `BookendedComponent`, which reports its profile's `total_thick`).

        Raises
        ------
        NotImplementedError
            If this component has no `.thick` attribute and doesn't
            override this method.
        """
        thick = getattr(self, "thick", None)
        if thick is None:
            msg = (
                f"{type(self).__name__} has no `.thick` attribute and doesn't "
                "override geometric_thickness() -- Structure's depth-profile "
                "methods need one or the other."
            )
            raise NotImplementedError(msg)
        return float(thick.value or 0.0)

    def __or__(self, other: Component | Structure) -> Structure:
        if isinstance(other, Structure):
            return Structure(self, *other.components)
        return Structure(self, other)

    def __ror__(self, other: Component) -> Structure:
        return Structure(other, self)


class MultiRowComponent(Component):
    """A `Component` that materializes into more than one packed slab row.

    Ordinary `Component`s (`Slab`, in particular) contribute exactly one row
    per energy evaluation, which `Structure.materialize_at` assumes by
    default. A `MultiRowComponent` instead resolves into a variable number of
    rows — e.g. a smoothly-varying orientation/density profile materialized
    onto an adaptive microslab mesh (see
    `refloxide.model.BookendedComponent`). `Structure.materialize_at`
    detects `MultiRowComponent` instances with `isinstance` and concatenates
    `slab_rows_at`/`tensor_rows_at`'s output instead of treating them as a
    single row.

    Subclass this and implement `slab_rows_at`/`tensor_rows_at`/`parameters`.
    `slab_row_at`/`tensor_at`/`row_and_tensor_at` aren't meaningful for a
    component with no single row, so the inherited abstract contract is
    satisfied here with concrete implementations that raise `TypeError`.
    """

    @abstractmethod
    def slab_rows_at(self, energy_ev: float) -> NDArray[np.float64]:
        """This component's `(n_rows, 4)` packed slab rows at `energy_ev`."""
        raise NotImplementedError

    @abstractmethod
    def tensor_rows_at(self, energy_ev: float) -> NDArray[np.complex128]:
        """This component's `(n_rows, 3, 3)` laboratory tensors at `energy_ev`."""
        raise NotImplementedError

    def rows_and_tensors_at(
        self, energy_ev: float
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """This component's packed rows and raw tensors, computed together.

        Default implementation calls `slab_rows_at`/`tensor_rows_at`
        independently — correct but, if the two share expensive work (e.g.
        `BookendedComponent` resolving its profile's full tensor stack),
        redoes it twice. `Structure.materialize_at` calls this instead of
        the two separately; override here too if a custom
        `MultiRowComponent` has an expensive computation shared between them.
        """
        return self.slab_rows_at(energy_ev), self.tensor_rows_at(energy_ev)

    def slab_row_at(self, energy_ev: float) -> NDArray[np.float64]:  # noqa: ARG002
        msg = (
            f"{type(self).__name__} is a MultiRowComponent with no single "
            "row -- call slab_rows_at(energy_ev) instead."
        )
        raise TypeError(msg)

    def tensor_at(self, energy_ev: float) -> NDArray[np.complex128]:  # noqa: ARG002
        msg = (
            f"{type(self).__name__} is a MultiRowComponent with no single "
            "row -- call tensor_rows_at(energy_ev) instead."
        )
        raise TypeError(msg)

    def row_and_tensor_at(
        self,
        energy_ev: float,  # noqa: ARG002
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        msg = (
            f"{type(self).__name__} is a MultiRowComponent -- "
            "Structure.materialize_at calls slab_rows_at/tensor_rows_at "
            "directly instead of row_and_tensor_at."
        )
        raise TypeError(msg)


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

    def row_and_tensor_at(
        self, energy_ev: float
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Resolve `sld.tensor_at` once, deriving both the row and raw tensor from it.

        `slab_row_at`/`tensor_at` each independently ask `sld` for its
        tensor; calling both back to back (as `ReflectModel` used to, via
        `Structure.slab_rows_at` + `Structure.tensor_rows_at`) computed every
        scatterer's tensor twice per evaluation. This is the single-lookup
        path `Structure.materialize_at` uses instead.
        """
        tensor = self.sld.tensor_at(energy_ev)
        row = np.asarray(
            optics.tensor_to_slab_row(
                float(self.thick.value or 0.0), float(self.rough.value or 0.0), tensor
            ),
            dtype=np.float64,
        )
        return row, tensor

    @property
    def parameters(self) -> Parameters:
        self._parameters.name = self.name
        return self._parameters

    def __repr__(self) -> str:
        return f"Slab({self.thick!r}, {self.sld!r}, {self.rough!r}, name={self.name!r})"


class SLDProfile(NamedTuple):
    """Depth-resolved, roughness-broadened optical-constant profile.

    Returned by `Structure.sld_profile_at` — see that method's docstring
    for how the roughness broadening is constructed.

    Parameters
    ----------
    z : NDArray[np.float64]
        Depth (angstrom), `0` at the fronting/first-real-layer interface,
        positive toward the backing medium.
    delta, beta : NDArray[np.float64]
        Isotropic (mean-of-diagonal) real/imaginary index at each `z`.
    delta_xx, beta_xx, delta_zz, beta_zz : NDArray[np.float64]
        Per-axis real/imaginary index at each `z` — identical to
        `delta`/`beta` for an isotropic material, distinct for a uniaxial
        one.
    """

    z: NDArray[np.float64]
    delta: NDArray[np.float64]
    beta: NDArray[np.float64]
    delta_xx: NDArray[np.float64]
    beta_xx: NDArray[np.float64]
    delta_zz: NDArray[np.float64]
    beta_zz: NDArray[np.float64]


def _erf_broadened_step(
    z: NDArray[np.float64],
    boundaries: NDArray[np.float64],
    roughness: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum of error-function-broadened steps — the standard SLD-profile visualization.

    Nevot-Croce roughness enters the actual reflectivity calculation as a
    multiplicative Debye-Waller-like factor on the reflection amplitude, not
    by literally smoothing the index profile between layers —
    `Structure.materialize_at` always feeds the Rust kernel sharp-step rows,
    regardless of this function. The standard way to VISUALIZE what a given
    roughness means, though, is to broaden each interface's step by an
    error function of width `sigma` (the same convention refnx's own
    `Structure.sld_profile` uses) — mathematically consistent with the
    Gaussian-roughness assumption Nevot-Croce is itself built on, just
    applied to the profile instead of to the reflection amplitude.

    Parameters
    ----------
    z : NDArray[np.float64]
        Depth axis to evaluate on.
    boundaries : NDArray[np.float64]
        Interface positions; `boundaries[i]` separates `values[i]` from
        `values[i + 1]`.
    roughness : NDArray[np.float64]
        Roughness (angstrom) at each interface in `boundaries`, same length.
    values : NDArray[np.float64]
        One value per region, `len(boundaries) + 1` entries.

    Returns
    -------
    NDArray[np.float64]
        Broadened profile, same shape as `z`.
    """
    from scipy.special import erf

    profile = np.full_like(z, values[0])
    for i, boundary in enumerate(boundaries):
        sigma = max(float(roughness[i]), 1e-6)
        step = values[i + 1] - values[i]
        profile = profile + step * 0.5 * (
            1.0 + erf((z - boundary) / (sigma * np.sqrt(2.0)))
        )
    return profile


class _NamedProfileSegment(NamedTuple):
    """One component's contribution to a run of scalar values for one named key.

    `rough` is that component's own `Slab.rough` — by convention (matching
    `Slab`/`sld_profile_at`) the roughness of the interface immediately
    ABOVE this component, consulted when broadening the boundary between
    this segment and the one before it.
    """

    z_start: float
    mask: NDArray[np.bool_]
    value: float
    rough: Parameter


def _broaden_named_run(
    z: NDArray[np.float64],
    sharp: NDArray[np.float64],
    run: list[_NamedProfileSegment],
) -> NDArray[np.float64]:
    """Erf-broaden one maximal run of consecutive scalar-valued segments.

    Mirrors `_erf_broadened_step`'s convention exactly, restricted to this
    run's own union of depth ranges — never touches a segment outside the
    run (a `BookendedComponent`'s already-continuous values, or a gap where
    no component in that range defines this key at all), which is left
    exactly as `sharp` already has it.
    """
    boundaries = np.array([seg.z_start for seg in run[1:]], dtype=np.float64)
    roughness = np.array(
        [max(float(seg.rough.value or 0.0), 1e-6) for seg in run[1:]], dtype=np.float64
    )
    values = np.array([seg.value for seg in run], dtype=np.float64)

    run_mask = np.zeros(z.shape, dtype=np.bool_)
    for seg in run:
        run_mask |= seg.mask

    broadened = sharp.copy()
    broadened[run_mask] = _erf_broadened_step(
        z[run_mask], boundaries, roughness, values
    )
    return broadened


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
        """Stacked `(n_total_rows, 4)` packed slab rows at `energy_ev`.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV.

        Returns
        -------
        NDArray[np.float64]
            One `[thickness, delta, beta, roughness]` row per component, in
            stack order, except a `MultiRowComponent` (see
            `refloxide.model.BookendedComponent`) contributes as many rows as
            it materializes — the `layers` argument
            `refloxide.tmm.uniaxial_reflectivity` expects.
        """
        rows = [
            component.slab_rows_at(energy_ev)
            if isinstance(component, MultiRowComponent)
            else component.slab_row_at(energy_ev)[np.newaxis, :]
            for component in self.components
        ]
        return np.concatenate(rows, axis=0).astype(np.float64)

    def tensor_rows_at(self, energy_ev: float) -> NDArray[np.complex128]:
        """Stacked `(n_total_rows, 3, 3)` laboratory tensors at `energy_ev`.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV.

        Returns
        -------
        NDArray[np.complex128]
            One `(3, 3)` tensor per component, in stack order, except a
            `MultiRowComponent` contributes as many tensors as it
            materializes — the `tensor` argument
            `refloxide.tmm.uniaxial_reflectivity` expects (the full
            anisotropic tensor, not `slab_rows_at`'s isotropic average).
        """
        tensors = [
            component.tensor_rows_at(energy_ev)
            if isinstance(component, MultiRowComponent)
            else component.tensor_at(energy_ev)[np.newaxis, :, :]
            for component in self.components
        ]
        return np.concatenate(tensors, axis=0).astype(np.complex128)

    def materialize_at(
        self, energy_ev: float
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Stacked slab rows and tensors at `energy_ev`, each component resolved once.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.complex128]]
            `(rows, tensors)`, shaped as `slab_rows_at`/`tensor_rows_at`
            would return individually. Unlike calling those two methods back
            to back, an ordinary `Component` is asked for its tensor exactly
            once (via `Component.row_and_tensor_at`) -- the path
            `ReflectModel` uses, since every scatterer's `tensor_at` can be
            an expensive, non-cached lookup (e.g. `MaterialSLD` calling
            `periodictable.xsf.index_of_refraction`). A `MultiRowComponent`
            has no single row to resolve once, but is asked for its whole
            row/tensor stack together (via `MultiRowComponent.
            rows_and_tensors_at`) rather than via two separate calls, for
            the same reason.
        """
        rows = []
        tensors = []
        for component in self.components:
            if isinstance(component, MultiRowComponent):
                row, tensor = component.rows_and_tensors_at(energy_ev)
                rows.append(row)
                tensors.append(tensor)
            else:
                row, tensor = component.row_and_tensor_at(energy_ev)
                rows.append(row[np.newaxis, :])
                tensors.append(tensor[np.newaxis, :, :])
        return (
            np.concatenate(rows, axis=0).astype(np.float64),
            np.concatenate(tensors, axis=0).astype(np.complex128),
        )

    def depth_grid(
        self, *, num_points: int = 1024, pad: float = 20.0
    ) -> NDArray[np.float64]:
        """Depth axis (angstrom) spanning this structure's real layers.

        `0` at the fronting/first-real-layer interface, extended by `pad`
        on each side into the fronting/backing media. Geometry-only
        (component thickness), independent of energy — build once and
        pass the same `z` to `sld_profile_at`/`density_profile_at`/
        `orientation_profile_at` for a consistent depth axis across all
        three, rather than letting each build its own.

        Parameters
        ----------
        num_points : int, optional
        pad : float, optional
            Extra depth (angstrom) shown into the fronting/backing media
            on each side.

        Returns
        -------
        NDArray[np.float64]
        """
        total = float(sum(c.geometric_thickness() for c in self.components))
        return np.linspace(-pad, total + pad, num_points)

    def sld_profile_at(
        self,
        energy_ev: float,
        z: NDArray[np.float64] | None = None,
        *,
        num_points: int = 1024,
        pad: float = 20.0,
    ) -> SLDProfile:
        """Depth-resolved, roughness-broadened `(delta, beta)` profile at `energy_ev`.

        See `refloxide.model.SLDProfile` and `_erf_broadened_step`'s
        docstring for how the broadening is constructed — a visualization
        convention laid on top of `materialize_at`'s sharp-step packed
        rows, not a change to what the Rust kernel is actually given.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV.
        z : NDArray[np.float64], optional
            Depth axis; defaults to `self.depth_grid(num_points=num_points, pad=pad)`.
        num_points, pad : optional
            Forwarded to `depth_grid` when `z` is omitted.

        Returns
        -------
        SLDProfile
        """
        if z is None:
            z = self.depth_grid(num_points=num_points, pad=pad)
        rows, tensor = self.materialize_at(energy_ev)
        thickness = rows[:, 0]
        roughness = rows[:, 3]
        boundaries = np.concatenate([[0.0], np.cumsum(thickness[1:-1])])
        sigma = roughness[1:]

        return SLDProfile(
            z=z,
            delta=_erf_broadened_step(z, boundaries, sigma, rows[:, 1]),
            beta=_erf_broadened_step(z, boundaries, sigma, rows[:, 2]),
            delta_xx=_erf_broadened_step(z, boundaries, sigma, tensor[:, 0, 0].real),
            beta_xx=_erf_broadened_step(z, boundaries, sigma, tensor[:, 0, 0].imag),
            delta_zz=_erf_broadened_step(z, boundaries, sigma, tensor[:, 2, 2].real),
            beta_zz=_erf_broadened_step(z, boundaries, sigma, tensor[:, 2, 2].imag),
        )

    def _named_depth_walk(
        self, z: NDArray[np.float64], *, roughness: bool = False
    ) -> dict[str, NDArray[np.float64]]:
        """Every component's named depth-resolved quantities, walked once.

        Each component contributes over its own depth range: a
        `BookendedComponent` contributes `"density"`/`"orientation"` from
        its own continuous profile; a `Slab` contributes `"density"`/
        `"orientation"` from `Scatterer.effective_density`/
        `effective_rotation`, plus whatever extra named scalars
        `Scatterer.named_profile_values` returns (e.g.
        `MixedUniTensorSLD`'s per-component `"vf_0"`/`"vf_1"`). The
        first/last components are treated as semi-infinite fronting/backing
        media, filling the padding on each side of `z`.

        Two components that both define the same key (e.g. two graded
        mixed layers, each with its own `"vf_0"`) occupy disjoint depth
        ranges by construction, so they compose into ONE trace per key
        spanning the whole structure rather than colliding — exactly what
        `Structure.plot.param` needs to draw "each volume fraction as its
        own trace of depth" across a many-layer film. A key is `NaN`
        wherever no component in that depth range defines it.

        Parameters
        ----------
        z : NDArray[np.float64]
        roughness : bool, optional
            When `True`, erf-broaden each key's own component-to-component
            steps by that boundary's own `Slab.rough` — the same
            Nevot-Croce-consistent visualization convention
            `sld_profile_at` uses for the optical constants, applied here
            to every named quantity instead. Broadening only spans a
            maximal run of CONSECUTIVE `Slab` components that all define a
            given key as a plain scalar — a `BookendedComponent` (already
            continuous) or a component that doesn't define the key at all
            breaks a run, and is left exactly as its sharp value there.
        """
        results: dict[str, NDArray[np.float64]] = {}
        current_runs: dict[str, list[_NamedProfileSegment]] = {}
        completed_runs: dict[str, list[list[_NamedProfileSegment]]] = {}

        def set_value(
            key: str, mask: NDArray[np.bool_], value: float | NDArray[np.float64]
        ) -> None:
            if key not in results:
                results[key] = np.full(z.shape, np.nan, dtype=np.float64)
            results[key][mask] = value

        def break_run(key: str) -> None:
            run = current_runs.pop(key, None)
            if run:
                completed_runs.setdefault(key, []).append(run)

        cursor = 0.0
        last = len(self.components) - 1
        for i, component in enumerate(self.components):
            if i == 0:
                z_start, z_end = -np.inf, 0.0
            elif i == last:
                z_start, z_end = cursor, np.inf
            else:
                z_start, z_end = cursor, cursor + component.geometric_thickness()
            mask = (z >= z_start) & (z < z_end)

            defined_this_component: set[str] = set()
            if isinstance(component, BookendedComponent):
                local_z = z[mask] - z_start
                set_value(
                    "density",
                    mask,
                    np.asarray(component.profile.local_density(local_z)),
                )
                set_value(
                    "orientation",
                    mask,
                    np.asarray(component.profile.orientation(local_z)),
                )
            elif isinstance(component, Slab):
                scalars: dict[str, float] = {}
                density = component.sld.effective_density()
                if density is not None:
                    scalars["density"] = density
                rotation = component.sld.effective_rotation()
                if rotation is not None:
                    scalars["orientation"] = rotation
                scalars.update(component.sld.named_profile_values())

                for key, value in scalars.items():
                    set_value(key, mask, value)
                    current_runs.setdefault(key, []).append(
                        _NamedProfileSegment(z_start, mask, value, component.rough)
                    )
                    defined_this_component.add(key)

            for key in list(current_runs):
                if key not in defined_this_component:
                    break_run(key)

            if i not in (0, last):
                cursor = z_end

        for key in list(current_runs):
            break_run(key)

        if roughness:
            for key, runs in completed_runs.items():
                current = results[key]
                for run in runs:
                    if len(run) >= 2:
                        current = _broaden_named_run(z, current, run)
                results[key] = current

        return results

    def named_profiles_at(
        self,
        z: NDArray[np.float64] | None = None,
        *,
        num_points: int = 1024,
        pad: float = 20.0,
        roughness: bool = False,
    ) -> dict[str, NDArray[np.float64]]:
        """Every named depth-resolved quantity this structure's components define.

        Always includes `"density"`/`"orientation"` (see
        `density_profile_at`/`orientation_profile_at`), plus any extra keys
        a component's `Scatterer.named_profile_values` defines — e.g.
        `MixedUniTensorSLD` contributes `"vf_0"`, `"vf_1"`, ... for its
        blended components. Backing `Structure.plot.param`'s regex
        matching.

        Parameters
        ----------
        z : NDArray[np.float64], optional
            Depth axis; defaults to `self.depth_grid(num_points=num_points, pad=pad)`.
        num_points, pad : optional
            Forwarded to `depth_grid` when `z` is omitted.
        roughness : bool, optional
            When `True`, erf-broaden each key's own interfaces by that
            boundary's roughness — see `_named_depth_walk`. `False` (sharp
            steps) by default.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            Each array is `NaN` wherever no component in that depth range
            defines that key.
        """
        if z is None:
            z = self.depth_grid(num_points=num_points, pad=pad)
        return self._named_depth_walk(z, roughness=roughness)

    def density_profile_at(
        self,
        z: NDArray[np.float64] | None = None,
        *,
        num_points: int = 1024,
        pad: float = 20.0,
        roughness: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Depth-resolved mass density (g/cm^3), `NaN` wherever a component has none.

        `MaterialSLD`/`UniTensorSLD` contribute their own `.density`;
        `MixedUniTensorSLD` contributes its volume-fraction-weighted
        average; a `BookendedComponent` contributes its own continuous
        `local_density(depth)` profile. Anything else — a custom
        `Scatterer` that doesn't override `effective_density` — contributes
        `NaN` over its depth range, so a gap is visible rather than a
        silently wrong flat line.

        Parameters
        ----------
        z : NDArray[np.float64], optional
            Depth axis; defaults to `self.depth_grid(num_points=num_points, pad=pad)`.
        num_points, pad : optional
            Forwarded to `depth_grid` when `z` is omitted.
        roughness : bool, optional
            When `True`, erf-broaden each Slab-to-Slab density step by that
            interface's own roughness (see `_named_depth_walk`); `False`
            (sharp steps) by default.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            `(z, density)`.
        """
        if z is None:
            z = self.depth_grid(num_points=num_points, pad=pad)
        profiles = self._named_depth_walk(z, roughness=roughness)
        density = profiles.get("density", np.full(z.shape, np.nan, dtype=np.float64))
        return z, density

    def orientation_profile_at(
        self,
        z: NDArray[np.float64] | None = None,
        *,
        num_points: int = 1024,
        pad: float = 20.0,
        roughness: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Depth-resolved molecular tilt (radians), `NaN` wherever a component has none.

        `UniTensorSLD` contributes its own `.rotation`; `MixedUniTensorSLD`
        contributes its volume-fraction-weighted average; a
        `BookendedComponent` contributes its own continuous
        `orientation(depth)` profile. `MaterialSLD` (isotropic — no
        orientation concept) and any custom `Scatterer` that doesn't
        override `effective_rotation` contribute `NaN`.

        Parameters
        ----------
        z : NDArray[np.float64], optional
            Depth axis; defaults to `self.depth_grid(num_points=num_points, pad=pad)`.
        num_points, pad : optional
            Forwarded to `depth_grid` when `z` is omitted.
        roughness : bool, optional
            When `True`, erf-broaden each Slab-to-Slab tilt step by that
            interface's own roughness (see `_named_depth_walk`); `False`
            (sharp steps) by default.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            `(z, orientation)`.
        """
        if z is None:
            z = self.depth_grid(num_points=num_points, pad=pad)
        profiles = self._named_depth_walk(z, roughness=roughness)
        orientation = profiles.get(
            "orientation", np.full(z.shape, np.nan, dtype=np.float64)
        )
        return z, orientation

    @property
    def plot(self) -> StructurePlot:
        """Depth-profile plotting accessor: `structure.plot.oc(...)`, `.param(...)`.

        Returns
        -------
        StructurePlot
            See `StructurePlot.oc`/`StructurePlot.param`.
        """
        return StructurePlot(self)


class StructurePlot:
    """Depth-profile plotting for one `Structure` — `structure.plot.oc(...)`/`.param()`.

    Get one via the `Structure.plot` property; not meant to be constructed
    directly. Requires `matplotlib`, imported lazily inside each method
    (not a hard runtime dependency of `refloxide.model` itself).

    Parameters
    ----------
    structure : Structure
    """

    def __init__(self, structure: Structure) -> None:
        self._structure = structure

    def oc(
        self,
        energy_ev: float,
        *,
        z: NDArray[np.float64] | None = None,
        num_points: int = 1024,
        pad: float = 20.0,
        ax: Axes | None = None,
        difference: bool = False,
        inset: bool = True,
    ) -> tuple[Figure, Axes] | tuple[Figure, tuple[Axes, Axes]]:
        """Plot the depth-resolved index of refraction at `energy_ev`.

        Same layout convention as pyref's `Structure.plot`: the isotropic
        delta/beta (solid) plus each axis's own `xx`/`zz` component
        (dashed/dotted), roughness-broadened per interface — see
        `Structure.sld_profile_at`.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV.
        z : NDArray[np.float64], optional
            Depth axis; defaults to `Structure.depth_grid(num_points, pad)`.
        num_points, pad : optional
            Forwarded to `depth_grid` when `z` is omitted.
        ax : matplotlib.axes.Axes, optional
            Axes to draw the index-of-refraction traces on. A new
            figure/axes pair is created when omitted. When `ax` is given
            directly and `difference=True`, the difference is always drawn
            as a twin axis (`inset=True`'s behavior) — a caller-supplied
            single `ax` doesn't give this method a figure to add a second
            subplot to.
        difference : bool, optional
            Also plot `delta_xx - delta_zz` (the birefringence/dichroism)
            — on a right-hand twin axis when `inset=True` (pyref's own
            convention: filled curve, y-limits matched to the primary
            axis, its own color/label), or as a separate subplot below the
            main plot when `inset=False`.
        inset : bool, optional
            Only consulted when `difference=True` and `ax` is omitted; see
            above.

        Returns
        -------
        tuple[Figure, Axes] or tuple[Figure, tuple[Axes, Axes]]
            `(fig, ax)`, or `(fig, (ax, ax_difference))` when
            `difference=True`.
        """
        import matplotlib.pyplot as plt

        structure = self._structure
        if z is None:
            z = structure.depth_grid(num_points=num_points, pad=pad)
        profile = structure.sld_profile_at(energy_ev, z=z)

        ax_diff: Axes | None = None
        use_inset = inset
        if ax is not None:
            fig = cast("Figure", ax.figure)
            use_inset = True
        elif difference and not inset:
            fig, (ax, ax_diff) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        else:
            fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(
            profile.z, profile.delta, color="C0", lw=0.9, zorder=20, label=r"$\delta$"
        )
        ax.plot(
            profile.z,
            profile.delta_xx,
            color="C0",
            ls="--",
            zorder=10,
            label=r"$\delta_{xx}$",
        )
        ax.plot(
            profile.z,
            profile.delta_zz,
            color="C0",
            ls=":",
            zorder=10,
            label=r"$\delta_{zz}$",
        )
        ax.plot(
            profile.z, profile.beta, color="C2", lw=0.9, zorder=20, label=r"$\beta$"
        )
        ax.plot(
            profile.z,
            profile.beta_xx,
            color="C2",
            ls="--",
            zorder=10,
            label=r"$\beta_{xx}$",
        )
        ax.plot(
            profile.z,
            profile.beta_zz,
            color="C2",
            ls=":",
            zorder=10,
            label=r"$\beta_{zz}$",
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Index of refraction")
        ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
        ax.legend(fontsize="small")

        if not difference:
            return fig, ax

        dichroism = profile.delta_xx - profile.delta_zz
        if use_inset:
            ax_diff = ax.twinx()
            ax_diff.plot(profile.z, dichroism, color="C1", zorder=20)
            ax_diff.fill_between(
                profile.z,
                dichroism,
                color="C1",
                alpha=0.5,
                zorder=5,
                label=r"$\delta_{xx} - \delta_{zz}$",
            )
            ax_diff.set_ylabel(r"$\delta_{xx} - \delta_{zz}$")
            ax_diff.tick_params(axis="y", labelcolor="C1", color="C1")
            ax_diff.spines["right"].set_color("C1")
            ax_diff.set_ylim(ax.get_ylim())
            ax_diff.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax_diff.legend(fontsize="small", loc="upper right")
        else:
            assert ax_diff is not None  # guaranteed by the subplot branch above
            ax_diff.plot(profile.z, dichroism, color="C1")
            ax_diff.fill_between(profile.z, dichroism, color="C1", alpha=0.5)
            ax_diff.set_ylabel(r"$\delta_{xx} - \delta_{zz}$")
            ax_diff.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
        fig.tight_layout()
        return fig, (ax, ax_diff)

    def param(
        self,
        pattern: str,
        *,
        z: NDArray[np.float64] | None = None,
        num_points: int = 1024,
        pad: float = 20.0,
        roughness: bool = False,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot every named depth-profile quantity matching a regex, one trace each.

        Parameters
        ----------
        pattern : str
            Regular expression (`re.search`) matched against
            `Structure.named_profiles_at`'s keys — `"density"`,
            `"orientation"`, and any scatterer-specific extras (e.g.
            `MixedUniTensorSLD`'s `"vf_0"`/`"vf_1"`). `"vf_"` plots every
            blended component's volume fraction as its own trace across
            the whole structure; `"vf_0"` plots only the first.
        z : NDArray[np.float64], optional
            Depth axis; defaults to `Structure.depth_grid(num_points, pad)`.
        num_points, pad : optional
            Forwarded to `depth_grid` when `z` is omitted.
        roughness : bool, optional
            When `True`, erf-broaden each matched trace's own Slab-to-Slab
            steps by that interface's own roughness — the same
            Nevot-Croce-consistent convention `Structure.plot.oc` uses for
            the optical constants (see `Structure.named_profiles_at`).
            `False` (sharp steps) by default.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure/axes pair is created when
            omitted.

        Returns
        -------
        tuple[Figure, Axes]

        Raises
        ------
        ValueError
            If `pattern` matches none of this structure's named-profile
            keys.
        """
        import re

        import matplotlib.pyplot as plt

        structure = self._structure
        if z is None:
            z = structure.depth_grid(num_points=num_points, pad=pad)
        profiles = structure.named_profiles_at(z=z, roughness=roughness)
        matched = {
            name: values
            for name, values in profiles.items()
            if re.search(pattern, name)
        }
        if not matched:
            msg = (
                f"no depth-profile quantity matches {pattern!r} -- available: "
                f"{sorted(profiles)}"
            )
            raise ValueError(msg)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = cast("Figure", ax.figure)
        for name, values in sorted(matched.items()):
            ax.plot(z, values, label=name)
        ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
        ax.legend(fontsize="small")
        return fig, ax


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
        self._tensor_cache: dict[tuple[float, float], NDArray[np.complex128]] = {}

    def tensor_at(self, energy_ev: float) -> NDArray[np.complex128]:
        eff_ev = float(energy_ev) + float(self.energy_offset.value or 0.0)
        density = float(self.density.value or 0.0)
        key = (eff_ev, density)
        cached = self._tensor_cache.get(key)
        if cached is not None:
            return cached

        sldc = xsf.index_of_refraction(
            self._formula, density=density, energy=eff_ev * 1e-3
        )
        if hasattr(sldc, "item"):
            sldc = sldc.item()
        n = complex(1.0) - complex(sldc)
        tensor = np.asarray(optics.isotropic_lab_tensor(n), dtype=np.complex128)

        # Bounded so a fit that varies density/energy_offset every iteration
        # (a cache miss on every call) can't grow this without limit.
        if len(self._tensor_cache) >= 256:
            self._tensor_cache.clear()
        self._tensor_cache[key] = tensor
        return tensor

    @property
    def parameters(self) -> Parameters:
        self._parameters.name = self.name
        return self._parameters

    def __complex__(self) -> complex:
        """Isotropic SLD (`delta + i*beta`) at this scatterer's nominal `energy`."""
        tensor = self.tensor_at(self.energy)
        return complex((2 * tensor[0, 0] + tensor[2, 2]) / 3)

    def effective_density(self) -> float | None:
        return float(self.density.value or 0.0)

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

    def effective_density(self) -> float | None:
        return float(self.density.value or 0.0)

    def effective_rotation(self) -> float | None:
        return float(self.rotation.value or 0.0)

    def __repr__(self) -> str:
        return f"UniTensorSLD(name={self.name!r})"


class MixedUniTensorSLD(Scatterer):
    """Volume-fraction-weighted mixture of uniaxial materials from tabulated OOC tables.

    Each component contributes its own tabulated optical-constants source,
    density, molecular tilt, and energy offset. The laboratory tensor is the
    volume-fraction-weighted sum of every component's own laboratory-frame
    diagonal (`refloxide.optics.uniaxial_lab_tensor` applied per component,
    then mixed) — mixing happens after each component's own uniaxial
    projection, not on the raw molecular indices before projection.

    Parameters
    ----------
    oocs : sequence of (OpticalConstants, polars.DataFrame, str, or pathlib.Path)
        One optical-constants source per component, each resolved via
        `refloxide.data.OpticalConstants.from_source` — components that
        reference the same underlying file or `OpticalConstants` share the
        one cached, loaded table.
    vf : sequence of float or refnx.analysis.Parameter
        Volume fraction per component, same length as `oocs`. Each is an
        independent, freely-varying `Parameter` bounded to `(0, 1)` — not
        constrained to sum to 1; callers who want a strict partition should
        fix all but one and add a `refnx` constraint on the last.
    rotation : sequence of float or refnx.analysis.Parameter
        Polar rotation of each component's molecular frame, in radians.
    density : sequence of float or refnx.analysis.Parameter
        Mass-density scale per component (g/cm^3).
    energy : float, optional
        Nominal photon energy (eV) used only by `__complex__`'s quick
        isotropic-SLD summary — not used by `tensor_at`.
    energy_offset : sequence of float or refnx.analysis.Parameter, optional
        Energy offset (eV) per component, added to whatever `energy_ev`
        `tensor_at` is called with before that component's table lookup.
        Defaults to zero for every component.
    name : str, optional

    Raises
    ------
    ValueError
        If `oocs`, `vf`, `rotation`, `density`, and (when given)
        `energy_offset` are not all the same length.
    """

    def __init__(
        self,
        oocs: Sequence[OpticalConstants | pl.DataFrame | str | Path],
        vf: Sequence[float],
        rotation: Sequence[float],
        density: Sequence[float],
        *,
        energy: float = 250.0,
        energy_offset: Sequence[float] | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        n = len(oocs)
        if energy_offset is None:
            energy_offset = [0.0] * n
        lengths = {len(oocs), len(vf), len(rotation), len(density), len(energy_offset)}
        if lengths != {n}:
            msg = (
                "MixedUniTensorSLD component sequences must all be the same "
                f"length: oocs={len(oocs)}, vf={len(vf)}, rotation={len(rotation)}, "
                f"density={len(density)}, energy_offset={len(energy_offset)}"
            )
            raise ValueError(msg)

        self.oocs = [OpticalConstants.from_source(ooc) for ooc in oocs]
        self.vf = [
            possibly_create_parameter(
                v, name=f"{name}_vf_{i}", vary=True, bounds=(0.0, 1.0)
            )
            for i, v in enumerate(vf)
        ]
        self.rotation = [
            possibly_create_parameter(
                r, name=f"{name}_rotation_{i}", vary=True, bounds=(-np.pi, np.pi)
            )
            for i, r in enumerate(rotation)
        ]
        self.density = [
            possibly_create_parameter(
                d, name=f"{name}_density_{i}", vary=True, bounds=(0.0, 5.0 * d)
            )
            for i, d in enumerate(density)
        ]
        self.energy_offset = [
            possibly_create_parameter(
                eo, name=f"{name}_energy_offset_{i}", vary=True, bounds=(-0.01, 0.01)
            )
            for i, eo in enumerate(energy_offset)
        ]
        self.energy = float(energy)
        self._parameters = Parameters(name=name)
        self._parameters.extend(self.vf)
        self._parameters.extend(self.rotation)
        self._parameters.extend(self.density)
        self._parameters.extend(self.energy_offset)

    def tensor_at(self, energy_ev: float) -> NDArray[np.complex128]:
        n_o = complex(0.0)
        n_e = complex(0.0)
        components = zip(
            self.oocs,
            self.vf,
            self.rotation,
            self.density,
            self.energy_offset,
            strict=True,
        )
        for ooc, vf, rotation, density, energy_offset in components:
            eff_ev = float(energy_ev) + float(energy_offset.value or 0.0)
            n_mol_xx, n_mol_zz = ooc.molecular_index_at(
                eff_ev, float(density.value or 0.0)
            )
            rotation_rad = float(rotation.value or 0.0)
            component_tensor = np.asarray(
                optics.uniaxial_lab_tensor(n_mol_xx, n_mol_zz, rotation_rad),
                dtype=np.complex128,
            )
            weight = float(vf.value or 0.0)
            n_o += component_tensor[0, 0] * weight
            n_e += component_tensor[2, 2] * weight
        return np.diag(np.array([n_o, n_o, n_e], dtype=np.complex128))

    @property
    def parameters(self) -> Parameters:
        self._parameters.name = self.name
        return self._parameters

    def __complex__(self) -> complex:
        """Isotropic SLD (`delta + i*beta`) at this scatterer's nominal `energy`."""
        tensor = self.tensor_at(self.energy)
        return complex((2 * tensor[0, 0] + tensor[2, 2]) / 3)

    def _volume_fraction_weighted(self, values: Sequence[Parameter]) -> float | None:
        weights = np.array([float(v.value or 0.0) for v in self.vf])
        raw = np.array([float(x.value or 0.0) for x in values])
        total = float(weights.sum())
        if total == 0.0:
            return None
        return float(np.sum(weights * raw) / total)

    def effective_density(self) -> float | None:
        """Volume-fraction-weighted average density across every blended component."""
        return self._volume_fraction_weighted(self.density)

    def effective_rotation(self) -> float | None:
        """Volume-fraction-weighted average tilt across every blended component."""
        return self._volume_fraction_weighted(self.rotation)

    def named_profile_values(self) -> dict[str, float]:
        """Each blended component's own vf, keyed `"vf_0"`, `"vf_1"`, ..."""
        return {f"vf_{i}": float(v.value or 0.0) for i, v in enumerate(self.vf)}

    def __repr__(self) -> str:
        return f"MixedUniTensorSLD(n={len(self.oocs)}, name={self.name!r})"


class BookendedComponent(MultiRowComponent):
    """Adapts a `BookendedOrientationProfile` to refloxide.model's `Structure` protocol.

    `BookendedOrientationProfile` (`refloxide.pxr.energy.bookended`)
    implements the legacy `refloxide.pxr.plugin.structure.Component`
    protocol (`.slabs()`/`.tensor()`) so it keeps working with the
    load-bearing `pxr.plugin` stack unchanged. This is a thin wrapper around
    the SAME profile instance — same live `Parameters`, same OOC binding —
    exposed through `refloxide.model.MultiRowComponent` instead, so a
    book-ended film composes into a `refloxide.model.Structure` with `|`
    alongside ordinary `MaterialSLD`/`UniTensorSLD` slabs and fits through
    `refloxide.model.ReflectModel`/`refloxide.objective.Objective` like any
    other component.

    Parameters
    ----------
    profile : refloxide.pxr.energy.bookended.BookendedOrientationProfile
        The profile to wrap. Its OOC table and nominal energy may be bound
        already (`ooc=`/`energy=` at construction) or deferred via
        `profile.bind_ooc(...)` before this component is ever evaluated —
        same deferred-by-default contract as the profile itself.
    """

    def __init__(self, profile: BookendedOrientationProfile) -> None:
        super().__init__(name=profile.name)
        self.profile = profile

    def rows_and_tensors_at(
        self, energy_ev: float
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Resolve `profile.tensor` once, deriving both rows and tensors from it.

        `slab_rows_at`/`tensor_rows_at` each independently ask the profile
        for its full tensor stack; calling both back to back (as
        `Structure.materialize_at` used to, for every `MultiRowComponent`)
        computed the whole orientation/density/tensor pipeline twice. Row
        packing is vectorized -- equivalent to calling
        `optics.tensor_to_slab_row` once per microslab (`n_avg =
        mean(diag(tensor))`, matching the crate-wide convention), but done
        for every row at once in numpy instead of one Rust FFI call per
        microslab, which stops being negligible once `num_slabs` reaches
        the hundreds to thousands.
        """
        thicknesses = self.profile.slab_thick
        tensors = np.asarray(self.profile.tensor(energy_ev), dtype=np.complex128)
        roughness = float(self.profile.surface_roughness.value or 0.0)
        n_avg = (tensors[:, 0, 0] + tensors[:, 1, 1] + tensors[:, 2, 2]) / 3.0
        rows = np.empty((tensors.shape[0], 4), dtype=np.float64)
        rows[:, 0] = thicknesses
        rows[:, 1] = n_avg.real
        rows[:, 2] = n_avg.imag
        rows[:, 3] = 0.0
        rows[0, 3] = roughness
        return rows, tensors

    def slab_rows_at(self, energy_ev: float) -> NDArray[np.float64]:
        return self.rows_and_tensors_at(energy_ev)[0]

    def tensor_rows_at(self, energy_ev: float) -> NDArray[np.complex128]:
        return self.rows_and_tensors_at(energy_ev)[1]

    @property
    def parameters(self) -> Parameters:
        return self.profile.parameters

    def geometric_thickness(self) -> float:
        return float(self.profile.total_thick.value or 0.0)

    def __repr__(self) -> str:
        return f"BookendedComponent({self.profile!r})"


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
    `refloxide.tmm.uniaxial_reflectivity` (Rust) instead of
    `refloxide.python.tmm.uniaxial_reflectivity` — the actual
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


def _is_isotropic_tensor(tensor: NDArray[np.complex128], rtol: float = 1e-9) -> bool:
    """Whether a `(3, 3)` tensor's diagonal is uniform to within `rtol`.

    The fully-fused bookended Rust path (`_fused_bookended_reflectivity`)
    carries only an isotropic `[thickness, delta, beta, roughness]` row for
    its fronting/backing components, not a separate tensor the way the
    general assembled path does — correct only when those components
    really are isotropic (true for `MaterialSLD`, generally false for a
    rotated `UniTensorSLD`/`MixedUniTensorSLD`).
    """
    diag = np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2]])
    scale = max(1e-30, float(np.max(np.abs(diag))))
    return bool(np.max(np.abs(diag - diag[0])) <= rtol * scale)


class _FusedBookendedPlan(NamedTuple):
    """A `Structure` confirmed eligible for the fully-Rust-fused bookended path."""

    component: BookendedComponent
    fronting_row: NDArray[np.float64]
    backing_rows: NDArray[np.float64]


def _plan_fused_bookended(
    structure: Structure, energy_ev: float
) -> _FusedBookendedPlan | None:
    """Check whether `structure` qualifies for the fused Rust bookended kernel.

    That kernel does mesh generation, the orientation/density profile, and
    per-microslab tensor construction entirely in Rust — skipping
    `Structure.materialize_at`'s Python/numpy intermediates for the whole
    structure, not just the profile — but its signature only accepts one
    isotropic fronting row and one-or-more isotropic backing rows (see
    `refloxide.rust.bookended_uniaxial_reflectivity`'s `.pyi` signature),
    not the general per-component tensor stack. This matches when:

    - `structure` has exactly one `BookendedComponent`, immediately preceded
      by exactly one ordinary fronting component and followed by one or more
      ordinary backing components (no other `MultiRowComponent` present).
    - The profile's OOC table uses linear interpolation (the Rust kernel's
      only supported mode; `pchip` falls back to the assembled path).
    - The fronting and every backing component's tensor is actually
      isotropic at `energy_ev` (checked directly, not assumed from type).

    Returns `None` for anything else, in which case `ReflectModel` falls
    back to the general, always-correct assembled path.
    """
    indices = [
        i
        for i, c in enumerate(structure.components)
        if isinstance(c, BookendedComponent)
    ]
    if len(indices) != 1 or indices[0] != 1:
        return None
    component = structure.components[1]
    if not isinstance(component, BookendedComponent):
        return None
    if component.profile.anchor.interp != "linear":
        return None
    before = structure.components[:1]
    after = structure.components[2:]
    if len(before) != 1 or not after:
        return None
    if any(isinstance(c, MultiRowComponent) for c in after):
        return None

    fronting_row, fronting_tensor = before[0].row_and_tensor_at(energy_ev)
    if not _is_isotropic_tensor(fronting_tensor):
        return None
    backing_rows = []
    for c in after:
        row, tensor = c.row_and_tensor_at(energy_ev)
        if not _is_isotropic_tensor(tensor):
            return None
        backing_rows.append(row)
    return _FusedBookendedPlan(
        component=component,
        fronting_row=fronting_row,
        backing_rows=np.asarray(backing_rows, dtype=np.float64),
    )


def _fused_bookended_reflectivity(
    plan: _FusedBookendedPlan,
    q: NDArray[np.float64],
    energy_ev: float,
    *,
    parallel: bool,
) -> NDArray[np.float64]:
    """Evaluate reflectivity through the fully-Rust-fused bookended kernel.

    Mesh generation, the orientation/density profile, and per-microslab
    tensor construction all happen inside one Rust call instead of building
    numpy intermediates in Python first — the same physics as
    `BookendedComponent.rows_and_tensors_at` fed through
    `refloxide.tmm.uniaxial_reflectivity`, just without materializing any of
    it on the Python side.
    """
    profile = plan.component.profile
    anchor = profile.anchor
    query_ev = profile.probe_at(energy_ev).effective_ev
    refl, _tran = tmm.bookended_uniaxial_reflectivity(
        np.asarray(q, dtype=np.float64),
        anchor.energy_ev,
        anchor.n_xx,
        anchor.n_ixx,
        anchor.n_zz,
        anchor.n_izz,
        query_ev,
        total_thick=float(profile.total_thick.value or 0.0),
        surface_roughness=float(profile.surface_roughness.value or 0.0),
        tau_si=float(profile.tau_si.value or 0.0),
        tau_vac=float(profile.tau_vac.value or 0.0),
        alpha_bulk=float(profile.alpha_bulk.value or 0.0),
        alpha_si=float(profile.alpha_si.value or 0.0),
        alpha_vac=float(profile.alpha_vac.value or 0.0),
        density_bulk=float(profile.density_bulk.value or 1.0),
        density_si=float(profile.density_si.value or 0.0),
        density_vac=float(profile.density_vac.value or 0.0),
        num_slabs=int(profile.num_slabs),
        mesh_constant=float(profile.mesh_constant),
        fronting=np.asarray(plan.fronting_row, dtype=np.float64),
        backing=plan.backing_rows,
        parallel=parallel,
    )
    return np.asarray(refl, dtype=np.float64)


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
    that original historically used only the pure-Python TMM
    (`refloxide.python.tmm`), so this port is also a real speed fix, not
    just a rename.

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

        # Smearing isn't wired into the fused bookended path yet -- only try
        # it dq-free; _kernel_call's own smearing branch stays the fallback.
        fused_plan = None
        if dq < 0.5:
            fused_plan = _plan_fused_bookended(self.structure, energy_ev)
        materialized: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None

        def kernel(q_eff: NDArray[np.float64]) -> NDArray[np.float64]:
            nonlocal materialized
            if fused_plan is not None:
                return _fused_bookended_reflectivity(
                    fused_plan, q_eff, energy_ev, parallel=self.parallel
                )
            if materialized is None:
                materialized = self.structure.materialize_at(energy_ev)
            layers, tensor = materialized
            return self._kernel_call(q_eff, layers, tensor, energy_ev, dq)

        if theta_s == theta_p:
            q_eff = _theta_shifted_q(q_arr, energy_ev, theta_s) + q_off
            refl = kernel(q_eff)
            s, p = refl[:, 0, 0], refl[:, 1, 1]
        else:
            q_s = _theta_shifted_q(q_arr, energy_ev, theta_s) + q_off
            q_p = _theta_shifted_q(q_arr, energy_ev, theta_p) + q_off
            s = kernel(q_s)[:, 0, 0]
            p = kernel(q_p)[:, 1, 1]

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
        scale_s = float(self.scale_s.value or 1.0)
        scale_p = float(self.scale_p.value or 1.0)
        bkg = float(self.bkg.value or 0.0)

        fused_plan = _plan_fused_bookended(self.structure, float(energies_arr[0]))
        if fused_plan is not None:
            s_cols = []
            p_cols = []
            for e in energies_arr:
                refl = _fused_bookended_reflectivity(
                    fused_plan, q_eff, float(e), parallel=self.parallel
                )
                s_cols.append(refl[:, 0, 0])
                p_cols.append(refl[:, 1, 1])
            return Reflectivity(
                s=scale_s * np.column_stack(s_cols) + bkg,
                p=scale_p * np.column_stack(p_cols) + bkg,
            )

        materialized = [self.structure.materialize_at(float(e)) for e in energies_arr]
        layers = np.stack([rows for rows, _tensor in materialized], axis=0)
        tensor = np.stack([tensor for _rows, tensor in materialized], axis=0)
        refl, _tran = tmm.uniaxial_reflectivity_batch(
            q_eff, layers, tensor, energies_arr, parallel=self.parallel
        )
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
