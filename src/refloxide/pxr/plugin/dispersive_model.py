"""Multi-energy reflectivity models with per-energy instrument parameters.

:class:`DispersiveReflectModel` holds one
:class:`~refloxide.pxr.energy.structure.DispersiveStructure`
and independent or linked instrument parameters for each photon energy. Energy-band
queries configure whether a field is fixed, shared across energies, or fit per energy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from refnx.analysis import Parameters

from refloxide.pxr.energy.structure import DispersiveStructure  # noqa: TC001
from refloxide.pxr.layout import reflectivity_for_pol
from refloxide.pxr.plugin.dispersive_instrument import (
    BOOKENDED_FILM_PARAM_NAMES,
    INSTRUMENT_FIELDS,
    REFLECT_MODEL_INSTRUMENTATION,
    EnergyInstrumentSlice,
    InstrumentField,
    InstrumentFieldQuery,
    InstrumentParameterView,
    ResolvedInstrument,
    copy_parameter_state,
    make_instrument_channel,
    resolve_instrument,
    safely_setp_param,
)
from refloxide.pxr.plugin.model import ReflectModel, reflectivity

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from refloxide.pxr.energy.bookended import BookendedOrientationProfile

PolKind = Literal["s", "p", "sp", "ps"]
BackendKind = Literal["uni", "bi"]

__all__ = [
    "BOOKENDED_FILM_PARAM_NAMES",
    "REFLECT_MODEL_INSTRUMENTATION",
    "DispersiveReflectModel",
    "DispersiveReflectObjective",
    "InstrumentFieldQuery",
    "InstrumentParameterView",
    "resolve_instrument",
    "safely_setp_param",
    "select",
]


class DispersiveReflectModel:
    """Energy-independent reflectivity model with per-energy instrument parameters.

    One :class:`~refloxide.pxr.energy.structure.DispersiveStructure` is shared across
    all photon energies. Instrument parameters (scales, offsets, backgrounds) live
    in :class:`EnergyInstrumentSlice` channels keyed by energy. Configure bands with
    ``model.theta_offset_s.where(...).setp(...)`` (same vocabulary as pyref
    ``ReflectModel`` ``setp`` calls in the fitting notebooks).

    Structure geometry (including
    :class:`~refloxide.pxr.energy.bookended.BookendedOrientationProfile` film
    parameters) is shared across energies automatically because there is a single
    stack instance.

    Parameters
    ----------
    structure
        Deferred-energy stack evaluated at each channel energy.
    energies
        Distinct photon energies (eV) with instrument channels.
    pol
        Default polarization when :meth:`model` is called without overriding.
    name
        Label for the refnx :class:`~refnx.analysis.Parameters` tree.
    backend
        Uniaxial (``'uni'``) or biaxial (``'bi'``) reflectivity backend.
    instrument_defaults
        Optional default floats for newly created per-energy parameters.
    """

    def __init__(
        self,
        structure: DispersiveStructure,
        energies: Sequence[float],
        *,
        pol: PolKind = "s",
        name: str = "",
        backend: BackendKind = "uni",
        instrument_defaults: dict[str, float] | None = None,
    ) -> None:
        unique = sorted({float(e) for e in energies})
        if not unique:
            msg = "DispersiveReflectModel requires at least one energy channel"
            raise ValueError(msg)
        self._structure = structure
        self._energies: tuple[float, ...] = tuple(unique)
        self._pol: PolKind = pol
        self.name = name
        self.backend = backend
        self._defaults = dict(instrument_defaults or {})
        self._phi = 0.0
        self._active_energy = unique[0]
        self._parameters: Parameters | None = None
        self._channels = {
            energy: make_instrument_channel(
                energy,
                defaults=self._defaults,
                energy_tag=self._energy_tag(energy),
            )
            for energy in self._energies
        }
        self._instrument_views: dict[InstrumentField, InstrumentParameterView] = {
            field: InstrumentParameterView(self, field) for field in INSTRUMENT_FIELDS
        }
        self._bind_structure(structure)

    @staticmethod
    def _energy_tag(energy_ev: float) -> str:
        return f"{energy_ev:.6g}".replace(".", "p")

    def _bind_structure(self, structure: DispersiveStructure) -> None:
        structure.structure_energy_offset.setp(vary=False, value=0.0)
        structure.lock_energy_offsets()
        self._rebuild_parameters()

    @property
    def energies(self) -> tuple[float, ...]:
        """Sorted distinct photon energies (eV) with instrument channels."""
        return self._energies

    @property
    def structure(self) -> DispersiveStructure:
        """Shared deferred-energy stack."""
        return self._structure

    @structure.setter
    def structure(self, structure: DispersiveStructure) -> None:
        self._structure = structure
        self._bind_structure(structure)

    def instrument_at(self, energy_ev: float) -> EnergyInstrumentSlice:
        """Return the instrument channel for ``energy_ev`` (eV)."""
        key = float(energy_ev)
        channel = self._channels.get(key)
        if channel is None:
            msg = (
                f"No instrument channel for energy {key} eV; "
                f"known energies: {list(self._energies)}"
            )
            raise KeyError(msg)
        return channel

    def instrument_query(self, field: InstrumentField) -> InstrumentParameterView:
        """Return the query view for ``field`` (alias: ``model.<field>``)."""
        if field not in INSTRUMENT_FIELDS:
            msg = f"Unknown instrument field {field!r}"
            raise ValueError(msg)
        return self._instrument_views[field]

    def bookended_film(self) -> BookendedOrientationProfile | None:
        """Return the graded book-ended film component when present."""
        from refloxide.pxr.energy.fused import find_bookended_profile

        located = find_bookended_profile(self._structure)
        return None if located is None else located[1]

    def link_bookended_film(
        self,
        reference_energy: float,
        *,
        param_names: tuple[str, ...] = BOOKENDED_FILM_PARAM_NAMES,
    ) -> DispersiveReflectModel:
        """No-op when geometry is already shared; kept for notebook parity.

        With :class:`DispersiveReflectModel` the film lives on the single shared
        structure, so book-ended parameters are already global. This method exists
        so graded-fitting notebooks can call the same symbol as
        ``link_bookended_film_to_reference`` without rebuilding per-energy stacks.
        """
        if self.bookended_film() is None:
            msg = "Structure has no BookendedOrientationProfile component"
            raise ValueError(msg)
        del reference_energy, param_names
        return self

    def _copy_channel_from_slice(
        self,
        destination: EnergyInstrumentSlice,
        source: EnergyInstrumentSlice,
        *,
        vary: bool | None,
    ) -> None:
        for name in REFLECT_MODEL_INSTRUMENTATION:
            copy_parameter_state(
                destination.parameter(name),
                source.parameter(name),
                vary=vary,
            )

    def copy_instrumentation_from(
        self,
        source: ReflectModel | DispersiveReflectModel,
        *,
        energies: Sequence[float] | None = None,
        vary: bool | None = None,
    ) -> DispersiveReflectModel:
        """Copy scales and offsets from another model onto matching channels.

        Mirrors :func:`~utils.graded_objective.copy_reflect_model_instrumentation`.
        """
        targets = (
            list(self._energies) if energies is None else [float(e) for e in energies]
        )
        if isinstance(source, DispersiveReflectModel):
            for energy in targets:
                self._copy_channel_from_slice(
                    self.instrument_at(energy),
                    source.instrument_at(energy),
                    vary=vary,
                )
            return self
        if source.energy is None:
            msg = "ReflectModel source must have energy set"
            raise ValueError(msg)
        channel = self.instrument_at(float(source.energy))
        for name in REFLECT_MODEL_INSTRUMENTATION:
            copy_parameter_state(
                channel.parameter(name),
                getattr(source, name),
                vary=vary,
            )
        return self

    def set_active_energy(self, energy_ev: float) -> None:
        """Select the channel used by legacy single-energy property accessors."""
        self._active_energy = float(energy_ev)

    @property
    def energy(self) -> float:
        """Active photon energy for single-channel Objective use."""
        return self._active_energy

    @energy.setter
    def energy(self, value: float) -> None:
        self.set_active_energy(value)

    @property
    def pol(self) -> PolKind:
        """Default output polarization."""
        return self._pol

    @pol.setter
    def pol(self, value: PolKind) -> None:
        self._pol = value

    @property
    def phi(self) -> float:
        """Azimuthal angle (degrees) for biaxial backend."""
        return self._phi

    @phi.setter
    def phi(self, value: float) -> None:
        self._phi = float(value)

    def _rebuild_parameters(self) -> None:
        instrument_root = Parameters(name="instrument per energy")
        for energy in self._energies:
            channel = self._channels[energy]
            block = Parameters(name=f"instrument@{self._energy_tag(energy)}eV")
            block.extend(channel.parameter(field) for field in INSTRUMENT_FIELDS)
            instrument_root.extend([block])
        root = Parameters(name=self.name or "dispersive_reflect_model")
        root.extend([instrument_root, self._structure.parameters])
        self._parameters = root

    @property
    def parameters(self) -> Parameters:
        """All structure and instrument parameters for refnx fitters."""
        self._rebuild_parameters()
        if self._parameters is None:
            msg = "DispersiveReflectModel parameters were not initialized"
            raise RuntimeError(msg)
        return self._parameters

    def logp(self) -> float:
        """Structure log-prior; instrument terms use refnx parameter bounds."""
        return float(self._structure.logp())

    def __call__(self, x: np.ndarray, p: np.ndarray | None = None, x_err=None):
        return self.model(x, p=p, x_err=x_err)

    def __repr__(self) -> str:
        return (
            f"DispersiveReflectModel({self._structure!r}, "
            f"energies={self._energies!r}, name={self.name!r})"
        )

    def _stack_arrays(
        self,
        energy_ev: float,
        instrument: ResolvedInstrument,
    ) -> tuple[np.ndarray, np.ndarray]:
        structure = self._structure
        if hasattr(structure, "materialize"):
            snap = structure.materialize(
                float(energy_ev),
                structure_offset_ev=instrument.energy_offset_ev,
            )
            return snap.layers, snap.tensors
        slabs = structure.slabs(energy=energy_ev)  # type: ignore[call-arg]
        tensor = structure.tensor(energy=energy_ev)
        if slabs is None:
            msg = "Structure returned no slabs"
            raise ValueError(msg)
        return (
            np.asarray(slabs, dtype=np.float64),
            np.asarray(tensor, dtype=np.complex128),
        )

    @staticmethod
    def _q_with_theta_offset(
        q: np.ndarray,
        wavelength: float,
        theta_offset_deg: float,
    ) -> np.ndarray:
        theta = np.arcsin(q * wavelength / (4 * np.pi)) * 180 / np.pi
        theta += theta_offset_deg
        return (4 * np.pi / wavelength) * np.sin(theta * np.pi / 180)

    def _resolve_q_grids(
        self,
        x: np.ndarray,
        wavelength: float,
        pol: PolKind,
        instrument: ResolvedInstrument,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if pol in ("sp", "ps"):
            concat_loc = int(np.argmax(np.abs(np.diff(x))))
            qvals_1 = x[: concat_loc + 1]
            qvals_2 = x[concat_loc + 1 :]
            num_q = max(len(x), concat_loc + 50)
            qvals_1 = self._q_with_theta_offset(
                qvals_1, wavelength, instrument.theta_offset_s
            )
            qvals_2 = self._q_with_theta_offset(
                qvals_2, wavelength, instrument.theta_offset_p
            )
            x = np.concatenate([qvals_1, qvals_2])
            qvals = np.linspace(float(np.min(x)), float(np.max(x)), num_q)
            return x, qvals, qvals_1, qvals_2
        if pol == "s":
            qvals = self._q_with_theta_offset(x, wavelength, instrument.theta_offset_s)
        elif pol == "p":
            qvals = self._q_with_theta_offset(x, wavelength, instrument.theta_offset_p)
        else:
            qvals = x
        return x, qvals, qvals, qvals

    def _model_at_energy(
        self,
        x: np.ndarray,
        energy_ev: float,
        *,
        pol: PolKind | None = None,
        x_err: np.ndarray | float | None = None,
        p: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if p is not None:
            self.parameters.pvals = np.asarray(p, dtype=np.float64)
        pol_use = pol if pol is not None else self._pol
        instrument = resolve_instrument(self, energy_ev)
        if x_err is None:
            x_err = instrument.dq
        dq = float(np.asarray(x_err).flat[0])
        slabs, tensor = self._stack_arrays(energy_ev, instrument)
        wavelength = 12398.42 / float(energy_ev)
        x = np.asarray(x, dtype=np.float64)
        _x, qvals, qvals_1, qvals_2 = self._resolve_q_grids(
            x, wavelength, pol_use, instrument
        )

        result = reflectivity(
            qvals + instrument.q_offset,
            slabs,
            tensor,
            energy=float(energy_ev),
            phi=self._phi,
            scale_s=instrument.scale_s,
            scale_p=instrument.scale_p,
            bkg=instrument.bkg,
            dq=dq,
            backend=self.backend,
        )
        if result is None:
            msg = "reflectivity returned None; check dq / backend"
            raise RuntimeError(msg)
        refl, _tran, _components = result
        output = reflectivity_for_pol(
            pol_use,
            refl,
            qvals,
            qvals_1,
            qvals_2,
        )
        return output, qvals, qvals_1, qvals_2

    def model(
        self,
        x: np.ndarray,
        p: np.ndarray | None = None,
        x_err=None,
        *,
        energy: float | None = None,
        pol: PolKind | None = None,
    ) -> np.ndarray:
        """Evaluate reflectivity at ``energy`` or the active channel energy."""
        energy_ev = float(self._active_energy if energy is None else energy)
        output, *_ = self._model_at_energy(
            np.asarray(x, dtype=np.float64),
            energy_ev,
            pol=pol,
            x_err=x_err,
            p=p,
        )
        return output

    def anisotropy(
        self,
        x: np.ndarray,
        p: np.ndarray | None = None,
        x_err=None,
        *,
        energy: float | None = None,
    ) -> np.ndarray:
        """Return ``(R_p - R_s) / (R_p + R_s)`` with per-channel theta offsets."""
        energy_ev = float(self._active_energy if energy is None else energy)
        instrument = resolve_instrument(self, energy_ev)
        wavelength = 12398.42 / energy_ev
        x = np.asarray(x, dtype=np.float64)
        q_s = self._q_with_theta_offset(x, wavelength, instrument.theta_offset_s)
        q_p = self._q_with_theta_offset(x, wavelength, instrument.theta_offset_p)
        r_s = self.model(q_s, p=p, x_err=x_err, energy=energy_ev, pol="s")
        r_p = self.model(q_p, p=p, x_err=x_err, energy=energy_ev, pol="p")
        return (r_p - r_s) / (r_p + r_s)

    @classmethod
    def from_reflect_models(
        cls,
        structure: DispersiveStructure,
        models: Sequence[ReflectModel],
    ) -> DispersiveReflectModel:
        """Build a dispersive model from single-energy :class:`ReflectModel` instances.

        Copies instrument parameter values from each source model into the
        matching energy channel.
        """
        energies = [float(m.energy) for m in models if m.energy is not None]
        out = cls(structure, energies, name=models[0].name if models else "")
        for model in models:
            if model.energy is None:
                continue
            out.copy_instrumentation_from(model, energies=[float(model.energy)])
        return out


for _field in INSTRUMENT_FIELDS:
    setattr(
        DispersiveReflectModel,
        _field,
        property(lambda self, field=_field: self._instrument_views[field]),
    )


class DispersiveReflectObjective:
    """Single-energy :class:`~refnx.analysis.Objective` bound to one model channel.

    .. deprecated::
        Prefer :class:`~refloxide.pxr.objective.ReflectivityObjective` with
        :class:`~refloxide.pxr.energy.model.CompiledReflectivityModel`.

    Parameters
    ----------
    model
        Parent :class:`DispersiveReflectModel`.
    data
        Reflectivity dataset for this energy.
    energy_ev
        Photon energy selecting the instrument channel before each evaluation.
    kwargs
        Forwarded to :class:`~refnx.analysis.Objective`.
    """

    def __init__(
        self,
        model: DispersiveReflectModel,
        data: Any,
        energy_ev: float,
        **kwargs: Any,
    ) -> None:
        from refnx.analysis import Objective

        self.energy_ev = float(energy_ev)
        self._objective = Objective(model, data, **kwargs)

    @property
    def model(self) -> DispersiveReflectModel:
        return self._objective.model

    @property
    def data(self) -> Any:
        return self._objective.data

    @property
    def parameters(self):
        return self._objective.parameters

    def varying_parameters(self):
        return self._objective.varying_parameters()

    def _run_at_channel(self, method: Callable[..., Any], pvals=None):
        self.model.set_active_energy(self.energy_ev)
        return method(pvals)

    def setp(self, pvals=None):
        return self._run_at_channel(self._objective.setp, pvals)

    def logl(self, pvals=None) -> float:
        return float(self._run_at_channel(self._objective.logl, pvals))

    def generative(self, pvals=None) -> np.ndarray:
        return self._run_at_channel(self._objective.generative, pvals)

    def residuals(self, pvals=None) -> np.ndarray:
        return self._run_at_channel(self._objective.residuals, pvals)

    def logp(self, pvals=None) -> float:
        return float(self._run_at_channel(self._objective.logp, pvals))


def select(
    target: DispersiveReflectModel | DispersiveReflectObjective | Any,
    slab_name: str,
    energy: float | None = None,
) -> Any:
    """Return the structure component whose name starts with ``slab_name``.

    Mirrors :func:`~utils.slab_builders.select` for graded-fitting notebooks, but
    reads from the single shared stack on :class:`DispersiveReflectModel`. When
    ``energy`` is given, the model's active channel is updated so instrument
    property views resolve consistently in single-energy objectives.

    Parameters
    ----------
    target
        :class:`DispersiveReflectModel`, :class:`DispersiveReflectObjective`, or
        any refnx objective exposing ``model.structure``.
    slab_name
        Prefix match on component ``name`` (e.g. ``"ZnPc"``, ``"Oxide"``).
    energy
        Optional photon energy (eV) for active-channel selection.

    Raises
    ------
    ValueError
        When no matching component exists.
    """
    if isinstance(target, DispersiveReflectObjective):
        dispersive_model = target.model
        if energy is None:
            energy = target.energy_ev
    elif isinstance(target, DispersiveReflectModel):
        dispersive_model = target
    else:
        dispersive_model = getattr(target, "model", None)
    if not isinstance(dispersive_model, DispersiveReflectModel):
        msg = f"select() expects DispersiveReflectModel; got {type(dispersive_model)!r}"
        raise TypeError(msg)
    if energy is not None:
        dispersive_model.set_active_energy(float(energy))
    for component in dispersive_model.structure.components:
        if component.name.startswith(slab_name):
            return component
    msg = f"No slab {slab_name!r} found in structure {dispersive_model.structure!r}"
    raise ValueError(msg)
