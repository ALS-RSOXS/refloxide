"""Shared and per-energy experiment correction parameters for ReflectModel.

Owns :class:`EnergyInstrumentChannel`, :class:`ExperimentCorrections`, and
:class:`InstrumentFieldView`. Does not evaluate reflectivity or own
structure geometry.

Scoping (native multi-energy path):

* Shared across all energies: ``energy_offset``, ``dq``, ``q_offset``.
* Per energy: ``scale_s``, ``scale_p``, ``bkg``, ``theta_offset_s``,
  ``theta_offset_p``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from refnx.analysis import Parameter, Parameters, possibly_create_parameter

PerEnergyField = Literal[
    "scale_s",
    "scale_p",
    "bkg",
    "theta_offset_s",
    "theta_offset_p",
]

PER_ENERGY_FIELDS: tuple[PerEnergyField, ...] = (
    "scale_s",
    "scale_p",
    "bkg",
    "theta_offset_s",
    "theta_offset_p",
)

EnergyPredicate = Callable[[float], bool]

_WHERE_SPECS: tuple[tuple[str, Callable[[float], EnergyPredicate]], ...] = (
    ("lt", lambda bound: lambda e: e < bound),
    ("le", lambda bound: lambda e: e <= bound),
    ("gt", lambda bound: lambda e: e > bound),
    ("ge", lambda bound: lambda e: e >= bound),
)


def safely_setp_param(param: Parameter, **kwargs: Any) -> None:
    """Apply :meth:`~refnx.analysis.Parameter.setp` with refnx constraint rules."""
    if kwargs.get("vary", False) and kwargs.get("constraint") is not None:
        kwargs = {**kwargs, "vary": None}
    param.setp(**kwargs)


def _float_param(param: Parameter | float) -> float:
    if isinstance(param, Parameter):
        return float(param.value or 0.0)
    return float(param)


def energy_tag(energy_ev: float) -> str:
    """Tag photon energy for parameter names (``285.1`` -> ``285p1``)."""
    return str(float(energy_ev)).replace(".", "p")


@dataclass(frozen=True, slots=True)
class _PerEnergyFieldSpec:
    field: PerEnergyField
    default: float
    vary: bool = True
    bounds: tuple[float, float] | None = None


_PER_ENERGY_SPECS: tuple[_PerEnergyFieldSpec, ...] = (
    _PerEnergyFieldSpec("scale_s", 1.0),
    _PerEnergyFieldSpec("scale_p", 1.0),
    _PerEnergyFieldSpec("bkg", 0.0),
    _PerEnergyFieldSpec("theta_offset_s", 0.0),
    _PerEnergyFieldSpec("theta_offset_p", 0.0),
)


@dataclass(slots=True)
class ResolvedChannel:
    """Scalar per-energy corrections for one reflectivity evaluation."""

    scale_s: float
    scale_p: float
    bkg: float
    theta_offset_s: float
    theta_offset_p: float


@dataclass(slots=True)
class EnergyInstrumentChannel:
    """Per-energy scale, background, and theta-offset parameters.

    Parameters
    ----------
    energy_ev
        Nominal photon energy in eV labelling this channel.
    scale_s, scale_p, bkg, theta_offset_s, theta_offset_p
        Per-energy instrument parameters registered with refnx.
    """

    energy_ev: float
    scale_s: Parameter
    scale_p: Parameter
    bkg: Parameter
    theta_offset_s: Parameter
    theta_offset_p: Parameter

    def parameter(self, field: PerEnergyField) -> Parameter:
        """Return the :class:`~refnx.analysis.Parameter` for ``field``."""
        return getattr(self, field)

    def resolved(self) -> ResolvedChannel:
        """Evaluate all per-energy parameters to plain floats."""
        return ResolvedChannel(
            scale_s=_float_param(self.scale_s),
            scale_p=_float_param(self.scale_p),
            bkg=_float_param(self.bkg),
            theta_offset_s=_float_param(self.theta_offset_s),
            theta_offset_p=_float_param(self.theta_offset_p),
        )

    def parameters(self) -> Parameters:
        """Collect this channel's parameters under one named block."""
        block = Parameters(name=f"instrument@{energy_tag(self.energy_ev)}eV")
        block.extend(
            [
                self.scale_s,
                self.scale_p,
                self.bkg,
                self.theta_offset_s,
                self.theta_offset_p,
            ]
        )
        return block


def make_energy_channel(
    energy_ev: float,
    *,
    defaults: dict[str, float] | None = None,
    name_prefix: str = "",
) -> EnergyInstrumentChannel:
    """Construct one :class:`EnergyInstrumentChannel` for ``energy_ev``."""
    defaults = defaults or {}
    tag = energy_tag(energy_ev)
    prefix = f"{name_prefix}_" if name_prefix else ""
    params: dict[str, Parameter] = {}

    for spec in _PER_ENERGY_SPECS:
        kwargs: dict[str, Any] = {}
        if spec.bounds is not None:
            kwargs["bounds"] = spec.bounds
        params[spec.field] = possibly_create_parameter(  # type: ignore[assignment]
            defaults.get(spec.field, spec.default),
            name=f"{prefix}{spec.field}@{tag}eV",
            vary=spec.vary,
            **kwargs,
        )
    return EnergyInstrumentChannel(energy_ev=float(energy_ev), **params)  # type: ignore[arg-type]


class ExperimentCorrections:
    """Shared energy offset / resolution plus per-energy scale and theta channels.

    Parameters
    ----------
    energies
        Photon energies (eV) for which to allocate per-energy channels.
        May be empty at construction; call :meth:`ensure_energies` later.
    name
        Prefix for shared parameter names.
    energy_offset, dq, q_offset
        Shared starting values.
    channel_defaults
        Optional overrides for per-energy field defaults
        (``scale_s``, ``scale_p``, ``bkg``, ``theta_offset_s``,
        ``theta_offset_p``).
    """

    def __init__(
        self,
        energies: Sequence[float] | None = None,
        *,
        name: str = "",
        energy_offset: float = 0.0,
        dq: float = 0.0,
        q_offset: float = 0.0,
        channel_defaults: dict[str, float] | None = None,
    ) -> None:
        prefix = f"{name}_" if name else ""
        self._name = name
        self._channel_defaults = dict(channel_defaults or {})
        self.energy_offset = possibly_create_parameter(
            energy_offset,
            name=f"{prefix}energy_offset",
            vary=False,
            bounds=(-1.0, 1.0),
        )
        self.dq = possibly_create_parameter(dq, name=f"{prefix}dq", vary=False)
        self.q_offset = possibly_create_parameter(
            q_offset, name=f"{prefix}q_offset", vary=False
        )
        self._channels: dict[float, EnergyInstrumentChannel] = {}
        if energies:
            self.ensure_energies(energies)

    @property
    def energies(self) -> list[float]:
        """Sorted list of energies that currently have a channel."""
        return sorted(self._channels)

    def ensure_energies(self, energies: Sequence[float]) -> None:
        """Create missing per-energy channels for each energy in ``energies``."""
        for energy in energies:
            key = float(energy)
            if key not in self._channels:
                self._channels[key] = make_energy_channel(
                    key,
                    defaults=self._channel_defaults,
                    name_prefix=self._name,
                )

    def channel_at(self, energy_ev: float) -> EnergyInstrumentChannel:
        """Return the channel for ``energy_ev``, creating it if absent."""
        key = float(energy_ev)
        self.ensure_energies([key])
        return self._channels[key]

    def resolved_at(self, energy_ev: float) -> ResolvedChannel:
        """Resolve per-energy scalars for ``energy_ev``."""
        return self.channel_at(energy_ev).resolved()

    def parameters(self) -> Parameters:
        """Shared params then one block per energy channel (sorted)."""
        root_name = "experiment_corrections" if not self._name else self._name
        root = Parameters(name=root_name)
        shared = Parameters(name="shared")
        shared.extend([self.energy_offset, self.dq, self.q_offset])
        root.append(shared)
        for energy in self.energies:
            root.append(self._channels[energy].parameters())
        return root

    def field_view(self, field: PerEnergyField) -> InstrumentFieldView:
        """Return an energy-queryable view of one per-energy field."""
        return InstrumentFieldView(self, field)


class InstrumentFieldView:
    """Energy-queryable view of one per-energy experiment-correction field.

    Examples
    --------
    >>> model.scale_s.at(285.1).setp(vary=True, bounds=(0.8, 1.2))
    >>> model.theta_offset_p.where(ge=283.0).setp(vary=True, bounds=(-0.1, 0.1))
    >>> model.scale_s.where(lt=280).link(anchor=250.0)
    """

    __slots__ = ("_corrections", "_field", "_predicates")

    def __init__(
        self, corrections: ExperimentCorrections, field: PerEnergyField
    ) -> None:
        self._corrections = corrections
        self._field = field
        self._predicates: list[EnergyPredicate] = []

    def _branch(self) -> InstrumentFieldView:
        view = InstrumentFieldView(self._corrections, self._field)
        view._predicates = list(self._predicates)
        return view

    def where(
        self,
        *,
        energy: float | None = None,
        lt: float | None = None,
        le: float | None = None,
        gt: float | None = None,
        ge: float | None = None,
        between: tuple[float, float] | None = None,
        energy_in: Iterable[float] | None = None,
    ) -> InstrumentFieldView:
        """Return a view whose next action applies only to matching channels."""
        view = self._branch()
        bounds = {"lt": lt, "le": le, "gt": gt, "ge": ge}
        if energy is not None:
            target = float(energy)
            view._predicates.append(lambda e, target=target: e == target)
        for name, factory in _WHERE_SPECS:
            bound = bounds[name]
            if bound is not None:
                view._predicates.append(factory(float(bound)))
        if between is not None:
            lo, hi = between
            view._predicates.append(lambda e, lo=lo, hi=hi: lo <= e <= hi)
        if energy_in is not None:
            allowed = {float(x) for x in energy_in}
            view._predicates.append(lambda e, allowed=allowed: e in allowed)
        return view

    def matching_energies(self) -> list[float]:
        """Return channel energies that satisfy every predicate."""
        if not self._predicates:
            return list(self._corrections.energies)
        return [
            channel_energy
            for channel_energy in self._corrections.energies
            if all(pred(channel_energy) for pred in self._predicates)
        ]

    def parameters(self) -> list[Parameter]:
        """Collect underlying refnx parameters for all matching channels."""
        return [
            self._corrections.channel_at(channel_energy).parameter(self._field)
            for channel_energy in self.matching_energies()
        ]

    def at(self, energy_ev: float) -> Parameter:
        """Return the single :class:`~refnx.analysis.Parameter` at ``energy_ev``.

        Creates the channel if it does not yet exist.
        """
        return self._corrections.channel_at(float(energy_ev)).parameter(self._field)

    def setp(self, **kwargs: Any) -> ExperimentCorrections:
        """Apply :meth:`~refnx.analysis.Parameter.setp` on every matching channel."""
        matched = self.matching_energies()
        if not matched and self._predicates:
            msg = f"no channels match predicates for field {self._field!r}"
            raise ValueError(msg)
        for param in self.parameters():
            safely_setp_param(param, **kwargs)
        return self._corrections

    set = setp

    def link(
        self,
        *,
        anchor: float | None = None,
        to: Parameter | None = None,
    ) -> ExperimentCorrections:
        """Constrain matching channels to one shared parameter.

        Parameters
        ----------
        anchor
            Master channel energy when ``to`` is omitted. Defaults to the
            lowest matching energy.
        to
            Optional external :class:`~refnx.analysis.Parameter` to constrain
            against.
        """
        matched = self.matching_energies()
        if not matched:
            return self._corrections
        if to is not None:
            master = to
            slaves = [param for param in self.parameters() if param is not master]
        else:
            anchor_e = float(anchor if anchor is not None else matched[0])
            if anchor_e not in matched:
                msg = f"anchor {anchor_e} is not among matched energies {matched}"
                raise ValueError(msg)
            master = self._corrections.channel_at(anchor_e).parameter(self._field)
            slaves = [
                self._corrections.channel_at(channel_energy).parameter(self._field)
                for channel_energy in matched
                if channel_energy != anchor_e
            ]
        for slave in slaves:
            safely_setp_param(slave, constraint=master, vary=None)
        return self._corrections

    link_shared = link
    constrain_to = link

    def unlink(self) -> ExperimentCorrections:
        """Remove cross-energy constraints on matching channels."""
        for param in self.parameters():
            safely_setp_param(param, constraint=None)
        return self._corrections
