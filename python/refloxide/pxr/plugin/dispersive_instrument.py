"""Per-energy instrument storage and energy-band parameter views.

Owns :class:`EnergyInstrumentSlice`, :class:`InstrumentParameterView`, and
instrument resolution for both :class:`~refloxide.pxr.plugin.model.ReflectModel`
and :class:`~refloxide.pxr.plugin.dispersive_model.DispersiveReflectModel`.
Does not evaluate reflectivity or own structure geometry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from refnx.analysis import Parameter, possibly_create_parameter

if TYPE_CHECKING:
    from refloxide.pxr.plugin.dispersive_model import DispersiveReflectModel

InstrumentField = Literal[
    "scale_s",
    "scale_p",
    "bkg",
    "dq",
    "q_offset",
    "energy_offset",
    "theta_offset_s",
    "theta_offset_p",
]

INSTRUMENT_FIELDS: tuple[InstrumentField, ...] = (
    "scale_s",
    "scale_p",
    "bkg",
    "dq",
    "q_offset",
    "energy_offset",
    "theta_offset_s",
    "theta_offset_p",
)

REFLECT_MODEL_INSTRUMENTATION: tuple[InstrumentField, ...] = INSTRUMENT_FIELDS

BOOKENDED_FILM_PARAM_NAMES: tuple[str, ...] = (
    "total_thick",
    "surface_roughness",
    "tau_si",
    "tau_vac",
    "alpha_bulk",
    "alpha_si",
    "alpha_vac",
    "density_bulk",
    "density_si",
    "density_vac",
    "energy_offset",
)

EnergyPredicate = Callable[[float], bool]

_WHERE_SPECS: tuple[tuple[str, Callable[[float], EnergyPredicate]], ...] = (
    ("lt", lambda bound: lambda e: e < bound),
    ("le", lambda bound: lambda e: e <= bound),
    ("gt", lambda bound: lambda e: e > bound),
    ("ge", lambda bound: lambda e: e >= bound),
)


class _HasInstrumentAt(Protocol):
    def instrument_at(self, energy_ev: float) -> EnergyInstrumentSlice: ...


class _HasInstrument(Protocol):
    @property
    def scale_s(self) -> Parameter: ...

    @property
    def scale_p(self) -> Parameter: ...

    @property
    def bkg(self) -> Parameter: ...

    @property
    def dq(self) -> Parameter | float: ...

    @property
    def q_offset(self) -> Parameter: ...

    @property
    def theta_offset_s(self) -> Parameter: ...

    @property
    def theta_offset_p(self) -> Parameter: ...


def safely_setp_param(param: Parameter, **kwargs: Any) -> None:
    """Apply :meth:`~refnx.analysis.Parameter.setp` with refnx constraint rules."""
    if kwargs.get("vary", False) and kwargs.get("constraint") is not None:
        kwargs = {**kwargs, "vary": None}
    param.setp(**kwargs)


def _float_param(param: Parameter | float) -> float:
    if isinstance(param, Parameter):
        return float(param.value or 0.0)
    return float(param)


def setp_kwargs_from_parameter(
    source: Parameter,
    *,
    vary: bool | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for copying one instrument parameter's fit state."""
    kwargs: dict[str, Any] = {"value": source.value}
    if vary is not None:
        kwargs["vary"] = vary
        return kwargs
    kwargs["vary"] = source.vary
    if source.constraint is not None:
        kwargs["constraint"] = source.constraint
        kwargs["vary"] = None
    if source.bounds is not None:
        kwargs["bounds"] = source.bounds
    return kwargs


def copy_parameter_state(
    destination: Parameter,
    source: Parameter,
    *,
    vary: bool | None = None,
) -> None:
    """Copy value, bounds, and constraint state from ``source`` onto ``destination``."""
    safely_setp_param(destination, **setp_kwargs_from_parameter(source, vary=vary))


@dataclass(frozen=True, slots=True)
class _ChannelFieldSpec:
    field: InstrumentField
    default: float
    vary: bool = True
    bounds: tuple[float, float] | None = None


_CHANNEL_SPECS: tuple[_ChannelFieldSpec, ...] = (
    _ChannelFieldSpec("scale_s", 1.0),
    _ChannelFieldSpec("scale_p", 1.0),
    _ChannelFieldSpec("bkg", 0.0),
    _ChannelFieldSpec("dq", 0.0, vary=False),
    _ChannelFieldSpec("q_offset", 0.0, vary=False),
    _ChannelFieldSpec("energy_offset", 0.0, vary=False, bounds=(-1.0, 1.0)),
    _ChannelFieldSpec("theta_offset_s", 0.0),
    _ChannelFieldSpec("theta_offset_p", 0.0),
)


@dataclass(slots=True)
class ResolvedInstrument:
    """Scalar instrument values for one reflectivity evaluation."""

    scale_s: float
    scale_p: float
    bkg: float
    dq: float
    q_offset: float
    theta_offset_s: float
    theta_offset_p: float
    energy_offset_ev: float


@dataclass(slots=True)
class EnergyInstrumentSlice:
    """Instrument :class:`~refnx.analysis.Parameter` objects for one photon energy.

    Parameters
    ----------
    energy_ev
        Nominal photon energy in eV labelling this channel.
    scale_s, scale_p, bkg, dq, q_offset, energy_offset, theta_offset_s, theta_offset_p
        Per-energy instrument parameters registered with refnx.
    """

    energy_ev: float
    scale_s: Parameter
    scale_p: Parameter
    bkg: Parameter
    dq: Parameter
    q_offset: Parameter
    energy_offset: Parameter
    theta_offset_s: Parameter
    theta_offset_p: Parameter

    def parameter(self, field: InstrumentField) -> Parameter:
        """Return the :class:`~refnx.analysis.Parameter` for ``field``."""
        return getattr(self, field)

    def resolved(self) -> ResolvedInstrument:
        """Evaluate all instrument parameters to plain floats."""
        return ResolvedInstrument(
            scale_s=_float_param(self.scale_s),
            scale_p=_float_param(self.scale_p),
            bkg=_float_param(self.bkg),
            dq=_float_param(self.dq),
            q_offset=_float_param(self.q_offset),
            theta_offset_s=_float_param(self.theta_offset_s),
            theta_offset_p=_float_param(self.theta_offset_p),
            energy_offset_ev=_float_param(self.energy_offset),
        )


def resolve_instrument(
    model: _HasInstrumentAt | _HasInstrument,
    energy_ev: float,
) -> ResolvedInstrument:
    """Resolve instrument scalars for ``energy_ev`` from any reflectivity model.

    Dispatches to :meth:`DispersiveReflectModel.instrument_at` when available;
    otherwise reads the single-channel fields on
    :class:`~refloxide.pxr.plugin.model.ReflectModel`.
    """
    instrument_at = getattr(model, "instrument_at", None)
    if callable(instrument_at):
        return instrument_at(float(energy_ev)).resolved()
    single = cast("_HasInstrument", model)
    energy_off = getattr(single, "energy_offset", Parameter(0.0))
    return ResolvedInstrument(
        scale_s=_float_param(single.scale_s),
        scale_p=_float_param(single.scale_p),
        bkg=_float_param(single.bkg),
        dq=_float_param(single.dq),
        q_offset=_float_param(single.q_offset),
        theta_offset_s=_float_param(single.theta_offset_s),
        theta_offset_p=_float_param(single.theta_offset_p),
        energy_offset_ev=_float_param(energy_off),
    )


def make_instrument_channel(
    energy_ev: float,
    *,
    defaults: dict[str, float],
    energy_tag: str,
) -> EnergyInstrumentSlice:
    """Construct one :class:`EnergyInstrumentSlice` for ``energy_ev``."""
    params: dict[str, Parameter] = {}

    def _parameter(
        name: str,
        default: float,
        *,
        vary: bool = True,
        bounds: tuple[float, float] | None = None,
    ) -> Parameter:
        kwargs: dict[str, Any] = {}
        if bounds is not None:
            kwargs["bounds"] = bounds
        return possibly_create_parameter(  # type: ignore[return-value]
            defaults.get(name, default),
            name=f"{name}@{energy_tag}eV",
            vary=vary,
            **kwargs,
        )

    for spec in _CHANNEL_SPECS:
        params[spec.field] = _parameter(
            spec.field,
            spec.default,
            vary=spec.vary,
            bounds=spec.bounds,
        )
    return EnergyInstrumentSlice(energy_ev=float(energy_ev), **params)  # type: ignore[arg-type]


class InstrumentParameterView:
    """Energy-queryable view of one instrument field on :class:`DispersiveReflectModel`.

    Mirrors notebook patterns such as ``o.model.energy_offset.setp(...)`` and
    ``constraint=reference.model.energy_offset``, but selects channels by photon
    energy before calling :meth:`~refnx.analysis.Parameter.setp`.

    Examples
    --------
    >>> model.theta_offset_s.where(lt=283.7).setp(vary=False)
    >>> model.theta_offset_s.where(between=(283.7, 290)).link(anchor_energy=283.7)
    >>> model.energy_offset.at(283.7).setp(vary=True, bounds=(-1, 1))
    """

    __slots__ = ("_field", "_model", "_predicates")

    def __init__(self, model: DispersiveReflectModel, field: InstrumentField) -> None:
        self._model = model
        self._field = field
        self._predicates: list[EnergyPredicate] = []

    def _branch(self) -> InstrumentParameterView:
        view = InstrumentParameterView(self._model, self._field)
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
    ) -> InstrumentParameterView:
        """Return a view whose next action applies only to matching channels."""
        view = self._branch()
        bounds = {"lt": lt, "le": le, "gt": gt, "ge": ge}
        if energy is not None:
            target = float(energy)
            view._predicates.append(lambda e, target=target: e == target)
        for _name, factory in _WHERE_SPECS:
            bound = bounds[_name]
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
            return list(self._model.energies)
        return [
            channel_energy
            for channel_energy in self._model.energies
            if all(pred(channel_energy) for pred in self._predicates)
        ]

    def parameters(self) -> list[Parameter]:
        """Collect underlying refnx parameters for all matching channels."""
        return [
            self._model.instrument_at(channel_energy).parameter(self._field)
            for channel_energy in self.matching_energies()
        ]

    def at(self, energy_ev: float) -> Parameter:
        """Return the single :class:`~refnx.analysis.Parameter` at ``energy_ev``."""
        return self.where(energy=energy_ev).parameters()[0]

    def setp(self, **kwargs: Any) -> DispersiveReflectModel:
        """Apply :meth:`~refnx.analysis.Parameter.setp` on every matching channel."""
        for param in self.parameters():
            safely_setp_param(param, **kwargs)
        return self._model

    set = setp

    def link(
        self,
        *,
        anchor_energy: float | None = None,
        to: Parameter | None = None,
    ) -> DispersiveReflectModel:
        """Constrain matching channels to one shared parameter.

        Parameters
        ----------
        anchor_energy
            Master channel energy when ``to`` is omitted. Defaults to the lowest
            matching energy (same convention as
            :func:`~utils.graded_objective.link_bookended_film_to_reference`).
        to
            Optional external :class:`~refnx.analysis.Parameter` to constrain
            against (notebook ``constraint=diag_obj.model.energy_offset`` style).
        """
        matched = self.matching_energies()
        if not matched:
            return self._model
        if to is not None:
            master = to
            slaves = [param for param in self.parameters() if param is not master]
        else:
            anchor = float(anchor_energy if anchor_energy is not None else matched[0])
            if anchor not in matched:
                msg = f"anchor_energy {anchor} is not among matched energies {matched}"
                raise ValueError(msg)
            master = self._model.instrument_at(anchor).parameter(self._field)
            slaves = [
                self._model.instrument_at(channel_energy).parameter(self._field)
                for channel_energy in matched
                if channel_energy != anchor
            ]
        for slave in slaves:
            safely_setp_param(slave, constraint=master, vary=None)
        return self._model

    link_shared = link
    constrain_to = link

    def unlink(self) -> DispersiveReflectModel:
        """Remove cross-energy constraints on matching channels."""
        for param in self.parameters():
            safely_setp_param(param, constraint=None)
        return self._model

    @property
    def value(self) -> float:
        """Value on the model's active energy channel (single-energy objectives)."""
        return _float_param(self.at(self._model.energy))


InstrumentFieldQuery = InstrumentParameterView
