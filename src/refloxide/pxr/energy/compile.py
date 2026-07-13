"""Compile plugin or legacy structures into deferred-energy stacks."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from refloxide.pxr.energy.migrate import upgrade_scatterer
from refloxide.pxr.energy.scatterers import (
    FreeTensorScatterer,
    FunctionScatterer,
    SymmetryKind,
)
from refloxide.pxr.energy.structure import DispersiveStructure
from refloxide.pxr.plugin.structure import (
    Component,
    Scatterer,
    Slab,
    Stack,
    Structure,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

PlanKind = Literal["default", "free_shared", "free_per_energy", "function"]


@dataclass(frozen=True, slots=True)
class SlabEnergyPlan:
    """Per-slab routing for :func:`compile_structure`.

    Parameters
    ----------
    kind
        Scatterer upgrade strategy for the slab SLD.
    energies
        Photon energies (eV) for ``free_per_energy`` groups.
    symmetry
        Tensor symmetry when building :class:`FreeTensorScatterer`.
    fn
        Callable for ``function`` slabs; signature
        ``fn(energy_ev, **hyperparams) -> (3, 3)``.
    hyperparams
        Initial hyperparameter values for ``function`` slabs.
    """

    kind: PlanKind = "default"
    energies: tuple[float, ...] | None = None
    symmetry: SymmetryKind = "uni"
    fn: Callable[..., NDArray] | None = None
    hyperparams: dict[str, float] | None = None


def _scatterer_from_plan(
    scatterer: Scatterer,
    plan: SlabEnergyPlan | None,
    *,
    slab_name: str,
) -> Scatterer:
    if plan is None or plan.kind == "default":
        return upgrade_scatterer(scatterer)
    match plan.kind:
        case "free_shared":
            from refloxide.pxr.plugin.structure import SLD

            if isinstance(scatterer, SLD):
                return FreeTensorScatterer.from_sld(
                    scatterer,
                    energies=None,
                    symmetry=plan.symmetry,
                )
            if isinstance(scatterer, FreeTensorScatterer):
                return scatterer
            return FreeTensorScatterer(symmetry=plan.symmetry, name=slab_name)
        case "free_per_energy":
            from refloxide.pxr.plugin.structure import SLD

            energies = plan.energies
            if energies is None or not energies:
                msg = "SlabEnergyPlan free_per_energy requires energies"
                raise ValueError(msg)
            if isinstance(scatterer, SLD):
                return FreeTensorScatterer.from_sld(
                    scatterer,
                    energies=energies,
                    symmetry=plan.symmetry,
                )
            return FreeTensorScatterer(
                symmetry=plan.symmetry,
                name=slab_name,
                energies=energies,
            )
        case "function":
            if plan.fn is None:
                msg = "SlabEnergyPlan function requires fn"
                raise ValueError(msg)
            return FunctionScatterer(
                plan.fn,
                hyperparams=plan.hyperparams,
                name=slab_name,
            )
        case _:
            msg = f"Unknown SlabEnergyPlan kind {plan.kind!r}"
            raise ValueError(msg)


def _compile_component(
    component: Component,
    plan: Mapping[str, SlabEnergyPlan] | None,
    *,
    default_nevot_croce: bool,
) -> Component:
    if isinstance(component, Slab) and isinstance(component.sld, Scatterer):
        slab_plan = None if plan is None else plan.get(component.name)
        upgraded = _scatterer_from_plan(
            component.sld,
            slab_plan,
            slab_name=component.name,
        )
        nevot = getattr(component, "enforce_nevot_croce", default_nevot_croce)
        if upgraded is component.sld and nevot == default_nevot_croce:
            return component
        return Slab(
            component.thick.value,
            upgraded,
            component.rough.value,
            name=component.name,
            enforce_nevot_croce=nevot,
        )
    if isinstance(component, Stack):
        children = [
            _compile_component(child, plan, default_nevot_croce=default_nevot_croce)
            for child in component.components
        ]
        if children == list(component.components):
            return component
        from refloxide.pxr.plugin.structure import Stack as PluginStack

        return PluginStack(children, name=component.name, repeats=component.repeats)
    return component


def compile_structure(
    structure: Structure | DispersiveStructure | Sequence[Component],
    *,
    plan: Mapping[str, SlabEnergyPlan] | None = None,
    default_nevot_croce: bool = True,
) -> DispersiveStructure:
    """Compile a plugin or deferred stack into a :class:`DispersiveStructure`.

    Parameters
    ----------
    structure
        Plugin :class:`~refloxide.pxr.plugin.structure.Structure`, existing
        :class:`~refloxide.pxr.energy.structure.DispersiveStructure`, or a
        component sequence.
    plan
        Optional mapping from slab name to :class:`SlabEnergyPlan` choosing
        shared or per-energy free tensors, callables, or default migration.
    default_nevot_croce
        Default :attr:`~refloxide.pxr.plugin.structure.Slab.enforce_nevot_croce`
        for newly built slabs when the source slab does not carry the flag.

    Returns
    -------
    DispersiveStructure
        Deferred-energy stack with scatterers upgraded per ``plan``.
    """
    if isinstance(structure, DispersiveStructure):
        if plan is None and default_nevot_croce:
            return structure
        components = [
            _compile_component(c, plan, default_nevot_croce=default_nevot_croce)
            for c in structure.components
        ]
        out = DispersiveStructure(
            *components,
            name=structure.name,
            reverse_structure=structure.reverse_structure,
            structure_energy_offset=structure.structure_energy_offset,
        )
        return out
    if isinstance(structure, Structure):
        components = [
            _compile_component(c, plan, default_nevot_croce=default_nevot_croce)
            for c in structure.components
        ]
        out = DispersiveStructure(
            *components,
            name=structure.name,
            reverse_structure=structure.reverse_structure,
        )
        with contextlib.suppress(ValueError, AttributeError):
            out.structure_energy_offset = structure.energy_offset
        return out
    components = [
        _compile_component(c, plan, default_nevot_croce=default_nevot_croce)
        for c in structure
    ]
    return DispersiveStructure(*components)
