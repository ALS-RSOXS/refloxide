"""Deferred-energy structures that materialize slabs and tensors at evaluation time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from refnx.analysis import Parameter, possibly_create_parameter

from refloxide.pxr.energy.probe import Probe
from refloxide.pxr.energy.scatterer import (
    DeferredScatterer,
    bind_scatterer_energy_offset,
)
from refloxide.pxr.plugin.structure import Component, Slab, Stack, Structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class StackSnapshot:
    """Materialized layer geometry and optical tensors at one photon energy.

    Parameters
    ----------
    layers
        Rows ``[thickness_A, delta, beta, roughness_A]`` with shape ``(n_layers, 4)``.
    tensors
        Per-layer ``(3, 3)`` complex index tensors, shape ``(n_layers, 3, 3)``.
    energy_ev
        Effective photon energy used for the materialization (eV).
    """

    layers: NDArray[np.float64]
    tensors: NDArray[np.complex128]
    energy_ev: float


def _tensor_to_slab_row(
    thick: float,
    rough: float,
    tensor: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Convert a diagonal tensor to refnx ``[d, delta, beta, sigma]`` layout."""
    n_avg = (tensor[0, 0] + tensor[1, 1] + tensor[2, 2]) / 3.0
    delta = float((1.0 - n_avg).real)
    beta = float((1.0 - n_avg).imag)
    return np.array([thick, delta, beta, rough], dtype=np.float64)


class DispersiveStructure(Structure):
    """Structure stack whose scatterers resolve optical data only at evaluation energy.

    The stack definition is built once; :meth:`materialize` and :meth:`tensor`
    resolve tabulated or formula optical constants at the requested photon energy
    without cloning components. Use :meth:`begin_materialization_batch` around
    multi-energy objectives so each distinct energy is materialized at most once
    per parameter vector.

    Parameters
    ----------
    components
        Initial :class:`~refloxide.pxr.plugin.structure.Component` sequence.
    name
        Human-readable label.
    reverse_structure
        When ``True``, reverse slab order like the plugin ``Structure`` class.
    structure_energy_offset
        Global offset in eV added to every probe before OOC lookup. Mirrors
        :class:`~refloxide.pxr.plugin.model.ReflectModel` ``energy_offset`` when set.
    """

    def __init__(
        self,
        *components: Component,
        name: str = "",
        reverse_structure: bool = False,
        structure_energy_offset: float | Parameter = 0.0,
    ) -> None:
        super().__init__(*components, name=name, reverse_structure=reverse_structure)
        self._structure_energy_offset = possibly_create_parameter(
            structure_energy_offset,
            name="structure_energy_offset",
            vary=True,
            bounds=(-1.0, 1.0),
        )
        self._materialization_cache: dict[tuple[float, float], StackSnapshot] | None = None

    @property
    def structure_energy_offset(self) -> Parameter:
        """Global energy offset (eV) applied before per-scatterer offsets."""
        return self._structure_energy_offset

    @structure_energy_offset.setter
    def structure_energy_offset(self, value: float | Parameter) -> None:
        self._structure_energy_offset = possibly_create_parameter(
            value,
            name="structure_energy_offset",
            vary=True,
            bounds=(-1.0, 1.0),
        )

    @property
    def energy_offset(self) -> Parameter:
        """Alias compatible with :class:`~refloxide.pxr.plugin.model.ReflectModel`."""
        return self.structure_energy_offset

    @energy_offset.setter
    def energy_offset(self, value: Parameter) -> None:
        self.structure_energy_offset = value
        self.lock_energy_offsets()

    def begin_materialization_batch(self) -> None:
        """Cache :meth:`materialize` results for repeated energies in one fit eval."""
        self._materialization_cache = {}

    def end_materialization_batch(self) -> None:
        """Drop the transient materialization cache."""
        self._materialization_cache = None

    def clear_materialization_cache(self) -> None:
        """Clear any cached snapshots (alias for :meth:`end_materialization_batch`)."""
        self.end_materialization_batch()

    def deferred_scatterers(self) -> list[DeferredScatterer]:
        """Collect deferred scatterers attached to slab components."""
        found: list[DeferredScatterer] = []

        def _walk(component: Component) -> None:
            if isinstance(component, OrientationSlab):
                if isinstance(component.scatterer, DeferredScatterer):
                    found.append(component.scatterer)
                return
            if isinstance(component, Slab) and isinstance(
                component.sld, DeferredScatterer
            ):
                found.append(component.sld)
                return
            if isinstance(component, Stack):
                for child in component.components:
                    _walk(child)

        for component in self.components:
            _walk(component)
        return found

    def lock_energy_offsets(self) -> None:
        """Bind every deferred scatterer ``energy_offset`` to the structure offset."""
        offset = self.structure_energy_offset
        for scatterer in self.deferred_scatterers():
            bind_scatterer_energy_offset(scatterer, offset)

    def validate_energy_contract(self) -> None:
        """Raise when any slab offset is not constrained to the structure offset."""
        struct_val = float(self.structure_energy_offset.value or 0.0)
        for scatterer in self.deferred_scatterers():
            slab_val = float(getattr(scatterer.energy_offset, "value", 0.0) or 0.0)
            constraint = getattr(scatterer.energy_offset, "constraint", None)
            mismatched = (
                constraint is not self.structure_energy_offset
                and slab_val != struct_val
            )
            if mismatched:
                msg = (
                    f"Scatterer {scatterer.name!r} energy_offset={slab_val} does not "
                    f"match structure offset={struct_val}; call lock_energy_offsets()"
                )
                raise ValueError(msg)

    def probe_at(self, base_energy_ev: float) -> Probe:
        """Build the probe for ``base_energy_ev`` plus structure offset."""
        off = self._structure_energy_offset.value
        struct_off = float(off) if off is not None else 0.0
        return Probe(
            base_energy_ev=float(base_energy_ev),
            structure_offset_ev=struct_off,
        )

    def _component_tensors(
        self,
        component: Component,
        probe: Probe,
    ) -> NDArray[np.complex128]:
        if isinstance(component, OrientationSlab):
            return component.tensors_at(probe)
        if isinstance(component, Slab):
            sld = component.sld
            if isinstance(sld, DeferredScatterer):
                return np.asarray([sld.tensor_at(probe)], dtype=np.complex128)
            if energy := probe.effective_ev:
                t = component.tensor(energy=energy)
                return np.asarray([t], dtype=np.complex128)
            return np.asarray([component.tensor()], dtype=np.complex128)
        if isinstance(component, Stack):
            parts = [self._component_tensors(c, probe) for c in component.components]
            tensor = np.concatenate(parts, axis=0)
            repeats = round(abs(component.repeats.value))  # type: ignore[union-attr]
            if repeats > 1:
                tensor = np.concatenate([tensor] * repeats, axis=0)
            return tensor
        if hasattr(component, "tensor"):
            t = component.tensor(energy=probe.effective_ev)
            return np.asarray(t, dtype=np.complex128)
        msg = (
            "Unsupported component type for energy materialization: "
            f"{type(component)!r}"
        )
        raise TypeError(msg)

    def _component_layers(
        self,
        component: Component,
        probe: Probe,
    ) -> NDArray[np.float64]:
        if isinstance(component, OrientationSlab):
            return component.layers_at(probe)
        if isinstance(component, Slab):
            thick = float(component.thick.value or 0.0)
            rough = float(component.rough.value or 0.0)
            sld = component.sld
            if isinstance(sld, DeferredScatterer):
                row = sld.slab_row_at(probe, thick, rough)
                return np.asarray([row], dtype=np.float64)
            row = component.slabs()
            return row if row is not None else np.empty((0, 4))
        if isinstance(component, Stack):
            parts = [self._component_layers(c, probe) for c in component.components]
            layers = np.concatenate(parts, axis=0)
            repeats = round(abs(component.repeats.value))  # type: ignore[union-attr]
            if repeats > 1:
                layers = np.concatenate([layers] * repeats, axis=0)
            return layers
        slabs = component.slabs(structure=self) if hasattr(component, "slabs") else None
        if slabs is None:
            return np.empty((0, 4))
        return np.asarray(slabs, dtype=np.float64)

    def _materialize_uncached(
        self,
        base_energy_ev: float,
        *,
        structure_offset_ev: float | None = None,
    ) -> StackSnapshot:
        if not len(self):
            msg = "DispersiveStructure has no components"
            raise ValueError(msg)
        if structure_offset_ev is not None:
            probe = Probe(
                base_energy_ev=float(base_energy_ev),
                structure_offset_ev=float(structure_offset_ev),
            )
        else:
            probe = self.probe_at(base_energy_ev)
        layer_parts = [self._component_layers(c, probe) for c in self.components]
        tensor_parts = [self._component_tensors(c, probe) for c in self.components]
        layers = np.concatenate(layer_parts, axis=0)
        tensors = np.concatenate(tensor_parts, axis=0)
        if self.reverse_structure:
            roughnesses = layers[1:, 3].copy()
            layers = np.flipud(layers)
            tensors = np.flip(tensors, axis=0)
            layers[1:, 3] = roughnesses[::-1]
            layers[0, 3] = 0.0
        return StackSnapshot(
            layers=layers,
            tensors=tensors,
            energy_ev=probe.effective_ev,
        )

    def materialize_at(self, base_energy_ev: float) -> StackSnapshot:
        """Resolve the stack at ``base_energy_ev`` without rebuilding components."""
        return self.materialize(base_energy_ev)

    def materialize(
        self,
        base_energy_ev: float,
        *,
        structure_offset_ev: float | None = None,
    ) -> StackSnapshot:
        """Resolve every layer at ``base_energy_ev`` plus configured offsets.

        Parameters
        ----------
        base_energy_ev
            Nominal photon energy in eV.
        structure_offset_ev
            When provided, overrides :attr:`structure_energy_offset` for this
            materialization only (used by per-energy instrument channels).
        """
        if structure_offset_ev is None:
            off_val = float(self._structure_energy_offset.value or 0.0)
        else:
            off_val = float(structure_offset_ev)
        key = (float(base_energy_ev), off_val)
        cache = self._materialization_cache
        if cache is not None and key in cache:
            return cache[key]
        snap = self._materialize_uncached(
            float(base_energy_ev),
            structure_offset_ev=off_val,
        )
        if cache is not None:
            cache[key] = snap
        return snap

    def materialize_many(
        self,
        energies: Sequence[float] | NDArray[np.float64],
    ) -> dict[float, StackSnapshot]:
        """Materialize the stack at each distinct energy in ``energies``.

        Reuses the transient batch cache when :meth:`begin_materialization_batch`
        is active so duplicate energies are not recomputed.
        """
        unique = sorted({float(e) for e in energies})
        return {energy: self.materialize(energy) for energy in unique}

    def tensor(self, energy: float | None = None) -> NDArray[np.complex128]:
        """Return tensors at ``energy`` (eV) with all configured offsets applied."""
        if energy is None:
            energy = 250.0
        return self.materialize(float(energy)).tensors

    def slabs(self, energy: float | None = None) -> NDArray[np.float64] | None:
        """Slab rows at ``energy`` (eV); defaults to 250 eV when omitted."""
        if not len(self):
            return None
        if energy is None:
            energy = 250.0
        return self.materialize(float(energy)).layers

    def _walk_slabs(self, components: list[Component]) -> list[Slab]:
        found: list[Slab] = []

        def _walk(component: Component) -> None:
            if isinstance(component, Slab):
                found.append(component)
                return
            if isinstance(component, Stack):
                for child in component.components:
                    _walk(child)

        for component in components:
            _walk(component)
        return found

    def logp_nevot_croce(self) -> float:
        """Return ``-inf`` when a flagged slab violates the Nevot-Croce thickness bound.

        For slabs with :attr:`~refloxide.pxr.plugin.structure.Slab.enforce_nevot_croce`
        set, requires ``thick >= sqrt(2*pi) * rough / 2`` in angstroms (same
        criterion as :class:`~refloxide.pxr.plugin.fitters.LogpExtra`).
        """
        for slab in self._walk_slabs(self.components):
            if not slab.enforce_nevot_croce:
                continue
            thick = float(slab.thick.value or 0.0)
            rough = float(slab.rough.value or 0.0)
            interface_limit = np.sqrt(2.0 * np.pi) * rough / 2.0
            if thick - interface_limit < 0.0:
                return -np.inf
        return 0.0

    def logp(self) -> float:
        """Structure log-prior including per-flag Nevot-Croce constraints."""
        base = float(super().logp())
        if not np.isfinite(base):
            return base
        nc = self.logp_nevot_croce()
        if not np.isfinite(nc):
            return nc
        return base + nc


class OrientationSlab(Component):
    """Depth-resolved uniaxial slab with orientations evaluated at one energy.

    Parameters
    ----------
    thick
        Total thickness in angstrom.
    scatterer
        Tabulated uniaxial scatterer supplying OOC data.
    orientations_rad
        Polar angles in radians, one per sub-layer; length sets sub-layer count.
    rough
        Nevot-Croce roughness between this block and the layer below (angstrom).
    name
        Component label.
    """

    def __init__(
        self,
        thick: float,
        scatterer: DeferredScatterer,
        orientations_rad: NDArray[np.float64],
        rough: float,
        *,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        from refloxide.pxr.energy.scatterer import OocUniTensorScatterer

        if not isinstance(scatterer, OocUniTensorScatterer):
            msg = "OrientationSlab requires OocUniTensorScatterer"
            raise TypeError(msg)
        self.thick = possibly_create_parameter(thick, name=f"{name}_thick")
        self.scatterer = scatterer
        self.orientations_rad = np.asarray(orientations_rad, dtype=np.float64).ravel()
        self.rough = possibly_create_parameter(rough, name=f"{name}_rough")
        n = self.orientations_rad.size
        if n < 1:
            msg = "orientations_rad must have at least one sample"
            raise ValueError(msg)
        self._sub_thick = float(thick) / float(n)
        self._parameters = scatterer.parameters

    def tensors_at(self, probe: Probe) -> NDArray[np.complex128]:
        """Sub-layer tensors with shape ``(n_sub, 3, 3)``."""
        return self.scatterer.tensor_batch_at(probe, self.orientations_rad)

    def layers_at(self, probe: Probe) -> NDArray[np.float64]:
        """Sub-layer slab rows; roughness applies on the bottom sub-layer only."""
        tensors = self.tensors_at(probe)
        rows = [
            _tensor_to_slab_row(
                self._sub_thick,
                float(self.rough.value or 0.0) if i == 0 else 0.0,
                tensors[i],
            )
            for i in range(tensors.shape[0])
        ]
        return np.stack(rows, axis=0)


EnergyDependentStructure = DispersiveStructure
EnergyOrientationSlab = OrientationSlab
