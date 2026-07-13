"""Compiled multi-energy reflectivity models with a Rust batch kernel.

Design decision: per-energy tensor materialization stays in Python
--------------------------------------------------------------------
``CompiledReflectivityModel._batch_kernel`` builds the ``(n_E, N, 3, 3)``
tensor and layer arrays in a Python loop over energies
(``_materialize_stack``, one call to ``DispersiveStructure.materialize``
per energy) before making a single call into the Rust batch kernel
(:func:`refloxide.rust.uniaxial_reflectivity_batch`). This is a deliberate
choice, not an oversight: benchmarking a representative structure (8-48
uniaxial layers, 200 q points x 50 energies) showed Python-side
materialization consistently at 5-8% of total wall time, and *shrinking*
as structure size grows (8 layers: 7.9%; 24 layers: 5.8%; 48 layers:
5.2% — Rust's per-(q, E) linear algebra scales faster than the O(layers)
Python packing loop). That is comfortably under the ~10% threshold where a
fused Rust entry point (taking the deferred scatterer plan and both arrays
directly, skipping the Python round-trip) would be worth the added
surface. Do not "fix" this split by moving materialization into Rust
without re-benchmarking first — see ``tmp/PLAN.md`` Task 1 for the
methodology if a future structure/energy-count combination changes this
picture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from refnx.analysis import Parameters

from refloxide.pxr.layout import apply_laboratory_scales, reflectivity_for_pol
from refloxide.pxr.plugin.dispersive_instrument import (
    INSTRUMENT_FIELDS,
    ResolvedInstrument,
    make_instrument_channel,
    resolve_instrument,
)
from refloxide.pxr.plugin.model import _smeared_reflectivity
from refloxide.rust import uniaxial_reflectivity, uniaxial_reflectivity_batch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from refloxide.pxr.energy.structure import DispersiveStructure

PolKind = Literal["s", "p", "sp", "ps"]


def _q_grids_equivalent(grids: Sequence[np.ndarray]) -> bool:
    """Return whether every q grid matches the first array element-wise."""
    if len(grids) <= 1:
        return bool(grids)
    ref = grids[0]
    return all(g.shape == ref.shape and np.array_equal(g, ref) for g in grids[1:])


def _energy_tag(energy_ev: float) -> str:
    return f"{energy_ev:.6g}".replace(".", "p")


def _q_with_theta_offset(
    q: np.ndarray,
    wavelength: float,
    theta_offset_deg: float,
) -> np.ndarray:
    theta = np.arcsin(q * wavelength / (4 * np.pi)) * 180 / np.pi
    theta += theta_offset_deg
    return (4 * np.pi / wavelength) * np.sin(theta * np.pi / 180)


def _resolve_q_grids(
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
        qvals_1 = _q_with_theta_offset(qvals_1, wavelength, instrument.theta_offset_s)
        qvals_2 = _q_with_theta_offset(qvals_2, wavelength, instrument.theta_offset_p)
        x = np.concatenate([qvals_1, qvals_2])
        qvals = np.linspace(float(np.min(x)), float(np.max(x)), num_q)
        return x, qvals, qvals_1, qvals_2
    if pol == "s":
        qvals = _q_with_theta_offset(x, wavelength, instrument.theta_offset_s)
    elif pol == "p":
        qvals = _q_with_theta_offset(x, wavelength, instrument.theta_offset_p)
    else:
        qvals = x
    return x, qvals, qvals, qvals


def _apply_channel_scaling(
    refl_block: np.ndarray,
    instrument: ResolvedInstrument,
    *,
    pol: PolKind,
    bkg: float,
) -> np.ndarray:
    block = np.asarray(refl_block, dtype=np.float64).copy()
    apply_laboratory_scales(block, instrument.scale_s, instrument.scale_p)
    curve = reflectivity_for_pol(
        pol,
        block,
        np.arange(block.shape[0], dtype=np.float64),
        np.arange(block.shape[0], dtype=np.float64),
        np.arange(block.shape[0], dtype=np.float64),
    )
    return curve + bkg


def _smeared_channel_reflectivity(
    q: np.ndarray,
    slabs: np.ndarray,
    tensor: np.ndarray,
    energy_ev: float,
    instrument: ResolvedInstrument,
    *,
    pol: PolKind,
    dq: float,
) -> np.ndarray:
    _x, qvals, _qvals_1, _qvals_2 = _resolve_q_grids(
        np.asarray(q, dtype=np.float64),
        12398.42 / float(energy_ev),
        pol,
        instrument,
    )
    q_kernel = qvals + instrument.q_offset
    if float(dq) == 0.0:
        refl, _tran, *_ = uniaxial_reflectivity(
            q_kernel,
            slabs,
            tensor,
            float(energy_ev),
            parallel=False,
        )
    else:
        refl, _tran, *_ = _smeared_reflectivity(
            q_kernel,
            slabs,
            tensor,
            float(energy_ev),
            0.0,
            float(dq),
            backend="uni",
        )
    return _apply_channel_scaling(
        refl,
        instrument,
        pol=pol,
        bkg=instrument.bkg,
    )


class CompiledReflectivityModel:
    """Vectorized ``(q, E)`` reflectivity with per-energy instrument channels.

    Parameters
    ----------
    structure
        Deferred-energy stack materialized once per distinct energy per evaluation.
    energies
        Sorted photon energies (eV) with instrument parameter channels.
    instrument_defaults
        Optional default floats for newly created per-energy parameters.
    parallel
        When ``True``, the Rust batch kernel parallelizes over ``(E, q)`` pairs.
        Default ``False`` for nested fitters.
    """

    def __init__(
        self,
        structure: DispersiveStructure,
        energies: Sequence[float],
        *,
        instrument_defaults: dict[str, float] | None = None,
        parallel: bool = False,
        name: str = "",
    ) -> None:
        unique = sorted({float(e) for e in energies})
        if not unique:
            msg = "CompiledReflectivityModel requires at least one energy"
            raise ValueError(msg)
        self._structure = structure
        self._energies: tuple[float, ...] = tuple(unique)
        self._defaults = dict(instrument_defaults or {})
        self._parallel = bool(parallel)
        self.name = name
        self._channels = {
            energy: make_instrument_channel(
                energy,
                defaults=self._defaults,
                energy_tag=_energy_tag(energy),
            )
            for energy in self._energies
        }
        self._bind_structure(structure)
        self._parameters: Parameters | None = None
        self._rebuild_parameters()

    def _bind_structure(self, structure: DispersiveStructure) -> None:
        structure.structure_energy_offset.setp(vary=False, value=0.0)
        structure.lock_energy_offsets()
        self._parameters = None

    def invalidate_parameters(self) -> None:
        """Drop the cached refnx parameter tree after structural model edits."""
        self._parameters = None

    @property
    def structure(self) -> DispersiveStructure:
        """Shared deferred-energy stack."""
        return self._structure

    @property
    def energies(self) -> tuple[float, ...]:
        """Sorted photon energies (eV) with instrument channels."""
        return self._energies

    @property
    def parallel(self) -> bool:
        """Whether the Rust batch kernel may use rayon."""
        return self._parallel

    def instrument_at(self, energy_ev: float):
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

    def _rebuild_parameters(self) -> None:
        if self._parameters is not None:
            return
        instrument_root = Parameters(name="instrument per energy")
        for energy in self._energies:
            channel = self._channels[energy]
            block = Parameters(name=f"instrument@{_energy_tag(energy)}eV")
            block.extend(channel.parameter(field) for field in INSTRUMENT_FIELDS)
            instrument_root.extend([block])
        root = Parameters(name=self.name or "compiled_reflectivity_model")
        root.extend([instrument_root, self._structure.parameters])
        self._parameters = root

    @property
    def parameters(self) -> Parameters:
        """All structure and instrument parameters for refnx fitters."""
        if self._parameters is None:
            self._rebuild_parameters()
        if self._parameters is None:
            msg = "CompiledReflectivityModel parameters were not initialized"
            raise RuntimeError(msg)
        return self._parameters

    def setp(self, pvals: np.ndarray | None = None) -> None:
        """Load parameter vector ``pvals`` into the model tree."""
        if pvals is not None:
            self.parameters.pvals = np.asarray(pvals, dtype=np.float64)

    def varying_parameters(self):
        """Parameters with ``vary=True`` for samplers and fitters."""
        return self.parameters.varying_parameters()

    def _materialize_stack(
        self,
        energy_ev: float,
        instrument: ResolvedInstrument,
    ) -> tuple[np.ndarray, np.ndarray]:
        snap = self._structure.materialize(
            float(energy_ev),
            structure_offset_ev=instrument.energy_offset_ev,
        )
        return snap.layers, snap.tensors

    def _batch_kernel(
        self,
        q: np.ndarray,
        energy_list: Sequence[float],
        *,
        pol: PolKind,
    ) -> dict[float, np.ndarray]:
        q_base = np.asarray(q, dtype=np.float64)
        unique = [float(e) for e in energy_list]
        q_eff: dict[float, np.ndarray] = {}
        instruments: dict[float, ResolvedInstrument] = {}
        layers_rows: list[np.ndarray] = []
        tensor_rows: list[np.ndarray] = []
        for energy in unique:
            instrument = resolve_instrument(self, energy)
            instruments[energy] = instrument
            wavelength = 12398.42 / energy
            _x, qvals, _q1, _q2 = _resolve_q_grids(
                q_base.copy(),
                wavelength,
                pol,
                instrument,
            )
            q_eff[energy] = qvals + instrument.q_offset
            layers, tensors = self._materialize_stack(energy, instrument)
            layers_rows.append(layers)
            tensor_rows.append(tensors)

        shared_q = _q_grids_equivalent(list(q_eff.values()))
        outputs: dict[float, np.ndarray] = {}
        if shared_q and len(unique) > 0:
            q_kernel = next(iter(q_eff.values()))
            layers_batch = np.stack(layers_rows, axis=0)
            tensor_batch = np.stack(tensor_rows, axis=0)
            energies_arr = np.asarray(unique, dtype=np.float64)
            refl, _tran = uniaxial_reflectivity_batch(
                q_kernel,
                layers_batch,
                tensor_batch,
                energies_arr,
                parallel=self._parallel,
            )
            for idx, energy in enumerate(unique):
                instrument = instruments[energy]
                dq = float(instrument.dq)
                block = refl[idx]
                if dq == 0.0:
                    outputs[energy] = _apply_channel_scaling(
                        block,
                        instrument,
                        pol=pol,
                        bkg=instrument.bkg,
                    )
                else:
                    outputs[energy] = _smeared_channel_reflectivity(
                        q_base,
                        layers_rows[idx],
                        tensor_rows[idx],
                        energy,
                        instrument,
                        pol=pol,
                        dq=dq,
                    )
            return outputs

        for energy in unique:
            instrument = instruments[energy]
            dq = float(instrument.dq)
            if dq == 0.0:
                refl, _tran = uniaxial_reflectivity(
                    q_eff[energy],
                    layers_rows[unique.index(energy)],
                    tensor_rows[unique.index(energy)],
                    energy,
                    parallel=self._parallel,
                )
                outputs[energy] = _apply_channel_scaling(
                    refl,
                    instrument,
                    pol=pol,
                    bkg=instrument.bkg,
                )
            else:
                outputs[energy] = _smeared_channel_reflectivity(
                    q_base,
                    layers_rows[unique.index(energy)],
                    tensor_rows[unique.index(energy)],
                    energy,
                    instrument,
                    pol=pol,
                    dq=dq,
                )
        return outputs

    def reflectivity(
        self,
        q: float | np.ndarray,
        energy: float | np.ndarray,
        *,
        pol: PolKind = "s",
    ) -> np.ndarray | float:
        """Evaluate reflectivity for scalar or vector ``q`` and ``energy``.

        When both ``q`` and ``energy`` are vectors, returns an array with shape
        ``(len(q), len(energy))``. Scalar inputs return a float.
        """
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        e_arr = np.atleast_1d(np.asarray(energy, dtype=np.float64))
        scalar_q = np.ndim(q) == 0
        scalar_e = np.ndim(energy) == 0
        curves = self._batch_kernel(q_arr, list(e_arr), pol=pol)
        if scalar_q and scalar_e:
            return float(curves[float(e_arr[0])][0])
        if scalar_e:
            return curves[float(e_arr[0])]
        if scalar_q:
            return np.array([curves[float(e)][0] for e in e_arr], dtype=np.float64)
        matrix = np.column_stack([curves[float(e)] for e in e_arr])
        return matrix


def compile_model(
    structure: DispersiveStructure,
    energies: Sequence[float],
    *,
    pol: PolKind = "s",
    instrument_defaults: dict[str, float] | None = None,
    parallel: bool = False,
    name: str = "",
) -> CompiledReflectivityModel:
    """Build a :class:`CompiledReflectivityModel` for ``energies`` (eV).

    Parameters
    ----------
    structure
        Deferred-energy stack shared across channels.
    energies
        Distinct photon energies with instrument parameter channels.
    pol
        Default polarization for :meth:`CompiledReflectivityModel.reflectivity`.
    instrument_defaults
        Optional default floats for per-energy instrument parameters.
    parallel
        Forwarded to the compiled model Rust batch path.
    name
        Label for the refnx parameter tree.

    Returns
    -------
    CompiledReflectivityModel
        Model ready for :class:`~refloxide.pxr.objective.ReflectivityObjective`.
    """
    del pol
    return CompiledReflectivityModel(
        structure,
        energies,
        instrument_defaults=instrument_defaults,
        parallel=parallel,
        name=name,
    )
