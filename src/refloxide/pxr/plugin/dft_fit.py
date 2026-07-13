"""Rebuild DFT-constrained multi-energy fits on :class:`DispersiveReflectModel`.

Utilities migrate pickled refnx / pyref :class:`~refnx.analysis.GlobalObjective`
bundles (as used in refl-analysis DFT fitting notebooks) onto one shared
:class:`~refloxide.pxr.energy.structure.DispersiveStructure` and a
:class:`~refloxide.pxr.plugin.dispersive_model.DispersiveReflectModel` evaluated
through :class:`~refloxide.pxr.plugin.batched_global.BatchedGlobalObjective`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from refloxide.pxr.energy.ooc import OocAnchor
from refloxide.pxr.energy.scatterers import (
    DispersiveMaterialSLD,
    FreeTensorScatterer,
    TabulatedUniTensorSLD,
)
from refloxide.pxr.energy.structure import DispersiveStructure
from refloxide.pxr.plugin.batched_global import (
    AnisotropyBatchTerm,
    BatchedGlobalObjective,
    ReflectivityBatchTerm,
)
from refloxide.pxr.plugin.dispersive_instrument import (
    INSTRUMENT_FIELDS,
    safely_setp_param,
)
from refloxide.pxr.plugin.dispersive_model import DispersiveReflectModel, select

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from refnx.analysis import GlobalObjective, Objective, Parameter

    from refloxide.pxr.energy.model import CompiledReflectivityModel
    from refloxide.pxr.objective import ReflectivityTerm
    from refloxide.pxr.plugin.model import ReflectModel

DIAGNOSTIC_ENERGY_EV = 283.7

MIN_INCIDENT_THETA_DEG = 0.02
THETA_OFFSET_MARGIN_DEG = 0.05

_TENSOR_LAYER_PREFIXES: tuple[str, ...] = ("Surface", "ZnPc", "Contamination")
_LabComponentBounds = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


def _param_bounds_tuple(param: Parameter) -> tuple[float, float]:
    """Return ``(lower, upper)`` bounds for a refnx parameter."""
    if param.bounds is None:
        msg = f"Parameter {param!r} has no bounds"
        raise ValueError(msg)
    return float(param.bounds.lb), float(param.bounds.ub)


def _unitensor_lab_components(
    sld: Any,
    density_bounds: tuple[float, float],
    rotation_bounds: tuple[float, float] | None,
) -> _LabComponentBounds:
    """Map UniTensorSLD density/rotation bounds to lab xx/zz intervals."""
    if rotation_bounds is None:
        rot = float(sld.rotation.value or 0.0)
        rotation_bounds = (rot, rot)
    rhos = density_bounds
    rots = rotation_bounds
    orig_rho = float(sld.density.value or 0.0)
    orig_rot = float(sld.rotation.value or 0.0)
    samples: list[tuple[float, float, float, float]] = []
    for rho in rhos:
        for rot in rots:
            sld.density.setp(value=rho)
            sld.rotation.setp(value=rot)
            tensor = sld.tensor
            n_xx = complex(tensor[0, 0])
            n_zz = complex(tensor[2, 2])
            samples.append((n_xx.real, n_xx.imag, n_zz.real, n_zz.imag))
    sld.density.setp(value=orig_rho)
    sld.rotation.setp(value=orig_rot)
    return (
        (min(s[0] for s in samples), max(s[0] for s in samples)),
        (min(s[1] for s in samples), max(s[1] for s in samples)),
        (min(s[2] for s in samples), max(s[2] for s in samples)),
        (min(s[3] for s in samples), max(s[3] for s in samples)),
    )


def apply_film_tensor_bounds_from_unitensor_slab(
    scatterer: FreeTensorScatterer,
    energy_ev: float,
    diag_slab: Any,
    *,
    density_bounds: tuple[float, float],
    rotation_bounds: tuple[float, float] | None = None,
) -> None:
    """Bound one film-layer tensor group from a diagnostic UniTensorSLD slab."""
    sld = diag_slab.sld
    xx_bounds, ixx_bounds, zz_bounds, izz_bounds = _unitensor_lab_components(
        sld,
        density_bounds,
        rotation_bounds,
    )
    group = scatterer.group_at(float(energy_ev))
    group.xx.setp(vary=True, bounds=xx_bounds)
    group.ixx.setp(vary=True, bounds=ixx_bounds)
    group.zz.setp(vary=True, bounds=zz_bounds)
    group.izz.setp(vary=True, bounds=izz_bounds)


def freeze_model_except_energy(
    model: CompiledReflectivityModel,
    structure: DispersiveStructure,
    energy_ev: float,
) -> None:
    """Fix instrument and film-tensor parameters away from ``energy_ev``."""
    target = float(energy_ev)
    for channel_energy in model.energies:
        channel = model.instrument_at(channel_energy)
        vary = abs(float(channel_energy) - target) <= 1e-3
        for field in INSTRUMENT_FIELDS:
            param = channel.parameter(field)
            if not vary:
                param.setp(vary=False)
    for slab in structure:
        sld = slab.sld
        if not isinstance(sld, FreeTensorScatterer) or sld.energies is None:
            continue
        for registered in sld.energies:
            group = sld.group_at(registered)
            active = abs(float(registered) - target) <= 1e-3
            for name in ("xx", "ixx", "zz", "izz"):
                getattr(group, name).setp(vary=active)


def configure_diagnostic_local_fit(
    model: CompiledReflectivityModel,
    structure: DispersiveStructure,
    diagnostic_objective: Objective,
    *,
    energy_ev: float = DIAGNOSTIC_ENERGY_EV,
    film_indices: tuple[int, int, int] = (1, 2, 3),
) -> None:
    """Scope a compiled model to a single-energy local fit at ``energy_ev``.

    Freezes instrument channels and film tensors at other energies, then maps
    diagnostic UniTensorSLD density/rotation bounds onto the active film tensor
    groups at ``energy_ev``.
    """
    freeze_model_except_energy(model, structure, energy_ev)
    old = diagnostic_objective.model.structure
    film_rules: tuple[tuple[tuple[float, float], tuple[float, float] | None], ...] = (
        ((1.5, 2.5), (0.0, np.pi / 2)),
        ((1.0, 2.0), None),
        ((0.0, 2.0), (0.0, np.pi / 2)),
    )
    for idx, rule in zip(film_indices, film_rules, strict=True):
        density_bounds, rotation_bounds = rule
        scatterer = structure[idx].sld
        if not isinstance(scatterer, FreeTensorScatterer):
            msg = f"structure[{idx}] must be FreeTensorScatterer for local film bounds"
            raise TypeError(msg)
        apply_film_tensor_bounds_from_unitensor_slab(
            scatterer,
            energy_ev,
            old[idx],
            density_bounds=density_bounds,
            rotation_bounds=rotation_bounds,
        )


def sync_instrument_from_bundle(
    model: CompiledReflectivityModel,
    bundle: GlobalObjective,
    *,
    exclude_energy: float | None = DIAGNOSTIC_ENERGY_EV,
) -> None:
    """Copy per-energy instrument parameters from a pickled bundle.

    When ``exclude_energy`` is set, that channel is left unchanged so a prior
    single-energy local fit can seed the global objective.
    """
    for obj in bundle.objectives:
        energy = float(obj.model.energy)
        if exclude_energy is not None and abs(energy - float(exclude_energy)) <= 1e-3:
            continue
        channel = model.instrument_at(energy)
        src = obj.model
        for field in INSTRUMENT_FIELDS:
            src_param = getattr(src, field)
            safely_setp_param(
                channel.parameter(field),
                value=src_param.value,
                vary=src_param.vary,
                bounds=src_param.bounds,
            )


def sync_instrument_vary_from_bundle(
    model: CompiledReflectivityModel,
    bundle: GlobalObjective,
) -> None:
    """Restore per-energy instrument ``vary`` flags from a pickled bundle."""
    for obj in bundle.objectives:
        energy = float(obj.model.energy)
        channel = model.instrument_at(energy)
        src = obj.model
        for field in INSTRUMENT_FIELDS:
            src_param = getattr(src, field)
            safely_setp_param(channel.parameter(field), vary=src_param.vary)


def reseed_film_tensors_from_bundle(
    structure: DispersiveStructure,
    bundle: GlobalObjective,
    *,
    film_indices: tuple[int, int, int] = (1, 2, 3),
    exclude_energy: float | None = DIAGNOSTIC_ENERGY_EV,
) -> None:
    """Reload per-energy laboratory tensors on film layers from a pickle bundle.

    Skips ``exclude_energy`` when set so a diagnostic local fit can seed one channel.
    """
    for obj in bundle.objectives:
        energy = float(obj.model.energy)
        if exclude_energy is not None and abs(energy - float(exclude_energy)) <= 1e-3:
            continue
        for idx in film_indices:
            scatterer = structure[idx].sld
            if not isinstance(scatterer, FreeTensorScatterer):
                msg = f"structure[{idx}] must be FreeTensorScatterer for tensor reseed"
                raise TypeError(msg)
            tensor = obj.model.structure[idx].sld.tensor
            scatterer.write_lab_tensor(energy, tensor)


def seed_global_film_tensor_bounds(
    structure: DispersiveStructure,
    bundle: GlobalObjective,
    *,
    film_indices: tuple[int, int, int] = (1, 2, 3),
) -> None:
    """Copy per-energy UniTensorSLD density/rotation bounds onto film tensor groups."""
    for obj in bundle.objectives:
        energy = float(obj.model.energy)
        for idx in film_indices:
            scatterer = structure[idx].sld
            if not isinstance(scatterer, FreeTensorScatterer):
                msg = (
                    f"structure[{idx}] must be FreeTensorScatterer "
                    "for global film bounds"
                )
                raise TypeError(msg)
            diag_slab = obj.model.structure[idx]
            sld = diag_slab.sld
            density_bounds = _param_bounds_tuple(sld.density)
            rotation_bounds = (
                _param_bounds_tuple(sld.rotation) if sld.rotation.vary else None
            )
            apply_film_tensor_bounds_from_unitensor_slab(
                scatterer,
                energy,
                diag_slab,
                density_bounds=density_bounds,
                rotation_bounds=rotation_bounds,
            )


def tighten_theta_offset_bounds_for_compiled(
    model: CompiledReflectivityModel,
    terms: Sequence[ReflectivityTerm],
    *,
    min_incident_theta_deg: float = MIN_INCIDENT_THETA_DEG,
    margin_deg: float = THETA_OFFSET_MARGIN_DEG,
) -> CompiledReflectivityModel:
    """Tighten ``theta_offset`` lower bounds on a :class:`CompiledReflectivityModel`."""
    min_theta: dict[tuple[float, Literal["s", "p"]], float] = {}
    for term in terms:
        key = (float(term.energy), term.pol)
        grazing = min_data_grazing_angle_deg(term.q, term.energy)
        prior = min_theta.get(key)
        min_theta[key] = grazing if prior is None else min(prior, grazing)

    for (energy, pol), data_min_deg in min_theta.items():
        field: Literal["theta_offset_s", "theta_offset_p"] = (
            "theta_offset_s" if pol == "s" else "theta_offset_p"
        )
        param = model.instrument_at(energy).parameter(field)
        safe_lb = min_incident_theta_deg + margin_deg - data_min_deg
        if param.bounds is None:
            lb_old, ub = -0.8, 0.8
        else:
            lb_old, ub = float(param.bounds.lb), float(param.bounds.ub)
        new_lb = max(lb_old, safe_lb)
        kwargs: dict[str, Any] = {"bounds": (new_lb, ub)}
        if param.value is not None and float(param.value) < new_lb:
            kwargs["value"] = new_lb
        safely_setp_param(param, **kwargs)
    return model


def min_data_grazing_angle_deg(q: np.ndarray, energy_ev: float) -> float:
    """Return the smallest laboratory grazing angle (degrees) in ``q``."""
    wavelength = 12398.42 / float(energy_ev)
    sin_theta = np.clip(
        np.asarray(q, dtype=np.float64) * wavelength / (4.0 * np.pi),
        0.0,
        1.0,
    )
    return float(np.degrees(np.arcsin(sin_theta)).min())


def tighten_theta_offset_bounds_from_terms(
    model: DispersiveReflectModel,
    terms: Sequence[ReflectivityBatchTerm],
    *,
    min_incident_theta_deg: float = MIN_INCIDENT_THETA_DEG,
    margin_deg: float = THETA_OFFSET_MARGIN_DEG,
) -> DispersiveReflectModel:
    """Tighten per-channel ``theta_offset`` lower bounds so no data point maps to q=0.

    The Rust Berreman kernel raises when the remapped scattering vector hits
    grazing incidence (singular vacuum dynamic matrix). This helper derives a
    safe lower bound from each reflectivity term's minimum data angle.

    Parameters
    ----------
    model
        Dispersive model whose instrument channels will be updated in place.
    terms
        Reflectivity batch terms supplying ``x``, ``pol``, and ``energy``.
    min_incident_theta_deg
        Minimum allowed ``theta_data + theta_offset`` in degrees.
    margin_deg
        Extra degrees subtracted from the analytic bound for numerical slack.

    Returns
    -------
    DispersiveReflectModel
        The same ``model`` instance for chaining.
    """
    min_theta: dict[tuple[float, Literal["s", "p"]], float] = {}
    for term in terms:
        key = (float(term.energy), term.pol)
        grazing = min_data_grazing_angle_deg(term.x, term.energy)
        prior = min_theta.get(key)
        min_theta[key] = grazing if prior is None else min(prior, grazing)

    for (energy, pol), data_min_deg in min_theta.items():
        field: Literal["theta_offset_s", "theta_offset_p"] = (
            "theta_offset_s" if pol == "s" else "theta_offset_p"
        )
        param = model.instrument_at(energy).parameter(field)
        safe_lb = min_incident_theta_deg + margin_deg - data_min_deg
        if param.bounds is None:
            lb_old, ub = -0.8, 0.8
        else:
            lb_old, ub = float(param.bounds.lb), float(param.bounds.ub)
        new_lb = max(lb_old, safe_lb)
        kwargs: dict[str, Any] = {"bounds": (new_lb, ub)}
        if param.value is not None and float(param.value) < new_lb:
            kwargs["value"] = new_lb
        safely_setp_param(param, **kwargs)
    return model


def load_fit_pickle(path: str | Path) -> GlobalObjective:
    """Load a pickled multi-energy :class:`~refnx.analysis.GlobalObjective`.

    Parameters
    ----------
    path
        Pickle file produced by refl-analysis fitting workflows.

    Returns
    -------
    GlobalObjective
        Bundle whose ``objectives`` entries carry ``model``, ``data``, and
        anisotropy metadata.

    Raises
    ------
    FileNotFoundError
        When ``path`` does not exist.
    TypeError
        When the unpickled object is not a global objective container.
    """
    file_path = Path(path)
    if not file_path.is_file():
        msg = f"Fitting pickle not found: {file_path}"
        raise FileNotFoundError(msg)
    with file_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not hasattr(bundle, "objectives"):
        msg = f"Pickle does not expose objectives; got {type(bundle)!r}"
        raise TypeError(msg)
    return bundle


def slab_base_name(slab: Any) -> str:
    """Strip the trailing ``_<energy>`` suffix from a fitted slab label."""
    return str(slab.name).rsplit("_", 1)[0]


def _rebuild_tensor_slab(old_slab: Any, ooc: OocAnchor) -> Any:
    scatterer = TabulatedUniTensorSLD(
        ooc=ooc,
        rotation=float(old_slab.sld.rotation.value),
        density=float(old_slab.sld.density.value),
        name=slab_base_name(old_slab),
    )
    return scatterer(float(old_slab.thick.value), float(old_slab.rough.value))


def _rebuild_isotropic_slab(
    old_slab: Any,
    formula: str,
) -> Any:
    scatterer = DispersiveMaterialSLD(
        formula,
        float(old_slab.sld.density.value),
        name=slab_base_name(old_slab),
    )
    return scatterer(float(old_slab.thick.value), float(old_slab.rough.value))


def structure_from_dft_diagnostic(
    diagnostic_objective: Objective,
    ooc: OocAnchor | pd.DataFrame,
    *,
    structure_name: str = "znpc_dft",
) -> DispersiveStructure:
    """Build one :class:`DispersiveStructure` from a diagnostic-energy objective.

    Replaces tabulated ZnPc-region layers with ``ooc`` while copying thickness,
    roughness, density, and rotation values from ``diagnostic_objective``.

    Parameters
    ----------
    diagnostic_objective
        Single-energy objective at the reference photon energy (typically
        283.7 eV).
    ooc
        DFT optical-constant table or :class:`OocAnchor`.
    structure_name
        Label stored on the returned structure.

    Returns
    -------
    DispersiveStructure
        Shared stack for all channels on a dispersive reflectivity model.
    """
    anchor = ooc if isinstance(ooc, OocAnchor) else OocAnchor.from_dataframe(ooc)
    old = diagnostic_objective.model.structure  # type: ignore[union-attr]
    components = old.components
    if len(components) < 6:
        msg = f"Expected at least six stack components; got {len(components)}"
        raise ValueError(msg)
    return DispersiveStructure(
        DispersiveMaterialSLD("", 0, name=slab_base_name(components[0]))(0, 0),
        _rebuild_tensor_slab(components[1], anchor),
        _rebuild_tensor_slab(components[2], anchor),
        _rebuild_tensor_slab(components[3], anchor),
        _rebuild_isotropic_slab(components[4], "SiO2"),
        _rebuild_isotropic_slab(components[5], "Si"),
        name=structure_name,
    )


def _copy_parameter_state(
    destination: Any,
    source: Any,
    *,
    constraint: Any | Literal[False] = False,
) -> None:
    kwargs: dict[str, Any] = {"value": source.value}
    if source.bounds is not None:
        kwargs["bounds"] = source.bounds
    if constraint is False:
        kwargs["constraint"] = None
    elif constraint is not None:
        kwargs["constraint"] = constraint
        kwargs["vary"] = None
    safely_setp_param(destination, **kwargs)


def apply_shared_slab_geometry_from_reference(
    structure: DispersiveStructure,
    reference: Any,
    *,
    film_indices: tuple[int, int, int] = (1, 2, 3),
    oxide_index: int = 4,
    substrate_index: int = 5,
    film_vary: bool = True,
    fix_substrate: bool = True,
) -> DispersiveStructure:
    """Copy shared slab thickness and roughness state from a reference stack.

    :class:`~refloxide.pxr.plugin.structure.Scatterer.__call__` sets bounds to
    ``(0, 2 * thick)`` using construction-time thickness. When a compiled stack
    is built from one pickle but seeded from another, callers must replace both
    ``value`` and ``bounds`` together (as in
    :func:`apply_dft_diagnostic_structure_constraints`); otherwise optimizers see
    incoherent boxes (for example oxide thickness above the upper bound).

    Parameters
    ----------
    structure
        Target :class:`~refloxide.pxr.energy.structure.DispersiveStructure`.
    reference
        Reference stack with the desired slab geometry, typically
        ``bundle.objectives[0].model.structure``.
    film_indices
        Surface, bulk, and interface slab indices on both stacks.
    oxide_index
        Oxide slab index on both stacks.
    substrate_index
        Substrate slab index on both stacks.
    film_vary
        When ``True``, mark film ``thick`` and ``rough`` as varying after copy.
    fix_substrate
        When ``True``, fix oxide and substrate ``thick`` and ``rough``.

    Returns
    -------
    DispersiveStructure
        The same ``structure`` instance for chaining.
    """
    ref_components = reference.components
    for idx in film_indices:
        dst = structure[idx]
        src = ref_components[idx]
        for attr in ("thick", "rough"):
            _copy_parameter_state(getattr(dst, attr), getattr(src, attr))
            getattr(dst, attr).setp(vary=film_vary)
    for idx in (oxide_index, substrate_index):
        dst = structure[idx]
        src = ref_components[idx]
        for attr in ("thick", "rough"):
            _copy_parameter_state(getattr(dst, attr), getattr(src, attr))
            getattr(dst, attr).setp(vary=not fix_substrate)
    return structure


def apply_dft_diagnostic_structure_constraints(
    model: DispersiveReflectModel,
    diagnostic_objective: Objective,
    *,
    diagnostic_energy: float = DIAGNOSTIC_ENERGY_EV,
) -> DispersiveReflectModel:
    """Apply fixed substrate / oxide slabs and tensor bounds from the diagnostic fit.

    Mirrors the 283.7 eV setup block in ``fit_dft_fix.ipynb``. Structure
    parameters are shared across energies on :class:`DispersiveReflectModel`, so
    this runs once rather than per-energy linking.

    Parameters
    ----------
    model
        Dispersive model whose ``structure`` was built from the same diagnostic
        objective.
    diagnostic_objective
        Source objective supplying parameter values and bounds.
    diagnostic_energy
        Photon energy (eV) labelling the diagnostic instrument channel.

    Returns
    -------
    DispersiveReflectModel
        The same ``model`` instance for chaining.
    """
    old = diagnostic_objective.model.structure  # type: ignore[union-attr]
    diag_model = diagnostic_objective.model  # type: ignore[union-attr]
    model.energy_offset.at(diagnostic_energy).setp(
        value=diag_model.energy_offset.value,
        vary=False,
    )
    select(model, "Vacuum").thick.setp(vary=False)
    select(model, "Vacuum").rough.setp(vary=False)
    select(model, "Substrate").thick.setp(vary=False)
    select(model, "Substrate").rough.setp(vary=False)
    select(model, "Oxide").thick.setp(vary=False)
    select(model, "Oxide").rough.setp(vary=False)
    select(model, "Oxide").sld.density.setp(vary=False)
    for label in _TENSOR_LAYER_PREFIXES:
        new_slab = select(model, label)
        old_slab = next(c for c in old.components if str(c.name).startswith(label))
        for attr in ("thick", "rough"):
            _copy_parameter_state(
                getattr(new_slab, attr),
                getattr(old_slab, attr),
            )
        for attr in ("density", "rotation"):
            _copy_parameter_state(
                getattr(new_slab.sld, attr),
                getattr(old_slab.sld, attr),
            )
    return model


def link_instrument_to_diagnostic(
    model: DispersiveReflectModel,
    *,
    diagnostic_energy: float = DIAGNOSTIC_ENERGY_EV,
) -> DispersiveReflectModel:
    """Constrain ``energy_offset`` on non-diagnostic channels to the reference energy.

    Parameters
    ----------
    model
        Model with per-energy instrument channels.
    diagnostic_energy
        Reference photon energy (eV).

    Returns
    -------
    DispersiveReflectModel
        The same ``model`` instance for chaining.
    """
    anchor = model.instrument_at(diagnostic_energy).energy_offset
    model.energy_offset.where(
        energy_in=[
            energy
            for energy in model.energies
            if abs(energy - diagnostic_energy) > 1e-6
        ]
    ).link(to=anchor)
    model.energy_offset.at(diagnostic_energy).setp(vary=False)
    return model


def build_dft_dispersive_model(
    bundle: GlobalObjective,
    ooc: OocAnchor | pd.DataFrame,
    *,
    diagnostic_energy: float = DIAGNOSTIC_ENERGY_EV,
    structure_name: str = "znpc_dft",
    apply_constraints: bool = True,
) -> DispersiveReflectModel:
    """Construct a dispersive model from a pickled DFT global fit bundle.

    Parameters
    ----------
    bundle
        Pickled global objective with one entry per photon energy.
    ooc
        DFT optical constants for the tabulated tensor layers.
    diagnostic_energy
        Reference energy for structure rebuild and instrument anchoring.
    structure_name
        Name assigned to the shared structure.
    apply_constraints
        When ``True``, run diagnostic structure and instrument linking helpers.

    Returns
    -------
    DispersiveReflectModel
        Model with shared structure and per-energy instrument parameters copied
        from each bundle objective.
    """
    diagnostic = next(
        obj
        for obj in bundle.objectives
        if abs(float(obj.model.energy) - diagnostic_energy) < 1e-3  # type: ignore[union-attr]
    )
    structure = structure_from_dft_diagnostic(
        diagnostic,
        ooc,
        structure_name=structure_name,
    )
    reflect_models: list[ReflectModel] = [
        obj.model  # type: ignore[misc]
        for obj in bundle.objectives
        if getattr(obj.model, "energy", None) is not None
    ]
    model = DispersiveReflectModel.from_reflect_models(structure, reflect_models)
    if reflect_models:
        model.name = str(getattr(reflect_models[0], "name", ""))
    model.pol = "sp"  # type: ignore[assignment]
    if apply_constraints:
        apply_dft_diagnostic_structure_constraints(
            model,
            diagnostic,
            diagnostic_energy=diagnostic_energy,
        )
        link_instrument_to_diagnostic(
            model,
            diagnostic_energy=diagnostic_energy,
        )
    return model


def batched_objective_from_fit_bundle(
    model: DispersiveReflectModel,
    bundle: GlobalObjective,
    *,
    logp_extra: Any | None = None,
    tighten_theta_bounds: bool = True,
) -> BatchedGlobalObjective:
    """Build :class:`BatchedGlobalObjective` from a pickled global fit bundle.

    Parameters
    ----------
    model
        Shared dispersive reflectivity model for every energy channel.
    bundle
        Source objectives supplying ``XrayReflectDataset`` s/p/anisotropy data.
    logp_extra
        Optional prior hook (for example
        :class:`~refloxide.pxr.plugin.fitters.LogpExtra`).
    tighten_theta_bounds
        When ``True``, narrow ``theta_offset`` lower bounds from data grazing
        angles so the Rust kernel never receives q=0 during optimization.

    Returns
    -------
    BatchedGlobalObjective
        Objective ready for :class:`~refloxide.pxr.plugin.batched_global.BatchedFitter`.
    """
    if not bundle.objectives:
        msg = "Fit bundle has no objectives"
        raise ValueError(msg)
    reference = bundle.objectives[0]
    terms: list[ReflectivityBatchTerm] = []
    anisotropy_terms: list[AnisotropyBatchTerm] = []
    for obj in bundle.objectives:
        energy = float(obj.model.energy)  # type: ignore[union-attr]
        data = obj.data
        if hasattr(data, "s") and hasattr(data, "p"):
            terms.append(
                ReflectivityBatchTerm.from_dataset(
                    x=data.s.x,  # type: ignore[union-attr]
                    y=data.s.y,  # type: ignore[union-attr]
                    y_err=data.s.y_err,  # type: ignore[union-attr]
                    pol="s",
                    energy=energy,
                    name=f"{getattr(obj, 'name', energy)}_s",
                )
            )
            terms.append(
                ReflectivityBatchTerm.from_dataset(
                    x=data.p.x,  # type: ignore[union-attr]
                    y=data.p.y,  # type: ignore[union-attr]
                    y_err=data.p.y_err,  # type: ignore[union-attr]
                    pol="p",
                    energy=energy,
                    name=f"{getattr(obj, 'name', energy)}_p",
                )
            )
        else:
            terms.append(
                ReflectivityBatchTerm.from_dataset(
                    x=data.x,
                    y=data.y,
                    y_err=getattr(data, "y_err", None),
                    pol="s" if obj.model.pol == "s" else "p",  # type: ignore[union-attr]
                    energy=energy,
                )
            )
        anisotropy = getattr(data, "anisotropy", None)
        if anisotropy is not None and anisotropy.x.size > 0:
            anisotropy_terms.append(
                AnisotropyBatchTerm(
                    x=anisotropy.x,
                    y=anisotropy.y,
                    energy=energy,
                    weight=float(getattr(obj, "logp_anisotropy_weight", 0.5)),
                    name=f"{getattr(obj, 'name', energy)}_anisotropy",
                )
            )
    if tighten_theta_bounds:
        tighten_theta_offset_bounds_from_terms(model, terms)
    objective = BatchedGlobalObjective(
        model,
        terms,
        anisotropy_terms=anisotropy_terms,
        transform=getattr(reference, "transform", None),
        use_weights=bool(getattr(reference, "weighted", True)),
        lnsigma=getattr(reference, "lnsigma", None),
        auxiliary_params=getattr(reference, "auxiliary_params", ()),
        alpha=getattr(reference, "alpha", None),
        logp_extra=logp_extra,
        name=getattr(bundle, "name", None) or "dft_batched",
        parallel_kernels=False,
    )
    return objective


def dft_fit_bundle_from_pickle(
    pickle_path: str | Path,
    ooc: OocAnchor | pd.DataFrame | str | Path,
    *,
    diagnostic_energy: float = DIAGNOSTIC_ENERGY_EV,
    attach_logp_extra: bool = True,
) -> tuple[DispersiveReflectModel, BatchedGlobalObjective]:
    """One-shot migration from pickle + OOC table to model and batched objective.

    Parameters
    ----------
    pickle_path
        Pickled :class:`~refnx.analysis.GlobalObjective` path.
    ooc
        DFT OOC anchor, dataframe, or CSV path.
    diagnostic_energy
        Reference photon energy (eV).
    attach_logp_extra
        When ``True``, attach interfacial-thickness
        :class:`~refloxide.pxr.plugin.fitters.LogpExtra`.

    Returns
    -------
    tuple
        ``(model, objective)`` ready for
        :class:`~refloxide.pxr.plugin.batched_global.BatchedFitter`.
    """
    bundle = load_fit_pickle(pickle_path)
    if isinstance(ooc, (str, Path)):
        anchor = OocAnchor.from_file(ooc)
    elif isinstance(ooc, OocAnchor):
        anchor = ooc
    else:
        anchor = OocAnchor.from_dataframe(ooc)
    model = build_dft_dispersive_model(
        bundle,
        anchor,
        diagnostic_energy=diagnostic_energy,
    )
    objective = batched_objective_from_fit_bundle(model, bundle)
    if attach_logp_extra:
        from refloxide.pxr.plugin.fitters import LogpExtra

        objective.logp_extra = LogpExtra(objective)
    return model, objective
