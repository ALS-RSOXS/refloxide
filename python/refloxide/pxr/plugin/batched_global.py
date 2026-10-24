"""Batched multi-energy global objectives for refnx fitters.

Evaluates many reflectivity datasets from one energy-parameterized
:class:`~refloxide.pxr.plugin.model.ReflectModel` in a single ``logl`` call:
for :class:`~refloxide.pxr.energy.structure.DispersiveStructure` stacks the
objective materializes slabs and tensors once per distinct photon energy per
parameter vector, then reuses those snapshots for every term at that energy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from refnx.analysis import (
    Objective,
    Parameter,
    is_parameter,
)
from refnx.dataset import Data1D

from refloxide.pxr.layout import apply_laboratory_scales, reflectivity_for_pol
from refloxide.pxr.plugin.dispersive_model import resolve_instrument
from refloxide.tmm import uniaxial_reflectivity as rust_uniaxial_reflectivity

if TYPE_CHECKING:
    from refloxide.pxr.energy.structure import StackSnapshot
    from refloxide.pxr.plugin.model import ReflectModel

PolKind = Literal["s", "p"]


@dataclass(frozen=True, slots=True)
class ReflectivityBatchTerm:
    """One reflectivity channel to include in a batched global objective.

    Parameters
    ----------
    x
        Scattering vector samples in inverse angstroms.
    y
        Measured reflectivity for ``pol`` at ``energy``.
    y_err
        Uncertainties on ``y``; use ones when unweighted.
    pol
        ``'s'`` or ``'p'`` laboratory channel (legacy plugin indexing via
        :func:`~refloxide.pxr.layout.reflectivity_for_pol`).
    energy
        Photon energy in eV for dispersive structure evaluation.
    lambda_
        Lagrange multiplier applied to this term's log-likelihood contribution.
    x_err
        Optional per-point ``dQ/Q`` smearing in percent; when omitted the
        model's constant ``dq`` is used.
    name
        Optional label for debugging and plotting.
    """

    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray
    pol: PolKind
    energy: float
    lambda_: float = 1.0
    x_err: np.ndarray | float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", np.asarray(self.x, dtype=np.float64))
        object.__setattr__(self, "y", np.asarray(self.y, dtype=np.float64))
        object.__setattr__(self, "y_err", np.asarray(self.y_err, dtype=np.float64))
        if self.x_err is not None:
            object.__setattr__(self, "x_err", np.asarray(self.x_err, dtype=np.float64))

    @property
    def weighted(self) -> bool:
        """Whether this term supplies experimental uncertainties."""
        return bool(np.all(self.y_err > 0) and np.all(np.isfinite(self.y_err)))

    @classmethod
    def from_dataset(
        cls,
        *,
        x: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray | None,
        pol: PolKind,
        energy: float,
        lambda_: float = 1.0,
        name: str | None = None,
    ) -> ReflectivityBatchTerm:
        """Build a term from bare arrays."""
        err = np.ones_like(y, dtype=np.float64) if y_err is None else y_err
        return cls(
            x=x,
            y=y,
            y_err=err,
            pol=pol,
            energy=float(energy),
            lambda_=float(lambda_),
            name=name,
        )


@dataclass(frozen=True, slots=True)
class AnisotropyBatchTerm:
    """Optional anisotropy residual appended after reflectivity terms."""

    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray | None = None
    energy: float = 0.0
    lambda_: float = 1.0
    weight: float = 0.5
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", np.asarray(self.x, dtype=np.float64))
        object.__setattr__(self, "y", np.asarray(self.y, dtype=np.float64))
        if self.y_err is not None:
            object.__setattr__(self, "y_err", np.asarray(self.y_err, dtype=np.float64))


def _q_grid_for_pol(
    x: np.ndarray,
    pol: PolKind,
    energy: float,
    *,
    theta_offset_s: float,
    theta_offset_p: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply laboratory theta offsets and return ``(qvals, qvals_1, qvals_2)``."""
    wavelength = 12398.42 / energy
    x = np.asarray(x, dtype=np.float64)
    if pol == "s":
        theta = np.arcsin(x * wavelength / (4 * np.pi)) * 180 / np.pi
        theta += theta_offset_s
        qvals = (4 * np.pi / wavelength) * np.sin(theta * np.pi / 180)
        return qvals, qvals, qvals
    theta = np.arcsin(x * wavelength / (4 * np.pi)) * 180 / np.pi
    theta += theta_offset_p
    qvals = (4 * np.pi / wavelength) * np.sin(theta * np.pi / 180)
    return qvals, qvals, qvals


def _gaussian_logl(
    y: np.ndarray,
    y_err: np.ndarray,
    model: np.ndarray,
    *,
    weighted: bool,
    lnsigma: float | None,
) -> float:
    if lnsigma is not None:
        var_y = y_err * y_err + np.exp(2 * float(lnsigma)) * model * model
    else:
        var_y = y_err * y_err
    terms = (y - model) ** 2 / var_y
    if weighted:
        terms = terms + np.log(2 * np.pi * var_y)
    if np.isnan(terms).any():
        msg = "BatchedGlobalObjective encountered NaN in log-likelihood terms"
        raise RuntimeError(msg)
    return float(-0.5 * np.sum(terms))


def _materialize_structure(
    structure: Any,
    energy: float,
    *,
    structure_offset_ev: float | None = None,
) -> StackSnapshot:
    """Materialize ``structure`` at ``energy`` with optional offset override."""
    if hasattr(structure, "materialize"):
        return structure.materialize(  # type: ignore[union-attr]
            energy,
            structure_offset_ev=structure_offset_ev,
        )
    layers = structure.slabs()  # type: ignore[union-attr]
    tensors = structure.tensor(energy=energy)  # type: ignore[union-attr]
    from refloxide.pxr.energy.structure import StackSnapshot

    return StackSnapshot(
        layers=np.asarray(layers, dtype=np.float64),
        tensors=np.asarray(tensors, dtype=np.complex128),
        energy_ev=float(energy),
    )


def _evaluate_reflectivity_term(
    model: ReflectModel,
    term: ReflectivityBatchTerm,
    *,
    parallel_kernels: bool,
    snapshot: StackSnapshot | None = None,
) -> np.ndarray:
    """Evaluate one batched term's model curve on ``term.x`` via Rust TMM."""
    energy = float(term.energy)
    instrument = resolve_instrument(model, energy)  # type: ignore[arg-type]
    structure = model.structure  # type: ignore[union-attr]
    if snapshot is None:
        snapshot = _materialize_structure(
            structure,
            energy,
            structure_offset_ev=instrument.energy_offset_ev,
        )
    slabs = snapshot.layers
    tensor = snapshot.tensors
    qvals, qvals_1, qvals_2 = _q_grid_for_pol(
        term.x,
        term.pol,
        energy,
        theta_offset_s=instrument.theta_offset_s,
        theta_offset_p=instrument.theta_offset_p,
    )
    dq_raw = term.x_err if term.x_err is not None else instrument.dq
    dq = float(np.asarray(dq_raw).flat[0])
    q_eval = np.asarray(qvals + instrument.q_offset, dtype=np.float64)
    if float(dq) == 0.0:
        refl, _tran = rust_uniaxial_reflectivity(
            q_eval,
            np.asarray(slabs, dtype=np.float64),
            np.asarray(tensor, dtype=np.complex128),
            energy,
            parallel=parallel_kernels,
        )
        refl = np.asarray(refl, dtype=np.float64)
        apply_laboratory_scales(refl, instrument.scale_s, instrument.scale_p)
        refl = refl + instrument.bkg
    else:
        from refloxide.pxr.plugin.model import reflectivity as plugin_reflectivity

        result = plugin_reflectivity(
            q_eval,
            slabs,
            tensor,
            energy,
            scale_s=instrument.scale_s,
            scale_p=instrument.scale_p,
            bkg=instrument.bkg,
            dq=dq,
            backend="uni",
        )
        if result is None:
            msg = "reflectivity returned None; check dq / backend"
            raise RuntimeError(msg)
        refl, _tran, _components = result
    return reflectivity_for_pol(
        term.pol,
        refl,
        qvals,
        qvals_1,
        qvals_2,
    )


def evaluate_reflectivity_batch(
    model: ReflectModel,
    terms: list[ReflectivityBatchTerm],
    *,
    parallel_kernels: bool = False,
    parallel_terms: bool = False,
    max_workers: int | None = None,
) -> dict[int, np.ndarray]:
    """Evaluate all reflectivity terms, materializing each distinct energy once.

    Parameters
    ----------
    model
        Shared :class:`~refloxide.pxr.plugin.model.ReflectModel` whose structure
        encodes the energy-dependent optical model.
    terms
        Reflectivity channels to evaluate.
    parallel_kernels
        When ``True``, pass ``parallel=True`` to the Rust kernel for each term's
        q-grid. Keep ``False`` when this runs inside MCMC or other multithreaded
        fitters.
    parallel_terms
        When ``True`` and ``parallel_kernels`` is ``False``, evaluate distinct
        terms on a :class:`~concurrent.futures.ThreadPoolExecutor`.
    max_workers
        Worker count for ``parallel_terms``; defaults to the number of terms.

    Returns
    -------
    dict
        Maps term index to the model reflectivity vector aligned with ``term.x``.
    """
    if not terms:
        return {}

    structure = model.structure  # type: ignore[union-attr]
    snapshots: dict[tuple[float, float], StackSnapshot] = {}
    batch_begin = getattr(structure, "begin_materialization_batch", None)
    batch_end = getattr(structure, "end_materialization_batch", None)
    if hasattr(structure, "materialize"):
        if batch_begin is not None:
            batch_begin()
        try:
            seen: set[tuple[float, float]] = set()
            for term in terms:
                instrument = resolve_instrument(model, float(term.energy))  # type: ignore[arg-type]
                key = (float(term.energy), instrument.energy_offset_ev)
                if key in seen:
                    continue
                seen.add(key)
                snapshots[key] = _materialize_structure(
                    structure,
                    key[0],
                    structure_offset_ev=key[1],
                )
        finally:
            if batch_end is not None:
                batch_end()

    def _one(idx_term: tuple[int, ReflectivityBatchTerm]) -> tuple[int, np.ndarray]:
        idx, term = idx_term
        instrument = resolve_instrument(model, float(term.energy))  # type: ignore[arg-type]
        snap_key = (float(term.energy), instrument.energy_offset_ev)
        snap = snapshots.get(snap_key)
        return idx, _evaluate_reflectivity_term(
            model,
            term,
            parallel_kernels=parallel_kernels,
            snapshot=snap,
        )

    if parallel_terms and not parallel_kernels and len(terms) > 1:
        workers = max_workers or len(terms)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(_one, enumerate(terms)))
        return dict(pairs)

    return dict(_one(item) for item in enumerate(terms))


class BatchedGlobalObjective(Objective):
    """Global objective that evaluates reflectivity terms in one batched pass.

    .. deprecated::
        Prefer :class:`~refloxide.pxr.objective.ReflectivityObjective` with
        :class:`~refloxide.pxr.energy.model.CompiledReflectivityModel`.

    Parameters
    ----------
    model
        Single :class:`~refloxide.pxr.plugin.model.ReflectModel` whose structure
        defines thicknesses, roughnesses, and energy-dependent optical tensors.
    terms
        Reflectivity datasets (typically one s and one p curve per energy).
    anisotropy_terms
        Optional anisotropy penalties using the same ``model.anisotropy`` path
        as :class:`~refloxide.pxr.plugin.fitters.AnisotropyObjective`.
    lambdas
        Per-term multipliers broadcast against ``terms`` (and anisotropy terms
        use their own ``lambda_`` fields).
    parallel_kernels
        Forwarded to :func:`refloxide.tmm.uniaxial_reflectivity` as
        ``parallel=`` when ``dq == 0``. Default ``False`` for nested fitters.
    parallel_terms
        Evaluate independent terms on a thread pool when ``True`` and
        ``parallel_kernels`` is ``False``.
    logp_extra
        Optional constraint hook with signature ``logp_extra(model, data)``.
        ``data`` is a stub built from the first reflectivity term.
    alpha, lnsigma, use_weights, transform, auxiliary_params, name
        Same semantics as :class:`refnx.analysis.Objective`.
    """

    def __init__(
        self,
        model: ReflectModel,
        terms: list[ReflectivityBatchTerm],
        *,
        anisotropy_terms: list[AnisotropyBatchTerm] | None = None,
        lambdas: np.ndarray | list[float] | None = None,
        parallel_kernels: bool = False,
        parallel_terms: bool = False,
        max_workers: int | None = None,
        logp_extra=None,
        lnsigma: float | Parameter | None = None,
        use_weights: bool = True,
        transform=None,
        auxiliary_params=(),
        name: str | None = None,
        alpha: float | Parameter | None = None,
    ) -> None:
        if not terms:
            msg = "BatchedGlobalObjective requires at least one ReflectivityBatchTerm"
            raise ValueError(msg)
        self.terms = list(terms)
        self.anisotropy_terms = list(anisotropy_terms or [])
        nobj = len(self.terms)
        if lambdas is not None:
            self.lambdas = np.broadcast_to(lambdas, (nobj,)).astype(float)
        else:
            self.lambdas = np.ones(nobj, dtype=float)
        self.parallel_kernels = bool(parallel_kernels)
        self.parallel_terms = bool(parallel_terms)
        self.max_workers = max_workers
        stub = Data1D(
            data=(self.terms[0].x, self.terms[0].y, self.terms[0].y_err),
            name=self.terms[0].name or "batched_stub",
        )
        super().__init__(
            model,
            stub,
            lnsigma=lnsigma,
            use_weights=use_weights,
            transform=transform,
            logp_extra=logp_extra,
            auxiliary_params=auxiliary_params,
            name=name or "batched_global",
            alpha=alpha,
        )
        weighted_flags = [t.weighted for t in self.terms]
        self._batched_weighted = np.array(weighted_flags, dtype=bool)
        if len(np.unique(self._batched_weighted)) > 1:
            msg = "All reflectivity terms must be consistently weighted or unweighted"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return (
            f"BatchedGlobalObjective({self.model!r}, {len(self.terms)} terms,"
            f" parallel_kernels={self.parallel_kernels!r})"
        )

    @property
    def weighted(self) -> bool:
        """Whether all terms use experimental uncertainties."""
        return bool(self._batched_weighted.all() and self._use_weights)

    @property
    def npoints(self) -> int:
        """Total number of reflectivity points across terms."""
        n = int(sum(t.y.size for t in self.terms))
        n += int(sum(t.x.size for t in self.anisotropy_terms))
        return n

    @classmethod
    def from_anisotropy_objectives(
        cls,
        objectives: list[Any],
        *,
        model: ReflectModel | None = None,
        lambdas: np.ndarray | list[float] | None = None,
        parallel_kernels: bool = False,
        parallel_terms: bool = False,
        **kwargs: Any,
    ) -> BatchedGlobalObjective:
        """Build a batched objective from anisotropy-style objectives.

        Each source objective contributes s- and p-pol reflectivity terms plus
        an optional anisotropy residual when ``data.anisotropy`` is populated.
        All objectives must share the same underlying structure parameters when
        ``model`` is omitted (the first objective's model is used).
        """
        from refloxide.pxr.plugin.fitters import AnisotropyObjective

        if not objectives:
            msg = "from_anisotropy_objectives requires at least one objective"
            raise ValueError(msg)
        if not all(isinstance(o, AnisotropyObjective) for o in objectives):
            msg = "from_anisotropy_objectives expects AnisotropyObjective instances"
            raise TypeError(msg)
        shared = model if model is not None else objectives[0].model
        terms: list[ReflectivityBatchTerm] = []
        anis: list[AnisotropyBatchTerm] = []
        term_lambdas: list[float] = []
        for obj_index, obj in enumerate(objectives):
            lam = 1.0
            if lambdas is not None:
                lam = float(np.broadcast_to(lambdas, (len(objectives),))[obj_index])
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
                        lambda_=lam,
                        name=f"{obj.name}_s" if obj.name else None,
                    )
                )
                term_lambdas.append(lam)
                terms.append(
                    ReflectivityBatchTerm.from_dataset(
                        x=data.p.x,  # type: ignore[union-attr]
                        y=data.p.y,  # type: ignore[union-attr]
                        y_err=data.p.y_err,  # type: ignore[union-attr]
                        pol="p",
                        energy=energy,
                        lambda_=lam,
                        name=f"{obj.name}_p" if obj.name else None,
                    )
                )
                term_lambdas.append(lam)
            else:
                terms.append(
                    ReflectivityBatchTerm.from_dataset(
                        x=data.x,
                        y=data.y,
                        y_err=getattr(data, "y_err", None),
                        pol="s" if obj.model.pol == "s" else "p",  # type: ignore[union-attr]
                        energy=energy,
                        lambda_=lam,
                        name=str(obj.name) if obj.name else None,
                    )
                )
                term_lambdas.append(lam)
            if hasattr(data, "anisotropy") and data.anisotropy.x.size > 0:  # type: ignore[union-attr]
                anis.append(
                    AnisotropyBatchTerm(
                        x=data.anisotropy.x,  # type: ignore[union-attr]
                        y=data.anisotropy.y,  # type: ignore[union-attr]
                        energy=energy,
                        lambda_=lam,
                        weight=float(obj.logp_anisotropy_weight),
                        name=f"{obj.name}_anisotropy" if obj.name else None,
                    )
                )
        init_lambdas = term_lambdas if lambdas is None else lambdas
        return cls(
            shared,
            terms,
            anisotropy_terms=anis,
            lambdas=init_lambdas,
            parallel_kernels=parallel_kernels,
            parallel_terms=parallel_terms,
            logp_extra=objectives[0].logp_extra,
            lnsigma=objectives[0].lnsigma,
            use_weights=objectives[0].weighted,
            transform=objectives[0].transform,
            auxiliary_params=objectives[0].auxiliary_params,
            name=kwargs.pop("name", None),
            alpha=objectives[0].alpha,
            **kwargs,
        )

    def generative(self, pvals=None) -> np.ndarray:
        """Concatenated model curves for all reflectivity terms."""
        self.setp(pvals)
        curves = evaluate_reflectivity_batch(
            self.model,
            self.terms,
            parallel_kernels=self.parallel_kernels,
            parallel_terms=self.parallel_terms,
            max_workers=self.max_workers,
        )
        parts = [curves[idx] for idx in range(len(self.terms))]
        return np.concatenate(parts)

    def residuals(self, pvals=None) -> np.ndarray:
        """Concatenated residuals with per-term ``lambdas`` scaling."""
        self.setp(pvals)
        curves = evaluate_reflectivity_batch(
            self.model,
            self.terms,
            parallel_kernels=self.parallel_kernels,
            parallel_terms=self.parallel_terms,
            max_workers=self.max_workers,
        )
        chunks: list[np.ndarray] = []
        lnsigma = float(self.lnsigma.value) if is_parameter(self.lnsigma) else None  # ty: ignore[unresolved-attribute]
        for idx, term in enumerate(self.terms):
            model_y = curves[idx]
            y, y_err, model_y = self._transform_term(term, model_y)
            if lnsigma is not None:
                s_n = np.sqrt(y_err * y_err + np.exp(2 * lnsigma) * model_y * model_y)
            else:
                s_n = y_err
            chunks.append(self.lambdas[idx] * (y - model_y) / s_n)
        return np.concatenate(chunks)

    def logl(self, pvals=None) -> float:
        """Batched log-likelihood with one structure evaluation per energy group."""
        self.setp(pvals)
        curves = evaluate_reflectivity_batch(
            self.model,
            self.terms,
            parallel_kernels=self.parallel_kernels,
            parallel_terms=self.parallel_terms,
            max_workers=self.max_workers,
        )
        lnsigma = float(self.lnsigma.value) if is_parameter(self.lnsigma) else None  # ty: ignore[unresolved-attribute]
        logl = 0.0
        for idx, term in enumerate(self.terms):
            model_y = curves[idx]
            y, y_err, model_y = self._transform_term(term, model_y)
            logl += float(self.lambdas[idx]) * _gaussian_logl(
                y,
                y_err,
                model_y,
                weighted=self.weighted,
                lnsigma=lnsigma,
            )
        extra = self.model.logp()
        if self.logp_extra is not None:
            extra += self.logp_extra(self.model, self.data)
        logl += extra
        for aterm in self.anisotropy_terms:
            saved = self.model.energy  # type: ignore[union-attr]
            try:
                self.model.energy = aterm.energy  # type: ignore[union-attr]
                model_a = self.model.anisotropy(aterm.x)  # type: ignore[union-attr]
            finally:
                self.model.energy = saved  # type: ignore[union-attr]
            resid = model_a - aterm.y
            if aterm.y_err is not None:
                resid = resid / aterm.y_err
            logl += (
                float(aterm.lambda_)
                * float(aterm.weight)
                * float(-0.5 * np.sum(resid * resid))
            )
        return logl

    def _transform_term(
        self,
        term: ReflectivityBatchTerm,
        model_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = term.y
        y_err = term.y_err if self.weighted else np.ones_like(y)
        if self.transform is None:
            return y, y_err, model_y
        model_t, _ = self.transform(term.x, model_y)
        y, y_err = self.transform(term.x, y, y_err)
        return y, y_err if self.weighted else np.ones_like(y), model_t

    def plot(self, pvals=None, **_kwargs):  # type: ignore[no-untyped-def]  # ty: ignore[invalid-method-override]
        """Plot each reflectivity term on shared axes (requires matplotlib)."""
        import matplotlib.pyplot as plt

        self.setp(pvals)
        curves = evaluate_reflectivity_batch(
            self.model,
            self.terms,
            parallel_kernels=self.parallel_kernels,
            parallel_terms=self.parallel_terms,
            max_workers=self.max_workers,
        )
        fig, ax = plt.subplots()
        for idx, term in enumerate(self.terms):
            label = term.name or f"term {idx}"
            ax.plot(term.x, term.y, "o", ms=3, label=f"{label} data")
            ax.plot(term.x, curves[idx], "-", label=f"{label} model")
        ax.set_yscale("log")
        ax.set_xlabel(r"$q\,(\mathrm{\AA}^{-1})$")
        ax.set_ylabel("Reflectivity")
        ax.legend()
        return fig, ax


class BatchedFitter:
    """Curve fitter entry point for :class:`BatchedGlobalObjective`.

    Thin wrapper around :class:`~refloxide.pxr.plugin.fitters.Fitter` so batched
    objectives use the same ``sample``, ``fit``, and ``to_arviz`` surface as the
    plugin :class:`~refloxide.pxr.plugin.fitters.Fitter`.
    """

    def __init__(
        self,
        objective: BatchedGlobalObjective,
        ntemps: int = -1,
        nwalkers: int | None = None,
        walkers_per_param: int = 10,
        **mcmc_kws: Any,
    ) -> None:
        from refloxide.pxr.plugin.fitters import Fitter

        self._fitter = Fitter(
            objective,
            ntemps=ntemps,
            nwalkers=nwalkers,
            walkers_per_param=walkers_per_param,
            **mcmc_kws,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fitter, name)
