"""Streamlined reflectivity objectives for refnx and emcee fitters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from refnx.analysis import Objective, Parameter, is_parameter
from refnx.dataset import Data1D

if TYPE_CHECKING:
    from collections.abc import Callable

    from refloxide.pxr.energy.model import CompiledReflectivityModel

PolKind = Literal["s", "p"]


@dataclass(frozen=True, slots=True)
class ReflectivityTerm:
    """One reflectivity channel in a global objective.

    Parameters
    ----------
    q
        Scattering vector samples in inverse angstroms.
    y
        Measured reflectivity at ``energy`` for ``pol``.
    y_err
        Uncertainties on ``y``; use ones when unweighted.
    energy
        Photon energy in eV.
    pol
        Laboratory polarization channel.
    lambda_
        Lagrange multiplier on this term's log-likelihood contribution.
    x_err
        Optional per-point ``dQ/Q`` smearing in percent.
    name
        Optional label for debugging.
    """

    q: np.ndarray
    y: np.ndarray
    y_err: np.ndarray
    energy: float
    pol: PolKind
    lambda_: float = 1.0
    x_err: np.ndarray | float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", np.asarray(self.q, dtype=np.float64))
        object.__setattr__(self, "y", np.asarray(self.y, dtype=np.float64))
        object.__setattr__(self, "y_err", np.asarray(self.y_err, dtype=np.float64))
        if self.x_err is not None:
            object.__setattr__(self, "x_err", np.asarray(self.x_err, dtype=np.float64))

    @property
    def weighted(self) -> bool:
        """Whether this term supplies experimental uncertainties."""
        return bool(np.all(self.y_err > 0) and np.all(np.isfinite(self.y_err)))


@dataclass(frozen=True, slots=True)
class _ReflectivityEvalBatch:
    """One vectorized kernel call covering several objective terms."""

    pol: PolKind
    q: np.ndarray
    energies: tuple[float, ...]
    term_indices: tuple[int, ...]


def _build_reflectivity_eval_batches(
    terms: list[ReflectivityTerm],
) -> tuple[_ReflectivityEvalBatch, ...]:
    """Group terms so each batch shares one ``q`` grid and polarization."""
    batches: list[_ReflectivityEvalBatch] = []
    for pol in ("s", "p"):
        indices = [idx for idx, term in enumerate(terms) if term.pol == pol]
        if not indices:
            continue
        q_ids = {id(terms[idx].q) for idx in indices}
        if len(q_ids) == 1:
            q = terms[indices[0]].q
            energies = tuple(sorted({float(terms[idx].energy) for idx in indices}))
            batches.append(
                _ReflectivityEvalBatch(
                    pol=pol,
                    q=q,
                    energies=energies,
                    term_indices=tuple(indices),
                )
            )
            continue
        grouped: dict[tuple[float, int], list[int]] = {}
        for idx in indices:
            key = (float(terms[idx].energy), id(terms[idx].q))
            grouped.setdefault(key, []).append(idx)
        for idxs in grouped.values():
            rep = terms[idxs[0]]
            batches.append(
                _ReflectivityEvalBatch(
                    pol=pol,
                    q=rep.q,
                    energies=(float(rep.energy),),
                    term_indices=tuple(idxs),
                )
            )
    return tuple(batches)


def gaussian_logl(
    y: np.ndarray,
    y_err: np.ndarray,
    model: np.ndarray,
    *,
    weighted: bool,
    lnsigma: float | None,
) -> float:
    """Gaussian log-likelihood for one reflectivity vector.

    Parameters
    ----------
    y, y_err, model
        Data, uncertainties, and model reflectivity aligned on ``q``.
    weighted
        When ``True``, include normalization terms in the Gaussian.
    lnsigma
        Optional fractional model-error scale passed to refnx objectives.
    """
    if lnsigma is not None:
        var_y = y_err * y_err + np.exp(2 * float(lnsigma)) * model * model
    else:
        var_y = y_err * y_err
    terms = (y - model) ** 2 / var_y
    if weighted:
        terms = terms + np.log(2 * np.pi * var_y)
    if np.isnan(terms).any():
        msg = "ReflectivityObjective encountered NaN in log-likelihood terms"
        raise RuntimeError(msg)
    return float(-0.5 * np.sum(terms))


class ReflectivityObjective(Objective):
    """Global reflectivity objective using :class:`CompiledReflectivityModel`.

    Subclasses :class:`~refnx.analysis.Objective` so
    :class:`~refnx.analysis.CurveFitter` receives ``covar``, ``logpost``, and
    related hooks without duplicating refnx boilerplate.

    Parameters
    ----------
    model
        Compiled multi-energy reflectivity model.
    terms
        Reflectivity datasets grouped by energy for batched evaluation.
    logp_extra
        Optional hook ``logp_extra(model, data)`` added to :meth:`logp`.
    lnsigma, use_weights, transform, auxiliary_params, name, alpha
        Same semantics as :class:`~refnx.analysis.Objective`.
    """

    def __init__(
        self,
        model: CompiledReflectivityModel,
        terms: list[ReflectivityTerm],
        *,
        logp_extra: Callable[..., float] | None = None,
        lnsigma: float | Parameter | None = None,
        use_weights: bool = True,
        transform: Callable[..., Any] | None = None,
        auxiliary_params: tuple[Parameter, ...] = (),
        name: str | None = None,
        alpha: float | Parameter | None = None,
    ) -> None:
        if not terms:
            msg = "ReflectivityObjective requires at least one ReflectivityTerm"
            raise ValueError(msg)
        self.terms = list(terms)
        weighted_flags = [t.weighted for t in self.terms]
        self._term_weighted = np.array(weighted_flags, dtype=bool)
        if len(np.unique(self._term_weighted)) > 1:
            msg = "All reflectivity terms must be consistently weighted or unweighted"
            raise ValueError(msg)
        stub = Data1D(
            data=(self.terms[0].q, self.terms[0].y, self.terms[0].y_err),
            name=self.terms[0].name or "reflectivity_stub",
        )
        super().__init__(
            model,
            stub,
            lnsigma=lnsigma,
            use_weights=use_weights,
            transform=transform,
            logp_extra=logp_extra,
            auxiliary_params=auxiliary_params,
            name=name or "reflectivity_objective",
            alpha=alpha,
        )
        self._eval_batches = _build_reflectivity_eval_batches(self.terms)

    @property
    def weighted(self) -> bool:
        """Whether all terms use experimental uncertainties."""
        return bool(self._term_weighted.all() and self._use_weights)

    def _evaluate_terms(self) -> list[np.ndarray]:
        compiled = self.model
        curves: list[np.ndarray | None] = [None] * len(self.terms)
        compiled.structure.begin_materialization_batch()
        try:
            for batch in self._eval_batches:
                block = compiled._batch_kernel(batch.q, batch.energies, pol=batch.pol)
                for idx in batch.term_indices:
                    curves[idx] = np.asarray(
                        block[float(self.terms[idx].energy)],
                        dtype=np.float64,
                    ).ravel()
            filled: list[np.ndarray] = []
            for curve in curves:
                if curve is None:
                    msg = "ReflectivityObjective left terms without model curves"
                    raise RuntimeError(msg)
                filled.append(curve)
            return filled
        finally:
            compiled.structure.end_materialization_batch()

    def generative(self, pvals: np.ndarray | None = None) -> np.ndarray:
        """Concatenated model curves for all terms."""
        self.setp(pvals)
        return np.concatenate(self._evaluate_terms())

    def residuals(self, pvals: np.ndarray | None = None) -> np.ndarray:
        """Weighted residuals with per-term ``lambda_`` scaling."""
        self.setp(pvals)
        curves = self._evaluate_terms()
        lnsigma = (
            float(self.lnsigma.value)  # ty: ignore[unresolved-attribute]
            if is_parameter(self.lnsigma)
            else None
        )
        chunks: list[np.ndarray] = []
        for idx, term in enumerate(self.terms):
            model_y = curves[idx]
            y, y_err, model_y = self._transform_term(term, model_y)
            if lnsigma is not None:
                s_n = np.sqrt(y_err * y_err + np.exp(2 * lnsigma) * model_y * model_y)
            else:
                s_n = y_err
            chunks.append(term.lambda_ * (y - model_y) / s_n)
        return np.concatenate(chunks)

    def logl(self, pvals: np.ndarray | None = None) -> float:
        """Log-likelihood summed over reflectivity terms."""
        self.setp(pvals)
        curves = self._evaluate_terms()
        lnsigma = (
            float(self.lnsigma.value)  # ty: ignore[unresolved-attribute]
            if is_parameter(self.lnsigma)
            else None
        )
        total = 0.0
        for idx, term in enumerate(self.terms):
            model_y = curves[idx]
            y, y_err, model_y = self._transform_term(term, model_y)
            total += float(term.lambda_) * gaussian_logl(
                y,
                y_err,
                model_y,
                weighted=self.weighted,
                lnsigma=lnsigma,
            )
        return total

    def logp(self, pvals: np.ndarray | None = None) -> float:
        """Parameter bounds prior plus structure Nevot-Croce constraints."""
        self.setp(pvals)
        total = float(self.model.structure.logp())
        if self.logp_extra is not None:
            total += float(self.logp_extra(self.model, self.data))
        return total

    def _transform_term(
        self,
        term: ReflectivityTerm,
        model_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = term.y
        y_err = term.y_err if self.weighted else np.ones_like(y)
        if self.transform is None:
            return y, y_err, model_y
        model_t, _ = self.transform(term.q, model_y)
        y, y_err = self.transform(term.q, y, y_err)
        return y, y_err if self.weighted else np.ones_like(y), model_t

    def __repr__(self) -> str:
        parallel = getattr(self.model, "parallel", False)
        return (
            f"ReflectivityObjective({self.model!r}, {len(self.terms)} terms, "
            f"parallel={parallel!r})"
        )
