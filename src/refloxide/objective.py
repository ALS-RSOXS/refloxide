"""refloxide's own reflectivity objective.

`Objective` subclasses `refnx.analysis.Objective` to reuse its parameter
bookkeeping (`setp`, `varying_parameters`, bounds-based `logp`) rather than
reimplementing it — but it is refloxide's own class, with its own
constructor and log-likelihood, informed by the design of `refnx`'s,
`pyref`'s, and `pypxr`'s Objective implementations rather than re-exported
from any of them. It accepts a single- or multi-energy, single- or
mixed-polarization `ReflectDataset` uniformly: there is no separate
`GlobalObjective`/`Term` concept to learn for the multi-energy case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from refnx.analysis import Objective as _RefnxObjective
from refnx.dataset import Data1D

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from refloxide.data import Pol, ReflectDataset
    from refloxide.model import ReflectModel


class _KernelBatch(NamedTuple):
    """One or more `(energy, pol)` groups sharing an identical `q` array.

    All energies in one batch are evaluated in a single `ReflectModel`
    call (the array-energy path) rather than one call per energy.
    """

    pol: Pol
    q: NDArray[np.float64]
    energies: tuple[float, ...]
    row_indices: tuple[NDArray[np.intp], ...]


def _build_kernel_batches(dataset: ReflectDataset) -> list[_KernelBatch]:
    """Group dataset rows so identical-`q`-grid energies share one kernel call.

    Real memory/speed fix, not cosmetic: without this, `Objective` would
    call `ReflectModel` once per `(energy, pol)` group even when many
    energies share the same `q` grid (a common multi-energy experimental
    setup), each call independently re-materializing the structure's
    layers/tensor arrays. Batching them lets `ReflectModel` use the Rust
    batch kernel (`uniaxial_reflectivity_batch`) directly, mirroring what
    `refloxide.pxr.objective.ReflectivityObjective`'s
    `_build_reflectivity_eval_batches` already did for the old Objective
    generation. Grouping is computed once here (data doesn't change across
    `logl()` calls), not per-evaluation.
    """
    batches: list[_KernelBatch] = []
    for pol in ("s", "p"):
        pol_entries = [
            (energy, indices)
            for energy, group_pol, indices in dataset.groups()
            if group_pol == pol
        ]
        buckets: list[
            tuple[NDArray[np.float64], list[tuple[float, NDArray[np.intp]]]]
        ] = []
        for energy, indices in pol_entries:
            q_vals = dataset.q[indices]
            for bucket_q, bucket_entries in buckets:
                if bucket_q.shape == q_vals.shape and np.array_equal(bucket_q, q_vals):
                    bucket_entries.append((energy, indices))
                    break
            else:
                buckets.append((q_vals, [(energy, indices)]))
        for q_vals, entries in buckets:
            batches.append(
                _KernelBatch(
                    pol=pol,  # type: ignore[arg-type]
                    q=q_vals,
                    energies=tuple(e for e, _ in entries),
                    row_indices=tuple(idx for _, idx in entries),
                )
            )
    return batches


class _AnisotropyPair(NamedTuple):
    """One energy's matched s/p rows, sharing an identical `q` array."""

    energy: float
    q: NDArray[np.float64]
    s_indices: NDArray[np.intp]
    p_indices: NDArray[np.intp]


def _build_anisotropy_pairs(dataset: ReflectDataset) -> list[_AnisotropyPair]:
    """Find, per energy, the s/p row pairs sharing an identical `q` array.

    Anisotropy `(R_p - R_s) / (R_p + R_s)` only makes sense where s and p
    were measured at the same q — energies with only one channel, or
    where s/p don't share a q grid, are silently excluded rather than
    raising, since a dataset can legitimately mix anisotropy-comparable
    and non-comparable energies.
    """
    by_energy: dict[float, dict[Pol, NDArray[np.intp]]] = {}
    for energy, pol, indices in dataset.groups():
        by_energy.setdefault(energy, {})[pol] = indices

    pairs: list[_AnisotropyPair] = []
    for energy, channels in sorted(by_energy.items()):
        if "s" not in channels or "p" not in channels:
            continue
        s_indices, p_indices = channels["s"], channels["p"]
        q_s, q_p = dataset.q[s_indices], dataset.q[p_indices]
        if q_s.shape == q_p.shape and np.array_equal(q_s, q_p):
            pairs.append(
                _AnisotropyPair(
                    energy=energy, q=q_s, s_indices=s_indices, p_indices=p_indices
                )
            )
    return pairs


def gaussian_logl(
    y: NDArray[np.float64],
    y_err: NDArray[np.float64],
    model: NDArray[np.float64],
    *,
    weighted: bool,
) -> float:
    """Gaussian log-likelihood for one reflectivity vector.

    Parameters
    ----------
    y, y_err, model : NDArray[np.float64]
        Data, uncertainties, and model reflectivity aligned on `q`.
    weighted : bool
        When `True`, include the `log(2*pi*y_err**2)` normalization term
        (a proper log-likelihood); when `False`, a bare weighted
        sum-of-squares (least-squares-equivalent, for unweighted fits).

    Returns
    -------
    float

    Raises
    ------
    RuntimeError
        If any term is non-finite (typically a zero or negative `y_err`).
    """
    var_y = y_err * y_err
    terms = (y - model) ** 2 / var_y
    if weighted:
        terms = terms + np.log(2 * np.pi * var_y)
    if not np.all(np.isfinite(terms)):
        msg = "Objective.logl encountered a non-finite term (check y_err > 0)"
        raise RuntimeError(msg)
    return float(-0.5 * np.sum(terms))


class Objective(_RefnxObjective):
    """Ties a `ReflectModel` to a `ReflectDataset` and a Gaussian log-likelihood.

    Groups the dataset by `(energy, pol)`, then further batches groups that
    share an identical `q` grid into a single `ReflectModel` call using its
    array-energy path (`_build_kernel_batches`) — N energies measured on
    the same q grid become one Rust batch-kernel call, not N. Each row's
    predicted value is read off the `s` or `p` channel its `pol` selects —
    the model itself never chooses a channel.

    Parameters
    ----------
    model : ReflectModel
    data : ReflectDataset
    use_weights : bool, optional
        When `True` (default), weight the log-likelihood by `1/r_err**2`
        and include the Gaussian normalization term.
    transform : callable, optional
        Same semantics as `refnx.analysis.Objective`'s `transform` — called
        as `transform(q, y)`/`transform(q, y, y_err)` before comparing data
        to model (e.g. `refnx.analysis.Transform("logY")`).
    name : str, optional
    anisotropy_weight : float, optional
        When `0.0` (the default), `logl()` is the standard, unnormalized
        Gaussian log-likelihood over every row. When nonzero (`0 < w <=
        1`), `logl()` instead blends that base likelihood with an extra
        term comparing `model.anisotropy(q, energy)` against the data's
        own `(r_p - r_s) / (r_p + r_s)` at every energy where `data` has
        matching s/p rows on the same `q` grid, then normalizes by the
        number of rows — `ll = (1 - w) * base + w * aniso_term`, `ll /=
        len(data)`. This exactly matches
        `refloxide.pxr.plugin.fitters.AnisotropyObjective`'s formula
        (including its per-point normalization, which only applies in this
        weighted mode — the `anisotropy_weight=0.0` default stays
        unnormalized, matching standard `refnx`/`CurveFitter` convention).

    Raises
    ------
    ValueError
        If `data` is empty, or `anisotropy_weight` is nonzero but no
        energy in `data` has matching s/p rows on the same `q` grid.
    """

    def __init__(
        self,
        model: ReflectModel,
        data: ReflectDataset,
        *,
        use_weights: bool = True,
        transform: Callable[..., Any] | None = None,
        name: str | None = None,
        anisotropy_weight: float = 0.0,
    ) -> None:
        if len(data) == 0:
            msg = "Objective requires a non-empty ReflectDataset"
            raise ValueError(msg)
        self._dataset = data
        self._groups = list(data.groups())
        self._batches = _build_kernel_batches(data)
        self.anisotropy_weight = float(anisotropy_weight)
        self._anisotropy_pairs: list[_AnisotropyPair] = []
        if self.anisotropy_weight:
            self._anisotropy_pairs = _build_anisotropy_pairs(data)
            if not self._anisotropy_pairs:
                msg = (
                    "anisotropy_weight is nonzero but no energy in data has "
                    "matching s/p rows sharing the same q grid"
                )
                raise ValueError(msg)
        stub = Data1D(
            data=(data.q, data.r, data.r_err), name=name or "reflectivity"
        )
        super().__init__(
            model,
            stub,
            use_weights=use_weights,
            transform=transform,
            name=name or "reflectivity_objective",
        )

    def _predicted(
        self, pvals: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Model reflectivity for every row, in the dataset's original order.

        Evaluates `self._batches`, precomputed once at construction — each
        batch is one `ReflectModel` call, covering every energy that shares
        that batch's `q` grid (see `_build_kernel_batches`).
        """
        self.setp(pvals)
        predicted = np.empty(len(self._dataset), dtype=np.float64)
        for batch in self._batches:
            if len(batch.energies) == 1:
                result = self.model(batch.q, batch.energies[0])
                predicted[batch.row_indices[0]] = (
                    result.s if batch.pol == "s" else result.p
                )
                continue
            result = self.model(batch.q, np.asarray(batch.energies, dtype=np.float64))
            matrix = result.s if batch.pol == "s" else result.p
            for col, indices in enumerate(batch.row_indices):
                predicted[indices] = matrix[:, col]
        return predicted

    def generative(
        self, pvals: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Model reflectivity for every row, in the dataset's original order."""
        return self._predicted(pvals)

    def _transformed(
        self, pvals: NDArray[np.float64] | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        predicted = self._predicted(pvals)
        y = self._dataset.r
        y_err = self._dataset.r_err if self.weighted else np.ones_like(y)
        if self.transform is None:
            return y, y_err, predicted
        model_t, _ = self.transform(self._dataset.q, predicted)
        y_t, y_err_t = self.transform(self._dataset.q, y, y_err)
        return y_t, (y_err_t if self.weighted else np.ones_like(y_t)), model_t

    def residuals(
        self, pvals: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Weighted residuals `(y - model) / y_err`.

        Transformed first if `self.transform` is set.
        """
        y, y_err, predicted = self._transformed(pvals)
        return (y - predicted) / y_err

    def _anisotropy_term(self) -> float:
        """`-0.5 * sum((model_anisotropy - data_anisotropy) ** 2)` over all pairs."""
        model_vals = []
        data_vals = []
        for pair in self._anisotropy_pairs:
            model_vals.append(self.model.anisotropy(pair.q, pair.energy))
            r_s = self._dataset.r[pair.s_indices]
            r_p = self._dataset.r[pair.p_indices]
            data_vals.append((r_p - r_s) / (r_p + r_s))
        model_aniso = np.concatenate(model_vals)
        data_aniso = np.concatenate(data_vals)
        return float(-0.5 * np.sum((model_aniso - data_aniso) ** 2))

    def logl(self, pvals: NDArray[np.float64] | None = None) -> float:
        """Log-likelihood over every row in the dataset.

        Standard, unnormalized Gaussian log-likelihood when
        `anisotropy_weight == 0.0` (the default). When nonzero, blends in
        the anisotropy term and normalizes by `len(data)` — see
        `anisotropy_weight`'s docstring on `__init__` for the exact
        formula and why the normalization only applies in this mode.
        """
        y, y_err, predicted = self._transformed(pvals)
        base = gaussian_logl(y, y_err, predicted, weighted=self.weighted)
        if not self.anisotropy_weight:
            return base
        weight = self.anisotropy_weight
        ll = base * (1.0 - weight) + self._anisotropy_term() * weight
        return ll / len(self._dataset)

    def __repr__(self) -> str:
        return (
            f"Objective({self.model!r}, {len(self._dataset)} points, "
            f"{len(self._groups)} energy/pol groups)"
        )
