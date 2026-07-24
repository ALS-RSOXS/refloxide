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

import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import numpy as np
from refnx.analysis import Objective as _RefnxObjective
from refnx.dataset import Data1D

from refloxide.data import OpticalConstants

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

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


def _iter_structure_slabs(structure: Any) -> list[Any]:
    """Collect slab leaves (objects with ``thick``, ``rough``, and ``sld``)."""
    found: list[Any] = []

    def walk(component: Any) -> None:
        if (
            getattr(component, "thick", None) is not None
            and getattr(component, "rough", None) is not None
            and getattr(component, "sld", None) is not None
        ):
            found.append(component)
            return
        for child in getattr(component, "components", ()) or ():
            walk(child)

    for component in getattr(structure, "components", ()) or ():
        walk(component)
    return found


def _nevot_croce_limit(rough: float) -> float:
    return float(np.sqrt(2.0 * np.pi) * rough / 2.0)


def nevot_croce_logp(model: ReflectModel) -> float:
    """Nevot-Croce support check independent of the fitter target.

    Enforces ``thick >= sqrt(2*pi) * rough / 2`` on every structure
    :class:`~refloxide.model.Slab` with ``enforce_nevot_croce`` set
    (finite films by default; fronting/backing with thick 0 skip).

    Returns
    -------
    float
        ``0.0`` when every flagged slab is allowed; ``-inf`` when any
        flagged slab violates the bound.

    Notes
    -----
    :class:`Objective` applies this from both :meth:`Objective.nll`
    (so ``CurveFitter(..., target='nll')`` cannot accept violations) and
    ``logp_extra`` (so ``target='nlpost'`` / MCMC still see it as prior
    support). Toggle with ``objective.nc_constraint``.
    """
    structure = getattr(model, "structure", None)
    if structure is None:
        return 0.0
    for slab in _iter_structure_slabs(structure):
        if not getattr(slab, "enforce_nevot_croce", False):
            continue
        thick = float(slab.thick.value or 0.0)
        rough = float(slab.rough.value or 0.0)
        if thick - _nevot_croce_limit(rough) < 0.0:
            return float(-np.inf)
    return 0.0


class LogpExtra:
    """Adapter that exposes :func:`nevot_croce_logp` as a refnx ``logp_extra``.

    Kept so MCMC / ``target='nlpost'`` paths that sum ``logp`` still see the
    Nevot-Croce hard constraint. Likelihood-only fits are covered separately
    by :meth:`Objective.nll`.
    """

    def __init__(self, objective: Objective) -> None:
        self.objective = objective

    def __call__(self, model: ReflectModel, data: Any) -> float:  # noqa: ARG002
        """Return ``0.0`` or ``-inf`` from :func:`nevot_croce_logp`."""
        return nevot_croce_logp(model)


NevotCroceLogp = LogpExtra


def _silence_polars_verbose() -> None:
    """Force quiet Polars streaming logs even if the shell exports POLARS_VERBOSE=1."""
    os.environ["POLARS_VERBOSE"] = "0"


def _warm_objective_caches(objective: Objective) -> None:
    """Re-register OpticalConstants and touch materialize once after unpickle."""
    structure = getattr(objective.model, "structure", None)
    if structure is None:
        return
    seen: set[int] = set()
    for slab in _iter_structure_slabs(structure):
        sld = getattr(slab, "sld", None)
        oocs: list[Any] = []
        single = getattr(sld, "ooc", None)
        if isinstance(single, OpticalConstants):
            oocs.append(single)
        many = getattr(sld, "oocs", None)
        if isinstance(many, list):
            oocs.extend(o for o in many if isinstance(o, OpticalConstants))
        for ooc in oocs:
            oid = id(ooc)
            if oid in seen:
                continue
            seen.add(oid)
            OpticalConstants._cache[ooc.source] = ooc
    energies = list(getattr(objective.model.corrections, "energies", []) or [])
    energy_off = float(objective.model.corrections.energy_offset.value or 0.0)
    for energy in energies[:1]:
        structure.materialize_at(float(energy) + energy_off)


class thread_workers:
    """SciPy DE map-like that evaluates population members on private Objective clones.

    Bare ``workers=8`` uses ``multiprocessing.Pool`` and pickles the objective
    into cold processes (periodictable / polars I/O storms). Pass this object
    instead (SciPy requires a map-like callable, so the instance is callable)::

        with thread_workers(8) as w:
            CurveFitter(objective).fit(method="differential_evolution", workers=w)

    Each worker thread owns a private pickled clone so concurrent ``setp`` /
    ``nll`` cannot race on shared ``Parameter`` state. Energies inside each
    ``nll`` stay serial so Rayon is not nested under the pool.

    Parameters
    ----------
    n : int
        Number of worker threads (>= 1).
    """

    def __init__(self, n: int) -> None:
        if int(n) < 1:
            msg = "thread_workers requires n >= 1"
            raise ValueError(msg)
        self._n = int(n)
        self._pool: ThreadPoolExecutor | None = None
        self._template: bytes | None = None
        self._local = threading.local()

    def bind(self, objective: Objective) -> Self:
        """Serialize ``objective`` once so each worker can load a private clone."""
        _silence_polars_verbose()
        self._template = pickle.dumps(objective, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def __enter__(self) -> Self:
        _silence_polars_verbose()
        self._pool = ThreadPoolExecutor(max_workers=self._n)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None

    def _clone(self) -> Objective:
        template = self._template
        if template is None:
            msg = "thread_workers.bind(objective) before map, or pass objective to map"
            raise RuntimeError(msg)
        clone = pickle.loads(template)
        _warm_objective_caches(clone)
        return clone

    def _worker_objective(self) -> Objective:
        obj = getattr(self._local, "objective", None)
        if obj is None:
            obj = self._clone()
            self._local.objective = obj
        return obj

    def map(
        self,
        func: Callable[..., Any],
        iterable: Iterable[Any],
    ) -> Iterator[Any]:
        """Map ``func`` over ``iterable`` on worker threads with private clones.

        When ``func`` is a bound ``Objective.nll`` / ``nlpost`` / ``logl``, each
        call runs on that worker's private clone so ``setp`` is thread-local.
        Other callables are invoked with the original ``func``.

        ``scipy.optimize.differential_evolution`` always wraps the ``func`` it
        is given in ``scipy._lib._util._FunctionWrapper`` before ever calling
        a map-like ``workers`` (see ``DifferentialEvolutionSolver.__init__``),
        even when called with no extra ``args``. That wrapper exposes neither
        ``__name__`` nor ``__self__``, so this method must unwrap it (via its
        ``.f``/``.args`` attributes) before checking whether the real
        underlying callable is a bound ``Objective`` cost method — checking
        directly on what SciPy hands this method never matches, silently
        falling through to calling the ORIGINAL shared, mutable objective
        concurrently from every worker thread, exactly the ``setp`` race this
        class exists to prevent.
        """
        if self._pool is None:
            msg = "thread_workers must be used as a context manager"
            raise RuntimeError(msg)

        inner_func = getattr(func, "f", func)
        extra_args = tuple(getattr(func, "args", ()) or ())
        method_name = getattr(inner_func, "__name__", None)
        owner = getattr(inner_func, "__self__", None)
        is_bound_cost = (
            method_name in {"nll", "nlpost", "logl", "logp", "logpost"}
            and owner is not None
        )
        if is_bound_cost and self._template is None:
            self.bind(owner)  # type: ignore[arg-type]

        if is_bound_cost:

            def run(x: Any) -> Any:
                return getattr(self._worker_objective(), str(method_name))(
                    x, *extra_args
                )

            return self._pool.map(run, iterable)
        return self._pool.map(func, iterable)

    def __call__(
        self,
        func: Callable[..., Any],
        iterable: Iterable[Any],
    ) -> Iterator[Any]:
        """SciPy ``MapWrapper`` requires a map-like callable; delegate to ``map``."""
        return self.map(func, iterable)


class Objective(_RefnxObjective):
    """Ties a `ReflectModel` to a `ReflectDataset` and a Gaussian log-likelihood.

    Groups the dataset by `(energy, pol)`, then further batches groups that
    share an identical `q` grid into a single `ReflectModel` call using its
    array-energy path (`_build_kernel_batches`) — N energies measured on
    the same q grid become one Rust batch-kernel call, not N. Each row's
    predicted value is read off the `s` or `p` channel its `pol` selects —
    the model itself never chooses a channel.

    Ensures the model's per-energy experiment-correction channels cover every
    energy present in ``data``.

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
    nc_constraint : bool, optional
        When `True` (default), enforce Nevot-Croce
        (``thick >= sqrt(2*pi)*rough/2`` on slabs with
        ``enforce_nevot_croce``) via :func:`nevot_croce_logp` in both
        :meth:`nll` and ``logp_extra``. That makes the check independent
        of :meth:`refnx.analysis.CurveFitter.fit`'s ``target``
        (``'nll'`` or ``'nlpost'``). Toggle mid-session with
        ``objective.nc_constraint = False``.

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
        nc_constraint: bool = True,
    ) -> None:
        if len(data) == 0:
            msg = "Objective requires a non-empty ReflectDataset"
            raise ValueError(msg)
        dataset_energies = sorted({float(e) for e, _pol, _idx in data.groups()})
        model.ensure_energies(dataset_energies)
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
        stub = Data1D(data=(data.q, data.r, data.r_err), name=name or "reflectivity")
        super().__init__(
            model,
            stub,
            use_weights=use_weights,
            transform=transform,
            name=name or "reflectivity_objective",
        )
        self._nc_constraint = bool(nc_constraint)
        self._nevot_croce = LogpExtra(self)
        self.logp_extra = self._nevot_croce if self._nc_constraint else None
        self._cached_y: NDArray[np.float64] | None = None
        self._cached_y_err: NDArray[np.float64] | None = None
        self._refresh_data_transform_cache()

    def _refresh_data_transform_cache(self) -> None:
        """Cache transformed measured ``y`` / ``y_err`` (dataset is fixed)."""
        y = self._dataset.r
        y_err = self._dataset.r_err if self.weighted else np.ones_like(y)
        if self.transform is None:
            self._cached_y = np.asarray(y, dtype=np.float64)
            self._cached_y_err = np.asarray(y_err, dtype=np.float64)
            return
        y_t, y_err_t = self.transform(self._dataset.q, y, y_err)
        self._cached_y = np.asarray(y_t, dtype=np.float64)
        if self.weighted:
            self._cached_y_err = np.asarray(y_err_t, dtype=np.float64)
        else:
            self._cached_y_err = np.ones_like(self._cached_y)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        _silence_polars_verbose()
        if getattr(self, "_cached_y", None) is None:
            self._refresh_data_transform_cache()
        _warm_objective_caches(self)

    @property
    def nc_constraint(self) -> bool:
        """When True, Nevot-Croce is enforced in ``nll`` and ``logp``."""
        return self._nc_constraint

    @nc_constraint.setter
    def nc_constraint(self, value: bool) -> None:
        self._nc_constraint = bool(value)
        self.logp_extra = self._nevot_croce if self._nc_constraint else None

    def nevot_croce_logp(self) -> float:
        """Nevot-Croce support for the current structure; ``0.0`` or ``-inf``."""
        if not self._nc_constraint:
            return 0.0
        return nevot_croce_logp(self.model)

    def nll(self, pvals: NDArray[np.float64] | None = None) -> float:
        """Negative log-likelihood with Nevot-Croce hard rejection.

        Applies :func:`nevot_croce_logp` before evaluating the Gaussian
        likelihood so ``CurveFitter.fit(target='nll')`` (the refnx default)
        cannot accept thick/rough pairs outside Nevot-Croce support.
        """
        self.setp(pvals)
        if not np.isfinite(self.nevot_croce_logp()):
            return float(np.inf)
        return float(-self.logl())

    def _predicted(
        self, pvals: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Model reflectivity for every row, in the dataset's original order.

        Multi-energy path:

        * Batches that already share an identical ``q`` across energies still
          use the array-energy ``ReflectModel`` call.
        * Singleton ``(energy, pol)`` batches are regrouped by energy, then
          materialized ALL AT ONCE via ``Structure.materialize_batch_at`` —
          one vectorized OOC interpolation / ``periodictable`` lookup per
          dispersive scatterer across every energy this dataset needs,
          instead of ``materialize_at`` redoing that lookup from scratch
          once per energy. ``energy_offset`` is applied as a single array
          add over every energy at once (it shifts every energy by the same
          amount), not recomputed per energy in the loop. Only the actual
          kernel call — genuinely per-(energy, pol) because real datasets
          rarely share one ``q`` grid across energies — still runs one at a
          time; energies stay serial there so DE ``workers`` can parallelize
          across population members without nested Rayon/thread pools.
        """
        self.setp(pvals)
        predicted = np.empty(len(self._dataset), dtype=np.float64)

        multi_q: list[_KernelBatch] = []
        by_energy: dict[float, dict[str, _KernelBatch]] = {}
        for batch in self._batches:
            if len(batch.energies) != 1:
                multi_q.append(batch)
                continue
            energy = float(batch.energies[0])
            by_energy.setdefault(energy, {})[batch.pol] = batch

        for batch in multi_q:
            result = self.model(batch.q, np.asarray(batch.energies, dtype=np.float64))
            matrix = result.s if batch.pol == "s" else result.p
            for col, indices in enumerate(batch.row_indices):
                predicted[indices] = matrix[:, col]

        if by_energy:
            energy_off = float(self.model.corrections.energy_offset.value or 0.0)
            base_energies = np.fromiter(by_energy.keys(), dtype=np.float64)
            oc_energies = base_energies + energy_off
            batch_layers, batch_tensor = self.model.structure.materialize_batch_at(
                oc_energies
            )
            for i, energy in enumerate(by_energy):
                pols = by_energy[energy]
                q_s = pols["s"].q if "s" in pols else None
                q_p = pols["p"].q if "p" in pols else None
                r_s, r_p = self.model.reflectivity_channels_at_energy(
                    energy,
                    q_s=q_s,
                    q_p=q_p,
                    layers=batch_layers[i],
                    tensor=batch_tensor[i],
                    parallel=bool(self.model.parallel),
                )
                if r_s is not None and "s" in pols:
                    predicted[pols["s"].row_indices[0]] = r_s
                if r_p is not None and "p" in pols:
                    predicted[pols["p"].row_indices[0]] = r_p
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
        if self._cached_y is None or self._cached_y_err is None:
            self._refresh_data_transform_cache()
        assert self._cached_y is not None and self._cached_y_err is not None
        if self.transform is None:
            return self._cached_y, self._cached_y_err, predicted
        model_t, _ = self.transform(self._dataset.q, predicted)
        return self._cached_y, self._cached_y_err, model_t

    def residuals(
        self, pvals: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Weighted residuals `(y - model) / y_err`.

        Transformed first if `self.transform` is set. When
        ``nc_constraint`` is active and Nevot-Croce is violated, returns a
        large finite residual vector so ``least_squares`` also rejects the
        point (``nll`` is not consulted on that path).
        """
        self.setp(pvals)
        if not np.isfinite(self.nevot_croce_logp()):
            return np.full(len(self._dataset), 1.0e12, dtype=np.float64)
        y, y_err, predicted = self._transformed(None)
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

    def logp(self, pvals: NDArray[np.float64] | None = None) -> float:
        """Log-prior: bounds from refnx plus optional Nevot-Croce ``logp_extra``."""
        total = float(super().logp(pvals))
        if not np.isfinite(total):
            return total
        if self.logp_extra is not None:
            total += float(self.logp_extra(self.model, self.data))
        return total

    def __repr__(self) -> str:
        return (
            f"Objective({self.model!r}, {len(self._dataset)} points, "
            f"{len(self._groups)} energy/pol groups)"
        )
