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
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Self, get_args

import numpy as np
from refnx._lib import flatten
from refnx._lib import unique as _unique
from refnx.analysis import Objective as _RefnxObjective
from refnx.analysis import Parameters
from refnx.dataset import Data1D

from refloxide import tmm
from refloxide.data import OpticalConstants
from refloxide.model import _plan_fused_bookended, _PointPlan

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from numpy.typing import NDArray
    from scipy.optimize import OptimizeResult

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


class _PointsJob(NamedTuple):
    """One parameter set's kernel inputs for the single-dispatch GPU path.

    ``points.channels`` holds one entry per energy (dataset rows, matching
    ``rows``) followed by one entry per anisotropy target.
    """

    energies: list[float]
    layers: NDArray[np.float64]
    tensor: NDArray[np.complex128]
    points: _PointPlan
    rows: list[tuple[NDArray[np.intp] | None, NDArray[np.intp] | None]]


class _AnisotropyPair(NamedTuple):
    """One energy's anisotropy target ``(R_p - R_s) / (R_p + R_s)`` on a q grid."""

    energy: float
    q: NDArray[np.float64]
    target: NDArray[np.float64]


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
            r_s, r_p = dataset.r[s_indices], dataset.r[p_indices]
            pairs.append(
                _AnisotropyPair(energy=energy, q=q_s, target=(r_p - r_s) / (r_p + r_s))
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


def _gaussian_terms(
    y: NDArray[np.float64],
    y_err: NDArray[np.float64],
    model: NDArray[np.float64],
    *,
    weighted: bool,
) -> NDArray[np.float64]:
    """Per-row log-likelihood terms whose sum is :func:`gaussian_logl`."""
    var_y = y_err * y_err
    terms = (y - model) ** 2 / var_y
    if weighted:
        terms = terms + np.log(2 * np.pi * var_y)
    if not np.all(np.isfinite(terms)):
        msg = "Objective.logl encountered a non-finite term (check y_err > 0)"
        raise RuntimeError(msg)
    return -0.5 * terms


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


def nevot_croce_violation(model: ReflectModel) -> float:
    """Squared Nevot-Croce violation ``sum(max(0, limit(rough) - thick) ** 2)``.

    Summed over the same flagged slabs as :func:`nevot_croce_logp`, in
    angstrom squared; ``0.0`` exactly when that check passes. Used as a
    smooth exterior penalty by gradient-based fitting, where the hard
    ``-inf`` wall would stall a line search.
    """
    structure = getattr(model, "structure", None)
    if structure is None:
        return 0.0
    total = 0.0
    for slab in _iter_structure_slabs(structure):
        if not getattr(slab, "enforce_nevot_croce", False):
            continue
        gap = _nevot_croce_limit(float(slab.rough.value or 0.0)) - float(
            slab.thick.value or 0.0
        )
        if gap > 0.0:
            total += gap * gap
    return total


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


def minimize_lbfgsb(
    objective: Objective,
    *,
    polish: bool = True,
    nc_penalty: float = 1e6,
    **options: Any,
) -> OptimizeResult:
    """L-BFGS-B on :meth:`Objective.nll_and_grad` with bound-normalized variables.

    Every varying parameter with finite bounds is mapped to ``[0, 1]``
    (parameters with infinite bounds keep unit scale), so gradients of
    parameters on very different scales (thicknesses, backgrounds, energy
    offsets) are comparably conditioned, and each iteration costs one
    value-plus-gradient kernel dispatch on ``objective.model.device``.

    On the GPU the descent runs on single-precision values and exact
    single-precision gradients, which reach the optimum's neighborhood
    quickly but can stall in stiff, strongly correlated valleys (tightly
    constrained backgrounds or offsets), where ``f32`` gradient error
    corrupts the quasi-Newton curvature pairs. With ``polish`` (default)
    the search then continues on the CPU in double precision, also with
    exact gradients, which typically needs few iterations from the GPU
    result. The model's ``device`` and ``parallel`` settings are restored
    afterwards, and the objective's parameters are left at the optimum.

    Parameters
    ----------
    objective : Objective
        Objective whose varying parameters define the search space.
    polish : bool, optional
        Finish a GPU run with a double-precision CPU L-BFGS-B stage.
        Ignored when the model already runs on the CPU.
    nc_penalty : float, optional
        Forwarded to :meth:`Objective.nll_and_grad`.
    **options
        Forwarded to ``scipy.optimize.minimize(..., method="L-BFGS-B")``
        (for example ``options={"maxiter": 500}`` or ``callback``).

    Returns
    -------
    scipy.optimize.OptimizeResult
        Result of the last stage, with ``x`` in physical parameter units;
        after a polish, ``gpu_result`` holds the single-precision stage and
        ``nfev`` / ``nit`` count both stages.
    """
    first = _lbfgsb_stage(objective, nc_penalty, options)
    model = objective.model
    if not polish or model.device != "gpu":
        return first
    device, parallel = model.device, model.parallel
    try:
        model.device, model.parallel = "cpu", True
        final = _lbfgsb_stage(objective, nc_penalty, options)
    finally:
        model.device, model.parallel = device, parallel
    final.gpu_result = first
    final.nfev += first.nfev
    final.nit += first.nit
    return final


def _lbfgsb_stage(
    objective: Objective, nc_penalty: float, options: dict[str, Any]
) -> OptimizeResult:
    """One bound-normalized L-BFGS-B run from the objective's current values."""
    from scipy.optimize import minimize

    varying = list(objective.varying_parameters())
    x0 = np.fromiter((float(p.value) for p in varying), dtype=np.float64)
    lb = np.array([p.bounds.lb for p in varying], dtype=np.float64)
    ub = np.array([p.bounds.ub for p in varying], dtype=np.float64)
    finite = np.isfinite(lb) & np.isfinite(ub) & (ub > lb)
    offset = np.where(finite, lb, 0.0)
    span = np.where(finite, ub - lb, 1.0)
    u_bounds = [
        (0.0, 1.0) if f else (lo, hi) for f, lo, hi in zip(finite, lb, ub, strict=True)
    ]

    def fun(u: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        value, grad = objective.nll_and_grad(offset + u * span, nc_penalty=nc_penalty)
        return value, grad * span

    result = minimize(
        fun,
        (x0 - offset) / span,
        jac=True,
        method="L-BFGS-B",
        bounds=u_bounds,
        **options,
    )
    result.x = offset + result.x * span
    objective.setp(result.x)
    return result


class gpu_batch:
    """Map-like that evaluates a whole population in one GPU dispatch.

    Pass as ``workers=`` to differential evolution or ``pool=`` to MCMC
    sampling; every generation (or walker ensemble) is prefetched through
    :meth:`Objective.prefetch`, then refnx's own cost function runs per
    candidate against the cached curves, so likelihoods, priors, and
    transforms are exactly those of the serial path::

        model.device = "gpu"
        fitter = CurveFitter(objective)
        fitter.fit("differential_evolution", workers=gpu_batch(objective))
        fitter.sample(1000, pool=gpu_batch(objective))

    Falls back to plain serial evaluation when the objective is not on the
    single-dispatch GPU path.

    Parameters
    ----------
    objective : Objective
        Objective whose ``model.device`` is ``"gpu"``.
    """

    def __init__(self, objective: Objective) -> None:
        self._objective = objective

    def map(self, func: Callable[..., Any], iterable: Iterable[Any]) -> list[Any]:
        """Prefetch ``iterable`` in one dispatch, then apply ``func`` to each item."""
        xs = [np.asarray(x, dtype=np.float64) for x in iterable]
        self._objective.prefetch(xs)
        try:
            return [func(x) for x in xs]
        finally:
            self._objective.clear_prefetch()

    def __call__(self, func: Callable[..., Any], iterable: Iterable[Any]) -> list[Any]:
        """SciPy ``workers`` calling convention; delegates to :meth:`map`."""
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
    anisotropy_targets : mapping of float to (array_like, array_like), optional
        Explicit anisotropy data ``{energy: (q, A)}`` replacing the targets
        derived from matched s/p rows, for data whose anisotropy was formed
        on its own grid (for example s and p interpolated onto a common
        ``q``). Every energy must be present in ``data``. Model anisotropy
        uses each channel's own theta offset, scale, and background.
    normalization : {"auto", "energy"}, optional
        ``"auto"`` (default) keeps the rules above. ``"energy"`` computes
        ``sum_E [(1 - w) * logl_E + w * aniso_E] / N_E`` with ``N_E`` the
        number of rows at energy ``E``, so each energy contributes a
        per-point average regardless of how many points it has; this is
        the sum of per-energy ``AnisotropyObjective`` terms combined in a
        refnx ``GlobalObjective``.
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
        anisotropy_targets: Mapping[float, tuple[Any, Any]] | None = None,
        normalization: Literal["auto", "energy"] = "auto",
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
        if normalization not in get_args(Literal["auto", "energy"]):
            msg = f"normalization must be 'auto' or 'energy', got {normalization!r}"
            raise ValueError(msg)
        self.normalization = normalization
        self._anisotropy_pairs: list[_AnisotropyPair] = []
        if self.anisotropy_weight and anisotropy_targets is not None:
            known = set(dataset_energies)
            for energy, (q_a, a_a) in sorted(anisotropy_targets.items()):
                if float(energy) not in known:
                    msg = f"anisotropy target energy {energy} is not in data"
                    raise ValueError(msg)
                q_arr = np.asarray(q_a, dtype=np.float64)
                a_arr = np.asarray(a_a, dtype=np.float64)
                if q_arr.shape != a_arr.shape or q_arr.ndim != 1:
                    msg = f"anisotropy target at {energy} eV needs matching 1-D q, A"
                    raise ValueError(msg)
                self._anisotropy_pairs.append(
                    _AnisotropyPair(energy=float(energy), q=q_arr, target=a_arr)
                )
        elif self.anisotropy_weight:
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

    def varying_parameters(self) -> Parameters:
        """Varying parameters, from a cached flattening of the parameter tree.

        Same selection as `refnx.analysis.Objective.varying_parameters`
        (``vary`` flags and constraint dependencies are re-read on every
        call, so toggling ``vary`` or bounds between fit stages needs no
        action), but the tree is flattened once instead of on every
        ``setp``/``logp``/``nll``, which dominates per-evaluation cost for
        many-layer structures. Call :meth:`refresh_parameters` after adding
        or removing parameters (for example editing the structure).
        """
        flat = self.__dict__.get("_flat_parameters")
        if flat is None:
            flat = list(flatten(self.parameters))
            self._flat_parameters = flat
        chosen: list[Any] = []
        for p in flat:
            if p.vary:
                chosen.append(p)
            elif len(p._deps):
                chosen.extend(d for d in p.dependencies() if d.vary)
        return Parameters(_unique(chosen))

    def refresh_parameters(self) -> None:
        """Invalidate the cached parameter flattening after structural edits."""
        self._flat_parameters = None

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

    def _likelihood_weights(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Row weights ``f`` and anisotropy-target weights ``g``.

        ``logl = sum_i f_i * t_i + sum_k g_k * (-0.5 * sum((A_k - A_k^data)^2))``
        with ``t_i`` the per-row Gaussian terms; shared by :meth:`logl` and
        :meth:`nll_and_grad` so value and gradient cannot drift apart.
        """
        n = len(self._dataset)
        w = float(self.anisotropy_weight)
        pairs = self._anisotropy_pairs
        if getattr(self, "normalization", "auto") == "energy":
            f = np.empty(n, dtype=np.float64)
            counts: dict[float, int] = {}
            for energy, _pol, idx in self._groups:
                counts[float(energy)] = counts.get(float(energy), 0) + len(idx)
            for energy, _pol, idx in self._groups:
                f[idx] = (1.0 - w) / counts[float(energy)]
            g = np.array([w / counts[p.energy] for p in pairs], dtype=np.float64)
            return f, g
        if not w:
            return np.ones(n, dtype=np.float64), np.zeros(len(pairs), dtype=np.float64)
        return (
            np.full(n, (1.0 - w) / n, dtype=np.float64),
            np.full(len(pairs), w / n, dtype=np.float64),
        )

    def _group_batches(
        self,
    ) -> tuple[list[_KernelBatch], dict[float, dict[str, _KernelBatch]]]:
        """Split batches into multi-energy ones and per-energy ``{pol: batch}``."""
        multi_q: list[_KernelBatch] = []
        by_energy: dict[float, dict[str, _KernelBatch]] = {}
        for batch in self._batches:
            if len(batch.energies) != 1:
                multi_q.append(batch)
                continue
            by_energy.setdefault(float(batch.energies[0]), {})[batch.pol] = batch
        return multi_q, by_energy

    def _param_key(self) -> bytes:
        """Exact byte key of the current varying parameter values."""
        return np.fromiter(
            (float(p.value) for p in self.varying_parameters()), dtype=np.float64
        ).tobytes()

    def _points_job(self, *, any_device: bool = False) -> _PointsJob | None:
        """Kernel inputs for the current parameters, or ``None`` if ineligible.

        Eligible when ``dq < 0.5``, every batch is single-energy, and the
        model runs on the GPU (or ``any_device``); the whole dataset then
        maps onto one :func:`refloxide.tmm.uniaxial_reflectivity_points` call.
        """
        if self.model.device != "gpu" and not any_device:
            return None
        if float(self.model.corrections.dq.value or 0.0) >= 0.5:
            return None
        multi_q, by_energy = self._group_batches()
        if multi_q or not by_energy:
            return None
        energies = list(by_energy)
        energy_off = float(self.model.corrections.energy_offset.value or 0.0)
        layers, tensor = self.model.structure.materialize_batch_at(
            np.asarray(energies, dtype=np.float64) + energy_off
        )
        pols = [by_energy[e] for e in energies]
        pairs = self._anisotropy_pairs if self.anisotropy_weight else []
        stack_of_energy = {e: i for i, e in enumerate(energies)}
        if any(p.energy not in stack_of_energy for p in pairs):
            return None
        points = self.model._prepare_points(
            energies + [p.energy for p in pairs],
            [p["s"].q if "s" in p else None for p in pols] + [p.q for p in pairs],
            [p["p"].q if "p" in p else None for p in pols] + [p.q for p in pairs],
            stack=list(range(len(energies)))
            + [stack_of_energy[p.energy] for p in pairs],
        )
        rows = [
            (
                p["s"].row_indices[0] if "s" in p else None,
                p["p"].row_indices[0] if "p" in p else None,
            )
            for p in pols
        ]
        return _PointsJob(energies, layers, tensor, points, rows)

    def _fill_job(
        self, job: _PointsJob, refl: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        """Dataset-ordered predictions and per-target model anisotropy for one job."""
        predicted = np.empty(len(self._dataset), dtype=np.float64)
        results = self.model._finish_points(refl, job.points)
        n_e = len(job.rows)
        for (s_rows, p_rows), (r_s, r_p) in zip(job.rows, results[:n_e], strict=True):
            if s_rows is not None and r_s is not None:
                predicted[s_rows] = r_s
            if p_rows is not None and r_p is not None:
                predicted[p_rows] = r_p
        aniso = []
        for r_s, r_p in results[n_e:]:
            assert r_s is not None and r_p is not None
            aniso.append((r_p - r_s) / (r_p + r_s))
        return predicted, aniso

    def prefetch(self, xs: Iterable[NDArray[np.float64]]) -> int:
        """Evaluate many parameter sets in one GPU dispatch and cache the curves.

        Each ``x`` is applied with ``setp``, its stack is materialized on the
        host, and every candidate's points are concatenated into a single
        :func:`refloxide.tmm.uniaxial_reflectivity_points` call. The predicted
        curves and model anisotropy are cached under the exact bytes of each
        ``x``, so subsequent ``nll``/``logl``/``logpost`` calls with that
        vector (in any order) skip the kernel; each cached result is consumed
        once. Parameter values are
        restored afterwards. Candidates whose materialization raises are
        skipped and evaluate normally later.

        Parameters
        ----------
        xs : iterable of NDArray[np.float64]
            Varying-parameter vectors, as passed to ``setp``.

        Returns
        -------
        int
            Number of candidates cached; ``0`` when the objective is not on
            the single-dispatch GPU path (see :meth:`_points_job`).
        """
        if self.model.device != "gpu":
            return 0
        saved = np.fromiter(
            (float(p.value) for p in self.varying_parameters()), dtype=np.float64
        )
        jobs: list[_PointsJob] = []
        keys: list[bytes] = []
        try:
            for x in xs:
                x_arr = np.asarray(x, dtype=np.float64)
                self.setp(x_arr)
                try:
                    job = self._points_job()
                except (ValueError, RuntimeError, FloatingPointError):
                    continue
                if job is None:
                    return 0
                jobs.append(job)
                keys.append(x_arr.tobytes())
        finally:
            self.setp(saved)
        if not jobs:
            return 0

        n_e = len(jobs[0].energies)
        refl, _tran = tmm.uniaxial_reflectivity_points(
            np.concatenate([j.points.q for j in jobs]),
            np.concatenate(
                [j.points.stack_index + k * n_e for k, j in enumerate(jobs)]
            ),
            np.concatenate([j.layers for j in jobs]),
            np.concatenate([j.tensor for j in jobs]),
            np.concatenate([np.asarray(j.energies, dtype=np.float64) for j in jobs]),
            device="gpu",
        )
        cache = getattr(self, "_prefetched", None)
        if cache is None:
            cache = self._prefetched = {}
        start = 0
        for key, job in zip(keys, jobs, strict=True):
            stop = start + job.points.q.size
            cache[key] = self._fill_job(job, refl[start:stop])
            start = stop
        return len(jobs)

    def _predict_all(
        self, pvals: NDArray[np.float64] | None = None
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        """Dataset-ordered predictions plus model anisotropy for every target.

        Uses a :meth:`prefetch` hit when available, then the single-dispatch
        points path, then per-energy evaluation (anisotropy via
        :meth:`ReflectModel.anisotropy`).
        """
        self.setp(pvals)
        prefetched = getattr(self, "_prefetched", None)
        if prefetched:
            key = (
                self._param_key()
                if pvals is None
                else np.asarray(pvals, dtype=np.float64).tobytes()
            )
            hit = prefetched.pop(key, None)
            if hit is not None:
                return hit
        job = self._points_job()
        if job is not None:
            refl, _tran = tmm.uniaxial_reflectivity_points(
                job.points.q,
                job.points.stack_index,
                job.layers,
                job.tensor,
                np.asarray(job.energies, dtype=np.float64),
                parallel=bool(self.model.parallel),
                device=self.model.device,
            )
            return self._fill_job(job, refl)
        predicted = self._predicted_rows()
        aniso = (
            [self.model.anisotropy(p.q, p.energy) for p in self._anisotropy_pairs]
            if self.anisotropy_weight
            else []
        )
        return predicted, aniso

    def nll_and_grad(
        self,
        pvals: NDArray[np.float64] | None = None,
        *,
        rel_step: float = 1e-4,
        nc_penalty: float = 1e6,
    ) -> tuple[float, NDArray[np.float64]]:
        """Negative log-likelihood and its exact gradient in one kernel dispatch.

        The kernel returns reflectance and forward-mode derivatives for every
        varying parameter at once (on :attr:`ReflectModel.device`; ``f32`` on
        the GPU, ``f64`` on the CPU), so the gradient carries the kernel's
        relative accuracy instead of finite-difference cancellation. How each
        parameter moves the kernel inputs (layer rows, tensors, per-point
        ``q``, scale, background) is taken from a central difference of the
        ``f64`` host-side materialization with step
        ``rel_step * max(|value|, 1)``. That difference only differentiates
        the host mapping, never the kernel: it is exact for the linear
        mappings (thickness, roughness, density, ``q_offset``, scale,
        background), second-order accurate for smooth ones (theta offsets),
        and averages the adjacent slopes of the piecewise tabulated optical
        constants when ``energy_offset`` straddles a table node. The step is
        large enough that ``f64`` roundoff in the materialization does not
        enter. Scale, background, the data transform, and the
        Gaussian likelihood are then differentiated analytically (the
        transform by an ``f64`` relative central difference per point).

        With ``nc_constraint`` active, points outside Nevot-Croce support
        return ``nll + nc_penalty * nevot_croce_violation(model)`` and its
        gradient instead of :meth:`nll`'s ``inf``: a hard wall stalls
        gradient line searches, while the exterior penalty steers back into
        support. Inside support the value equals :meth:`nll`.

        Parameters
        ----------
        pvals : NDArray[np.float64] or None
            Varying-parameter values; ``None`` uses the current values.
        rel_step : float, optional
            Relative host-side materialization step.
        nc_penalty : float, optional
            Weight of the squared Nevot-Croce violation (per angstrom
            squared) used outside support.

        Returns
        -------
        tuple of (float, NDArray[np.float64])
            ``nll`` (plus any Nevot-Croce penalty) and its gradient ordered
            like :meth:`varying_parameters`.

        Raises
        ------
        ValueError
            When ``dq >= 0.5``, a batch spans several energies, or a parameter
            changes the layer count; use a derivative-free or
            finite-difference method there.
        """
        self.setp(pvals)
        varying = list(self.varying_parameters())
        x0 = np.fromiter((float(p.value) for p in varying), dtype=np.float64)
        use_nc = bool(self._nc_constraint)
        penalty = nc_penalty * nevot_croce_violation(self.model) if use_nc else 0.0
        dpenalty = np.zeros_like(x0)
        base = self._points_job(any_device=True)
        if base is None:
            msg = (
                "nll_and_grad requires single-energy batches and dq < 0.5; "
                "use a finite-difference gradient for this objective"
            )
            raise ValueError(msg)

        n_dirs = len(varying)
        n_points = base.points.q.size
        n_entries = len(base.points.channels)
        dq = np.zeros((n_dirs, n_points))
        dlayers = np.zeros((n_dirs, *base.layers.shape))
        dtensor = np.zeros((n_dirs, *base.tensor.shape), dtype=np.complex128)
        dchan = np.zeros((n_dirs, n_entries, 3))
        try:
            for k in range(n_dirs):
                h = rel_step * max(abs(x0[k]), 1.0)
                shifted = []
                violation = []
                for sign in (1.0, -1.0):
                    x = x0.copy()
                    x[k] += sign * h
                    self.setp(x)
                    if use_nc:
                        violation.append(nevot_croce_violation(self.model))
                    job = self._points_job(any_device=True)
                    if job is None or job.layers.shape != base.layers.shape:
                        msg = f"parameter {varying[k].name!r} changes the stack layout"
                        raise ValueError(msg)
                    shifted.append(job)
                plus, minus = shifted
                inv = 1.0 / (2.0 * h)
                if use_nc:
                    dpenalty[k] = nc_penalty * (violation[0] - violation[1]) * inv
                dq[k] = (plus.points.q - minus.points.q) * inv
                dlayers[k] = (plus.layers - minus.layers) * inv
                dtensor[k] = (plus.tensor - minus.tensor) * inv
                for i, (cp, cm) in enumerate(
                    zip(plus.points.channels, minus.points.channels, strict=True)
                ):
                    dchan[k, i] = (
                        (cp[0].scale_s - cm[0].scale_s) * inv,
                        (cp[0].scale_p - cm[0].scale_p) * inv,
                        (cp[0].bkg - cm[0].bkg) * inv,
                    )
        finally:
            self.setp(x0)

        refl, jac = tmm.uniaxial_reflectivity_points_jvp(
            base.points.q,
            base.points.stack_index,
            base.layers,
            base.tensor,
            np.asarray(base.energies, dtype=np.float64),
            dq,
            dlayers,
            dtensor,
            parallel=bool(self.model.parallel),
            device=self.model.device,
        )

        def channel_values(i: int, sl: slice, col: int) -> tuple[Any, Any]:
            channel = base.points.channels[i][0]
            scale = channel.scale_s if col == 1 else channel.scale_p
            dscale = dchan[:, i, 0] if col == 1 else dchan[:, i, 1]
            r = refl[sl, col]
            value = scale * r + channel.bkg
            deriv = (
                scale * jac[:, sl, col]
                + dscale[:, None] * r[None, :]
                + dchan[:, i, 2][:, None]
            )
            return value, deriv

        predicted = np.empty(len(self._dataset), dtype=np.float64)
        dpred = np.zeros((n_dirs, len(self._dataset)), dtype=np.float64)
        for i, (s_rows, p_rows) in enumerate(base.rows):
            _channel, s_sl, p_sl = base.points.channels[i]
            for rows, sl, col in ((s_rows, s_sl, 1), (p_rows, p_sl, 0)):
                if rows is None or sl is None:
                    continue
                predicted[rows], dpred[:, rows] = channel_values(i, sl, col)

        y, y_err = self._cached_y, self._cached_y_err
        if y is None or y_err is None:
            self._refresh_data_transform_cache()
            y, y_err = self._cached_y, self._cached_y_err
        assert y is not None and y_err is not None
        if self.transform is None:
            model_t = predicted
            dtransform = np.ones_like(predicted)
        else:
            q_all = self._dataset.q
            model_t, _ = self.transform(q_all, predicted)
            step = 1e-6 * np.where(predicted != 0, np.abs(predicted), 1.0)
            t_plus, _ = self.transform(q_all, predicted + step)
            t_minus, _ = self.transform(q_all, predicted - step)
            dtransform = (t_plus - t_minus) / (2.0 * step)
        f_rows, g_pairs = self._likelihood_weights()
        terms = _gaussian_terms(y, y_err, model_t, weighted=self.weighted)
        logl = float(f_rows @ terms)
        grad = -(dpred @ (f_rows * (y - model_t) / (y_err * y_err) * dtransform))
        n_e = len(base.rows)
        for k, pair in enumerate(
            self._anisotropy_pairs if self.anisotropy_weight else []
        ):
            _channel, s_sl, p_sl = base.points.channels[n_e + k]
            assert s_sl is not None and p_sl is not None
            r_s, d_s = channel_values(n_e + k, s_sl, 1)
            r_p, d_p = channel_values(n_e + k, p_sl, 0)
            total = r_p + r_s
            model_a = (r_p - r_s) / total
            d_a = 2.0 * (r_s * d_p - r_p * d_s) / (total * total)
            resid = model_a - pair.target
            logl += float(g_pairs[k] * -0.5 * np.sum(resid * resid))
            grad += g_pairs[k] * (d_a @ resid)
        return float(-logl + penalty), dpenalty + grad

    def nll_grad(self, pvals: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """Gradient of :meth:`nll`, for ``jac=`` in scipy minimizers.

        See :meth:`nll_and_grad`. Pass as
        ``CurveFitter(objective).fit("L-BFGS-B", jac=objective.nll_grad)``.
        """
        return self.nll_and_grad(pvals)[1]

    def clear_prefetch(self) -> None:
        """Drop any unconsumed :meth:`prefetch` results."""
        self._prefetched = {}

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

        On the GPU, or with a :meth:`prefetch` hit, predictions come from
        the single-dispatch points path (see :meth:`_predict_all`).
        """
        return self._predict_all(pvals)[0]

    def _predicted_rows(self) -> NDArray[np.float64]:
        """Per-energy evaluation of every row at the current parameters."""
        predicted = np.empty(len(self._dataset), dtype=np.float64)
        multi_q, by_energy = self._group_batches()

        for batch in multi_q:
            result = self.model(batch.q, np.asarray(batch.energies, dtype=np.float64))
            matrix = result.s if batch.pol == "s" else result.p
            for col, indices in enumerate(batch.row_indices):
                predicted[indices] = matrix[:, col]

        if by_energy:
            energy_off = float(self.model.corrections.energy_offset.value or 0.0)
            dq = float(self.model.corrections.dq.value or 0.0)
            base_energies = np.fromiter(by_energy.keys(), dtype=np.float64)
            oc_energies = base_energies + energy_off
            # Fused bookended path rebuilds the film inside Rust and ignores
            # pre-materialized layers. Skip the expensive Python materialize
            # when that path will win; otherwise batch-materialize once.
            fused_eligible = (
                self.model.device == "cpu"
                and dq < 0.5
                and (
                    _plan_fused_bookended(self.model.structure, float(oc_energies[0]))
                    is not None
                )
            )
            batch_layers = None
            batch_tensor = None
            if not fused_eligible:
                batch_layers, batch_tensor = self.model.structure.materialize_batch_at(
                    oc_energies
                )
            for i, energy in enumerate(by_energy):
                pols = by_energy[energy]
                q_s = pols["s"].q if "s" in pols else None
                q_p = pols["p"].q if "p" in pols else None
                layers_i = None if batch_layers is None else batch_layers[i]
                tensor_i = None if batch_tensor is None else batch_tensor[i]
                r_s, r_p = self.model.reflectivity_channels_at_energy(
                    energy,
                    q_s=q_s,
                    q_p=q_p,
                    layers=layers_i,
                    tensor=tensor_i,
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

    def logl(self, pvals: NDArray[np.float64] | None = None) -> float:
        """Log-likelihood over every row in the dataset.

        Standard, unnormalized Gaussian log-likelihood when
        `anisotropy_weight == 0.0` and ``normalization="auto"`` (the
        defaults). Otherwise blends in the anisotropy term and normalizes
        as documented for ``anisotropy_weight`` / ``normalization`` on
        `__init__`.
        """
        predicted, aniso = self._predict_all(pvals)
        if self._cached_y is None or self._cached_y_err is None:
            self._refresh_data_transform_cache()
        assert self._cached_y is not None and self._cached_y_err is not None
        model_t = (
            predicted
            if self.transform is None
            else self.transform(self._dataset.q, predicted)[0]
        )
        f_rows, g_pairs = self._likelihood_weights()
        terms = _gaussian_terms(
            self._cached_y, self._cached_y_err, model_t, weighted=self.weighted
        )
        ll = float(f_rows @ terms)
        for g, pair, model_a in zip(
            g_pairs, self._anisotropy_pairs, aniso, strict=False
        ):
            resid = model_a - pair.target
            ll += float(g * -0.5 * np.sum(resid * resid))
        return ll

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
