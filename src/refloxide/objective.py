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

from typing import TYPE_CHECKING, Any

import numpy as np
from refnx.analysis import Objective as _RefnxObjective
from refnx.dataset import Data1D

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from refloxide.data import ReflectDataset
    from refloxide.model import ReflectModel


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

    Groups the dataset by `(energy, pol)` so each distinct energy triggers
    at most one `ReflectModel` evaluation regardless of how many rows share
    it, and reads each row's predicted value off the `s` or `p` channel its
    `pol` selects — the model itself never chooses a channel.

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

    Raises
    ------
    ValueError
        If `data` is empty.
    """

    def __init__(
        self,
        model: ReflectModel,
        data: ReflectDataset,
        *,
        use_weights: bool = True,
        transform: Callable[..., Any] | None = None,
        name: str | None = None,
    ) -> None:
        if len(data) == 0:
            msg = "Objective requires a non-empty ReflectDataset"
            raise ValueError(msg)
        self._dataset = data
        self._groups = list(data.groups())
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
        """Model reflectivity for every row, in the dataset's original order."""
        self.setp(pvals)
        predicted = np.empty(len(self._dataset), dtype=np.float64)
        for energy, pol, indices in self._groups:
            result = self.model(self._dataset.q[indices], energy)
            predicted[indices] = result.s if pol == "s" else result.p
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

    def logl(self, pvals: NDArray[np.float64] | None = None) -> float:
        """Gaussian log-likelihood summed over every row in the dataset."""
        y, y_err, predicted = self._transformed(pvals)
        return gaussian_logl(y, y_err, predicted, weighted=self.weighted)

    def __repr__(self) -> str:
        return (
            f"Objective({self.model!r}, {len(self._dataset)} points, "
            f"{len(self._groups)} energy/pol groups)"
        )
