"""ReflectModel channel layout for polarized ``(n_q, 2, 2)`` power matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def reflectmodel_layout(matrix: NDArray[Any]) -> NDArray[Any]:
    """Reorder native kernel output for :class:`pyref.fitting.model.ReflectModel`.

    Native refloxide and ``pyref.fitting.uniaxial`` kernels store ``R_ss`` at
    ``[:,0,0]``, ``R_sp`` at ``[:,0,1]``, ``R_ps`` at ``[:,1,0]``, and ``R_pp`` at
    ``[:,1,1]``. ``ReflectModel`` reads ``pol='s'`` from ``[:,1,1]`` and
    ``pol='p'`` from ``[:,0,0]``; this permutation places laboratory ``R_ss`` and
    ``R_pp`` on those diagonals and maps cross terms to ``[:,1,0]`` / ``[:,0,1]``.
    """
    out = np.empty_like(matrix)
    out[:, 0, 0] = matrix[:, 1, 1]
    out[:, 0, 1] = matrix[:, 1, 0]
    out[:, 1, 0] = matrix[:, 0, 1]
    out[:, 1, 1] = matrix[:, 0, 0]
    return out


def apply_reflectmodel_scales(
    refl: NDArray[np.float64],
    scale_s: float,
    scale_p: float,
) -> None:
    """Apply ``scale_s`` and ``scale_p`` to s and p diagonals in ReflectModel layout."""
    refl[:, 1, 1] = scale_s * refl[:, 1, 1]
    refl[:, 0, 0] = scale_p * refl[:, 0, 0]
