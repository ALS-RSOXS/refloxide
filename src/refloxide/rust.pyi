"""Type stubs for the Rust-backed ``refloxide.rust`` extension module.

The native implementation is produced from ``src/lib.rs`` via PyO3 and
``maturin``. Function shapes and conventions mirror
``refloxide.python.tmm.uniaxial_reflectivity`` so the two implementations
can be exchanged without changing call sites.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def uniaxial_reflectivity(
    q: NDArray[np.float64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energy_ev: float,
    parallel: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Compute uniaxial 4x4 reflection and transmission for a stratified medium.

    Parameters
    ----------
    q
        Scattering wavevectors in inverse angstroms. Shape ``(numpnts,)``.
    layers
        Per-layer rows ``[d, sld_real, sld_imag, sigma]`` of shape
        ``(nlayers, 4)``. The first and last rows describe the fronting and
        the backing; the fronting and backing thicknesses are ignored.
    tensor
        Per-layer 3x3 dispersion tensor of shape ``(nlayers, 3, 3)``. The
        Berreman dielectric is built as ``eps = conj(I - 2 * tensor)``.
    energy_ev
        Photon energy in eV.
    parallel
        When ``True`` (the default), the q-point loop runs on rayon's global
        thread pool. Pass ``False`` from within fitting routines that are
        themselves multi-threaded or multi-process (refnx workers, emcee
        walkers, ``multiprocessing.Pool``) to avoid CPU oversubscription.
        The pool size can also be capped globally with the environment
        variable ``RAYON_NUM_THREADS``.

    Returns
    -------
    refl
        Real power reflectance with shape ``(numpnts, 2, 2)``. Index
        layout matches ``refloxide.python.tmm.uniaxial_reflectivity``:
        ``refl[:, 0, 0] = R_ss``, ``refl[:, 1, 1] = R_pp``,
        ``refl[:, 0, 1] = R_sp``, ``refl[:, 1, 0] = R_ps``.
    tran
        Complex amplitude transmission with the same index layout.

    Raises
    ------
    ValueError
        Layer count mismatch, fewer than two slabs, malformed input shapes,
        or non-finite or non-positive ``energy_ev``.
    RuntimeError
        Dynamic matrix is singular at some (layer, q-index), with the
        offending indices reported in the exception message.
    """
    ...

def uniaxial_reflectivity_batch(
    q: NDArray[np.float64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energies_ev: NDArray[np.float64],
    parallel: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Batched uniaxial reflectivity over shared ``q`` and many energies.

    Parameters
    ----------
    q
        Scattering wavevectors, shape ``(n_q,)``.
    layers
        Per-energy slab rows, shape ``(n_E, N, 4)``.
    tensor
        Per-energy tensors, shape ``(n_E, N, 3, 3)``.
    energies_ev
        Photon energies in eV, shape ``(n_E,)``.
    parallel
        When ``True``, parallelize over flattened ``(energy, q)`` indices.

    Returns
    -------
    refl
        Power reflectance, shape ``(n_E, n_q, 2, 2)``.
    tran
        Complex transmission amplitudes with the same shape.
    """
    ...

def interp_ooc_linear(
    energy_ev: NDArray[np.float64],
    n_xx: NDArray[np.float64],
    n_ixx: NDArray[np.float64],
    n_zz: NDArray[np.float64],
    n_izz: NDArray[np.float64],
    query_ev: float,
) -> tuple[float, float, float, float]:
    """Piecewise-linear OOC lookup at one photon energy (eV).

    Returns ``(n_xx, n_ixx, n_zz, n_izz)``. Out-of-range ``query_ev`` clamps to
    the tabulated endpoints.
    """
    ...

def lab_tensor_diagonals_batch(
    n_mol_xx: complex,
    n_mol_zz: complex,
    orientations_rad: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Batch laboratory ``(3, 3)`` tensors for uniaxial molecular constants.

    Parameters
    ----------
    n_mol_xx, n_mol_zz
        Complex molecular indices along principal axes after density scaling.
    orientations_rad
        Polar rotations in radians, shape ``(n_sub,)``.

    Returns
    -------
    NDArray[np.complex128]
        Shape ``(n_sub, 3, 3)`` diagonal laboratory tensors.
    """
    ...

def isotropic_lab_tensor(n: complex) -> NDArray[np.complex128]:
    """Build a ``(3, 3)`` isotropic tensor with scalar index ``n`` on the diagonal."""
    ...

def molecular_index_at_ooc(
    energy_ev: NDArray[np.float64],
    n_xx: NDArray[np.float64],
    n_ixx: NDArray[np.float64],
    n_zz: NDArray[np.float64],
    n_izz: NDArray[np.float64],
    query_ev: float,
    density: float,
) -> tuple[complex, complex]:
    """Linear OOC lookup and density scaling to molecular ``(n_xx, n_zz)``."""
    ...

def uniaxial_lab_tensor(
    n_mol_xx: complex,
    n_mol_zz: complex,
    orientation_rad: float,
) -> NDArray[np.complex128]:
    """Laboratory ``(3, 3)`` tensor for one uniaxial orientation (radians)."""
    ...

def tensor_to_slab_row(
    thickness: float,
    roughness: float,
    tensor: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Pack refnx ``[d, delta, beta, sigma]`` from a laboratory ``(3, 3)`` tensor."""
    ...

def bookended_uniaxial_reflectivity(
    q: NDArray[np.float64],
    energy_ev: NDArray[np.float64],
    n_xx: NDArray[np.float64],
    n_ixx: NDArray[np.float64],
    n_zz: NDArray[np.float64],
    n_izz: NDArray[np.float64],
    query_ev: float,
    total_thick: float,
    surface_roughness: float,
    tau_si: float,
    tau_vac: float,
    alpha_bulk: float,
    alpha_si: float,
    alpha_vac: float,
    density_bulk: float,
    density_si: float,
    density_vac: float,
    num_slabs: int,
    mesh_constant: float,
    fronting: NDArray[np.float64],
    backing: NDArray[np.float64],
    parallel: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Fused book-ended film + substrate reflectivity (GIL released).

    Builds the adaptive microslab mesh, OOC lookup, laboratory tensors, and
    uniaxial transfer-matrix solve entirely in Rust. ``fronting`` is one row
    ``[d, delta, beta, sigma]``; ``backing`` has shape ``(n_backing, 4)``.
    """
    ...
