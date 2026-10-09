"""Type stubs for the Rust-backed ``refloxide.rust`` extension module.

The native implementation is produced from ``src/lib.rs`` via PyO3 and
``maturin``. Function shapes and conventions mirror
``refloxide.python.tmm.uniaxial_reflectivity`` so the two implementations
can be exchanged without changing call sites.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

def uniaxial_reflectivity(
    q: NDArray[np.float64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energy_ev: float,
    parallel: bool = True,
    device: Literal["cpu", "gpu"] = "cpu",
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
    device
        ``"cpu"`` (default) runs the double-precision 4x4 kernel.
        ``"gpu"`` runs the single-precision uniaxial-z recursion on a
        process-wide GPU context (Metal, Vulkan, or DX12 via wgpu): inputs
        are narrowed to ``float32`` on the device, reflectance agrees with
        the CPU kernel to about ``1e-4`` relative, cross-polarized terms are
        exactly zero, and ``parallel`` is ignored.

    Returns
    -------
    refl
        Real power reflectance with shape ``(numpnts, 2, 2)``. Index
        layout matches ``refloxide.python.tmm.uniaxial_reflectivity`` and
        Fresnel vacuum/substrate checks: ``refl[:, 0, 0] = R_pp``,
        ``refl[:, 1, 1] = R_ss``, ``refl[:, 0, 1] = R_sp``,
        ``refl[:, 1, 0] = R_ps``. ``ReflectModel`` remaps these into
        physically named ``Reflectivity.p`` / ``.s`` channels.
    tran
        Complex amplitude transmission with the same index layout. With
        ``device="gpu"`` transmission is not computed and every entry is
        ``nan + nanj``.

    Raises
    ------
    ValueError
        Layer count mismatch, fewer than two slabs, malformed input shapes,
        or non-finite or non-positive ``energy_ev``.
    RuntimeError
        Dynamic matrix is singular at some (layer, q-index), with the
        offending indices reported in the exception message; or, with
        ``device="gpu"``, no GPU is available or the build lacks GPU support.
    """
    ...

def uniaxial_reflectivity_batch(
    q: NDArray[np.float64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energies_ev: NDArray[np.float64],
    parallel: bool = True,
    device: Literal["cpu", "gpu"] = "cpu",
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
    device
        As in :func:`uniaxial_reflectivity`.

    Returns
    -------
    refl
        Power reflectance, shape ``(n_E, n_q, 2, 2)``.
    tran
        Complex transmission amplitudes with the same shape; ``nan + nanj``
        everywhere with ``device="gpu"``.
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

def uniaxial_reflectivity_points(
    q: NDArray[np.float64],
    stack_index: NDArray[np.int64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energies_ev: NDArray[np.float64],
    parallel: bool = True,
    device: Literal["cpu", "gpu"] = "cpu",
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Uniaxial reflectivity for independent ``(q, stack)`` points.

    Generalizes :func:`uniaxial_reflectivity_batch` to ragged q-grids:
    point ``i`` is evaluated at ``q[i]`` against stack ``stack_index[i]``.
    The usual layout is one stack per photon energy with every energy's
    measured q-grid concatenated, so a multi-energy objective is one call
    (and, on the GPU, one dispatch).

    Parameters
    ----------
    q
        Scattering wavevector per point, shape ``(n_points,)``.
    stack_index
        Non-negative stack index per point, shape ``(n_points,)``; each entry
        must be ``< n_stacks``.
    layers
        Per-stack slab rows, shape ``(n_stacks, N, 4)``.
    tensor
        Per-stack tensors, shape ``(n_stacks, N, 3, 3)``.
    energies_ev
        Photon energy per stack in eV, shape ``(n_stacks,)``.
    parallel
        When ``True``, parallelize over points on the CPU.
    device
        As in :func:`uniaxial_reflectivity`.

    Returns
    -------
    refl
        Power reflectance, shape ``(n_points, 2, 2)``, in the
        :func:`uniaxial_reflectivity` packing.
    tran
        Complex transmission amplitudes with the same shape; ``nan + nanj``
        everywhere with ``device="gpu"``.

    Raises
    ------
    ValueError
        Malformed shapes, negative or out-of-range ``stack_index``, or
        invalid energies.
    RuntimeError
        Singular dynamic matrix (CPU), or GPU unavailable.
    """
    ...

def uniaxial_reflectivity_points_jvp(
    q: NDArray[np.float64],
    stack_index: NDArray[np.int64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energies_ev: NDArray[np.float64],
    dq: NDArray[np.float64],
    dlayers: NDArray[np.float64],
    dtensor: NDArray[np.complex128],
    parallel: bool = True,
    device: Literal["cpu", "gpu"] = "cpu",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reflectance and forward-mode directional derivatives for ``(q, stack)`` points.

    Evaluates the uniaxial-z recursion (equal to the 4x4 kernel in this
    scope) with exact derivatives along ``n_dirs`` tangent directions in
    one call (one dispatch on the GPU). Direction ``k`` moves ``q`` by
    ``dq[k]``, the slab rows by ``dlayers[k]``, and the tensors by
    ``dtensor[k]``; tangents use the same packing as the values.

    Parameters
    ----------
    q, stack_index, layers, tensor, energies_ev
        As in :func:`uniaxial_reflectivity_points`.
    dq
        Tangent of ``q``, shape ``(n_dirs, n_points)``.
    dlayers
        Tangent of ``layers``, shape ``(n_dirs, n_stacks, N, 4)``.
    dtensor
        Tangent of ``tensor``, shape ``(n_dirs, n_stacks, N, 3, 3)``.
    parallel
        When ``True``, parallelize over ``(direction, point)`` on the CPU.
    device
        ``"cpu"`` (``float64``) or ``"gpu"`` (``float32``; derivatives keep
        the reflectance's relative accuracy).

    Returns
    -------
    refl
        ``[R_pp, R_ss]`` per point, shape ``(n_points, 2)``.
    jac
        ``[dR_pp, dR_ss]`` per direction and point, shape
        ``(n_dirs, n_points, 2)``.

    Raises
    ------
    ValueError
        Malformed shapes or invalid energies.
    RuntimeError
        GPU unavailable.
    """
    ...

def bookended_uniaxial_reflectivity(
    q: NDArray[np.float64],
    energy_ev: NDArray[np.float64],
    n_xx: NDArray[np.float64],
    n_ixx: NDArray[np.float64],
    n_zz: NDArray[np.float64],
    n_izz: NDArray[np.float64],
    query_ev: float,
    wavelength_ev: float,
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
    with_transmission: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Fused book-ended film + substrate reflectivity (GIL released).

    Builds the adaptive microslab mesh, OOC lookup, laboratory tensors, and
    uniaxial transfer-matrix solve entirely in Rust. ``fronting`` is one row
    ``[d, delta, beta, sigma]``; ``backing`` has shape ``(n_backing, 4)``.

    ``query_ev`` selects optical constants (may include ``energy_offset``).
    ``wavelength_ev`` sets the TMM wavevector and should be the nominal
    photon energy so fused and assembled paths stay aligned when those
    energies differ.

    With ``with_transmission=False`` the same stack is evaluated by the
    reflectance-only uniaxial-z recursion (equal to the 4x4 chain in this
    scope, more accurate, and several times faster); ``tran`` is then
    ``nan + nanj`` everywhere and cross-polarized reflectance is exactly
    zero. Use it wherever transmission is discarded, such as fitting.
    """
    ...
