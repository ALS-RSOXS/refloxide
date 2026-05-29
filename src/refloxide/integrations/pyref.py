"""Drop-in uniaxial reflectivity hooks for ``pyref.fitting`` workflows.

Maps refloxide's laboratory s/p power matrix (``R_ss`` at ``[:,0,0]``,
``R_pp`` at ``[:,1,1]`` after the Berreman cross-map in
:mod:`refloxide.pxr.tjf4x4`) onto the legacy ``pyref.fitting.uniaxial``
kernel layout expected by :class:`pyref.fitting.model.ReflectModel`.

Scope is **uniaxial only** (``phi = 0``, ``backend = "uni"``). Do not use
this adapter for biaxial or general-incidence claims.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.interpolate import splev, splrep

if TYPE_CHECKING:
    from numpy.typing import NDArray

from refloxide.pxr.tjf4x4 import uniaxial_reflectivity as _python_uniaxial

_FWHM = 2 * np.sqrt(2 * np.log(2.0))


def _swap_sp_block(matrix: NDArray[Any]) -> NDArray[Any]:
    """Reorder a ``(n_q, 2, 2)`` block from refloxide to pyref kernel layout."""
    out = np.empty_like(matrix)
    out[:, 0, 0] = matrix[:, 1, 1]
    out[:, 0, 1] = matrix[:, 1, 0]
    out[:, 1, 0] = matrix[:, 0, 1]
    out[:, 1, 1] = matrix[:, 0, 0]
    return out


def _solve_refloxide(
    q: NDArray[np.float64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energy: float,
    *,
    use_rust: bool,
    parallel: bool,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    if use_rust:
        from refloxide.rust import uniaxial_reflectivity as rust_uniaxial

        refl, tran = rust_uniaxial(
            q,
            layers,
            tensor,
            float(energy),
            parallel=parallel,
        )
        refl_arr = np.asarray(refl, dtype=np.float64)
        tran_arr = np.asarray(tran, dtype=np.complex128)
    else:
        refl_arr, tran_arr, *_ = _python_uniaxial(
            q,
            layers,
            tensor,
            float(energy),
            phi=0.0,
        )
        refl_arr = np.asarray(refl_arr, dtype=np.float64)
        tran_arr = np.asarray(tran_arr, dtype=np.complex128)

    return _swap_sp_block(refl_arr), _swap_sp_block(tran_arr)


def uniaxial_reflectivity(
    q: NDArray[np.float64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energy: float,
    *,
    use_rust: bool = True,
    parallel: bool = False,
    return_components: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.complex128], *tuple[Any, ...]]:
    """Evaluate uniaxial reflectivity in the ``pyref.fitting.uniaxial`` contract.

    Accepts the same positional arguments as
    ``pyref.fitting.uniaxial.uniaxial_reflectivity`` (``q``, ``layers``,
    ``tensor``, ``energy`` in eV). Internally routes through refloxide's
    uniaxial TJF 4x4 kernel and reorders the ``(n_q, 2, 2)`` power matrix so
    ``ReflectModel`` polarization indexing matches historical pyref fits.

    Parameters
    ----------
    q
        Scattering wavevectors in inverse angstroms, shape ``(n_q,)``.
    layers
        Slab table ``(n_layers, 4)`` with rows ``[thickness, sld_real,
        sld_imag, roughness]`` in refnx/pyref units (SLD scaled by
        ``1e-6`` Angstrom**-2).
    tensor
        Per-layer dispersion tensor ``(n_layers, 3, 3)`` with diagonals
        carrying :math:`\\delta + i\\beta` per principal axis before the
        ``epsilon = conj(I - 2 * tensor)`` conversion inside the kernel.
    energy
        Photon energy in eV.
    use_rust
        When ``True`` (default), call :func:`refloxide.rust.uniaxial_reflectivity`.
        When ``False``, use the pure-Python port in
        :mod:`refloxide.pxr.tjf4x4`.
    parallel
        Passed to the Rust kernel as ``parallel=``. Default ``False`` so
        refnx/emcee worker pools are not oversubscribed. Ignored when
        ``use_rust=False``.
    return_components
        When ``True`` and ``use_rust=False``, append the diagnostic arrays
        returned by the Python port (``kx``, ``ky``, ``kz``, polarization
        bases, transfer matrices). When ``True`` and ``use_rust=True``, the
        trailing tuple is empty because the extension returns only ``refl``
        and ``tran``.

    Returns
    -------
    refl
        Power reflectance ``(n_q, 2, 2)`` in pyref kernel layout
        (``[:,0,0]`` / ``[:,1,1]`` follow ``pyref.fitting.uniaxial``).
    tran
        Complex transmission amplitudes with the same index layout as
        ``refl``.
    components
        Optional trailing diagnostics when ``return_components=True``.

    Raises
    ------
    ValueError
        Propagated from the Rust kernel on malformed inputs.
    RuntimeError
        Propagated from the Rust kernel when the dynamic matrix is singular.

    Notes
    -----
    Laboratory-frame ``R_ss`` / ``R_pp`` from refloxide occupy the opposite
    diagonal corners relative to pyref's legacy kernel. This adapter applies
    the index swap so existing ``pyref.fitting.model.ReflectModel`` instances
    continue to select s/p channels via ``pol='s'`` / ``pol='p'`` without
    code changes.

    Examples
    --------
    Monkeypatch pyref before constructing models::

        import pyref.fitting.uniaxial as uni_mod
        from refloxide.integrations.pyref import patch_pyref

        patch_pyref(use_rust=True, parallel=False)
        assert uni_mod.uniaxial_reflectivity is not None
    """
    q_arr = np.asarray(q, dtype=np.float64)
    layers_arr = np.asarray(layers, dtype=np.float64)
    tensor_arr = np.asarray(tensor, dtype=np.complex128)
    refl, tran = _solve_refloxide(
        q_arr,
        layers_arr,
        tensor_arr,
        float(energy),
        use_rust=use_rust,
        parallel=parallel,
    )
    if return_components and not use_rust:
        *_, components = _python_uniaxial(
            q_arr,
            layers_arr,
            tensor_arr,
            float(energy),
            phi=0.0,
        )
        return refl, tran, *components
    return refl, tran


def reflectivity(
    q: np.ndarray,
    slabs: np.ndarray,
    tensor: np.ndarray,
    energy: float = 250.0,
    phi: float = 0.0,
    scale_s: float = 1.0,
    scale_p: float = 1.0,
    bkg: float = 0.0,
    dq: float = 0.0,
    backend: Literal["uni"] = "uni",
    *,
    use_rust: bool = True,
    parallel: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[Any]] | None:
    """Mirror ``pyref.fitting.model.reflectivity`` for uniaxial refloxide kernels.

    Applies scale, background, and optional constant ``dQ/Q`` smearing using
    the same recipe as pyref, but evaluates the stratified stack with
    :func:`uniaxial_reflectivity`.

    Parameters
    ----------
    q
        Scattering wavevectors in inverse angstroms.
    slabs
        Slab table with shape ``(2 + N, 4)``; see
        ``pyref.fitting.model.reflectivity`` for column semantics.
    tensor
        Per-layer ``(2 + N, 3, 3)`` dispersion tensor.
    energy
        Photon energy in eV.
    phi
        Accepted for API compatibility with pyref; ignored (uniaxial path
        fixes ``phi = 0``).
    scale_s, scale_p
        Independent scale factors applied to ``refl[:,0,0]`` and
        ``refl[:,1,1]`` before adding ``bkg``.
    bkg
        Constant background added to every reflectivity element.
    dq
        Constant ``dQ/Q`` resolution smearing in percent. ``0`` disables
        smearing.
    backend
        Must be ``"uni"``; other backends raise ``ValueError``.
    use_rust
        Forwarded to :func:`uniaxial_reflectivity`.
    parallel
        Forwarded to :func:`uniaxial_reflectivity`.

    Returns
    -------
    refl
        Smear-adjusted, scaled reflectivity matrix plus ``bkg``.
    tran
        Transmission amplitudes from the final kernel call inside the smear
        branch (unchanged by scaling).
    components
        Empty list when smearing is disabled; otherwise the trailing items
        from the last internal kernel call.

    Raises
    ------
    ValueError
        When ``backend`` is not ``"uni"``.
    """
    del phi
    if backend != "uni":
        msg = (
            "refloxide.integrations.pyref.reflectivity supports backend='uni' "
            f"only; got {backend!r}"
        )
        raise ValueError(msg)

    if not isinstance(dq, numbers.Real):
        return None

    if float(dq) == 0:
        refl, tran, *components = uniaxial_reflectivity(
            q,
            slabs,
            tensor,
            energy,
            use_rust=use_rust,
            parallel=parallel,
        )
        refl = np.asarray(refl, dtype=np.float64)
        refl[:, 0, 0] = scale_s * refl[:, 0, 0]
        refl[:, 1, 1] = scale_p * refl[:, 1, 1]
        return refl + bkg, tran, list(components)

    smear_refl, smear_tran, *components = _smeared_reflectivity(
        q,
        slabs,
        tensor,
        energy,
        float(dq),
        use_rust=use_rust,
        parallel=parallel,
    )
    smear_refl[:, 0, 0] = scale_s * smear_refl[:, 0, 0]
    smear_refl[:, 1, 1] = scale_p * smear_refl[:, 1, 1]
    return smear_refl + bkg, smear_tran, list(components)


def _smeared_reflectivity(
    q: np.ndarray,
    slabs: np.ndarray,
    tensor: np.ndarray,
    energy: float,
    resolution: float,
    *,
    use_rust: bool,
    parallel: bool,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    if resolution < 0.5:
        refl, tran, *components = uniaxial_reflectivity(
            q,
            slabs,
            tensor,
            energy,
            use_rust=use_rust,
            parallel=parallel,
        )
        return np.asarray(refl), tran, list(components)

    resolution /= 100
    gaussnum = 51
    gaussgpoint = (gaussnum - 1) / 2

    def gauss(x, s):
        return 1.0 / s / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2 / s / s)

    lowq = float(np.min(q))
    highq = float(np.max(q))
    if lowq <= 0:
        lowq = 1e-6

    start = np.log10(lowq) - 6 * resolution / _FWHM
    finish = np.log10(highq * (1 + 6 * resolution / _FWHM))
    interpnum = int(
        np.round(
            np.abs(1 * (np.abs(start - finish)))
            / (1.7 * resolution / _FWHM / gaussgpoint)
        )
    )
    xtemp = np.linspace(start, finish, interpnum)
    xlin = np.power(10.0, xtemp)

    gauss_x = np.linspace(-1.7 * resolution, 1.7 * resolution, gaussnum)
    gauss_y = gauss(gauss_x, resolution / _FWHM)
    refl, tran, *components = uniaxial_reflectivity(
        xlin,
        slabs,
        tensor,
        energy,
        use_rust=use_rust,
        parallel=parallel,
    )
    smeared_ss = np.convolve(refl[:, 0, 0], gauss_y, mode="same") * (
        gauss_x[1] - gauss_x[0]
    )
    smeared_pp = np.convolve(refl[:, 1, 1], gauss_y, mode="same") * (
        gauss_x[1] - gauss_x[0]
    )
    smeared_sp = np.convolve(refl[:, 0, 1], gauss_y, mode="same") * (
        gauss_x[1] - gauss_x[0]
    )
    smeared_ps = np.convolve(refl[:, 1, 0], gauss_y, mode="same") * (
        gauss_x[1] - gauss_x[0]
    )

    tck_ss = splrep(xlin, smeared_ss)
    smeared_output_ss = splev(q, tck_ss)

    tck_sp = splrep(xlin, smeared_sp)
    smeared_output_sp = splev(q, tck_sp)

    tck_ps = splrep(xlin, smeared_ps)
    smeared_output_ps = splev(q, tck_ps)

    tck_pp = splrep(xlin, smeared_pp)
    smeared_output_pp = splev(q, tck_pp)

    smeared_output = np.rollaxis(
        np.array(
            [
                [smeared_output_ss, smeared_output_sp],
                [smeared_output_ps, smeared_output_pp],
            ]
        ),
        2,
        0,
    )
    return smeared_output, tran, list(components)


def patch_pyref(
    *,
    use_rust: bool = True,
    parallel: bool = False,
    patch_reflectivity: bool = True,
) -> None:
    """Install refloxide uniaxial kernels into an imported ``pyref.fitting`` stack.

    Replaces ``pyref.fitting.uniaxial.uniaxial_reflectivity`` with a partial
    application of :func:`uniaxial_reflectivity`. Optionally replaces
    ``pyref.fitting.model.reflectivity`` so smearing and scaling stay on the
    refloxide code path.

    Parameters
    ----------
    use_rust
        When ``True`` (default), patched calls use
        :func:`refloxide.rust.uniaxial_reflectivity`.
    parallel
        Forwarded as ``parallel=`` to the Rust kernel. Default ``False`` for
        fitting loops that already parallelize across walkers or datasets.
    patch_reflectivity
        When ``True`` (default), also patch
        ``pyref.fitting.model.reflectivity``.

    Raises
    ------
    ImportError
        When ``pyref.fitting`` is not installed in the active environment.

    Notes
    -----
    Call this once at process startup before constructing
    :class:`pyref.fitting.model.ReflectModel` instances. The patch is
    process-global and affects every subsequent pyref reflectivity evaluation.
    """
    import importlib
    from functools import partial

    try:
        uni_mod = importlib.import_module("pyref.fitting.uniaxial")
        model_mod = importlib.import_module("pyref.fitting.model")
    except ModuleNotFoundError as exc:
        msg = (
            "pyref.fitting is not importable; install pyref or add it to "
            "PYTHONPATH before calling patch_pyref"
        )
        raise ImportError(msg) from exc

    patched_uni = partial(
        uniaxial_reflectivity,
        use_rust=use_rust,
        parallel=parallel,
    )
    uni_mod.uniaxial_reflectivity = patched_uni  # ty: ignore[unresolved-attribute]
    if patch_reflectivity:
        patched_refl = partial(
            reflectivity,
            use_rust=use_rust,
            parallel=parallel,
        )
        model_mod.reflectivity = patched_refl  # ty: ignore[unresolved-attribute]
