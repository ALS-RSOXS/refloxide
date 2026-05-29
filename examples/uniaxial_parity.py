"""Visual parity and microbenchmark for the Rust uniaxial kernel.

Mirrors the matplotlib comparison in ``refloxide.pxr.stacks.__main__`` but routes
the reflectivity calculation through the Rust-backed
``refloxide.rust.uniaxial_reflectivity`` rather than the pure-Python
``refloxide.pxr.tjf4x4`` port, and additionally times each stage so the cost of
structure construction can be separated from the cost of the kernel call. Run
with::

    uv run maturin develop --release
    uv run python examples/uniaxial_parity.py

Re-run ``maturin develop`` after any change to the Rust extension; an outdated
``.so`` can surface as missing keyword arguments on ``refloxide.rust`` APIs.
For optimization drivers that already parallelize across workers or threads,
pass ``parallel=False`` into :func:`rust_reflectivity` so each evaluation does
not also fan out across rayon's pool (avoids oversubscription).

The subplots overlay refnx with both Rust modes (sequential and parallel q) on
the same vacuum / polystyrene / silicon stack at 250 eV. The printed table times
``refloxide.rust`` twice: with ``parallel=False`` (single-threaded q-loop) and
``parallel=True`` (rayon over q), alongside refnx and the pure-Python kernel,
using mean and standard deviation in milliseconds across a configurable number of
repetitions.
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from periodictable.xsf import index_of_refraction

from refloxide.pxr.plugin.structure import MaterialSLD
from refloxide.pxr.stacks import Layer, Material, stack_slabs, stack_tensor
from refloxide.pxr.tjf4x4 import uniaxial_reflectivity as python_uniaxial_reflectivity
from refloxide.rust import uniaxial_reflectivity as rust_uniaxial_reflectivity


def rust_reflectivity(
    layers: list[Layer],
    q: np.ndarray,
    energy: float,
    *,
    parallel: bool = True,
):
    """Run the Rust kernel through the existing stack-to-array helpers.

    Parameters
    ----------
    layers
        Stack specification passed to ``stack_slabs`` / ``stack_tensor``.
    q
        Scattering wavevectors; coerced to ``float64`` contiguous semantics via
        ``np.asarray``.
    energy
        Photon energy in eV forwarded to the stack builders and kernel.
    parallel
        When ``True``, q-points are solved on rayon's global pool (default).
        Set ``False`` when an outer fitter or sampler already uses threads or
        processes so each objective evaluation stays single-threaded inside
        Rust.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(R_ss, R_pp)`` as real 1D arrays of length ``len(q)``.
    """
    slabs = np.asarray(stack_slabs(layers, energy=energy), dtype=np.float64)
    tensor = np.asarray(stack_tensor(layers, energy=energy), dtype=np.complex128)
    refl, _tran = rust_uniaxial_reflectivity(
        np.asarray(q, dtype=np.float64),
        slabs,
        tensor,
        float(energy),
        parallel,
    )
    return refl[:, 0, 0], refl[:, 1, 1]


def python_reflectivity(layers: list[Layer], q: np.ndarray, energy: float):
    """Run the pure-Python tjf4x4 kernel through the same helpers.

    Provides a third reference column in the timing table so the Rust speedup
    relative to the existing pure-Python kernel is visible alongside refnx.
    """
    slabs = np.asarray(stack_slabs(layers, energy=energy), dtype=np.float64)
    tensor = np.asarray(stack_tensor(layers, energy=energy), dtype=np.complex128)
    refl, _tran, *_ = python_uniaxial_reflectivity(
        np.asarray(q, dtype=np.float64), slabs, tensor, float(energy)
    )
    return refl[:, 0, 0], refl[:, 1, 1]


def time_callable(
    fn: Callable[[], Any],
    n: int,
    warmup: int = 1,
) -> tuple[float, float, np.ndarray]:
    """Time ``fn`` over ``n`` runs; return ``(mean, std, samples)`` in seconds.

    Disables Python's cyclic garbage collector during measurement to reduce
    timing jitter. Performs ``warmup`` untimed calls first to warm caches and
    JIT-like state in any underlying library.
    """
    for _ in range(warmup):
        fn()
    samples = np.empty(n, dtype=np.float64)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(n):
            t0 = time.perf_counter()
            fn()
            samples[i] = time.perf_counter() - t0
    finally:
        if gc_was_enabled:
            gc.enable()
    return float(samples.mean()), float(samples.std(ddof=0)), samples


def _ms(x: float) -> str:
    return f"{x * 1e3:.3f}"


def print_timing_table(
    rows: list[tuple[str, tuple[float, float], tuple[float, float]]],
    *,
    label_width: int = 32,
) -> None:
    """Pretty-print a two-column timing table in milliseconds.

    Each row is ``(label, (construct_mean, construct_std),
    (operate_mean, operate_std))`` with times in seconds.
    """
    impl = "Implementation"
    header = f"{impl:<{label_width}} {'Construct (ms)':<22} {'Operate (ms)':<22}"
    print(header)
    print("-" * len(header))
    for label, (cm, cs), (om, os) in rows:
        construct = f"{_ms(cm)} +/- {_ms(cs)}"
        operate = f"{_ms(om)} +/- {_ms(os)}"
        print(f"{label:<{label_width}} {construct:<22} {operate:<22}")


def main() -> None:
    energy = 250.0
    n_construct = 50
    n_operate = 30
    q = np.linspace(0.001, 0.25, 1000)

    def ps_sld(e: float):
        return index_of_refraction("C8H8", density=1, energy=e * 1e-3)

    def si_sld(e: float):
        return index_of_refraction("Si", density=2.33, energy=e * 1e-3)

    def build_refloxide() -> list[Layer]:
        vac = Layer(
            thickness=0.0,
            roughness=0.0,
            material=Material("scalar"),
            sld=complex(1, 0),
        )
        ps = Layer(
            thickness=200.0,
            roughness=5.0,
            material=Material("uniaxial"),
            sld=ps_sld,
        )
        si = Layer(
            thickness=0.0,
            roughness=0.5,
            material=Material("scalar"),
            sld=si_sld,
        )
        return [vac, ps, si]

    def build_refnx():
        vac = MaterialSLD("", 0, energy, name="vacuum")(0, 0)
        ps = MaterialSLD("C8H8", 1.0, energy, name="polystyrene")(200, 5.0)
        si = MaterialSLD("Si", 2.33, energy, name="silicon")(0, 0.5)
        return vac | ps | si

    construct_refloxide = time_callable(build_refloxide, n=n_construct)
    construct_refnx = time_callable(build_refnx, n=n_construct)

    refloxide_layers = build_refloxide()
    refnx_stack = build_refnx()

    operate_rust_seq = time_callable(
        lambda: rust_reflectivity(refloxide_layers, q, energy, parallel=False),
        n=n_operate,
    )
    operate_rust_par = time_callable(
        lambda: rust_reflectivity(refloxide_layers, q, energy, parallel=True),
        n=n_operate,
    )
    operate_python = time_callable(
        lambda: python_reflectivity(refloxide_layers, q, energy), n=n_operate
    )
    operate_refnx = time_callable(
        lambda: refnx_stack.reflectivity(q=q, energy=energy), n=n_operate
    )

    print(
        f"Stack: vacuum / polystyrene (200 A) / silicon at {energy:.0f} eV, "
        f"{q.size} q-points."
    )
    print(f"Repetitions: {n_construct} for construct, {n_operate} for operate.")
    print_timing_table(
        [
            (
                "refloxide-rust (sequential)",
                (construct_refloxide[0], construct_refloxide[1]),
                (operate_rust_seq[0], operate_rust_seq[1]),
            ),
            (
                "refloxide-rust (parallel q)",
                (construct_refloxide[0], construct_refloxide[1]),
                (operate_rust_par[0], operate_rust_par[1]),
            ),
            (
                "refloxide-python",
                (construct_refloxide[0], construct_refloxide[1]),
                (operate_python[0], operate_python[1]),
            ),
            (
                "refnx",
                (construct_refnx[0], construct_refnx[1]),
                (operate_refnx[0], operate_refnx[1]),
            ),
        ]
    )

    t_seq = operate_rust_seq[0]
    t_par = operate_rust_par[0]
    t_py = operate_python[0]
    t_rx = operate_refnx[0]
    rust_par_over_seq = t_seq / t_par if t_par > 0 else float("nan")
    rust_par_over_python = t_py / t_par if t_par > 0 else float("nan")
    rust_par_over_refnx = t_rx / t_par if t_par > 0 else float("nan")
    rust_seq_over_refnx = t_rx / t_seq if t_seq > 0 else float("nan")
    msg_par_vs_seq = (
        "Rust parallel q vs sequential q (operate time ratio): "
        f"{rust_par_over_seq:.2f}x"
    )
    print(msg_par_vs_seq)
    msg_par_vs_py = (
        f"Speedup of Rust (parallel q) over pure-Python kernel: "
        f"{rust_par_over_python:.2f}x"
    )
    print(msg_par_vs_py)
    msg_par_vs_rx = (
        f"Speedup of Rust (parallel q) over refnx plugin path: "
        f"{rust_par_over_refnx:.2f}x"
    )
    print(msg_par_vs_rx)
    msg_seq_vs_rx = (
        f"Speedup of Rust (sequential q) over refnx plugin path: "
        f"{rust_seq_over_refnx:.2f}x"
    )
    print(msg_seq_vs_rx)

    refl_ss_rust, refl_pp_rust = rust_reflectivity(
        refloxide_layers, q, energy, parallel=True
    )
    refl_pp_refnx, refl_ss_refnx, *_ = refnx_stack.reflectivity(q=q, energy=energy)

    refl_ss_rust_seq, refl_pp_rust_seq = rust_reflectivity(
        refloxide_layers, q, energy, parallel=False
    )
    abs_err_s = np.abs(refl_ss_rust - refl_ss_refnx)
    abs_err_p = np.abs(refl_pp_rust - refl_pp_refnx)
    abs_mode_s = np.max(np.abs(refl_ss_rust - refl_ss_rust_seq))
    abs_mode_p = np.max(np.abs(refl_pp_rust - refl_pp_rust_seq))
    print(
        "max |R_ss(rust, parallel) - R_ss(refnx)|: ",
        np.nanmax(abs_err_s),
    )
    print(
        "max |R_pp(rust, parallel) - R_pp(refnx)|: ",
        np.nanmax(abs_err_p),
    )
    print(
        "max |R_ss(rust, sequential) - R_ss(refnx)|: ",
        np.nanmax(np.abs(refl_ss_rust_seq - refl_ss_refnx)),
    )
    print(
        "max |R_pp(rust, sequential) - R_pp(refnx)|: ",
        np.nanmax(np.abs(refl_pp_rust_seq - refl_pp_refnx)),
    )
    print(
        "max |R_ss(rust, parallel) - R_ss(rust, sequential)|: ",
        abs_mode_s,
    )
    print(
        "max |R_pp(rust, parallel) - R_pp(rust, sequential)|: ",
        abs_mode_p,
    )

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))
    ax[0].plot(q, refl_ss_rust, label="refloxide-rust (parallel q) s", c="C0")
    ax[0].plot(
        q,
        refl_ss_rust_seq,
        label="refloxide-rust (sequential) s",
        ls=":",
        c="C2",
    )
    ax[0].plot(q, refl_ss_refnx, label="refnx s", ls="--", c="k")
    ax[1].plot(q, refl_pp_rust, label="refloxide-rust (parallel q) p", c="C1")
    ax[1].plot(
        q,
        refl_pp_rust_seq,
        label="refloxide-rust (sequential) p",
        ls=":",
        c="C3",
    )
    ax[1].plot(q, refl_pp_refnx, label="refnx p", ls="--", c="k")
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].set_ylabel(r"$R_{ss}$")
    ax[1].set_ylabel(r"$R_{pp}$")
    ax[1].set_xlabel(r"$q$ (1/Angstrom)")
    ax[0].legend()
    ax[1].legend()
    fig.suptitle(
        "Rust uniaxial kernel (sequential vs parallel q) vs refnx, "
        "vac / PS / Si at 250 eV"
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
