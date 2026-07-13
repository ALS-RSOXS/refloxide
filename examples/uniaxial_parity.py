"""Visual parity and microbenchmark for the Rust uniaxial kernel.

Builds a vacuum / polystyrene / silicon stack with
:class:`~refloxide.pxr.plugin.structure.MaterialSLD`, then times
``refloxide.tmm.uniaxial_reflectivity`` (sequential and parallel q), the
pure-Python ``refloxide.python.tmm`` kernel, and the plugin
``Structure.reflectivity`` path.
Run with::

    uv run maturin develop --release
    uv run python examples/uniaxial_parity.py

Re-run ``maturin develop`` after any change to the Rust extension; an outdated
``.so`` can surface as missing keyword arguments on ``refloxide.rust`` APIs.
For optimization drivers that already parallelize across workers or threads,
pass ``parallel=False`` into :func:`rust_reflectivity` so each evaluation does
not also fan out across rayon's pool (avoids oversubscription).
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

from refloxide.pxr.plugin.structure import MaterialSLD, Structure
from refloxide.python.tmm import uniaxial_reflectivity as python_uniaxial_reflectivity
from refloxide.tmm import uniaxial_reflectivity as rust_uniaxial_reflectivity


def _stack_at(energy_ev: float) -> Structure:
    vac = MaterialSLD("", 0, energy_ev, name="vacuum")(0, 0)
    ps = MaterialSLD("C8H8", 1.0, energy_ev, name="polystyrene")(200, 5.0)
    si = MaterialSLD("Si", 2.33, energy_ev, name="silicon")(0, 0.5)
    return vac | ps | si


def _arrays_at(structure: Structure, energy_ev: float) -> tuple[np.ndarray, np.ndarray]:
    slabs = np.asarray(structure.slabs(), dtype=np.float64)
    tensor = np.asarray(structure.tensor(energy=energy_ev), dtype=np.complex128)
    return slabs, tensor


def rust_reflectivity(
    structure: Structure,
    q: np.ndarray,
    energy_ev: float,
    *,
    parallel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the Rust kernel on ``structure`` arrays at ``energy_ev``."""
    slabs, tensor = _arrays_at(structure, energy_ev)
    refl, _tran = rust_uniaxial_reflectivity(
        np.asarray(q, dtype=np.float64),
        slabs,
        tensor,
        float(energy_ev),
        parallel,
    )
    return refl[:, 0, 0], refl[:, 1, 1]


def python_reflectivity(
    structure: Structure,
    q: np.ndarray,
    energy_ev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the pure-Python TMM kernel on the same arrays."""
    slabs, tensor = _arrays_at(structure, energy_ev)
    refl, _tran, *_ = python_uniaxial_reflectivity(
        np.asarray(q, dtype=np.float64), slabs, tensor, float(energy_ev)
    )
    return refl[:, 0, 0], refl[:, 1, 1]


def time_callable(
    fn: Callable[[], Any],
    n: int,
    warmup: int = 1,
) -> tuple[float, float, np.ndarray]:
    """Time ``fn`` over ``n`` runs; return ``(mean, std, samples)`` in seconds."""
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
    """Pretty-print construct/operate timing columns in milliseconds."""
    impl = "Implementation"
    header = f"{impl:<{label_width}} {'Construct (ms)':<22} {'Operate (ms)':<22}"
    print(header)
    print("-" * len(header))
    for label, (cm, cs), (om, os) in rows:
        construct = f"{_ms(cm)} +/- {_ms(cs)}"
        operate = f"{_ms(om)} +/- {_ms(os)}"
        print(f"{label:<{label_width}} {construct:<22} {operate:<22}")


def main() -> None:
    energy_ev = 250.0
    n_construct = 50
    n_operate = 30
    q = np.linspace(0.001, 0.25, 1000)

    construct = time_callable(lambda: _stack_at(energy_ev), n=n_construct)
    stack = _stack_at(energy_ev)

    operate_rust_seq = time_callable(
        lambda: rust_reflectivity(stack, q, energy_ev, parallel=False),
        n=n_operate,
    )
    operate_rust_par = time_callable(
        lambda: rust_reflectivity(stack, q, energy_ev, parallel=True),
        n=n_operate,
    )
    operate_python = time_callable(
        lambda: python_reflectivity(stack, q, energy_ev), n=n_operate
    )
    operate_refnx = time_callable(
        lambda: stack.reflectivity(q=q, energy=energy_ev), n=n_operate
    )

    print(
        f"Stack: vacuum / polystyrene (200 A) / silicon at {energy_ev:.0f} eV, "
        f"{q.size} q-points."
    )
    print(f"Repetitions: {n_construct} for construct, {n_operate} for operate.")
    print_timing_table(
        [
            (
                "refloxide-rust (sequential)",
                (construct[0], construct[1]),
                (operate_rust_seq[0], operate_rust_seq[1]),
            ),
            (
                "refloxide-rust (parallel q)",
                (construct[0], construct[1]),
                (operate_rust_par[0], operate_rust_par[1]),
            ),
            (
                "refloxide-python",
                (construct[0], construct[1]),
                (operate_python[0], operate_python[1]),
            ),
            (
                "plugin Structure.reflectivity",
                (construct[0], construct[1]),
                (operate_refnx[0], operate_refnx[1]),
            ),
        ]
    )

    t_seq = operate_rust_seq[0]
    t_par = operate_rust_par[0]
    t_py = operate_python[0]
    t_rx = operate_refnx[0]
    print(
        "Rust parallel q vs sequential q (operate time ratio): "
        f"{(t_seq / t_par) if t_par > 0 else float('nan'):.2f}x"
    )
    print(
        "Speedup of Rust (parallel q) over pure-Python kernel: "
        f"{(t_py / t_par) if t_par > 0 else float('nan'):.2f}x"
    )
    print(
        "Speedup of Rust (parallel q) over plugin path: "
        f"{(t_rx / t_par) if t_par > 0 else float('nan'):.2f}x"
    )
    print(
        "Speedup of Rust (sequential q) over plugin path: "
        f"{(t_rx / t_seq) if t_seq > 0 else float('nan'):.2f}x"
    )

    refl_ss_rust, refl_pp_rust = rust_reflectivity(stack, q, energy_ev, parallel=True)
    refl_pp_refnx, refl_ss_refnx, *_ = stack.reflectivity(q=q, energy=energy_ev)
    refl_ss_rust_seq, refl_pp_rust_seq = rust_reflectivity(
        stack, q, energy_ev, parallel=False
    )
    print(
        "max |R_ss(rust, parallel) - R_ss(plugin)|: ",
        np.nanmax(np.abs(refl_ss_rust - refl_ss_refnx)),
    )
    print(
        "max |R_pp(rust, parallel) - R_pp(plugin)|: ",
        np.nanmax(np.abs(refl_pp_rust - refl_pp_refnx)),
    )
    print(
        "max |R_ss(rust, sequential) - R_ss(plugin)|: ",
        np.nanmax(np.abs(refl_ss_rust_seq - refl_ss_refnx)),
    )
    print(
        "max |R_pp(rust, sequential) - R_pp(plugin)|: ",
        np.nanmax(np.abs(refl_pp_rust_seq - refl_pp_refnx)),
    )
    print(
        "max |R_ss(rust, parallel) - R_ss(rust, sequential)|: ",
        np.max(np.abs(refl_ss_rust - refl_ss_rust_seq)),
    )
    print(
        "max |R_pp(rust, parallel) - R_pp(rust, sequential)|: ",
        np.max(np.abs(refl_pp_rust - refl_pp_rust_seq)),
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
    ax[0].plot(q, refl_ss_refnx, label="plugin s", ls="--", c="k")
    ax[1].plot(q, refl_pp_rust, label="refloxide-rust (parallel q) p", c="C1")
    ax[1].plot(
        q,
        refl_pp_rust_seq,
        label="refloxide-rust (sequential) p",
        ls=":",
        c="C3",
    )
    ax[1].plot(q, refl_pp_refnx, label="plugin p", ls="--", c="k")
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].set_ylabel(r"$R_{ss}$")
    ax[1].set_ylabel(r"$R_{pp}$")
    ax[1].set_xlabel(r"$q$ (1/Angstrom)")
    ax[0].legend()
    ax[1].legend()
    fig.suptitle(
        "Rust uniaxial kernel (sequential vs parallel q) vs plugin path, "
        "vac / PS / Si at 250 eV"
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
