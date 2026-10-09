"""Cross-backend wall-time and peak-heap benchmark for uniaxial reflectivity.

Always times ``refloxide`` CPU (``parallel=False`` / ``True``) and GPU when an
adapter is present. Optional comparison backends (skipped when not installed):

* ``refnx`` Abeles on an isotropic twin (dev/plugin extra only; not a package
  runtime dependency)
* ``pypxr`` when importable
* in-tree ``refloxide.pxr.plugin`` when that path imports (needs the plugin/
  dev extra for ``refnx``)

Metrics per backend and problem size: median wall time after warmup, peak
Python heap via ``tracemalloc``, and throughput in q-points per second.

Run from a synced dev environment::

    uv sync --group dev
    uv run python examples/gpu_backend_bench.py

Writes ``examples/results/gpu_backend_bench_results.md`` (gitignored under
``examples/results/``).
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

HC_EV_ANGSTROM = 12398.4193
ENERGY_EV = 250.0
RESULTS_PATH = (
    Path(__file__).resolve().parent / "results" / "gpu_backend_bench_results.md"
)


@dataclass(frozen=True)
class BenchCase:
    """One (n_q, n_film_layers) problem size."""

    n_q: int
    n_film: int


@dataclass(frozen=True)
class BenchRow:
    """One measured backend/size result."""

    backend: str
    n_q: int
    n_layers: int
    median_s: float
    peak_kib: float
    points_per_s: float
    notes: str = ""


def _delta_beta_to_refnx_sld(delta: float, beta: float, energy_ev: float) -> complex:
    wavelength = HC_EV_ANGSTROM / energy_ev
    factor = 2.0 * np.pi / wavelength**2 * 1.0e6
    return complex(delta * factor, beta * factor)


def _graded_arrays(n_film: int) -> tuple[np.ndarray, np.ndarray]:
    """Build vacuum / graded uniaxial film / substrate arrays without refnx."""
    dz = 200.0 / max(n_film, 1)
    rows: list[list[float]] = [[0.0, 0.0, 0.0, 0.0]]
    tensors: list[np.ndarray] = [np.diag([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])]
    for i in range(n_film):
        frac = i / max(n_film - 1, 1)
        delta_o = 1.8e-3 * (1.0 + 0.15 * frac)
        beta_o = 1.0e-4 * (1.0 + 0.2 * frac)
        delta_e = 2.2e-3 * (1.0 - 0.1 * frac)
        beta_e = 1.2e-4 * (1.0 + 0.1 * frac)
        rows.append([dz, delta_o, beta_o, 2.0 if i == 0 else 0.0])
        n_o = complex(delta_o, beta_o)
        n_e = complex(delta_e, beta_e)
        tensors.append(np.diag([n_o, n_o, n_e]))
    rows.append([0.0, 5.97e-3, 4.25e-3, 0.5])
    n_b = complex(5.97e-3, 4.25e-3)
    tensors.append(np.diag([n_b, n_b, n_b]))
    layers = np.asarray(rows, dtype=np.float64)
    tensor = np.asarray(tensors, dtype=np.complex128)
    return layers, tensor


def _try_refnx_model(layers: np.ndarray, energy_ev: float):
    try:
        from refnx.reflect import SLD, ReflectModel, Structure
    except ImportError:
        return None

    structure = Structure()
    for i, row in enumerate(layers):
        thick = float(row[0])
        delta = float(row[1])
        beta = float(row[2])
        sigma = float(row[3])
        sld = SLD(_delta_beta_to_refnx_sld(delta, beta, energy_ev), name=f"L{i}")
        slab_thick = thick if 0 < i < len(layers) - 1 else 0.0
        structure |= sld(slab_thick, sigma)
    return ReflectModel(structure, dq=0.0)


def _try_pypxr_callable(
    layers: np.ndarray, tensor: np.ndarray, energy_ev: float
) -> Callable[[np.ndarray], np.ndarray] | None:
    try:
        import pypxr  # type: ignore[import-not-found]
    except ImportError:
        return None
    solve = getattr(pypxr, "uniaxial_reflectivity", None) or getattr(
        pypxr, "reflectivity", None
    )
    if solve is None:
        return None

    def _call(q: np.ndarray) -> np.ndarray:
        out = solve(q, layers, tensor, energy_ev)
        if isinstance(out, tuple):
            return np.asarray(out[0])
        return np.asarray(out)

    return _call


def _try_pxr_plugin_callable(
    layers: np.ndarray, tensor: np.ndarray, energy_ev: float
) -> Callable[[np.ndarray], np.ndarray] | None:
    try:
        from refloxide.python.tmm import uniaxial_reflectivity as python_uniaxial
    except ImportError:
        return None

    def _call(q: np.ndarray) -> np.ndarray:
        refl, _tran, *_ = python_uniaxial(q, layers, tensor, energy_ev)
        return np.asarray(refl)

    return _call


def _measure(
    fn: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    peaks: list[float] = []
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        samples.append(elapsed)
        peaks.append(peak / 1024.0)
    return statistics.median(samples), statistics.median(peaks)


def _gpu_probe(layers: np.ndarray, tensor: np.ndarray) -> tuple[bool, str]:
    from refloxide.tmm import uniaxial_reflectivity

    q = np.array([0.05], dtype=np.float64)
    try:
        uniaxial_reflectivity(
            q, layers, tensor, ENERGY_EV, parallel=False, device="gpu"
        )
    except RuntimeError as exc:
        return False, str(exc)
    return True, "wgpu adapter present"


def _skipped_row(
    backend: str, case: BenchCase, n_layers: int, notes: str
) -> BenchRow:
    return BenchRow(
        backend=backend,
        n_q=case.n_q,
        n_layers=n_layers,
        median_s=float("nan"),
        peak_kib=float("nan"),
        points_per_s=float("nan"),
        notes=notes,
    )


def run_case(
    case: BenchCase,
    *,
    repeats: int,
    warmup: int,
    gpu_ok: bool,
    gpu_note: str,
) -> list[BenchRow]:
    layers, tensor = _graded_arrays(case.n_film)
    q = np.linspace(0.001, 0.25, case.n_q, dtype=np.float64)
    n_layers = int(layers.shape[0])
    rows: list[BenchRow] = []

    def add(
        backend: str,
        fn: Callable[[], object],
        *,
        notes: str = "",
    ) -> None:
        median_s, peak_kib = _measure(fn, repeats=repeats, warmup=warmup)
        rows.append(
            BenchRow(
                backend=backend,
                n_q=case.n_q,
                n_layers=n_layers,
                median_s=median_s,
                peak_kib=peak_kib,
                points_per_s=(case.n_q / median_s) if median_s > 0 else float("nan"),
                notes=notes,
            )
        )

    refnx_model = _try_refnx_model(layers, ENERGY_EV)
    if refnx_model is None:
        rows.append(
            _skipped_row(
                "refnx Abeles (isotropic)",
                case,
                n_layers,
                "skipped: refnx not installed (dev/plugin extra)",
            )
        )
    else:
        add("refnx Abeles (isotropic)", lambda: refnx_model(q))

    pypxr_fn = _try_pypxr_callable(layers, tensor, ENERGY_EV)
    if pypxr_fn is not None:
        add("pypxr", lambda: pypxr_fn(q))
    else:
        pxr_fn = _try_pxr_plugin_callable(layers, tensor, ENERGY_EV)
        if pxr_fn is None:
            rows.append(
                _skipped_row(
                    "python.tmm / pxr.plugin",
                    case,
                    n_layers,
                    "skipped: pypxr and pure-Python TMM unavailable",
                )
            )
        else:
            add(
                "refloxide.python.tmm",
                lambda: pxr_fn(q),
                notes="pypxr not installed; pure-Python polarized TMM",
            )

    from refloxide.tmm import uniaxial_reflectivity

    add(
        "refloxide CPU parallel=False",
        lambda: uniaxial_reflectivity(
            q, layers, tensor, ENERGY_EV, parallel=False, device="cpu"
        ),
    )
    add(
        "refloxide CPU parallel=True",
        lambda: uniaxial_reflectivity(
            q, layers, tensor, ENERGY_EV, parallel=True, device="cpu"
        ),
    )

    if not gpu_ok:
        rows.append(
            _skipped_row("refloxide GPU", case, n_layers, f"skipped: {gpu_note}")
        )
    else:
        add(
            "refloxide GPU",
            lambda: uniaxial_reflectivity(
                q, layers, tensor, ENERGY_EV, parallel=False, device="gpu"
            ),
            notes=gpu_note,
        )

    return rows


def _format_table(rows: list[BenchRow]) -> str:
    header = (
        "| backend | n_q | n_layers | median_ms | peak_KiB | points/s | notes |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    body = []
    for r in rows:
        median_ms = "n/a" if not np.isfinite(r.median_s) else f"{r.median_s * 1e3:.3f}"
        peak = "n/a" if not np.isfinite(r.peak_kib) else f"{r.peak_kib:.1f}"
        thr = "n/a" if not np.isfinite(r.points_per_s) else f"{r.points_per_s:,.0f}"
        body.append(
            "| {backend} | {n_q} | {n_layers} | {median_ms} | {peak} | {thr} |"
            " {notes} |".format(
                backend=r.backend,
                n_q=r.n_q,
                n_layers=r.n_layers,
                median_ms=median_ms,
                peak=peak,
                thr=thr,
                notes=r.notes,
            )
        )
    return header + "\n".join(body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=["256:8", "1024:8", "1024:32", "4096:32"],
        help="Problem sizes as n_q:n_film (film microslabs before substrate)",
    )
    args = parser.parse_args()
    cases = [
        BenchCase(n_q=int(a), n_film=int(b))
        for size in args.sizes
        for a, b in [size.split(":", 1)]
    ]

    probe_layers, probe_tensor = _graded_arrays(4)
    gpu_ok, gpu_note = _gpu_probe(probe_layers, probe_tensor)

    all_rows: list[BenchRow] = []
    for case in cases:
        all_rows.extend(
            run_case(
                case,
                repeats=args.repeats,
                warmup=args.warmup,
                gpu_ok=gpu_ok,
                gpu_note=gpu_note,
            )
        )

    table = _format_table(all_rows)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    cpu = platform.processor() or "cpu"
    machine = f"{platform.system()} {platform.machine()} / {cpu}"
    report = (
        f"# GPU backend bench results\n\n"
        f"- generated: `{stamp}`\n"
        f"- machine: `{machine}`\n"
        f"- energy_ev: `{ENERGY_EV}`\n"
        f"- repeats/warmup: `{args.repeats}` / `{args.warmup}`\n"
        f"- gpu: `{gpu_note}`\n\n"
        f"{table}"
    )
    print(report)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
