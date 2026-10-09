"""Cross-backend wall-time and memory benchmark for uniaxial reflectivity.

Headline metric: **10_000 film microslabs** at a fixed q-grid (default 256
points). Reports median wall time, peak Python heap (``tracemalloc``), and
process RSS delta (``resource``) per backend.

Always times ``refloxide`` CPU (``parallel=False`` / ``True``) and GPU when an
adapter is present. Optional comparison backends (skipped when not installed):

* ``refnx`` Abeles on an isotropic twin — scalar 2x2 Abeles, not polarized
  4x4; included as a cheap isotropic baseline, not an apples-to-apples peer
* ``pypxr`` when importable
* ``refloxide.python.tmm`` (pure-Python polarized) — off by default at large
  ``n_film`` (enable with ``--include-python-tmm``)

``refnx`` is a **dev/plugin** extra only; it is not a runtime dependency.

Run::

    uv sync --group dev
    uv run python examples/gpu_backend_bench.py
    uv run python examples/gpu_backend_bench.py --plot

Writes under ``examples/results/`` (gitignored). With ``--plot``, also writes
committed README assets under ``docs/assets/performance/``.
"""

from __future__ import annotations

import argparse
import csv
import platform
import resource
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
HEADLINE_N_FILM = 10_000
DEFAULT_N_Q = 256
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ASSETS_DIR = (
    Path(__file__).resolve().parents[1] / "docs" / "assets" / "performance"
)
PYTHON_TMM_MAX_FILM_DEFAULT = 256

BACKEND_ORDER = (
    "refnx Abeles (isotropic)",
    "pypxr",
    "refloxide.python.tmm",
    "refloxide CPU parallel=False",
    "refloxide CPU parallel=True",
    "refloxide GPU",
)

BACKEND_COLORS = {
    "refnx Abeles (isotropic)": "#6b7280",
    "pypxr": "#9ca3af",
    "refloxide.python.tmm": "#a16207",
    "refloxide CPU parallel=False": "#1d4ed8",
    "refloxide CPU parallel=True": "#2563eb",
    "refloxide GPU": "#0f766e",
}


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
    n_film: int
    median_s: float
    peak_heap_kib: float
    rss_delta_mib: float
    points_per_s: float
    notes: str = ""


def _graded_arrays(n_film: int) -> tuple[np.ndarray, np.ndarray]:
    """Build vacuum / graded uniaxial film / substrate arrays without refnx."""
    dz = 200.0 / max(n_film, 1)
    n_total = n_film + 2
    layers = np.zeros((n_total, 4), dtype=np.float64)
    tensor = np.zeros((n_total, 3, 3), dtype=np.complex128)
    for i in range(n_film):
        frac = i / max(n_film - 1, 1)
        delta_o = 1.8e-3 * (1.0 + 0.15 * frac)
        beta_o = 1.0e-4 * (1.0 + 0.2 * frac)
        delta_e = 2.2e-3 * (1.0 - 0.1 * frac)
        beta_e = 1.2e-4 * (1.0 + 0.1 * frac)
        row = i + 1
        layers[row, 0] = dz
        layers[row, 1] = delta_o
        layers[row, 2] = beta_o
        layers[row, 3] = 2.0 if i == 0 else 0.0
        n_o = complex(delta_o, beta_o)
        n_e = complex(delta_e, beta_e)
        tensor[row, 0, 0] = n_o
        tensor[row, 1, 1] = n_o
        tensor[row, 2, 2] = n_e
    layers[-1, 1] = 5.97e-3
    layers[-1, 2] = 4.25e-3
    layers[-1, 3] = 0.5
    n_b = complex(5.97e-3, 4.25e-3)
    tensor[-1, 0, 0] = n_b
    tensor[-1, 1, 1] = n_b
    tensor[-1, 2, 2] = n_b
    return layers, tensor


def _try_refnx_callable(
    layers: np.ndarray, energy_ev: float
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Abeles on an isotropic twin via refnx's array API (no Structure build)."""
    try:
        from refnx.reflect import reflectivity
    except ImportError:
        return None

    wavelength = HC_EV_ANGSTROM / energy_ev
    factor = 2.0 * np.pi / wavelength**2 * 1.0e6
    slabs = np.zeros((layers.shape[0], 4), dtype=np.float64)
    slabs[:, 0] = layers[:, 0]
    slabs[0, 0] = 0.0
    slabs[-1, 0] = 0.0
    slabs[:, 1] = layers[:, 1] * factor
    slabs[:, 2] = layers[:, 2] * factor
    slabs[:, 3] = layers[:, 3]

    def _call(q: np.ndarray) -> np.ndarray:
        return reflectivity(q, slabs, scale=1.0, bkg=0.0, dq=0.0, threads=1)

    return _call


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


def _try_python_tmm_callable(
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


def _rss_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: KiB; macOS: bytes.
    if platform.system() == "Darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def _measure(
    fn: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    heaps: list[float] = []
    rss_deltas: list[float] = []
    for _ in range(repeats):
        rss_before = _rss_mib()
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = _rss_mib()
        samples.append(elapsed)
        heaps.append(peak / 1024.0)
        rss_deltas.append(max(0.0, rss_after - rss_before))
    return (
        statistics.median(samples),
        statistics.median(heaps),
        statistics.median(rss_deltas),
    )


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
        n_film=case.n_film,
        median_s=float("nan"),
        peak_heap_kib=float("nan"),
        rss_delta_mib=float("nan"),
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
    include_python_tmm: bool,
) -> list[BenchRow]:
    print(f"building stack n_film={case.n_film} n_q={case.n_q} ...", flush=True)
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
        print(f"  timing {backend} ...", flush=True)
        median_s, peak_heap_kib, rss_delta_mib = _measure(
            fn, repeats=repeats, warmup=warmup
        )
        rows.append(
            BenchRow(
                backend=backend,
                n_q=case.n_q,
                n_layers=n_layers,
                n_film=case.n_film,
                median_s=median_s,
                peak_heap_kib=peak_heap_kib,
                rss_delta_mib=rss_delta_mib,
                points_per_s=(case.n_q / median_s) if median_s > 0 else float("nan"),
                notes=notes,
            )
        )

    refnx_fn = _try_refnx_callable(layers, ENERGY_EV)
    if refnx_fn is None:
        rows.append(
            _skipped_row(
                "refnx Abeles (isotropic)",
                case,
                n_layers,
                "skipped: refnx not installed (dev/plugin extra)",
            )
        )
    else:
        add(
            "refnx Abeles (isotropic)",
            lambda: refnx_fn(q),
            notes="scalar Abeles 2x2 on isotropic twin; not polarized TMM",
        )

    pypxr_fn = _try_pypxr_callable(layers, tensor, ENERGY_EV)
    if pypxr_fn is not None:
        add("pypxr", lambda: pypxr_fn(q))
    elif include_python_tmm or case.n_film <= PYTHON_TMM_MAX_FILM_DEFAULT:
        pxr_fn = _try_python_tmm_callable(layers, tensor, ENERGY_EV)
        if pxr_fn is None:
            rows.append(
                _skipped_row(
                    "refloxide.python.tmm",
                    case,
                    n_layers,
                    "skipped: pure-Python TMM unavailable",
                )
            )
        else:
            add(
                "refloxide.python.tmm",
                lambda: pxr_fn(q),
                notes="pure-Python polarized TMM",
            )
    else:
        rows.append(
            _skipped_row(
                "refloxide.python.tmm",
                case,
                n_layers,
                f"skipped: n_film>{PYTHON_TMM_MAX_FILM_DEFAULT} "
                "(pass --include-python-tmm)",
            )
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
        "| backend | n_film | n_q | n_layers | median_ms | heap_KiB |"
        " rss_delta_MiB | points/s | notes |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    body = []
    for r in rows:
        median_ms = "n/a" if not np.isfinite(r.median_s) else f"{r.median_s * 1e3:.3f}"
        heap = "n/a" if not np.isfinite(r.peak_heap_kib) else f"{r.peak_heap_kib:.1f}"
        rss = "n/a" if not np.isfinite(r.rss_delta_mib) else f"{r.rss_delta_mib:.2f}"
        thr = "n/a" if not np.isfinite(r.points_per_s) else f"{r.points_per_s:,.0f}"
        body.append(
            f"| {r.backend} | {r.n_film} | {r.n_q} | {r.n_layers} | {median_ms} |"
            f" {heap} | {rss} | {thr} | {r.notes} |"
        )
    return header + "\n".join(body) + "\n"


def _write_csv(path: Path, rows: list[BenchRow]) -> None:
    fieldnames = [
        "backend",
        "n_film",
        "n_q",
        "n_layers",
        "median_s",
        "peak_heap_kib",
        "rss_delta_mib",
        "points_per_s",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "backend": r.backend,
                    "n_film": r.n_film,
                    "n_q": r.n_q,
                    "n_layers": r.n_layers,
                    "median_s": r.median_s,
                    "peak_heap_kib": r.peak_heap_kib,
                    "rss_delta_mib": r.rss_delta_mib,
                    "points_per_s": r.points_per_s,
                    "notes": r.notes,
                }
            )


def _headline_rows(rows: list[BenchRow], n_film: int) -> list[BenchRow]:
    selected = [r for r in rows if r.n_film == n_film and np.isfinite(r.median_s)]
    order = {name: i for i, name in enumerate(BACKEND_ORDER)}
    return sorted(selected, key=lambda r: order.get(r.backend, 999))


def _style_axes(ax) -> None:
    ax.set_facecolor("#fafafa")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#374151")


def write_plots(
    rows: list[BenchRow],
    *,
    out_dir: Path,
    n_film: int,
    machine: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    headline = _headline_rows(rows, n_film)
    if not headline:
        msg = f"no finite rows for n_film={n_film}"
        raise RuntimeError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [r.backend.replace("refloxide ", "") for r in headline]
    colors = [BACKEND_COLORS.get(r.backend, "#4b5563") for r in headline]
    times_ms = [r.median_s * 1e3 for r in headline]
    # Prefer RSS delta when nonzero; otherwise fall back to heap peak (MiB).
    mem_mib = []
    mem_label = "RSS delta (MiB)"
    if any(r.rss_delta_mib > 0.05 for r in headline):
        mem_mib = [r.rss_delta_mib for r in headline]
    else:
        mem_mib = [r.peak_heap_kib / 1024.0 for r in headline]
        mem_label = "Peak Python heap (MiB)"

    written: list[Path] = []

    def _annotate_bars(ax, values: list[float], fmt: str) -> None:
        ymax = max(values) if values else 1.0
        for xi, value in zip(x, values, strict=True):
            ax.text(
                xi,
                value + 0.02 * ymax,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#374151",
            )

    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    _style_axes(ax)
    x = np.arange(len(labels))
    ax.bar(x, times_ms, color=colors, width=0.72, zorder=2)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Median wall time (ms)")
    ax.set_title(
        f"Forward model at {n_film:,} film slabs / {headline[0].n_q} q-points",
        fontsize=11,
        color="#111827",
    )
    _annotate_bars(ax, times_ms, "{:.1f}")
    ax.text(
        0.99,
        0.98,
        machine,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#6b7280",
    )
    path_time = out_dir / "bench_wall_time.png"
    fig.savefig(path_time, dpi=200, facecolor="white")
    plt.close(fig)
    written.append(path_time)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    _style_axes(ax)
    ax.bar(x, mem_mib, color=colors, width=0.72, zorder=2)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel(mem_label)
    ax.set_title(
        f"Python-visible memory at {n_film:,} film slabs "
        f"/ {headline[0].n_q} q-points",
        fontsize=11,
        color="#111827",
    )
    _annotate_bars(ax, mem_mib, "{:.3f}")
    ax.text(
        0.99,
        0.02,
        "tracemalloc; Rust/GPU buffers are outside this meter",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#6b7280",
    )
    path_mem = out_dir / "bench_memory.png"
    fig.savefig(path_mem, dpi=200, facecolor="white")
    plt.close(fig)
    written.append(path_mem)

    scaling = [r for r in rows if r.backend == "refloxide CPU parallel=True"]
    scaling = [r for r in scaling if np.isfinite(r.median_s)]
    if len(scaling) >= 2:
        scaling = sorted(scaling, key=lambda r: r.n_film)
        fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
        _style_axes(ax)
        for backend, color in (
            ("refnx Abeles (isotropic)", BACKEND_COLORS["refnx Abeles (isotropic)"]),
            (
                "refloxide CPU parallel=True",
                BACKEND_COLORS["refloxide CPU parallel=True"],
            ),
            ("refloxide GPU", BACKEND_COLORS["refloxide GPU"]),
        ):
            series = sorted(
                [r for r in rows if r.backend == backend and np.isfinite(r.median_s)],
                key=lambda r: r.n_film,
            )
            if not series:
                continue
            ax.plot(
                [r.n_film for r in series],
                [r.median_s * 1e3 for r in series],
                marker="o",
                color=color,
                label=backend.replace("refloxide ", ""),
                linewidth=1.8,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Film microslabs")
        ax.set_ylabel("Median wall time (ms)")
        ax.set_title("Scaling with stack depth", fontsize=11, color="#111827")
        ax.legend(frameon=False, fontsize=8)
        path_scale = out_dir / "bench_scaling.png"
        fig.savefig(path_scale, dpi=200, facecolor="white")
        plt.close(fig)
        written.append(path_scale)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--n-q",
        type=int,
        default=DEFAULT_N_Q,
        help="q-grid length for each case",
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=None,
        help=(
            "Optional n_q:n_film overrides. Default is a short scaling sweep "
            f"ending at the headline {HEADLINE_N_FILM} slabs."
        ),
    )
    parser.add_argument(
        "--include-python-tmm",
        action="store_true",
        help="Time pure-Python TMM even for large n_film",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write PNG figures into docs/assets/performance/",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory for committed README/docs plot assets",
    )
    args = parser.parse_args()

    if args.sizes:
        cases = [
            BenchCase(n_q=int(a), n_film=int(b))
            for size in args.sizes
            for a, b in [size.split(":", 1)]
        ]
    else:
        n_q = args.n_q
        cases = [
            BenchCase(n_q=n_q, n_film=100),
            BenchCase(n_q=n_q, n_film=1_000),
            BenchCase(n_q=n_q, n_film=HEADLINE_N_FILM),
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
                include_python_tmm=args.include_python_tmm,
            )
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = _format_table(all_rows)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    cpu = platform.processor() or "cpu"
    machine = f"{platform.system()} {platform.machine()} / {cpu}"
    report = (
        f"# GPU backend bench results\n\n"
        f"- generated: `{stamp}`\n"
        f"- machine: `{machine}`\n"
        f"- energy_ev: `{ENERGY_EV}`\n"
        f"- headline n_film: `{HEADLINE_N_FILM}`\n"
        f"- repeats/warmup: `{args.repeats}` / `{args.warmup}`\n"
        f"- gpu: `{gpu_note}`\n\n"
        f"{table}"
    )
    print(report)
    md_path = RESULTS_DIR / "gpu_backend_bench_results.md"
    csv_path = RESULTS_DIR / "gpu_backend_bench_results.csv"
    md_path.write_text(report, encoding="utf-8")
    _write_csv(csv_path, all_rows)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")

    if args.plot:
        written = write_plots(
            all_rows,
            out_dir=args.assets_dir,
            n_film=HEADLINE_N_FILM,
            machine=machine,
        )
        # Also copy CSV next to assets for CI/README provenance.
        assets_csv = args.assets_dir / "gpu_backend_bench_results.csv"
        args.assets_dir.mkdir(parents=True, exist_ok=True)
        assets_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
        assets_md = args.assets_dir / "gpu_backend_bench_results.md"
        assets_md.write_text(report, encoding="utf-8")
        for path in written:
            print(f"Wrote {path}")
        print(f"Wrote {assets_csv}")
        print(f"Wrote {assets_md}")


if __name__ == "__main__":
    main()
