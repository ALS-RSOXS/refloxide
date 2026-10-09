"""Uniaxial backend bench: serial CPU, parallel CPU, GPU, PyPXR/refnx plugin.

Compares the same uniaxial stack across:

* ``refloxide`` CPU ``parallel=False``
* ``refloxide`` CPU ``parallel=True``
* ``refloxide`` GPU (``device="gpu"``, skipped when no adapter)
* **PyPXR / refnx plugin** — in-tree ``refloxide.pxr.plugin.model.reflectivity``
  (pure-Python uniaxial TMM; the polarized refnx-plugin path)

Headline sizes: **1 film slab** and **10_000 film slabs** (plus vacuum /
substrate rows). Metrics: median wall time and peak Python heap
(``tracemalloc``).

``refnx`` is required only for the plugin import graph (dev/plugin extras),
not as a core runtime dependency of the Rust kernels.

Run::

    uv sync --group dev --group plugin
    uv run python examples/gpu_backend_bench.py --plot
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

ENERGY_EV = 250.0
DEFAULT_N_Q = 256
HEADLINE_FILMS = (1, 10_000)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ASSETS_DIR = (
    Path(__file__).resolve().parents[1] / "docs" / "assets" / "performance"
)

BACKEND_SERIAL = "refloxide CPU serial"
BACKEND_PARALLEL = "refloxide CPU parallel"
BACKEND_GPU = "refloxide GPU"
BACKEND_PYPXR = "PyPXR / refnx plugin"

BACKEND_ORDER = (BACKEND_SERIAL, BACKEND_PARALLEL, BACKEND_GPU, BACKEND_PYPXR)
BACKEND_COLORS = {
    BACKEND_SERIAL: "#1d4ed8",
    BACKEND_PARALLEL: "#2563eb",
    BACKEND_GPU: "#0f766e",
    BACKEND_PYPXR: "#6b7280",
}
BACKEND_SHORT = {
    BACKEND_SERIAL: "CPU serial",
    BACKEND_PARALLEL: "CPU parallel",
    BACKEND_GPU: "GPU",
    BACKEND_PYPXR: "PyPXR plugin",
}


@dataclass(frozen=True)
class BenchCase:
    """One (n_q, n_film) problem size."""

    n_q: int
    n_film: int


@dataclass(frozen=True)
class BenchRow:
    """One measured backend/size result."""

    backend: str
    n_q: int
    n_film: int
    n_layers: int
    median_s: float
    peak_heap_mib: float
    rss_mib: float
    notes: str = ""


def _graded_arrays(n_film: int) -> tuple[np.ndarray, np.ndarray]:
    """Vacuum / graded uniaxial film / substrate without refnx."""
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


def _try_pypxr_plugin_callable(
    layers: np.ndarray, tensor: np.ndarray, energy_ev: float
) -> Callable[[np.ndarray], object] | None:
    """PyPXR-shaped refnx plugin uniaxial path (in-tree ``pxr.plugin``)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from refloxide.pxr.plugin.model import reflectivity as plugin_reflectivity
    except ImportError as exc:
        print(f"PyPXR plugin unavailable: {exc}", flush=True)
        return None

    def _call(q: np.ndarray) -> object:
        out = plugin_reflectivity(
            q,
            layers,
            tensor,
            energy=energy_ev,
            dq=0.0,
            backend="uni",
            parallel=False,
        )
        if out is None:
            msg = "pxr.plugin.reflectivity returned None"
            raise RuntimeError(msg)
        return out

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
    heaps: list[float] = []
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        samples.append(elapsed)
        heaps.append(peak / (1024.0 * 1024.0))
    return statistics.median(samples), statistics.median(heaps)


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


BACKEND_RSS_KEY = {
    BACKEND_SERIAL: "serial",
    BACKEND_PARALLEL: "parallel",
    BACKEND_GPU: "gpu",
    BACKEND_PYPXR: "pypxr",
}


def _subprocess_rss_mib(
    *,
    backend_key: str,
    n_film: int,
    n_q: int,
) -> float:
    """Peak RSS of a fresh interpreter running one evaluation of ``backend_key``."""
    code = _rss_worker_source(backend_key=backend_key, n_film=n_film, n_q=n_q)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    if proc.returncode != 0:
        err = proc.stderr[-400:] if proc.stderr else ""
        print(f"RSS probe failed for {backend_key}: {err}", flush=True)
        return float("nan")
    try:
        return float(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return float("nan")


def _rss_worker_source(*, backend_key: str, n_film: int, n_q: int) -> str:
    return f"""
import platform
import resource
import warnings

import numpy as np

ENERGY_EV = {ENERGY_EV!r}
n_film = {n_film}
n_q = {n_q}
backend = {backend_key!r}

def graded(n_film):
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
        layers[row] = [dz, delta_o, beta_o, 2.0 if i == 0 else 0.0]
        n_o = complex(delta_o, beta_o)
        n_e = complex(delta_e, beta_e)
        tensor[row, 0, 0] = n_o
        tensor[row, 1, 1] = n_o
        tensor[row, 2, 2] = n_e
    layers[-1, 1:] = [5.97e-3, 4.25e-3, 0.5]
    n_b = complex(5.97e-3, 4.25e-3)
    tensor[-1, 0, 0] = tensor[-1, 1, 1] = tensor[-1, 2, 2] = n_b
    return layers, tensor

layers, tensor = graded(n_film)
q = np.linspace(0.001, 0.25, n_q, dtype=np.float64)
if backend == "serial":
    from refloxide.tmm import uniaxial_reflectivity
    uniaxial_reflectivity(q, layers, tensor, ENERGY_EV, parallel=False, device="cpu")
elif backend == "parallel":
    from refloxide.tmm import uniaxial_reflectivity
    uniaxial_reflectivity(q, layers, tensor, ENERGY_EV, parallel=True, device="cpu")
elif backend == "gpu":
    from refloxide.tmm import uniaxial_reflectivity
    uniaxial_reflectivity(q, layers, tensor, ENERGY_EV, parallel=False, device="gpu")
elif backend == "pypxr":
    warnings.simplefilter("ignore", DeprecationWarning)
    from refloxide.pxr.plugin.model import reflectivity
    reflectivity(q, layers, tensor, energy=ENERGY_EV, dq=0.0, backend="uni", parallel=False)
else:
    raise SystemExit("unknown backend")
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024)
"""


def _skipped(
    backend: str, case: BenchCase, n_layers: int, notes: str
) -> BenchRow:
    return BenchRow(
        backend=backend,
        n_q=case.n_q,
        n_film=case.n_film,
        n_layers=n_layers,
        median_s=float("nan"),
        peak_heap_mib=float("nan"),
        rss_mib=float("nan"),
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
    print(f"building stack n_film={case.n_film} n_q={case.n_q} ...", flush=True)
    layers, tensor = _graded_arrays(case.n_film)
    q = np.linspace(0.001, 0.25, case.n_q, dtype=np.float64)
    n_layers = int(layers.shape[0])
    rows: list[BenchRow] = []

    def add(backend: str, fn: Callable[[], object], *, notes: str = "") -> None:
        print(f"  timing {backend} ...", flush=True)
        median_s, peak_heap_mib = _measure(fn, repeats=repeats, warmup=warmup)
        rss_key = BACKEND_RSS_KEY[backend]
        print(f"  rss probe {backend} ...", flush=True)
        rss_mib = _subprocess_rss_mib(
            backend_key=rss_key, n_film=case.n_film, n_q=case.n_q
        )
        rows.append(
            BenchRow(
                backend=backend,
                n_q=case.n_q,
                n_film=case.n_film,
                n_layers=n_layers,
                median_s=median_s,
                peak_heap_mib=peak_heap_mib,
                rss_mib=rss_mib,
                notes=notes,
            )
        )

    from refloxide.tmm import uniaxial_reflectivity

    add(
        BACKEND_SERIAL,
        lambda: uniaxial_reflectivity(
            q, layers, tensor, ENERGY_EV, parallel=False, device="cpu"
        ),
    )
    add(
        BACKEND_PARALLEL,
        lambda: uniaxial_reflectivity(
            q, layers, tensor, ENERGY_EV, parallel=True, device="cpu"
        ),
    )

    if not gpu_ok:
        rows.append(_skipped(BACKEND_GPU, case, n_layers, f"skipped: {gpu_note}"))
    else:
        add(
            BACKEND_GPU,
            lambda: uniaxial_reflectivity(
                q, layers, tensor, ENERGY_EV, parallel=False, device="gpu"
            ),
            notes=gpu_note,
        )

    plugin_fn = _try_pypxr_plugin_callable(layers, tensor, ENERGY_EV)
    if plugin_fn is None:
        rows.append(
            _skipped(
                BACKEND_PYPXR,
                case,
                n_layers,
                "skipped: install refnx via uv sync --group plugin",
            )
        )
    else:
        add(
            BACKEND_PYPXR,
            lambda: plugin_fn(q),
            notes="pxr.plugin uniaxial (pure-Python TMM)",
        )

    return rows


def _format_table(rows: list[BenchRow]) -> str:
    header = (
        "| backend | n_film | n_q | n_layers | median_ms | heap_MiB |"
        " rss_MiB | notes |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    lines = []
    for r in rows:
        ms = "n/a" if not np.isfinite(r.median_s) else f"{r.median_s * 1e3:.3f}"
        heap = "n/a" if not np.isfinite(r.peak_heap_mib) else f"{r.peak_heap_mib:.3f}"
        rss = "n/a" if not np.isfinite(r.rss_mib) else f"{r.rss_mib:.1f}"
        lines.append(
            f"| {r.backend} | {r.n_film} | {r.n_q} | {r.n_layers} | {ms} |"
            f" {heap} | {rss} | {r.notes} |"
        )
    return header + "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[BenchRow]) -> None:
    fields = [
        "backend",
        "n_film",
        "n_q",
        "n_layers",
        "median_s",
        "peak_heap_mib",
        "rss_mib",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "backend": r.backend,
                    "n_film": r.n_film,
                    "n_q": r.n_q,
                    "n_layers": r.n_layers,
                    "median_s": r.median_s,
                    "peak_heap_mib": r.peak_heap_mib,
                    "rss_mib": r.rss_mib,
                    "notes": r.notes,
                }
            )


def _rows_for(rows: list[BenchRow], n_film: int) -> list[BenchRow]:
    order = {name: i for i, name in enumerate(BACKEND_ORDER)}
    selected = [r for r in rows if r.n_film == n_film and np.isfinite(r.median_s)]
    return sorted(selected, key=lambda r: order.get(r.backend, 999))


def _style_axes(ax) -> None:
    ax.set_facecolor("#fafafa")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#374151")


def _bar_panel(
    ax,
    panel_rows: list[BenchRow],
    *,
    values: list[float],
    ylabel: str,
    title: str,
    fmt: str,
) -> None:
    _style_axes(ax)
    labels = [BACKEND_SHORT[r.backend] for r in panel_rows]
    colors = [BACKEND_COLORS[r.backend] for r in panel_rows]
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.72, zorder=2)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, color="#111827")
    ymax = max(values) if values else 1.0
    for xi, value in zip(x, values, strict=True):
        ax.text(
            xi,
            value + 0.03 * ymax,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#374151",
        )


def write_plots(
    rows: list[BenchRow],
    *,
    out_dir: Path,
    machine: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), layout="constrained")
    for col, n_film in enumerate(HEADLINE_FILMS):
        panel = _rows_for(rows, n_film)
        if not panel:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        times_ms = [r.median_s * 1e3 for r in panel]
        mem_mib = [
            r.rss_mib if np.isfinite(r.rss_mib) else r.peak_heap_mib for r in panel
        ]
        slab_label = "1 uniaxial slab" if n_film == 1 else f"{n_film:,} uniaxial slabs"
        _bar_panel(
            axes[0, col],
            panel,
            values=times_ms,
            ylabel="Median wall time (ms)",
            title=f"Speed — {slab_label}",
            fmt="{:.2f}",
        )
        _bar_panel(
            axes[1, col],
            panel,
            values=mem_mib,
            ylabel="Peak process RSS (MiB)",
            title=f"Memory — {slab_label}",
            fmt="{:.1f}",
        )

    fig.suptitle(
        f"Uniaxial forward model ({DEFAULT_N_Q} q-points) — {machine}",
        fontsize=11,
        color="#111827",
    )
    fig.text(
        0.5,
        0.01,
        "PyPXR / refnx plugin = in-tree pxr.plugin uniaxial path "
        "(pure-Python TMM). Memory is peak RSS in a fresh subprocess.",
        ha="center",
        fontsize=7.5,
        color="#6b7280",
    )
    path = out_dir / "bench_uniaxial_grid.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    written.append(path)

    # Also emit the four single-metric panels used by the README layout.
    for n_film, tag in ((1, "1slab"), (HEADLINE_FILMS[1], "10k")):
        panel = _rows_for(rows, n_film)
        if not panel:
            continue
        slab_label = "1 uniaxial slab" if n_film == 1 else f"{n_film:,} uniaxial slabs"
        for kind, values, ylabel, fmt, fname in (
            (
                "speed",
                [r.median_s * 1e3 for r in panel],
                "Median wall time (ms)",
                "{:.2f}",
                f"bench_wall_time_{tag}.png",
            ),
            (
                "memory",
                [
                    r.rss_mib if np.isfinite(r.rss_mib) else r.peak_heap_mib
                    for r in panel
                ],
                "Peak process RSS (MiB)",
                "{:.1f}",
                f"bench_memory_{tag}.png",
            ),
        ):
            _ = kind
            fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
            _bar_panel(
                ax,
                panel,
                values=values,
                ylabel=ylabel,
                title=f"{ylabel.split('(')[0].strip()} — {slab_label}",
                fmt=fmt,
            )
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
            out = out_dir / fname
            fig.savefig(out, dpi=200, facecolor="white")
            plt.close(fig)
            written.append(out)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--n-q", type=int, default=DEFAULT_N_Q)
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=None,
        help="Optional n_q:n_film overrides (default: 1 and 10000 film slabs)",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    args = parser.parse_args()

    if args.sizes:
        cases = [
            BenchCase(n_q=int(a), n_film=int(b))
            for size in args.sizes
            for a, b in [size.split(":", 1)]
        ]
    else:
        cases = [BenchCase(n_q=args.n_q, n_film=n) for n in HEADLINE_FILMS]

    probe_layers, probe_tensor = _graded_arrays(1)
    gpu_ok, gpu_note = _gpu_probe(probe_layers, probe_tensor)

    all_rows: list[BenchRow] = []
    for case in cases:
        # Fewer repeats for the expensive pure-Python 10k path.
        repeats = args.repeats
        if case.n_film >= 10_000:
            repeats = min(repeats, 3)
        all_rows.extend(
            run_case(
                case,
                repeats=repeats,
                warmup=args.warmup,
                gpu_ok=gpu_ok,
                gpu_note=gpu_note,
            )
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    cpu = platform.processor() or "cpu"
    machine = f"{platform.system()} {platform.machine()} / {cpu}"
    table = _format_table(all_rows)
    report = (
        f"# Uniaxial backend bench\n\n"
        f"- generated: `{stamp}`\n"
        f"- machine: `{machine}`\n"
        f"- energy_ev: `{ENERGY_EV}`\n"
        f"- n_q: `{args.n_q if not args.sizes else 'mixed'}`\n"
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
        written = write_plots(all_rows, out_dir=args.assets_dir, machine=machine)
        args.assets_dir.mkdir(parents=True, exist_ok=True)
        (args.assets_dir / "gpu_backend_bench_results.csv").write_text(
            csv_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (args.assets_dir / "gpu_backend_bench_results.md").write_text(
            report, encoding="utf-8"
        )
        for path in written:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
