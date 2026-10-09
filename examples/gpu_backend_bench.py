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
BACKEND_SHORT = {
    BACKEND_SERIAL: "CPU serial",
    BACKEND_PARALLEL: "CPU parallel",
    BACKEND_GPU: "GPU",
    BACKEND_PYPXR: "PyPXR",
}

# Ruff-style single accent: ALS soft-X-ray violet (synchrotron / RSOXS adjacent).
THEME = {
    "light": {
        "text": "#1c1c1e",
        "muted": "#6e6e73",
        "grid": "#d8d8dc",
        "bar": "#6B3FA0",
        "winner": "#1c1c1e",
    },
    "dark": {
        "text": "#f5f5f7",
        "muted": "#8e8e93",
        "grid": "#3a3a3c",
        "bar": "#9B6DFF",
        "winner": "#ffffff",
    },
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
    reflectivity(
        q, layers, tensor, energy=ENERGY_EV, dq=0.0, backend="uni", parallel=False
    )
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


def _ruff_axes(ax, theme: dict[str, object]) -> None:
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(str(theme["grid"]))
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=str(theme["muted"]), length=0, labelsize=9)
    ax.yaxis.set_ticks_position("none")


def _mem_value(row: BenchRow) -> float:
    return row.rss_mib if np.isfinite(row.rss_mib) else row.peak_heap_mib


def _save_transparent(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=220, facecolor="none", edgecolor="none", transparent=True)
    plt.close(fig)


def _format_duration(seconds: float) -> str:
    if seconds < 1.0:
        ms = seconds * 1e3
        return f"{ms:.2f}ms" if ms < 10 else f"{ms:.1f}ms"
    if seconds < 10.0:
        return f"{seconds:.2f}s"
    return f"{seconds:.1f}s"


def _nice_cap(values: list[float], *, headroom: float = 1.08) -> float:
    peak = max(values) if values else 1.0
    if peak <= 0:
        return 1.0
    exp = 10 ** np.floor(np.log10(peak))
    for mult in (1.0, 1.25, 1.5, 2.0, 2.5, 5.0, 10.0):
        cand = mult * exp
        if cand >= peak * headroom:
            return float(cand)
    return float(peak * headroom)


def _second_ticks(xlim: float) -> tuple[list[float], list[str]]:
    use_ms = xlim < 0.05
    if use_ms:
        xlim_ms = xlim * 1e3
        step_ms = _nice_cap([xlim_ms], headroom=1.0) / 4.0
        if step_ms <= 0:
            step_ms = xlim_ms / 2.0
        positions_ms = list(np.arange(0.0, xlim_ms + step_ms * 0.25, step_ms))
        if positions_ms[-1] < xlim_ms - 1e-9:
            positions_ms.append(xlim_ms)
        positions = [p / 1e3 for p in positions_ms]
        labels = [
            "0ms" if p == 0.0 else f"{p:.2g}ms".replace("msms", "ms")
            for p in positions_ms
        ]
        return positions, labels
    if xlim <= 0.1:
        step = xlim / 4.0
    elif xlim <= 1.0:
        step = 0.25
    elif xlim <= 3.0:
        step = 0.5
    else:
        step = max(1.0, round(xlim / 3.0))
    positions = list(np.arange(0.0, xlim + step * 0.5, step))
    if positions[-1] < xlim - 1e-12:
        positions.append(xlim)
    labels = []
    for t in positions:
        if t == 0.0:
            labels.append("0s")
        elif t >= 1.0 and abs(t - round(t)) < 1e-9:
            labels.append(f"{round(t)}s")
        else:
            labels.append(f"{t:.2g}s")
    return positions, labels


def _memory_ticks(xlim: float) -> tuple[list[float], list[str]]:
    positions = [0.0, xlim / 2.0, xlim]
    labels = ["0", f"{xlim / 2:.0f}", f"{xlim:.0f}"]
    return positions, labels


def _horizontal_bars(
    *,
    labels: list[str],
    values: list[float],
    value_tags: list[str],
    winner_idx: int,
    theme: dict[str, object],
    xlim: float,
    tick_labels: list[str],
    tick_positions: list[float],
    figsize: tuple[float, float],
) -> object:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    _ruff_axes(ax, theme)
    y = np.arange(len(labels))
    drawn = [min(v, xlim) for v in values]
    ax.barh(y, drawn, color=str(theme["bar"]), height=0.58, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    for tick, idx in zip(ax.get_yticklabels(), range(len(labels)), strict=True):
        if idx == winner_idx:
            tick.set_fontweight("bold")
            tick.set_color(str(theme["winner"]))
        else:
            tick.set_fontweight("regular")
            tick.set_color(str(theme["muted"]))
    has_overflow = any(v > xlim for v in values)
    view_max = xlim if has_overflow else xlim * 1.28
    ax.set_xlim(0.0, view_max)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.grid(axis="x", color=str(theme["grid"]), linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    pad = 0.015 * xlim
    for yi, raw, tag in zip(y, values, value_tags, strict=True):
        if raw >= xlim:
            ax.text(
                xlim - pad,
                yi,
                tag,
                va="center",
                ha="right",
                fontsize=9,
                color=str(theme["text"]),
                zorder=3,
            )
        else:
            ax.text(
                raw + pad,
                yi,
                tag,
                va="center",
                ha="left",
                fontsize=9,
                color=str(theme["muted"]),
                zorder=3,
                clip_on=False,
            )
    fig.subplots_adjust(left=0.24, right=0.99, top=0.97, bottom=0.24)
    return fig


def write_hero_plot(
    rows: list[BenchRow],
    *,
    out_dir: Path,
    n_film: int = HEADLINE_FILMS[1],
) -> list[Path]:
    """Ruff-style horizontal hero; light + dark variants for GitHub themes."""
    ranked = sorted(_rows_for(rows, n_film), key=lambda r: r.median_s)
    if not ranked:
        msg = f"no finite rows for n_film={n_film}"
        raise RuntimeError(msg)

    panel = list(reversed(ranked))
    labels = [BACKEND_SHORT[r.backend] for r in panel]
    times_s = [r.median_s for r in panel]
    winner_idx = int(np.argmin(times_s))
    mid = sorted(times_s)[-2] if len(times_s) >= 2 else times_s[0]
    xlim = max(_nice_cap([mid], headroom=1.2), 1.0)
    tick_positions, tick_labels = _second_ticks(xlim)
    tags = [
        f"> {tick_labels[-1]}" if t > xlim else _format_duration(t) for t in times_s
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for mode, theme in THEME.items():
        fig = _horizontal_bars(
            labels=labels,
            values=times_s,
            value_tags=tags,
            winner_idx=winner_idx,
            theme=theme,
            xlim=xlim,
            tick_labels=tick_labels,
            tick_positions=tick_positions,
            figsize=(5.8, 2.15),
        )
        path = out_dir / f"bench_hero_{mode}.png"
        _save_transparent(fig, path)
        written.append(path)
    return written


def write_mini_plot(
    rows: list[BenchRow],
    *,
    out_dir: Path,
    n_film: int,
    metric: str,
    stem: str,
) -> list[Path]:
    """Compact Ruff-style horizontal bars for the details block."""
    ranked = sorted(
        _rows_for(rows, n_film),
        key=lambda r: r.median_s if metric == "time" else _mem_value(r),
    )
    if not ranked:
        return []
    panel = list(reversed(ranked))
    labels = [BACKEND_SHORT[r.backend] for r in panel]
    if metric == "time":
        values = [r.median_s for r in panel]
        mid = sorted(values)[-2] if len(values) >= 2 else values[0]
        xlim = _nice_cap([mid], headroom=1.2)
        if max(values) > xlim:
            xlim = max(xlim, 1.0) if mid >= 0.2 else _nice_cap(values, headroom=1.05)
        winner_idx = int(np.argmin(values))
        tick_positions, tick_labels = _second_ticks(xlim)
        tags = [
            f"> {tick_labels[-1]}" if v > xlim else _format_duration(v) for v in values
        ]
    else:
        values = [_mem_value(r) for r in panel]
        xlim = _nice_cap(values, headroom=1.05)
        winner_idx = int(np.argmin(values))
        tick_positions, tick_labels = _memory_ticks(xlim)
        tags = [f"{v:.0f} MiB" for v in values]

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for mode, theme in THEME.items():
        fig = _horizontal_bars(
            labels=labels,
            values=values,
            value_tags=tags,
            winner_idx=winner_idx,
            theme=theme,
            xlim=xlim,
            tick_labels=tick_labels,
            tick_positions=tick_positions,
            figsize=(4.6, 1.9),
        )
        path = out_dir / f"{stem}_{mode}.png"
        _save_transparent(fig, path)
        written.append(path)
    return written


def write_plots(
    rows: list[BenchRow],
    *,
    out_dir: Path,
    machine: str,
) -> list[Path]:
    _ = machine
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.extend(write_hero_plot(rows, out_dir=out_dir))
    written.extend(
        write_mini_plot(
            rows,
            out_dir=out_dir,
            n_film=1,
            metric="time",
            stem="bench_time_1slab",
        )
    )
    written.extend(
        write_mini_plot(
            rows,
            out_dir=out_dir,
            n_film=1,
            metric="rss",
            stem="bench_rss_1slab",
        )
    )
    written.extend(
        write_mini_plot(
            rows,
            out_dir=out_dir,
            n_film=HEADLINE_FILMS[1],
            metric="time",
            stem="bench_time_10k",
        )
    )
    written.extend(
        write_mini_plot(
            rows,
            out_dir=out_dir,
            n_film=HEADLINE_FILMS[1],
            metric="rss",
            stem="bench_rss_10k",
        )
    )
    return written


def _load_rows_from_csv(path: Path) -> list[BenchRow]:
    rows: list[BenchRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            rows.append(
                BenchRow(
                    backend=item["backend"],
                    n_q=int(item["n_q"]),
                    n_film=int(item["n_film"]),
                    n_layers=int(item["n_layers"]),
                    median_s=float(item["median_s"]),
                    peak_heap_mib=float(item["peak_heap_mib"]),
                    rss_mib=float(item["rss_mib"]),
                    notes=item.get("notes", ""),
                )
            )
    return rows


def best_results_markdown(
    rows: list[BenchRow], *, n_film: int = HEADLINE_FILMS[1]
) -> str:
    """Compact comparison table sorted best-first by wall time."""
    panel = sorted(_rows_for(rows, n_film), key=lambda r: r.median_s)
    if not panel:
        return ""
    ref = max(panel, key=lambda r: r.median_s)
    lines = [
        "| Backend | Time | vs PyPXR | RSS |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in panel:
        ms = r.median_s * 1e3
        speedup = ref.median_s / r.median_s if r.median_s > 0 else float("nan")
        vs = "-" if r.backend == ref.backend else f"{speedup:,.0f}x"
        rss = _mem_value(r)
        name = BACKEND_SHORT[r.backend]
        lines.append(
            f"| **{name}** | **{ms:.1f} ms** | {vs} | {rss:.0f} MiB |"
        )
    return "\n".join(lines)


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
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Skip timing; regenerate plots from an existing results CSV",
    )
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "gpu_backend_bench_results.csv"
    md_path = RESULTS_DIR / "gpu_backend_bench_results.md"
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    cpu = platform.processor() or "cpu"
    machine = f"{platform.system()} {platform.machine()} / {cpu}"

    if args.from_csv is not None:
        all_rows = _load_rows_from_csv(args.from_csv)
        gpu_note = "from csv"
        report = (
            f"# Uniaxial backend bench\n\n"
            f"- regenerated plots: `{stamp}`\n"
            f"- source: `{args.from_csv}`\n"
            f"- machine: `{machine}`\n\n"
            f"{_format_table(all_rows)}\n"
            f"## Best results ({HEADLINE_FILMS[1]:,} slabs)\n\n"
            f"{best_results_markdown(all_rows)}\n"
        )
    else:
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

        all_rows = []
        for case in cases:
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
        report = (
            f"# Uniaxial backend bench\n\n"
            f"- generated: `{stamp}`\n"
            f"- machine: `{machine}`\n"
            f"- energy_ev: `{ENERGY_EV}`\n"
            f"- n_q: `{args.n_q if not args.sizes else 'mixed'}`\n"
            f"- repeats/warmup: `{args.repeats}` / `{args.warmup}`\n"
            f"- gpu: `{gpu_note}`\n\n"
            f"{_format_table(all_rows)}\n"
            f"## Best results ({HEADLINE_FILMS[1]:,} slabs)\n\n"
            f"{best_results_markdown(all_rows)}\n"
        )
        _write_csv(csv_path, all_rows)
        print(f"Wrote {csv_path}")

    print(report)
    md_path.write_text(report, encoding="utf-8")
    print(f"Wrote {md_path}")

    if args.plot or args.from_csv is not None:
        written = write_plots(all_rows, out_dir=args.assets_dir, machine=machine)
        args.assets_dir.mkdir(parents=True, exist_ok=True)
        source_csv = args.from_csv if args.from_csv is not None else csv_path
        if not csv_path.exists() and args.from_csv is not None:
            _write_csv(csv_path, all_rows)
        (args.assets_dir / "gpu_backend_bench_results.csv").write_text(
            source_csv.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (args.assets_dir / "gpu_backend_bench_results.md").write_text(
            report, encoding="utf-8"
        )
        (args.assets_dir / "best_results.md").write_text(
            best_results_markdown(all_rows) + "\n", encoding="utf-8"
        )
        for path in written:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
