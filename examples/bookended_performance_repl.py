"""Interactive showcase: forward-modeling performance of the book-ended profile.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/bookended_performance_repl.py

Forward modeling only -- no fitting, no pyref comparison. This picks up
directly from `bookended_repl.py`'s structure/fitting showcase and the
performance investigation that followed it: `BookendedOrientationProfile`
composed via `refloxide.model.BookendedComponent` (see
`tests/test_bookended_model_integration.py`) is Rust-backed and about 10x
faster per call than the same profile run through the legacy, pure-Python
`pxr.plugin.model.ReflectModel` path. This script demonstrates that
capability directly:

1. Speed vs `num_slabs` -- the adaptive microslab mesh is the one knob that
   trades resolution for raw per-call cost; timing it across a range shows
   the cost is close to linear (fixed overhead + a roughly constant
   per-microslab cost), so cutting `num_slabs` is a real, predictable lever.
2. A molecular-tilt sweep (`alpha_bulk`), the same kind of "augment one
   parameter, replot" exercise as the rotation sweeps in
   `uni_tensor_znpc_repl.py`/`mixed_uni_tensor_znpc_repl.py` -- but for the
   whole graded profile at once, and timed, since the point here is
   demonstrating that a fast forward model makes this kind of exploratory
   sweep (or, later, an optimizer inner loop) cheap.
3. A relaxation-length sweep (`tau_vac`), the other kind of book-ended shape
   parameter (how fast the vacuum-side book-end relaxes into the bulk),
   shown the same way.
4. In-place parameter mutation vs rebuilding the profile/structure from
   scratch on every sweep step -- the same physics, timed both ways, to
   show why a sweep (or a fitter) should mutate `Parameter.value` in an
   existing model rather than reconstruct one per evaluation.
5. Two verified, independent speedups stacked together: `ReflectModel(...,
   parallel=True)` (Rayon-parallel over q, ~3.5-5x here, growing with
   `num_slabs`) and a better-tuned `mesh_constant` (fewer microslabs for the
   same accuracy, verified against a very fine reference mesh, not assumed).
   Profiling behind this script showed >99% of forward-model time is inside
   the Rust kernel itself at any reasonable `num_slabs`, so those two are
   the only levers that touch the dominant cost -- everything else
   (avoiding redundant profile-tensor recomputation, vectorizing the
   per-microslab row-packing loop, both now fixed in
   `refloxide.pxr.energy.bookended`/`refloxide.model.BookendedComponent`)
   was Python-side overhead too small to move the needle at typical
   `num_slabs`, though it stops being negligible in the hundreds-to-thousands
   range this script also exercises.
6. The fully-Rust-fused bookended path (`refloxide.tmm.
   bookended_uniaxial_reflectivity`): mesh generation, the orientation/
   density profile, and per-microslab tensor construction all happen inside
   one Rust call, skipping `Structure.materialize_at`'s Python/numpy
   intermediates entirely. `ReflectModel` uses it automatically whenever the
   structure qualifies (see `refloxide.model._plan_fused_bookended`) --
   nothing to opt into. Measured here both ways: wall-clock is a wash
   (confirms finding 5 -- the Rust kernel itself already dominates, so
   skipping Python glue barely moves total time), but peak Python-heap
   memory per call drops sharply and grows far more slowly with
   `num_slabs`, since the whole mesh/profile/tensor pipeline no longer
   allocates numpy arrays on the Python side at all.
"""
# %%
from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import refloxide.model as refloxide_model_module
from refloxide.model import BookendedComponent, MaterialSLD, ReflectModel
from refloxide.pxr.energy.bookended import BookendedOrientationProfile
from refloxide.pxr.energy.ooc import OocAnchor

# %% Locate the DFT-computed ZnPc optical constants in the sibling refl-analysis repo

ZNPC_DFT_CSV = (
    Path.home()
    / "projects"
    / "refl-analysis"
    / "@models"
    / "optical"
    / "znpc"
    / "dft.csv"
)
if not ZNPC_DFT_CSV.exists():
    msg = (
        f"ZnPc DFT optical constants not found at {ZNPC_DFT_CSV} -- this "
        "example expects a sibling refl-analysis checkout at ../refl-analysis "
        "next to refloxide."
    )
    raise FileNotFoundError(msg)

ENERGY_EV = 285.1  # carbon K-edge pi* resonance
Q = np.linspace(0.015, 0.25, 150)
ZNPC_OOC = OocAnchor.from_file(ZNPC_DFT_CSV)

PARAMS = {
    "total_thick": 180.0,
    "surface_roughness": 4.0,
    "density_bulk": 1.61,
    "density_si": 1.55,
    "density_vac": 1.45,
    "tau_si": 15.0,
    "tau_vac": 10.0,
    "alpha_bulk": 0.9,
    "alpha_si": 0.3,
    "alpha_vac": 1.4,
}


def build_model(
    num_slabs: int, *, parallel: bool = False, mesh_constant: float = 0.1
) -> tuple[ReflectModel, BookendedOrientationProfile]:
    profile = BookendedOrientationProfile(
        ooc=ZNPC_OOC,
        energy=ENERGY_EV,
        num_slabs=num_slabs,
        mesh_constant=mesh_constant,
        name="ZnPc",
        **PARAMS,
    )
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    oxide = MaterialSLD("SiO2", density=2.2, name="oxide")(8, 3)
    si = MaterialSLD("Si", density=2.33, name="si")(0, 3)
    structure = vacuum | BookendedComponent(profile) | oxide | si
    return ReflectModel(structure, parallel=parallel), profile


def time_it(fn, n: int = 200, warmup: int = 5) -> float:
    """Mean seconds per call, after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


# %% 1. Speed vs num_slabs -- the adaptive-mesh resolution/cost trade-off

SLAB_COUNTS = (3, 5, 7, 15, 30, 50, 100, 150, 200, 1000)
slab_timings = []
slab_reflectivity = {}
for num_slabs in SLAB_COUNTS:
    model, _profile = build_model(num_slabs)
    # n=50 rather than time_it's default 200 -- the largest slab counts here
    # cost ~7 ms/call, and this loop times 8 of them; 50 reps is already
    # stable to within a couple percent, so 200 would just be paying 4x the
    # wall-clock for a demo plot that doesn't need that precision.
    t = time_it(lambda m=model: m(Q, ENERGY_EV), n=50)
    slab_timings.append(t * 1e3)
    slab_reflectivity[num_slabs] = model(Q, ENERGY_EV)
    print(f"num_slabs={num_slabs:>4d}: {t * 1e3:8.4f} ms/call")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(SLAB_COUNTS, slab_timings, "o-")
ax.set_xlabel("num_slabs")
ax.set_ylabel("ms / forward-model call")
ax.set_title("Forward-model cost vs microslab count (near-linear)")
fig.tight_layout()
plt.show()

# %% 1b. Reflectivity comparison across num_slabs -- does coarsening cost accuracy?

# Cutting num_slabs is only a free lunch if the reflectivity barely moves --
# reuse the SAME structures just timed above and check that directly,
# against the finest mesh as the reference.
reference_slabs = SLAB_COUNTS[-1]
reference_r = slab_reflectivity[reference_slabs]
slab_colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(SLAB_COUNTS)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for num_slabs, color in zip(SLAB_COUNTS, slab_colors, strict=True):
    r = slab_reflectivity[num_slabs]
    axes[0].plot(Q, r.s, color=color, label=f"num_slabs={num_slabs}")
    rel_dev = np.abs(r.s - reference_r.s) / reference_r.s
    axes[1].plot(Q, rel_dev, color=color, label=f"num_slabs={num_slabs}")

axes[0].set_yscale("log")
axes[0].set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
axes[0].set_ylabel("s-pol reflectivity")
axes[0].legend(fontsize="small", ncol=2)
axes[0].set_title("Reflectivity vs num_slabs (same profile, different mesh)")

axes[1].set_yscale("log")
axes[1].set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
axes[1].set_ylabel(f"relative |s - s(num_slabs={reference_slabs})|")
axes[1].set_title("Convergence: deviation from the finest mesh")
fig.tight_layout()
plt.show()

print(f"relative deviation from num_slabs={reference_slabs} (max over q):")
for num_slabs in SLAB_COUNTS[:-1]:
    r = slab_reflectivity[num_slabs]
    rel_dev = np.abs(r.s - reference_r.s) / reference_r.s
    print(f"  num_slabs={num_slabs:>4d}: {rel_dev.max():.2e}")
print()

# %% 1c. Structure visualization -- depth profile, even though the fused path
# skips `Structure.materialize_at`'s Python/numpy intermediates for reflectivity
#
# `Structure.plot.oc`/`.param` call `materialize_at` directly, independent
# of whichever path `ReflectModel` used for the actual reflectivity call
# above (fused Rust or assembled) -- so the visualization capability is
# available on this fused-path model exactly as it is everywhere else in
# the codebase.

viz_model, _viz_profile = build_model(num_slabs=30)
viz_structure = viz_model.structure
viz_structure.plot.oc(ENERGY_EV, pad=15.0, difference=True)
plt.show()
viz_structure.plot.param("density|orientation", pad=15.0)
plt.show()

# %% 2. Molecular-tilt sweep (alpha_bulk) -- augment one parameter, replot, time it

model, profile = build_model(num_slabs=30)
depth = np.linspace(0.0, PARAMS["total_thick"], 300)

alpha_bulk_values = np.linspace(0.0, np.pi / 2, 7)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

t0 = time.perf_counter()
for value, color in zip(alpha_bulk_values, colors, strict=False):
    profile.alpha_bulk.value = value
    r = model(Q, ENERGY_EV)
    axes[0].plot(Q, r.s, color=color, label=f"alpha_bulk={value:.2f} rad")
    axes[1].plot(depth, np.rad2deg(profile.orientation(depth)), color=color)
t_sweep = time.perf_counter() - t0

axes[0].set_yscale("log")
axes[0].set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
axes[0].set_ylabel("s-pol reflectivity")
axes[0].legend(fontsize="small")
axes[1].set_xlabel("Depth from vacuum interface (A)")
axes[1].set_ylabel("Molecular tilt (deg)")
axes[1].set_title("Orientation profile for each alpha_bulk")
fig.suptitle(
    f"Bulk molecular-tilt sweep: {len(alpha_bulk_values)} evaluations in "
    f"{t_sweep * 1e3:.2f} ms ({t_sweep / len(alpha_bulk_values) * 1e3:.3f} ms/eval)"
)
fig.tight_layout()
plt.show()
profile.alpha_bulk.value = PARAMS["alpha_bulk"]  # restore

print(
    f"alpha_bulk sweep: {len(alpha_bulk_values)} evals in {t_sweep * 1e3:.2f} ms "
    f"({t_sweep / len(alpha_bulk_values) * 1e3:.3f} ms/eval)\n"
)

# %% 3. Relaxation-length sweep (tau_vac) -- the other book-ended shape parameter

tau_vac_values = np.linspace(2.0, 40.0, 7)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

t0 = time.perf_counter()
for value, color in zip(tau_vac_values, colors, strict=False):
    profile.tau_vac.value = value
    r = model(Q, ENERGY_EV)
    axes[0].plot(Q, r.s, color=color, label=f"tau_vac={value:.1f} A")
    axes[1].plot(depth, np.rad2deg(profile.orientation(depth)), color=color)
t_sweep_tau = time.perf_counter() - t0

axes[0].set_yscale("log")
axes[0].set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
axes[0].set_ylabel("s-pol reflectivity")
axes[0].legend(fontsize="small")
axes[1].set_xlabel("Depth from vacuum interface (A)")
axes[1].set_ylabel("Molecular tilt (deg)")
axes[1].set_title("Orientation profile for each tau_vac")
n_tau = len(tau_vac_values)
fig.suptitle(
    f"Vacuum-side relaxation-length sweep: {n_tau} evaluations in "
    f"{t_sweep_tau * 1e3:.2f} ms ({t_sweep_tau / n_tau * 1e3:.3f} ms/eval)"
)
fig.tight_layout()
plt.show()
profile.tau_vac.value = PARAMS["tau_vac"]  # restore

print(
    f"tau_vac sweep: {len(tau_vac_values)} evals in {t_sweep_tau * 1e3:.2f} ms "
    f"({t_sweep_tau / len(tau_vac_values) * 1e3:.3f} ms/eval)\n"
)

# %% 4. In-place mutation vs rebuilding the profile/structure from scratch every step

N_STEPS = 30
sweep_values = np.linspace(0.0, np.pi / 2, N_STEPS)

t0 = time.perf_counter()
for value in sweep_values:
    profile.alpha_bulk.value = value
    model(Q, ENERGY_EV)
t_mutate = time.perf_counter() - t0

t0 = time.perf_counter()
for value in sweep_values:
    step_params = dict(PARAMS, alpha_bulk=value)
    step_model, _step_profile = build_model(num_slabs=30)
    for name, val in step_params.items():
        getattr(_step_profile, name).value = val
    step_model(Q, ENERGY_EV)
t_rebuild = time.perf_counter() - t0

profile.alpha_bulk.value = PARAMS["alpha_bulk"]  # restore

print(f"{N_STEPS}-step alpha_bulk sweep:")
print(
    f"  mutate Parameter.value in place:      {t_mutate * 1e3:8.2f} ms total "
    f"({t_mutate / N_STEPS * 1e3:.3f} ms/step)"
)
print(
    f"  rebuild profile+structure each step:   {t_rebuild * 1e3:8.2f} ms total "
    f"({t_rebuild / N_STEPS * 1e3:.3f} ms/step)"
)
print(f"  rebuilding is {t_rebuild / t_mutate:.1f}x slower for the same sweep")

# %% 5a. parallel=True -- Rayon-parallel over q, the single biggest lever here

# Profiling this whole script showed >99% of forward-model time is inside
# the Rust kernel itself (refloxide.rust.uniaxial_reflectivity), at any
# num_slabs worth using. parallel=True is the one switch that touches that
# dominant cost directly -- safe here since this is a standalone forward
# model, not nested inside an outer-parallel fitter/walker loop (that's the
# case where parallel=False is the right call, to avoid oversubscribing
# the same cores twice over).
PARALLEL_SLAB_COUNTS = (30, 100, 300, 1000)
seq_timings = []
par_timings = []
for num_slabs in PARALLEL_SLAB_COUNTS:
    seq_model, _ = build_model(num_slabs, parallel=False)
    par_model, _ = build_model(num_slabs, parallel=True)
    t_seq = time_it(lambda m=seq_model: m(Q, ENERGY_EV), n=50)
    t_par = time_it(lambda m=par_model: m(Q, ENERGY_EV), n=50)
    seq_timings.append(t_seq * 1e3)
    par_timings.append(t_par * 1e3)
    print(
        f"num_slabs={num_slabs:>5d}: sequential={t_seq * 1e3:8.4f} ms  "
        f"parallel={t_par * 1e3:8.4f} ms  speedup={t_seq / t_par:.2f}x"
    )

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(PARALLEL_SLAB_COUNTS, seq_timings, "o-", label="parallel=False (default)")
ax.plot(PARALLEL_SLAB_COUNTS, par_timings, "o-", label="parallel=True")
ax.set_xlabel("num_slabs")
ax.set_ylabel("ms / forward-model call")
ax.set_title("parallel=True: Rayon over q-points, standalone forward model")
ax.legend()
fig.tight_layout()
plt.show()
print()

# %% 5b. Put both verified wins together: parallel=True + a tuned mesh_constant

# num_slabs=24 with mesh_constant=0.05 was found (by direct comparison
# against a num_slabs=2000 reference, not by assumption) to match
# num_slabs=30/mesh_constant=0.10's own accuracy -- so this is an
# apples-to-apples speedup, not a coarser-and-therefore-faster trade.
reference_model, _ = build_model(2000, parallel=True, mesh_constant=0.1)
reference_r = reference_model(Q, ENERGY_EV)

baseline_model, _ = build_model(30, parallel=False, mesh_constant=0.1)
tuned_model, _ = build_model(24, parallel=True, mesh_constant=0.05)

t_baseline = time_it(lambda: baseline_model(Q, ENERGY_EV))
t_tuned = time_it(lambda: tuned_model(Q, ENERGY_EV))

r_baseline = baseline_model(Q, ENERGY_EV)
r_tuned = tuned_model(Q, ENERGY_EV)
dev_baseline = np.max(np.abs(r_baseline.s - reference_r.s) / reference_r.s)
dev_tuned = np.max(np.abs(r_tuned.s - reference_r.s) / reference_r.s)

print("baseline vs tuned, both accuracy-checked against a num_slabs=2000 reference:")
print(
    f"  baseline (num_slabs=30, mesh_constant=0.10, sequential): "
    f"{t_baseline * 1e3:8.4f} ms/call, max relative deviation = {dev_baseline:.2e}"
)
print(
    f"  tuned    (num_slabs=24, mesh_constant=0.05, parallel=True): "
    f"{t_tuned * 1e3:8.4f} ms/call, max relative deviation = {dev_tuned:.2e}"
)
print(f"  combined speedup at matched accuracy: {t_baseline / t_tuned:.2f}x")

# %% 6. The fully-Rust-fused bookended path: wall-clock is a wash, memory is not

# ReflectModel already used the fused path for every call above whenever the
# structure qualified (see refloxide.model._plan_fused_bookended) -- this
# cell isolates its effect by forcing the assembled path off via monkeypatch
# and comparing both wall-clock and peak Python-heap memory for the same
# structure and the same call.
FUSED_SLAB_COUNTS = (30, 100, 300, 1000)
fused_timings = []
assembled_timings = []
fused_mem = []
assembled_mem = []

for num_slabs in FUSED_SLAB_COUNTS:
    model, _ = build_model(num_slabs, parallel=True)

    t_fused = time_it(lambda m=model: m(Q, ENERGY_EV), n=50)
    for _ in range(5):
        model(Q, ENERGY_EV)
    tracemalloc.start()
    model(Q, ENERGY_EV)
    _, peak_fused = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    original_plan_fn = refloxide_model_module._plan_fused_bookended
    refloxide_model_module._plan_fused_bookended = lambda *_a, **_k: None
    t_assembled = time_it(lambda m=model: m(Q, ENERGY_EV), n=50)
    for _ in range(5):
        model(Q, ENERGY_EV)
    tracemalloc.start()
    model(Q, ENERGY_EV)
    _, peak_assembled = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    refloxide_model_module._plan_fused_bookended = original_plan_fn

    fused_timings.append(t_fused * 1e3)
    assembled_timings.append(t_assembled * 1e3)
    fused_mem.append(peak_fused)
    assembled_mem.append(peak_assembled)
    print(
        f"num_slabs={num_slabs:>5d}: "
        f"time assembled={t_assembled * 1e3:7.4f} ms  fused={t_fused * 1e3:7.4f} ms  "
        f"(speedup={t_assembled / t_fused:.2f}x)   "
        f"peak-mem assembled={peak_assembled:>7,} B  fused={peak_fused:>6,} B  "
        f"(reduction={peak_assembled / max(peak_fused, 1):.1f}x)"
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(
    FUSED_SLAB_COUNTS, assembled_timings, "o-", label="assembled (materialize_at)"
)
axes[0].plot(FUSED_SLAB_COUNTS, fused_timings, "o-", label="fused (Rust end-to-end)")
axes[0].set_xlabel("num_slabs")
axes[0].set_ylabel("ms / forward-model call")
axes[0].set_title("Wall-clock: a wash (kernel-bound either way)")
axes[0].legend()

axes[1].plot(FUSED_SLAB_COUNTS, assembled_mem, "o-", label="assembled")
axes[1].plot(FUSED_SLAB_COUNTS, fused_mem, "o-", label="fused")
axes[1].set_xlabel("num_slabs")
axes[1].set_ylabel("peak Python-heap bytes / call")
axes[1].set_yscale("log")
axes[1].set_title("Memory: fused stays flat, assembled grows with num_slabs")
axes[1].legend()
fig.tight_layout()
plt.show()
