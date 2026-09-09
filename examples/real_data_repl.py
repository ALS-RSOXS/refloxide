"""Interactive multi-energy fit of real ZnPc XRR with UniTensorSLD slabs.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/real_data_repl.py

Loads all energies from ``reflectivity_data.parquet`` (sibling
``refl-analysis`` checkout). Each energy is a legacy pyref s/p concat
(split where ``Q`` resets); first chunk remaps to native ``pol="p"``,
second to native ``pol="s"`` (legacy labels are inverted relative to the
Rust kernel).

Stack: vacuum / ZnPc surface / bulk / interface / SiO2 / Si. Each ZnPc
layer is a ``UniTensorSLD`` on ``dft.csv``. Experiment corrections:
shared ``energy_offset``; per-energy ``scale_s/p`` and ``theta_offset_s/p``.
"""

# %% 0
from __future__ import annotations

import json
import os
import pickle

# IDE shells often export POLARS_VERBOSE=1; force quiet streaming IO.
os.environ["POLARS_VERBOSE"] = "0"

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from refnx.analysis import CurveFitter, Transform

from refloxide.data import OpticalConstants, ReflectDataset
from refloxide.model import MaterialSLD, ReflectModel, UniTensorSLD
from refloxide.objective import Objective, thread_workers

# %% Paths + starting geometry

ZNPC_DFT_CSV = Path.home() / "projects/refl-analysis/@models/optical/znpc/dft.csv"
ZNPC_DATA = (
    Path.home() / "projects/refl-analysis/@data/xrr/znpc/reflectivity_data.parquet"
)
for path in (ZNPC_DFT_CSV, ZNPC_DATA):
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- expects sibling refl-analysis checkout"
        )

Q_MIN = 0.0001
# Geometry must satisfy NC at start: thick >= sqrt(2*pi)*rough/2.
#
# thick/rough (every layer) and density (Oxide/Substrate only) below come
# from the free-tensor GlobalObjective's own fit (see
# dft_free_hybrid_refit_repl.py's transplant step, same source/convention)
# rather than the 283.7-eV-only fit used previously. The free-tensor
# model's Surface/ZnPc/Contamination layers are `pyref.fitting.structure.
# SLD` -- a raw per-energy diagonal tensor with no density/rotation
# parameter at all -- so density/rotation for SURFACE/BULK/INTERFACE are
# NOT touched here; there is no free-tensor value to pull for them.
SURFACE = {"thick": 9.98852, "rough": 6.22017, "density": 2.54155, "rotation": 1.03105}
BULK = {"thick": 180.554, "rough": 12.302, "density": 1.69, "rotation": 1.26212}
INTERFACE = {
    "thick": 11.0031,
    "rough": 7.53376,
    "density": 0.906035,
    "rotation": 0.769367,
}
OXIDE = {"thick": 8.83717, "rough": 5.09216, "density": 2.27249}
SUBSTRATE = {"rough": 1.2, "density": 2.37014}

# %% Load reflectivity_data: all energies, legacy concat -> native pol

parts = []
for _, g in pl.read_parquet(ZNPC_DATA).group_by("energy", maintain_order=True):
    q = g["Q"].to_numpy()
    cut = np.flatnonzero(np.diff(q) < 0)
    i = int(cut[0] + 1) if cut.size else len(q)
    # legacy first=.s (native R_pp), second=.p (native R_ss)
    pol = np.where(np.arange(len(q)) < i, "p", "s")
    parts.append(g.with_columns(pl.Series("pol", pol)))

frame = (
    pl.concat(parts)
    .rename({"Q": "q", "R": "r", "dR": "r_err"})
    .filter(pl.col("q") >= Q_MIN)
    .select("q", "energy", "pol", "r", "r_err")
)
dataset = ReflectDataset.from_polars(frame)
energies = sorted(frame["energy"].unique().to_list())
print(
    f"ReflectDataset: {len(dataset)} rows, "
    f"{len(list(dataset.groups()))} groups, energies={energies}"
)

# %% Measured data overview

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for energy, pol, indices in dataset.groups():
    ax = axes[0] if pol == "s" else axes[1]
    ax.errorbar(
        dataset.q[indices],
        dataset.r[indices],
        yerr=dataset.r_err[indices],
        fmt=".",
        ms=2,
        lw=0.5,
        alpha=0.7,
        label=f"{energy:.1f} eV",
    )
axes[0].set_title("measured R_s")
axes[1].set_title("measured R_p")
for ax in axes:
    ax.set_yscale("log")
    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.legend(fontsize="x-small", ncol=2)
axes[0].set_ylabel("Reflectivity")
fig.suptitle(f"ZnPc reflectivity_data ({len(energies)} energies)")
fig.tight_layout()
plt.show()

# %% UniTensor stack (one shared OpticalConstants across ZnPc layers)

znpc_ooc = OpticalConstants.from_file(ZNPC_DFT_CSV)


def build_structure():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    surface = UniTensorSLD(
        znpc_ooc,
        density=SURFACE["density"],
        rotation=SURFACE["rotation"],
        name="ZnPc_surface",
    )(SURFACE["thick"], SURFACE["rough"])
    bulk = UniTensorSLD(
        znpc_ooc,
        density=BULK["density"],
        rotation=BULK["rotation"],
        name="ZnPc_bulk",
    )(BULK["thick"], BULK["rough"])
    interface = UniTensorSLD(
        znpc_ooc,
        density=INTERFACE["density"],
        rotation=INTERFACE["rotation"],
        name="ZnPc_interface",
    )(INTERFACE["thick"], INTERFACE["rough"])
    oxide = MaterialSLD("SiO2", density=OXIDE["density"], name="oxide")(
        OXIDE["thick"], OXIDE["rough"]
    )
    substrate = MaterialSLD("Si", density=SUBSTRATE["density"], name="substrate")(
        0, SUBSTRATE["rough"]
    )
    return vacuum | surface | bulk | interface | oxide | substrate


model = ReflectModel(build_structure(), energies=energies, parallel=False)
structure = model.structure
surface = structure.slab("ZnPc_surface")
bulk = structure.slab("ZnPc_bulk")
interface = structure.slab("ZnPc_interface")
assert surface.sld.ooc is bulk.sld.ooc is interface.sld.ooc is znpc_ooc
assert surface.enforce_nevot_croce and bulk.enforce_nevot_croce
assert interface.enforce_nevot_croce and structure.slab("oxide").enforce_nevot_croce
assert not structure.slab("vacuum").enforce_nevot_croce
assert not structure.slab("substrate").enforce_nevot_croce

print(structure)
structure.plot.oc(283.7, difference=True)
plt.show()
structure.plot.param("density|orientation")
plt.show()

# %% Freeze defaults, free geometry + experiment corrections

oxide = structure.slab("oxide")
substrate = structure.slab("substrate")

for param in model.parameters.flattened():
    param.vary = False

bulk.thick.setp(value = 180, vary=True, bounds=(150, 190))
bulk.sld.density.setp(vary=False, bounds=(1.5, 1.7))
bulk.sld.rotation.setp(vary=True, bounds=(0.0, np.pi / 2))

# NC: thick >= sqrt(2*pi)*rough/2 — keep lower thick bounds above that floor at
# the upper rough bound so DE box constraints cannot enter the forbidden region.
surface.thick.setp(vary=True, bounds=(0, 25.0))
surface.rough.setp(vary=True, bounds=(0.0, 9.0))
surface.sld.density.setp(value = 2.1, vary=True, bounds=(1.6, 2.5))
surface.sld.rotation.setp(value = np.pi/4, vary=True, bounds=(0.0, np.pi / 2))

# interface.sld.density's ceiling is capped strictly below bulk's OWN
# floor (1.5) -- the interface region physically represents a less-dense,
# incompletely-packed transition zone near the substrate, not a second
# bulk-like layer. The 21-energy fit's own result (0.906 -> 1.253 for the
# same interface parameter, single- vs multi-energy) crept up toward that
# 1.5 floor inside the old (0.0, 1.7) bound; 1.3 keeps room to explore
# without ever letting "interface" become indistinguishable from "bulk".
interface.thick.setp(vary=True, bounds=(0, 25.0))
interface.rough.setp(vary=True, bounds=(0.0, 9.0))
interface.sld.density.setp(value = 0.5, vary=True, bounds=(0.5, 1.3))
interface.sld.rotation.setp(value = np.pi/4, vary=True, bounds=(0.0, np.pi / 2))

oxide.thick.setp(vary=False, bounds=(8, 12))
oxide.rough.setp(vary=False, bounds=(3, 6))
# Previously FROZEN at a single hardcoded value (2.3) -- any real
# mismatch between the true SiO2 density and that fixed guess had nowhere
# to go except into the adjacent interface layer's own free density,
# which is exactly the cross-talk between these two neighboring layers
# this comment block is now guarding against on both sides. Freed with a
# tight, physically-anchored bound (native/thermal SiO2 is well
# established in the 2.2-2.3 g/cm^3 range) instead of either staying
# frozen or opening up to the same wide (2.2, 2.4) span used before.
oxide.sld.density.setp(vary=False, bounds=(2.2, 2.3))

substrate.rough.setp(vary=False, bounds=(1.1, 1.5))
substrate.sld.density.setp(vary=False, bounds=(2.3, 2.5))

# Shared model energy_offset (scatterer-local offsets stay frozen at 0).
model.energy_offset.setp(value=0.003, vary=False, bounds=(-0.5, 0.5))
model.scale_s.where(between=(250, 300)).setp(vary=True, bounds=(0.7, 1.3))
model.scale_p.where(between=(250, 300)).setp(vary=True, bounds=(0.7, 1.3))
model.theta_offset_s.where(between=(250, 300)).setp(vary=True, bounds=(-0.05, 0.05))
model.theta_offset_p.where(between=(250, 300)).setp(vary=True, bounds=(-0.05, 0.05))

# DE thread_workers parallelizes the population; keep Rayon off to avoid nested pools.
model.parallel = False

objective = Objective(model, dataset, transform=Transform("logY"), nc_constraint=True)
print(f"nc_constraint={objective.nc_constraint}")
print(f"energies: {len(energies)}, batches: {len(objective._batches)}")
print(f"varying parameters: {len(objective.varying_parameters())}")
assert np.isfinite(objective.logp()), "starting geometry must satisfy NC prior"

# %% Fit (nll path; NC enforced via Objective.nll; DE pop via thread_workers)

logl_before = float(objective.logl())
print(f"bulk thick start = {bulk.thick.value:.1f}")
print(f"logl before fit: {logl_before:.3f}")
print(f"nll before fit:  {float(objective.nll()):.3f}")

# Bare workers=int uses multiprocessing (pickle / polars IPC storms).
# thread_workers maps the DE population on private Objective clones.
with thread_workers(12) as workers:
    workers.bind(objective)

    t0 = time.perf_counter()
    CurveFitter(objective).fit(
        method="differential_evolution",
        popsize=12,
        polish=True,
        seed=1,
        workers=workers,
        updating="deferred",
    )
    print(f"DE maxiter=40 with thread_workers(8): {time.perf_counter() - t0:.2f} s")

logl_after = float(objective.logl())
recovered = float(bulk.thick.value or 0.0)
print(f"recovered bulk thick = {recovered:.2f} A")
print(f"energy_offset = {model.energy_offset.value:.4f} eV")
print(f"logl after fit:  {logl_after:.3f}")
print(f"delta logl = {logl_after - logl_before:.3f}")
assert np.isfinite(logl_before) and np.isfinite(logl_after)
assert np.isfinite(objective.logp()), "fit exited outside NC prior support"
for slab in (surface, bulk, interface, oxide):
    thick = float(slab.thick.value or 0.0)
    rough = float(slab.rough.value or 0.0)
    limit = np.sqrt(2.0 * np.pi) * rough / 2.0
    assert thick >= limit, f"{slab.name}: thick={thick:.3f} < NC limit {limit:.3f}"
# %% Overlay data vs fit at a few energies

show_energies = (250.0, 283.7, 285.1)
fig, axes = plt.subplots(len(show_energies), 2, figsize=(11, 3.2 * len(show_energies)))
for row, energy in enumerate(show_energies):
    for col, pol in enumerate(("s", "p")):
        ax = axes[row, col]
        for e, p, indices in dataset.groups():
            if e != energy or p != pol:
                continue
            q = dataset.q[indices]
            ax.errorbar(
                q,
                dataset.r[indices],
                yerr=dataset.r_err[indices],
                fmt=".",
                ms=2,
                lw=0.5,
                color="0.4",
                label="data",
            )
            pred = model(q, energy)
            ax.plot(
                q,
                pred.s if pol == "s" else pred.p,
                color="C0",
                lw=1.5,
                label="fit",
            )
        ax.set_yscale("log")
        ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        ax.set_title(f"{energy:.1f} eV, {pol}-pol")
        if col == 0:
            ax.set_ylabel("Reflectivity")
        ax.legend(fontsize="x-small")
fig.suptitle(f"UniTensor fit, bulk thick = {recovered:.1f} A", y=1.01)
fig.tight_layout()
plt.show()

# %% Orientation / density profiles after fit

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
structure.plot.param("orientation", ax=axes[0], roughness=True)
structure.plot.param("density", ax=axes[1], roughness=True)
fig.tight_layout()
plt.show()
# %%
print(objective.varying_parameters())

# %% Compare against the original DFT-fit GlobalObjective (legacy pyref)
#
# The legacy fit (extracted once into a portable summary -- see
# dft_model_comparison_repl.py for the full extraction/parity-check story)
# used the SAME dft.csv OOC anchor and the SAME 283.7 eV anchor energy for
# an equivalent vacuum/surface/bulk/interface/oxide/substrate stack, just
# under different layer names: its "Contamination" layer is the same
# physical role as this script's own "interface" (both sit between bulk
# ZnPc and the oxide). This is an INDEPENDENT re-fit against comparable
# data, not a reproduction of the legacy pickle -- exact agreement isn't
# expected; the question is how far the two geometries diverge.

DFT_DIR = Path.home() / "projects/refl-analysis/@models/xrr/znpc/dft"
DFT_SUMMARY_PATH = DFT_DIR / "dft_en_offset_new2_summary.json"
DFT_DATA_PATH = DFT_DIR / "dft_en_offset_new2_data.parquet"
for path in (DFT_SUMMARY_PATH, DFT_DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- run the one-time extraction described in "
            "dft_model_comparison_repl.py first"
        )

dft_summary = json.loads(DFT_SUMMARY_PATH.read_text())
dft_frame = pl.read_parquet(DFT_DATA_PATH)
dft_layers = {layer["name"]: layer for layer in dft_summary["layers"]}

ROLE_MAP = {
    "ZnPc_surface": ("Surface", surface),
    "ZnPc_bulk": ("ZnPc", bulk),
    "ZnPc_interface": ("Contamination", interface),
}

print(f"{'role':<16}{'param':<10}{'ours':>12}{'legacy DFT':>12}{'delta':>12}")
for our_name, (legacy_name, slab) in ROLE_MAP.items():
    legacy = dft_layers[legacy_name]
    params = [
        ("thick", float(slab.thick.value or 0.0), legacy["thick"]),
        ("rough", float(slab.rough.value or 0.0), legacy["rough"]),
        ("density", float(slab.sld.density.value or 0.0), legacy["density"]),
        ("rotation", float(slab.sld.rotation.value or 0.0), legacy["rotation"]),
    ]
    for param, ours, legacy_v in params:
        print(
            f"{our_name:<16}{param:<10}{ours:>12.4f}{legacy_v:>12.4f}"
            f"{ours - legacy_v:>12.4f}"
        )

legacy_oxide = dft_layers["Oxide"]
for param, ours in (
    ("thick", float(oxide.thick.value or 0.0)),
    ("rough", float(oxide.rough.value or 0.0)),
    ("density", float(oxide.sld.density.value or 0.0)),
):
    legacy_v = legacy_oxide[param]
    delta = ours - legacy_v
    print(f"{'oxide':<16}{param:<10}{ours:>12.4f}{legacy_v:>12.4f}{delta:>12.4f}")

# %% Reflectivity overlay: data vs. our new fit vs. the legacy DFT fit, at
# the same representative energies used earlier -- 250/283.7/285.1 eV are
# anchor points common to both fits.

fig, axes = plt.subplots(len(show_energies), 2, figsize=(11, 3.2 * len(show_energies)))
for row, energy in enumerate(show_energies):
    legacy_group = dft_frame.filter(pl.col("energy") == energy)
    for col, pol in enumerate(("s", "p")):
        ax = axes[row, col]
        for e, p, indices in dataset.groups():
            if e != energy or p != pol:
                continue
            q = dataset.q[indices]
            ax.errorbar(
                q,
                dataset.r[indices],
                yerr=dataset.r_err[indices],
                fmt=".",
                ms=2,
                lw=0.5,
                color="0.4",
                label="data",
            )
            pred = model(q, energy)
            ax.plot(
                q,
                pred.s if pol == "s" else pred.p,
                color="C0",
                lw=1.5,
                label="our fit",
            )
        legacy_sub = legacy_group.filter(pl.col("pol") == pol)
        if not legacy_sub.is_empty():
            ax.plot(
                legacy_sub["q"].to_numpy(),
                legacy_sub["legacy_pred"].to_numpy(),
                color="k",
                ls="--",
                lw=1.2,
                label="legacy DFT fit",
            )
        ax.set_yscale("log")
        ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        ax.set_title(f"{energy:.1f} eV, {pol}-pol")
        if col == 0:
            ax.set_ylabel("Reflectivity")
        ax.legend(fontsize="x-small")
fig.suptitle("Our re-fit vs. the legacy DFT-fit GlobalObjective", y=1.01)
fig.tight_layout()
plt.show()

# %% Depth profile overlay: our fitted structure vs. the legacy DFT-fit's
# own structure, same dft.csv OOC anchor, same depth axis convention.


def build_legacy_structure():
    slabs = []
    for layer in dft_summary["layers"]:
        if layer["kind"] == "uniaxial":
            sld = UniTensorSLD(
                znpc_ooc,
                density=layer["density"],
                rotation=layer["rotation"],
                name=layer["name"],
            )
        else:
            sld = MaterialSLD(
                layer["formula"], density=layer["density"], name=layer["name"]
            )
        slabs.append(sld(layer["thick"], layer["rough"]))
    built = slabs[0]
    for slab in slabs[1:]:
        built = built | slab
    return built


legacy_structure = build_legacy_structure()
z_ours = structure.depth_grid(num_points=2000, pad=20.0)
z_legacy = legacy_structure.depth_grid(num_points=2000, pad=20.0)

_, density_ours = structure.density_profile_at(z_ours, roughness=True)
_, orientation_ours = structure.orientation_profile_at(z_ours, roughness=True)
_, density_legacy = legacy_structure.density_profile_at(z_legacy, roughness=True)
_, orientation_legacy = legacy_structure.orientation_profile_at(
    z_legacy, roughness=True
)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].plot(z_ours, orientation_ours, color="C0", label="our fit")
axes[0].plot(z_legacy, orientation_legacy, color="k", ls="--", label="legacy DFT fit")
axes[0].set_title("Orientation")
axes[0].set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
axes[0].legend(fontsize="x-small")

axes[1].plot(z_ours, density_ours, color="C0", label="our fit")
axes[1].plot(z_legacy, density_legacy, color="k", ls="--", label="legacy DFT fit")
axes[1].set_title("Density")
axes[1].set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
axes[1].legend(fontsize="x-small")
fig.tight_layout()
plt.show()

# %% Save the fitted objective as its own pickle -- a NEW name/location,
# distinct from both legacy inputs this run compared against
# (dft_en_offset_new2.pkl, free_en_offset_init_2.pkl). This is refloxide's
# own `Objective` (picklable via the same `pickle.dumps`/`loads` round-trip
# `thread_workers` already relies on internally -- confirmed directly,
# `Objective.__setstate__`/`_warm_objective_caches` re-register the
# OpticalConstants cache on load), so re-loading this file needs only
# refloxide -- no pyref/refnx-legacy dependency, unlike the two inputs.

FIT_OUT_DIR = Path.home() / "projects/refl-analysis/@models/xrr/znpc/refloxide"
FIT_OUT_DIR.mkdir(parents=True, exist_ok=True)
FIT_OUT_PATH = FIT_OUT_DIR / "refit-refloxide.pkl"

with FIT_OUT_PATH.open("wb") as f:
    pickle.dump(objective, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"saved fitted objective to {FIT_OUT_PATH}")
