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

import os

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
SURFACE = {"thick": 12.0, "rough": 7.0, "density": 2.0, "rotation": 0.5}
BULK = {"thick": 179.0, "rough": 13.0, "density": 1.61, "rotation": np.pi / 2}
INTERFACE = {"thick": 14.0, "rough": 8.0, "density": 0.1, "rotation": 0.0}

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
    .filter((pl.col("q") >= Q_MIN) & (pl.col("energy") > 250))
    .select("q", "energy", "pol", "r", "r_err")
)
dataset = ReflectDataset.from_polars(frame)
energies = sorted(frame["energy"].unique().to_list())
assert len(energies) >= 20, f"expected 21 energies, got {len(energies)}: {energies}"
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
    oxide = MaterialSLD("SiO2", density=2.3, name="oxide")(8.8, 5.09)
    substrate = MaterialSLD("Si", density=2.36, name="substrate")(0, 1.2)
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

bulk.thick.setp(value = 180, vary=True, bounds=(175, 190))
bulk.sld.density.setp(vary=True, bounds=(1.2, 1.8))
bulk.sld.rotation.setp(vary=True, bounds=(0.0, np.pi / 2))

# NC: thick >= sqrt(2*pi)*rough/2 — keep lower thick bounds above that floor at
# the upper rough bound so DE box constraints cannot enter the forbidden region.
surface.thick.setp(vary=True, bounds=(12.0, 25.0))
surface.rough.setp(vary=True, bounds=(0.0, 9.0))
surface.sld.density.setp(vary=True, bounds=(0.0, 3.0))
surface.sld.rotation.setp(vary=True, bounds=(0.0, np.pi / 2))

interface.thick.setp(vary=True, bounds=(12.0, 25.0))
interface.rough.setp(vary=True, bounds=(0.0, 9.0))
interface.sld.density.setp(vary=True, bounds=(0.0, 3.0))
interface.sld.rotation.setp(vary=True, bounds=(0.0, np.pi / 2))

oxide.thick.setp(vary=True, bounds=(8, 12))
oxide.rough.setp(vary=True, bounds=(3, 6))
oxide.sld.density.setp(vary=True, bounds=(2.2, 2.4))

substrate.rough.setp(vary=True, bounds=(1.1, 1.5))
substrate.sld.density.setp(vary=True, bounds=(2.3, 2.5))

# Shared model energy_offset (scatterer-local offsets stay frozen at 0).
model.energy_offset.setp(value=0.003, vary=True, bounds=(-0.5, 0.5))
model.scale_s.where(between=(250, 300)).setp(vary=True, bounds=(0.7, 1.3))
model.scale_p.where(between=(250, 300)).setp(vary=True, bounds=(0.7, 1.3))
model.theta_offset_s.where(between=(250, 300)).setp(vary=False, bounds=(-0.05, 0.05))
model.theta_offset_p.where(between=(250, 300)).setp(vary=False, bounds=(-0.05, 0.05))

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
assert logl_after > logl_before
assert 150.0 <= recovered <= 220.0
print("OK: full 21-energy UniTensor fit recovered a plausible NC-valid bulk.\n")
# # %%
# 
# Add back in the the experiment correction terms
model = objective.model
model.energy_offset.setp(value=0.003, vary=True, bounds=(-0.5, 0.5))
model.scale_s.where(between=(250, 300)).setp(vary=True, bounds=(0.7, 1.3))
model.scale_p.where(between=(250, 300)).setp(vary=True, bounds=(0.7, 1.3))
model.theta_offset_s.where(between=(250, 300)).setp(vary=True, bounds=(-0.05, 0.05))
model.theta_offset_p.where(between=(250, 300)).setp(vary=True, bounds=(-0.05, 0.05))

with thread_workers(12) as workers:
    workers.bind(objective)

    t0 = time.perf_counter()
    CurveFitter(objective).fit(
        method="L-BFGS-B",
        options = {
            "workers": workers,
        },
    )
    print(f"DE maxiter=40 with thread_workers(8): {time.perf_counter() - t0:.2f} s")


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
