"""Interactive showcase: book-ended film on a multi-energy fit.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/bookended_multi_energy_repl.py

Companion to ``bookended_repl.py`` (single-energy mechanics). Here the same
``BookendedOrientationProfile`` / ``BookendedComponent`` stack is probed at
many energies near the carbon K-edge. Geometry and orientation/density shape
are shared across energies; only the OOC (and isotropic CXRO layers) change
with photon energy when ``ReflectModel`` rematerializes the structure.

Three things:

1. Build the deferred-energy bookended stack and plot orientation/density
   once (energy-independent), then optical constants at several energies.
2. Forward-model reflectivity with the array-energy path
   ``ReflectModel(q, energies)`` and synthesize a noisy multi-energy s+p
   ``ReflectDataset``.
3. One short global fit: freeze everything, free ``total_thick`` from a
   wrong start, recover it against all energies at once via
   ``refloxide.objective.Objective`` (batches energies that share a ``q``
   grid into one Rust kernel call). Multi-energy thickness logl is
   needle-sharp, so DE uses ``polish=True`` to finish with a local step.
"""

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from refnx.analysis import CurveFitter

from refloxide.data import ReflectDataset
from refloxide.model import BookendedComponent, MaterialSLD, ReflectModel
from refloxide.objective import Objective
from refloxide.pxr.energy.bookended import BookendedOrientationProfile
from refloxide.pxr.energy.ooc import OocAnchor

# %% Locate DFT ZnPc optical constants (sibling refl-analysis checkout)

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

ZNPC_OOC = OocAnchor.from_file(ZNPC_DFT_CSV)
Q = np.linspace(0.015, 0.22, 100)
ENERGIES_EV = np.asarray(
    [280.0, 283.0, 284.5, 285.1, 286.0, 288.0],
    dtype=np.float64,
)

TRUE_TOTAL_THICK = 180.0
TRUE_SURFACE_ROUGHNESS = 4.0
TRUE_DENSITY_BULK = 1.61
TRUE_DENSITY_SI = 1.55
TRUE_DENSITY_VAC = 1.45
TRUE_TAU_SI = 15.0
TRUE_TAU_VAC = 10.0
TRUE_ALPHA_BULK = 0.9
TRUE_ALPHA_SI = 0.3
TRUE_ALPHA_VAC = 1.4

# %% Bookended stack: OOC bound, energy left deferred for rematerialization


def make_profile(*, name: str = "ZnPc") -> BookendedOrientationProfile:
    return BookendedOrientationProfile(
        ooc=ZNPC_OOC,
        energy=None,
        num_slabs=20,
        mesh_constant=0.1,
        name=name,
        total_thick=TRUE_TOTAL_THICK,
        surface_roughness=TRUE_SURFACE_ROUGHNESS,
        density_bulk=TRUE_DENSITY_BULK,
        density_si=TRUE_DENSITY_SI,
        density_vac=TRUE_DENSITY_VAC,
        tau_si=TRUE_TAU_SI,
        tau_vac=TRUE_TAU_VAC,
        alpha_bulk=TRUE_ALPHA_BULK,
        alpha_si=TRUE_ALPHA_SI,
        alpha_vac=TRUE_ALPHA_VAC,
    )


def build_structure(profile: BookendedOrientationProfile):
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    oxide = MaterialSLD("SiO2", density=2.2, name="oxide")(8, 3)
    si = MaterialSLD("Si", density=2.33, name="si")(0, 3)
    return vacuum | BookendedComponent(profile) | oxide | si


def freeze_entire_structure(model: ReflectModel) -> None:
    for p in model.structure.parameters.flattened():
        if p.constraint is None:
            p.vary = False


true_profile = make_profile()
true_model = ReflectModel(build_structure(true_profile))
true_structure = true_model.structure

print(
    f"energies (eV): {ENERGIES_EV.tolist()}\n"
    f"q points: {len(Q)}, num_slabs: {true_profile.num_slabs}"
)

# %% 1. Energy-independent shape: orientation and density vs depth

depth = np.linspace(0.0, TRUE_TOTAL_THICK, 300)
fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
axes[0].plot(depth, np.rad2deg(true_profile.orientation(depth)), color="C0")
axes[0].set_ylabel("Molecular tilt (deg)")
axes[0].set_title("Book-ended profile (shared across all energies)")
axes[1].plot(depth, true_profile.local_density(depth), color="C1")
axes[1].set_ylabel(r"Density (g/cm$^3$)")
axes[1].set_xlabel("Depth from vacuum interface (A)")
fig.tight_layout()
plt.show()

# %% 1b. Optical constants rematerialize with energy

demo_energies = [280.0, 285.1, 288.0]
for e in demo_energies:
    true_structure.plot.oc(e, pad=15.0, difference=True)
    plt.gcf().suptitle(f"Structure OC at {e:.1f} eV", y=1.02)
    plt.show()

# %% 2. Array-energy forward model + synthetic multi-energy s+p data

r_true = true_model(Q, ENERGIES_EV)
assert r_true.s.shape == (len(Q), len(ENERGIES_EV))
assert r_true.p.shape == (len(Q), len(ENERGIES_EV))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for i, e in enumerate(ENERGIES_EV):
    axes[0].plot(Q, r_true.s[:, i], label=f"{e:.1f} eV")
    axes[1].plot(Q, r_true.p[:, i], label=f"{e:.1f} eV")
axes[0].set_title("R_ss (array-energy path)")
axes[1].set_title("R_pp (array-energy path)")
for ax in axes:
    ax.set_yscale("log")
    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.legend(fontsize="x-small", ncol=2)
axes[0].set_ylabel("Reflectivity")
fig.tight_layout()
plt.show()

rng = np.random.default_rng(0)
q_rows: list[np.ndarray] = []
e_rows: list[np.ndarray] = []
pol_rows: list[np.ndarray] = []
r_rows: list[np.ndarray] = []
err_rows: list[np.ndarray] = []
for i, e in enumerate(ENERGIES_EV):
    for pol, col in (("s", r_true.s[:, i]), ("p", r_true.p[:, i])):
        noisy = col * (1.0 + rng.normal(0.0, 0.01, size=Q.shape))
        err = np.maximum(col * 0.02, 1e-8)
        q_rows.append(Q)
        e_rows.append(np.full(Q.shape, e))
        pol_rows.append(np.full(Q.shape, pol, dtype=object))
        r_rows.append(noisy)
        err_rows.append(err)

dataset = ReflectDataset(
    q=np.concatenate(q_rows),
    energy=np.concatenate(e_rows),
    pol=np.concatenate(pol_rows),
    r=np.concatenate(r_rows),
    r_err=np.concatenate(err_rows),
)
print(
    f"ReflectDataset: {len(dataset)} rows, "
    f"{len(list(dataset.groups()))} (energy, pol) groups"
)

# %% 3. Multi-energy fit: freeze all, free total_thick from a wrong start

fit_profile = make_profile()
fit_model = ReflectModel(build_structure(fit_profile))
freeze_entire_structure(fit_model)

fit_profile.total_thick.value = 140.0
fit_profile.total_thick.setp(vary=True, bounds=(100.0, 250.0))
fit_profile.alpha_si.setp(vary=True, bounds=(0.0, np.pi/2))
fit_profile.tau_si.setp(vary=True, bounds=(0.0, 20))

objective = Objective(fit_model, dataset, anisotropy_weight=0.4)
print(f"\ntotal_thick start = {fit_profile.total_thick.value:.1f} (true = 180.0)")
print(f"logl before fit: {objective.logl():.3f}")
print(
    f"Objective batches: {len(objective._batches)} "
    f"(expect 2: one s-pol and one p-pol spanning all energies)"
)

CurveFitter(objective).fit(
    method="differential_evolution",
    maxiter=60,
    popsize=15,
    polish=True,
    seed=1,
)

print(
    f"recovered total_thick = {fit_profile.total_thick.value:.2f} "
    f"(true = {TRUE_TOTAL_THICK:.2f})"
)
print(f"logl after fit:  {objective.logl():.3f}")

# %% Overlay: true vs recovered at a couple of energies

r_fit = fit_model(Q, ENERGIES_EV)
show = [0, 3, 5]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for i in show:
    e = ENERGIES_EV[i]
    axes[0].plot(Q, r_true.s[:, i], color="k", lw=1.5, alpha=0.7)
    axes[0].plot(Q, r_fit.s[:, i], "--", label=f"{e:.1f} eV")
    axes[1].plot(Q, r_true.p[:, i], color="k", lw=1.5, alpha=0.7)
    axes[1].plot(Q, r_fit.p[:, i], "--", label=f"{e:.1f} eV")
axes[0].set_title("R_ss: true (solid) vs fit (dashed)")
axes[1].set_title("R_pp: true (solid) vs fit (dashed)")
for ax in axes:
    ax.set_yscale("log")
    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.legend(fontsize="small")
axes[0].set_ylabel("Reflectivity")
fig.tight_layout()
plt.show()
