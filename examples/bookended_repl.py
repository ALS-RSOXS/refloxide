"""Interactive showcase: the book-ended orientation/density profile, and how to fit it.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/bookended_repl.py

`refloxide.pxr.energy.bookended.BookendedOrientationProfile`: a single
component that represents a whole graded organic film as one smoothly
varying orientation/density profile between two book-ends (the vacuum-side
value and the substrate-side value), materialized into `num_slabs` adaptive
microslabs, rather than a handful of discrete, uniform sublayers. Real,
DFT-derived ZnPc optical constants are used, as in the other `*_repl.py`
examples.

The profile composes into `refloxide.model.Structure`/`ReflectModel` via
`refloxide.model.BookendedComponent`, a thin adapter (see
`tests/test_bookended_model_integration.py` for the parity check against the
legacy `pxr.plugin` stack the profile also still supports).
`refloxide.model.ReflectModel` is Rust-backed by construction.

The profile has nine shape parameters (`total_thick`, `surface_roughness`,
`density_bulk`/`density_si`/`density_vac`, `tau_si`/`tau_vac`,
`alpha_bulk`/`alpha_si`/`alpha_vac`), and fitting all nine at once from a
cold start is a real, separate problem (parameter correlation, staging
strategy, etc.) -- deliberately out of scope here. This script instead
isolates the basic mechanic a bigger fit is built out of: hold everything
else fixed at its known-correct value via `Parameter.vary = False`, free
just one or two parameters, and confirm the fit actually recovers them from
a wrong starting guess.

Three things:

1. Build the profile and structure, plot its orientation(depth) and
   density(depth) curves directly -- what "book-ended" actually looks like.
2. Compute its reflectivity and synthesize a noisy s+p dataset from it.
3. Two small fits against that dataset, everything else held at its true
   value: first a single parameter (`total_thick`), then two at once
   (`alpha_bulk` and `tau_vac`, which are physically correlated -- the
   vacuum-side tilt and how fast it relaxes into the bulk both shape the
   same part of the profile).
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
Q = np.linspace(0.015, 0.22, 150)
ZNPC_OOC = OocAnchor.from_file(ZNPC_DFT_CSV)

# %% "True" film: book-ended orientation/density, tilt/relaxation differ per book-end

TRUE_PARAMS = {
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


def build_structure(profile: BookendedOrientationProfile):
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    oxide = MaterialSLD("SiO2", density=2.2, name="oxide")(8, 3)
    si = MaterialSLD("Si", density=2.33, name="si")(0, 3)
    return vacuum | BookendedComponent(profile) | oxide | si


true_profile = BookendedOrientationProfile(
    ooc=ZNPC_OOC,
    energy=ENERGY_EV,
    num_slabs=30,
    mesh_constant=0.1,
    name="ZnPc",
    **TRUE_PARAMS,
)
true_model = ReflectModel(build_structure(true_profile))

# %% 1. What "book-ended" looks like: orientation and density vs depth

depth = np.linspace(0.0, TRUE_PARAMS["total_thick"], 300)
fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
axes[0].plot(depth, np.rad2deg(true_profile.orientation(depth)), color="C0")
axes[0].axhline(np.rad2deg(TRUE_PARAMS["alpha_vac"]), ls=":", color="0.5", lw=1)
axes[0].axhline(np.rad2deg(TRUE_PARAMS["alpha_si"]), ls=":", color="0.5", lw=1)
axes[0].set_ylabel("Molecular tilt (deg)")
axes[0].set_title(
    "Book-ended orientation profile: vacuum-side -> bulk -> substrate-side"
)

axes[1].plot(depth, true_profile.local_density(depth), color="C1")
axes[1].axhline(TRUE_PARAMS["density_vac"], ls=":", color="0.5", lw=1)
axes[1].axhline(TRUE_PARAMS["density_si"], ls=":", color="0.5", lw=1)
axes[1].set_ylabel(r"Density (g/cm$^3$)")
axes[1].set_xlabel("Depth from vacuum interface (A)")
fig.tight_layout()
plt.show()

# %% 1b. The whole-structure view: `Structure.plot.oc`/`Structure.plot.param`
#
# The plot above is the bare `BookendedOrientationProfile` in isolation.
# `Structure.plot.oc`/`.param` walk the FULL structure instead -- vacuum,
# the book-ended film, the SiO2 oxide, and the Si substrate -- with each
# interface's optical-constant step broadened by an error function of
# width sigma = that interface's own roughness (the standard
# NC-consistent SLD-profile convention), plus the xx/zz dichroism on a
# right-hand twin axis (`difference=True`). Density and orientation are
# NaN over the isotropic vacuum/oxide/Si regions, which have no molecular
# tilt.

true_structure = true_model.structure
true_structure.plot.oc(ENERGY_EV, pad=15.0, difference=True)
plt.show()
true_structure.plot.param("density|orientation", pad=15.0)
plt.show()

# %% 2. Reflectivity of the true film, and a synthetic noisy s+p dataset from it

r_true = true_model(Q, ENERGY_EV)
r_s_true, r_p_true = r_true.s, r_true.p

rng = np.random.default_rng(0)
r_s = r_s_true * (1 + rng.normal(0, 0.01, size=Q.shape))
r_p = r_p_true * (1 + rng.normal(0, 0.01, size=Q.shape))
err_s = r_s_true * 0.02
err_p = r_p_true * 0.02

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(Q, r_s, yerr=err_s, fmt=".", ms=3, alpha=0.5, label="synthetic data (s)")
ax.errorbar(Q, r_p, yerr=err_p, fmt=".", ms=3, alpha=0.5, label="synthetic data (p)")
ax.plot(Q, r_s_true, color="C0", label="true model (s)")
ax.plot(Q, r_p_true, color="C1", label="true model (p)")
ax.set_yscale("log")
ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
ax.set_ylabel("Reflectivity")
ax.legend(fontsize="small")
ax.set_title(f"True book-ended ZnPc film, {ENERGY_EV:.1f} eV")
fig.tight_layout()
plt.show()

dataset = ReflectDataset(
    q=np.concatenate([Q, Q]),
    energy=np.full(2 * len(Q), ENERGY_EV),
    pol=np.concatenate(
        [np.full(Q.shape, "s", dtype=object), np.full(Q.shape, "p", dtype=object)]
    ),
    r=np.concatenate([r_s, r_p]),
    r_err=np.concatenate([err_s, err_p]),
)

# %% 3. A fit profile that starts EXACTLY at the true values, then held fixed

BOUNDS = {
    "total_thick": (100.0, 250.0),
    "alpha_bulk": (0.0, np.pi / 2),
    "tau_vac": (2.0, 40.0),
}


def freeze_entire_structure(model: ReflectModel) -> None:
    """Freeze every parameter in the structure, not just the bookended profile's.

    `Scatterer.__call__` (`MaterialSLD(...)(thick, rough)`) always sets the
    resulting slab's `thick`/`rough` to `vary=True`, and `MaterialSLD.density`
    defaults `vary=True` too -- the vacuum/oxide/substrate slabs surrounding
    the profile are not exempt. Freezing only `profile.parameters` leaves
    those on by default, which both slows `differential_evolution` down (a
    needlessly larger search space) and triggers refnx's "parameter has no
    effect on residuals" warning for `vacuum_thick` (vacuum's thickness is
    physically meaningless).
    """
    for p in model.structure.parameters.flattened():
        if p.constraint is None:
            p.vary = False


fit_profile = BookendedOrientationProfile(
    ooc=ZNPC_OOC,
    energy=ENERGY_EV,
    num_slabs=30,
    mesh_constant=0.1,
    name="ZnPc",
    **TRUE_PARAMS,
)
fit_model = ReflectModel(build_structure(fit_profile))
freeze_entire_structure(fit_model)  # every parameter starts fixed at its true value

objective = Objective(fit_model, dataset, anisotropy_weight=0.4)

# %% 3a. Hold everything fixed, free just one parameter: total_thick

fit_profile.total_thick.value = 140.0  # wrong on purpose; everything else stays true
fit_profile.total_thick.setp(vary=True, bounds=BOUNDS["total_thick"])

print(f"total_thick perturbed to {fit_profile.total_thick.value:.1f} (true = 180.0)")
print(f"logl before fit: {objective.logl():.3f}")
CurveFitter(objective).fit(
    method="differential_evolution", maxiter=40, polish=False, seed=1
)
print(
    f"recovered total_thick = {fit_profile.total_thick.value:.2f} "
    f"(true = {TRUE_PARAMS['total_thick']:.2f})"
)
print(f"logl after fit:  {objective.logl():.3f}\n")

fit_profile.total_thick.vary = False
fit_profile.total_thick.value = TRUE_PARAMS[
    "total_thick"
]  # restore before the next demo

# %% 3b. Hold everything fixed, free two correlated parameters: alpha_bulk, tau_vac

fit_profile.alpha_bulk.value = 0.3  # wrong; true is 0.9
fit_profile.tau_vac.value = 25.0  # wrong; true is 10.0
fit_profile.alpha_bulk.setp(vary=True, bounds=BOUNDS["alpha_bulk"])
fit_profile.tau_vac.setp(vary=True, bounds=BOUNDS["tau_vac"])

print(
    f"alpha_bulk perturbed to {fit_profile.alpha_bulk.value:.2f} rad (true = 0.90), "
    f"tau_vac perturbed to {fit_profile.tau_vac.value:.1f} A (true = 10.0)"
)
print(f"logl before fit: {objective.logl():.3f}")
CurveFitter(objective).fit(
    method="differential_evolution", maxiter=60, polish=False, seed=1
)
print(
    f"recovered alpha_bulk = {fit_profile.alpha_bulk.value:.3f} rad "
    f"(true = {TRUE_PARAMS['alpha_bulk']:.3f})"
)
print(
    f"recovered tau_vac    = {fit_profile.tau_vac.value:.2f} A "
    f"(true = {TRUE_PARAMS['tau_vac']:.2f})"
)
print(f"logl after fit:  {objective.logl():.3f}")

# %% 3c. Same fit, but the fitting model only gets a deliberately coarse num_slabs
#
# The synthetic "measured" data still comes from the well-resolved
# num_slabs=30 true_model above -- only the model DOING the fitting is
# under-resolved. bookended_performance_repl.py's convergence study found
# num_slabs=7 already deviates ~560% from a converged mesh, so this isn't a
# subtle effect: a too-coarse mesh caps how well the model can ever match
# the data, no matter what alpha_bulk/tau_vac end up at.

COARSE_NUM_SLABS = 8

coarse_profile = BookendedOrientationProfile(
    ooc=ZNPC_OOC,
    energy=ENERGY_EV,
    num_slabs=COARSE_NUM_SLABS,
    mesh_constant=0.1,
    name="ZnPc",
    **TRUE_PARAMS,
)
coarse_model = ReflectModel(build_structure(coarse_profile))
freeze_entire_structure(coarse_model)

coarse_profile.alpha_bulk.value = 0.3  # same wrong start as 3b, for a fair comparison
coarse_profile.tau_vac.value = 25.0
coarse_profile.alpha_bulk.setp(vary=True, bounds=BOUNDS["alpha_bulk"])
coarse_profile.tau_vac.setp(vary=True, bounds=BOUNDS["tau_vac"])

coarse_objective = Objective(coarse_model, dataset, anisotropy_weight=0.4)

print(f"\nSame fit, but num_slabs={COARSE_NUM_SLABS} instead of 30:")
print(f"logl before fit: {coarse_objective.logl():.3f}")
CurveFitter(coarse_objective).fit(
    method="differential_evolution", maxiter=60, polish=False, seed=1
)
print(
    f"recovered alpha_bulk = {coarse_profile.alpha_bulk.value:.3f} rad "
    f"(true = {TRUE_PARAMS['alpha_bulk']:.3f})"
)
print(
    f"recovered tau_vac    = {coarse_profile.tau_vac.value:.2f} A "
    f"(true = {TRUE_PARAMS['tau_vac']:.2f})"
)
print(f"logl after fit:  {coarse_objective.logl():.3f}")
print(
    f"num_slabs=30 fit reached logl={objective.logl():.3f}; "
    f"num_slabs={COARSE_NUM_SLABS} caps out at logl={coarse_objective.logl():.3f} "
    "-- the gap is mesh resolution, not the optimizer"
)

# %% Plot: true vs the num_slabs=30 fit vs the coarse num_slabs fit

r_fit = fit_model(Q, ENERGY_EV)
r_coarse = coarse_model(Q, ENERGY_EV)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Q, r_s_true, color="k", lw=2, label="true (s)")
ax.plot(Q, r_fit.s, "--", color="C0", label="fit (s): num_slabs=30")
ax.plot(Q, r_coarse.s, ":", color="C3", label=f"fit (s): num_slabs={COARSE_NUM_SLABS}")
ax.set_yscale("log")
ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
ax.set_ylabel("Reflectivity")
ax.legend(fontsize="small")
ax.set_title("Recovered reflectivity: well-resolved vs. deliberately coarse mesh")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    depth, np.rad2deg(true_profile.orientation(depth)), color="k", lw=2, label="true"
)
ax.plot(
    depth,
    np.rad2deg(fit_profile.orientation(depth)),
    "--",
    color="C0",
    label="fit: num_slabs=30",
)
ax.plot(
    depth,
    np.rad2deg(coarse_profile.orientation(depth)),
    ":",
    color="C3",
    label=f"fit: num_slabs={COARSE_NUM_SLABS}",
)
ax.set_ylabel("Molecular tilt (deg)")
ax.set_xlabel("Depth from vacuum interface (A)")
ax.legend(fontsize="small")
ax.set_title("Recovered orientation profile: well-resolved vs. coarse mesh")
fig.tight_layout()
plt.show()
