"""Port of refl-analysis's ``fit_orientation_graded.ipynb`` -- multi-energy.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/bookended_real_data_fit_repl.py

Fits the book-ended ZnPc orientation/density profile against real XRR data
at ALL 21 energies in the legacy DFT-fit extraction (carbon K-edge region,
250-289 eV), using refloxide's native ``BookendedComponent``/``Objective``
directly -- no ``pyref.fitting`` patching, no
``utils.models``/``utils.refloxide_profile`` layer from refl-analysis.

Companion to ``real_data_repl.py`` (discrete surface/bulk/interface
``UniTensorSLD`` slabs, same real multi-energy dataset and per-energy
correction pattern) and ``bookended_multi_energy_repl.py`` (same book-ended
multi-energy mechanics, but on synthetic data). Here, the DFT-fit template's
three discrete slabs at its 283.7 eV anchor energy are collapsed into one
``BookendedComponent`` wrapping a ``BookendedOrientationProfile`` built via
``refloxide.pxr.energy.bookended.bookended_from_three_slabs`` -- with its
energy left deferred (``energy=None``) so the SAME shared geometry
(``total_thick``, ``tau_si``/``tau_vac``, ``alpha_*``, ``density_*``)
rematerializes its Rust-backed OOC interpolation at every one of the 21
energies `ReflectModel` evaluates, exactly like
``bookended_multi_energy_repl.py``'s deferred-energy profile.

Data + starting geometry come from the same one-time extraction
(``dft_en_offset_new2_summary.json`` / ``_data.parquet``, sibling
``refl-analysis`` checkout) that ``dft_model_comparison_repl.py`` uses --
see that script's docstring for how it was produced and why unpickling the
legacy ``pyref``/``refnx`` ``GlobalObjective`` directly isn't needed (or
possible, per AGENTS.md: refloxide drops the ``pyref`` dependency).

Three things:

1. Rebuild the DFT-fit template's surface/bulk/interface slabs at the 283.7
   eV anchor energy, collapse them into a deferred-energy book-ended
   profile, and compare the two structures' orientation/density profiles
   directly (energy-independent -- shape doesn't change with photon energy).
2. Load the real measured s+p data at all 21 energies and seed a
   `ReflectModel` with one per-energy instrument-correction channel each
   (label-swapped the same way `dft_model_comparison_repl.py` documents);
   the shared `energy_offset` is common to every energy.
3. Two staged `differential_evolution` fits -- shared geometry + densities,
   then opening the substrate-/vacuum-side tilt -- against all 21 energies
   at once, then a final polish pass, each followed by a data-vs-fit
   overlay at a few representative energies.
"""

# %% 0
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

os.environ["POLARS_VERBOSE"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from refnx.analysis import CurveFitter, Transform

from refloxide.data import OpticalConstants, ReflectDataset
from refloxide.model import BookendedComponent, MaterialSLD, ReflectModel, UniTensorSLD
from refloxide.objective import Objective, thread_workers
from refloxide.pxr.energy.bookended import bookended_from_three_slabs
from refloxide.pxr.energy.ooc import OocAnchor

# %% Paths (sibling refl-analysis checkout; same extraction dft_model_comparison_repl.py uses)

DFT_DIR = Path.home() / "projects/refl-analysis/@models/xrr/znpc/dft"
SUMMARY_PATH = DFT_DIR / "dft_en_offset_new2_summary.json"
DATA_PATH = DFT_DIR / "dft_en_offset_new2_data.parquet"
for path in (SUMMARY_PATH, DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- run the one-time extraction (in the "
            "refl-analysis venv, which has pyref) against "
            f"{DFT_DIR / 'dft_en_offset_new2.pkl'} first, or see "
            "dft_model_comparison_repl.py for the extraction story"
        )

summary = json.loads(SUMMARY_PATH.read_text())
ANCHOR_ENERGY = float(summary["anchor_energy"])  # 283.7 eV; seeds the shared geometry
NUM_SLABS_FIT = 24
NUM_SLABS_PLOT = 100
SHOW_ENERGIES = (250.0, 283.7, 285.1)  # low, anchor, and pi*-resonance-adjacent
print(
    f"anchor energy: {ANCHOR_ENERGY} eV, "
    f"layers: {[layer['name'] for layer in summary['layers']]}"
)

# %% Rebuild the DFT-fit template structure at the anchor energy (same values, not a re-fit)
#
# Two distinct optical-constant wrappers share the same underlying CSV:
# `OpticalConstants` (discrete `UniTensorSLD` slabs, as in
# `dft_model_comparison_repl.py`) and `OocAnchor` (the book-ended profile's
# own Rust-side linear interpolator, as in `bookended_multi_energy_repl.py`).

znpc_oc = OpticalConstants.from_file(summary["ooc_csv"])
znpc_anchor = OocAnchor.from_file(summary["ooc_csv"])


def build_template_structure():
    slabs = []
    for layer in summary["layers"]:
        if layer["kind"] == "uniaxial":
            sld = UniTensorSLD(
                znpc_oc,
                density=layer["density"],
                rotation=layer["rotation"],
                name=layer["name"],
            )
        else:
            sld = MaterialSLD(
                layer["formula"], density=layer["density"], name=layer["name"]
            )
        slabs.append(sld(layer["thick"], layer["rough"]))
    structure = slabs[0]
    for slab in slabs[1:]:
        structure = structure | slab
    return structure


template_structure = build_template_structure()
surface_slab = template_structure.slab("Surface")
bulk_slab = template_structure.slab("ZnPc")
interface_slab = template_structure.slab("Contamination")
oxide_layer = next(layer for layer in summary["layers"] if layer["name"] == "Oxide")
substrate_layer = next(
    layer for layer in summary["layers"] if layer["name"] == "Substrate"
)

# %% Collapse surface/bulk/interface into one book-ended profile, energy deferred
#
# `energy=None` (bookended_from_three_slabs's own default): this profile's
# total_thick/tau_*/alpha_*/density_* shape is shared across every energy the
# multi-energy `ReflectModel` below evaluates -- only the OOC lookup inside
# `rows_and_tensors_at` changes per energy, same deferred-energy contract
# `bookended_multi_energy_repl.py` demonstrates on synthetic data.

profile_fit = bookended_from_three_slabs(
    surface_slab,
    bulk_slab,
    interface_slab,
    znpc_anchor,
    energy=None,
    num_slabs=NUM_SLABS_FIT,
    mesh_constant=0.1,
    name="ZnPc",
)
profile_plot = bookended_from_three_slabs(
    surface_slab,
    bulk_slab,
    interface_slab,
    znpc_anchor,
    energy=ANCHOR_ENERGY,
    num_slabs=NUM_SLABS_PLOT,
    mesh_constant=0.1,
    name="ZnPc",
)
print(
    f"book-ended profile: total_thick={profile_fit.total_thick.value:.2f} A, "
    f"alpha_bulk={profile_fit.alpha_bulk.value:.3f} rad"
)


def build_structure(profile):
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    oxide = MaterialSLD("SiO2", density=oxide_layer["density"], name="oxide")(
        oxide_layer["thick"], oxide_layer["rough"]
    )
    substrate = MaterialSLD("Si", density=substrate_layer["density"], name="substrate")(
        substrate_layer["thick"], substrate_layer["rough"]
    )
    return vacuum | BookendedComponent(profile) | oxide | substrate


structure_fit = build_structure(profile_fit)
structure_plot = build_structure(profile_plot)

# %% Graded book-ended profile vs. the DFT three-slab template it was collapsed from
#
# Orientation/density vs. depth is energy-independent for a book-ended
# profile (only the OOC changes with photon energy), so this comparison
# holds regardless of which energy `structure_fit`/`profile_fit` end up
# evaluated at during the multi-energy fit below.

fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharey=True)
structure_plot.plot.param("density|orientation", ax=axes[0], roughness=True)
axes[0].set_title(f"book-ended profile ({NUM_SLABS_PLOT} slabs)")
template_structure.plot.param("density|orientation", ax=axes[1], roughness=True)
axes[1].set_title("DFT three-slab template")
fig.tight_layout()
plt.show()

# %% Load real s+p data at ALL 21 energies, build the multi-energy model

frame = pl.read_parquet(DATA_PATH)
dataset = ReflectDataset.from_polars(frame)
energies = sorted(frame["energy"].unique().to_list())
print(
    f"ReflectDataset: {len(dataset)} rows, {len(energies)} energies, "
    f"{len(list(dataset.groups()))} (energy, pol) groups"
)

model = ReflectModel(structure_fit, energies=energies, parallel=False)


def freeze_entire_structure(model: ReflectModel) -> None:
    """Freeze every parameter in the structure, not just the profile's.

    `Scatterer.__call__` (`MaterialSLD(...)(thick, rough)`) always sets the
    resulting slab's `thick`/`rough` to `vary=True` by default -- the
    vacuum/oxide/substrate slabs surrounding the profile are not exempt.
    """
    for p in model.structure.parameters.flattened():
        if p.constraint is None:
            p.vary = False


freeze_entire_structure(model)  # oxide/substrate/vacuum + profile all start fixed

# %% Free the profile's shared geometry (one set of values fit against all 21 energies)
#
# oxide/substrate stay fixed at the DFT template's values throughout.

profile_fit.total_thick.setp(vary=True, bounds=(160.0, 210.0))
profile_fit.surface_roughness.setp(vary=True, bounds=(0.0, 10.0))
profile_fit.tau_si.setp(vary=True, bounds=(0.5, 15.0))
profile_fit.tau_vac.setp(vary=True, bounds=(0.5, 30.0))
profile_fit.alpha_bulk.setp(vary=True, bounds=(0.0, np.pi / 2))
profile_fit.density_bulk.setp(vary=True, bounds=(1.0, 2.5))
profile_fit.density_si.setp(value=0.5, vary=True, bounds=(0.0, 1.3))
profile_fit.density_vac.setp(value=2.1, vary=True, bounds=(1.5, 2.5))

# %% Seed per-energy instrument corrections from the DFT-fit template's own values
#
# Legacy pyref label note (see dft_model_comparison_repl.py): scale_s/scale_p
# map straight across, but theta_offset_s/theta_offset_p are inverted
# relative to the native refloxide s/p channels. energy_offset is shared
# across every energy (confirmed identical in every correction below).

energy_offsets = {c["energy_offset"] for c in summary["corrections"]}
assert len(energy_offsets) == 1, "expected one shared energy_offset across all energies"
model.energy_offset.setp(value=energy_offsets.pop(), vary=False)

SCALE_DELTA = 0.15
SCALE_LO, SCALE_HI = 0.75, 1.55
THETA_DELTA = 0.20

for c in summary["corrections"]:
    e = c["energy"]
    sc_s, sc_p = float(c["scale_s"]), float(c["scale_p"])
    th_s0 = float(c["theta_offset_p"])
    th_p0 = float(c["theta_offset_s"])
    model.scale_s.at(e).setp(
        value=sc_s,
        vary=True,
        bounds=(max(SCALE_LO, sc_s - SCALE_DELTA), min(SCALE_HI, sc_s + SCALE_DELTA)),
    )
    model.scale_p.at(e).setp(
        value=sc_p,
        vary=True,
        bounds=(max(SCALE_LO, sc_p - SCALE_DELTA), min(SCALE_HI, sc_p + SCALE_DELTA)),
    )
    model.theta_offset_s.at(e).setp(
        value=th_s0,
        vary=True,
        bounds=(th_s0 - THETA_DELTA, th_s0 + THETA_DELTA),
    )
    model.theta_offset_p.at(e).setp(
        value=th_p0,
        vary=True,
        bounds=(th_p0 - THETA_DELTA, th_p0 + THETA_DELTA),
    )
    model.bkg.at(e).setp(value=0.0, vary=False)

# The manuscript's AnisotropyObjective added an anisotropy_weight=0.5 term, but
# refloxide.objective.Objective's own anisotropy_weight (see its docstring)
# requires s and p to share an IDENTICAL q grid at that energy -- none of
# these 21 real-data energies have matching s/p q grids (e.g. 145 vs 151
# rows at 283.7 eV), so that term does not apply and is left at its 0.0
# default (plain Gaussian log-likelihood).
#
# nc_constraint=False: the legacy DFT-fit template's own frozen Oxide layer
# (thick=6.879, rough=8.127) fails refloxide's default Nevot-Croce prior
# (thick >= sqrt(2*pi)*rough/2 = 10.19) -- oxide.thick/rough never vary in
# this script, so this is a property of the legacy geometry being carried
# over for a faithful comparison, not something a fit could or should fix.
objective = Objective(model, dataset, transform=Transform("logY"), nc_constraint=False)
print(f"logl before fit: {objective.logl():.3f}")
print(f"varying parameters: {len(objective.varying_parameters())}")
print(f"Objective batches: {len(objective._batches)}")
assert np.isfinite(objective.logp())

# %% Data vs. starting (template-derived) book-ended model, at a few representative energies


def plot_data_vs_model(model, title, show_energies=SHOW_ENERGIES):
    fig, axes = plt.subplots(
        len(show_energies), 2, figsize=(11, 3.2 * len(show_energies))
    )
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
                    q, pred.s if pol == "s" else pred.p, color="C0", lw=1.5, label="fit"
                )
            ax.set_yscale("log")
            ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
            ax.set_title(f"{energy:.1f} eV, {pol}-pol")
            if col == 0:
                ax.set_ylabel("Reflectivity")
            ax.legend(fontsize="x-small")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    plt.show()


plot_data_vs_model(model, "Starting model, all 21 energies")

# %% Fit 1: shared geometry + densities, against all 21 energies at once

x0 = [p.value for p in objective.varying_parameters()]
with thread_workers(12) as workers:
    workers.bind(objective)
    CurveFitter(objective).fit(
        method="differential_evolution",
        workers=workers,
        updating="deferred",
        maxiter=50,
        popsize=10,
        polish=False,
        x0=x0,
    )
print(f"logl after fit 1: {objective.logl():.3f}")
plot_data_vs_model(model, "Fit 1: shared geometry + densities")
structure_fit.plot.param("density|orientation", roughness=True)
plt.show()

# %% Fit 2: open the substrate-/vacuum-side tilt and relaxation lengths

profile_fit.tau_si.setp(vary=True, bounds=(0.5, 25.0))
profile_fit.tau_vac.setp(vary=True, bounds=(0.5, 25.0))
profile_fit.alpha_si.setp(vary=True, bounds=(0.0, np.pi / 4))
profile_fit.alpha_vac.setp(vary=True, bounds=(0.0, np.pi / 2))

with thread_workers(12) as workers:
    workers.bind(objective)
    CurveFitter(objective).fit(
        method="differential_evolution",
        workers=workers,
        updating="deferred",
        maxiter=50,
        popsize=10,
        polish=False,
    )
print(f"logl after fit 2: {objective.logl():.3f}")
plot_data_vs_model(model, "Fit 2: + alpha_si, alpha_vac")
structure_fit.plot.param("density|orientation", roughness=True)
plt.show()

# %% Fit 3: final polish, same free parameters as fit 2

x0 = [p.value for p in objective.varying_parameters()]
with thread_workers(12) as workers:
    workers.bind(objective)
    CurveFitter(objective).fit(
        method="differential_evolution",
        workers=workers,
        updating="deferred",
        maxiter=30,
        popsize=10,
        polish=True,
        x0=x0,
    )
print(f"logl after fit 3: {objective.logl():.3f}")
print(f"varying parameters:\n{objective.varying_parameters()}")
plot_data_vs_model(model, "Fit 3: polish")
structure_fit.plot.param("density|orientation", roughness=True)
plt.show()
assert np.isfinite(objective.logl())

# %% Reflectivity overlay: data vs. our fit vs. the legacy DFT-fit's own prediction,
# at the same representative energies used earlier.

fig, axes = plt.subplots(len(SHOW_ENERGIES), 2, figsize=(11, 3.2 * len(SHOW_ENERGIES)))
for row, energy in enumerate(SHOW_ENERGIES):
    legacy_group = frame.filter(pl.col("energy") == energy)
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
            order = np.argsort(legacy_sub["q"].to_numpy())
            ax.plot(
                legacy_sub["q"].to_numpy()[order],
                legacy_sub["legacy_pred"].to_numpy()[order],
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
fig.suptitle("Multi-energy book-ended fit vs. legacy DFT three-slab fit", y=1.01)
fig.tight_layout()
plt.show()

# %% Depth profile: fitted book-ended film vs. the DFT three-slab template

z_fit = structure_fit.depth_grid(num_points=2000, pad=20.0)
z_template = template_structure.depth_grid(num_points=2000, pad=20.0)
_, density_fit = structure_fit.density_profile_at(z_fit, roughness=True)
_, orientation_fit = structure_fit.orientation_profile_at(z_fit, roughness=True)
_, density_template = template_structure.density_profile_at(z_template, roughness=True)
_, orientation_template = template_structure.orientation_profile_at(
    z_template, roughness=True
)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(z_fit, np.rad2deg(orientation_fit), color="C0", label="book-ended fit")
axes[0].plot(
    z_template,
    np.rad2deg(orientation_template),
    color="k",
    ls="--",
    label="DFT template",
)
axes[0].set_title("Orientation")
axes[0].set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
axes[0].set_ylabel("Molecular tilt (deg)")
axes[0].legend(fontsize="x-small")

axes[1].plot(z_fit, density_fit, color="C0", label="book-ended fit")
axes[1].plot(z_template, density_template, color="k", ls="--", label="DFT template")
axes[1].set_title("Density")
axes[1].set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
axes[1].set_ylabel(r"Density (g/cm$^3$)")
axes[1].legend(fontsize="x-small")
fig.tight_layout()
plt.show()

# %% Save the fitted objective (refloxide-native; no pyref/refnx-legacy dependency to reload)

FIT_OUT_DIR = Path.home() / "projects/refl-analysis/@models/xrr/znpc/refloxide"
FIT_OUT_DIR.mkdir(parents=True, exist_ok=True)
FIT_OUT_PATH = FIT_OUT_DIR / "graded_orientation_multi_energy_refit-refloxide.pkl"
with FIT_OUT_PATH.open("wb") as f:
    pickle.dump(objective, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"saved fitted objective to {FIT_OUT_PATH}")
