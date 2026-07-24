"""Hybrid re-fit: seed the DFT model's geometry from the free-tensor model.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/dft_free_hybrid_refit_repl.py

Loads BOTH legacy extractions built by ``dft_model_comparison_repl.py``/
``free_tensor_model_comparison_repl.py`` (see those scripts' docstrings for
the extraction pipeline and label conventions -- this script assumes both
have already been run) and combines them:

1. Rebuild the DFT model's structure exactly as
   ``dft_model_comparison_repl.py`` does (``UniTensorSLD`` for
   Surface/ZnPc/Contamination, ``MaterialSLD`` for Vacuum/Oxide/Substrate).
2. Overwrite EVERY layer's thickness and roughness with the free-tensor
   model's own fitted geometry -- confirmed identical across all 21
   energies in both pickles, so there is exactly one number per layer to
   transplant. Overwrite density too, but only for the three isotropic
   layers (Vacuum/Oxide/Substrate): the free-tensor model's
   Surface/ZnPc/Contamination layers are ``pyref.fitting.structure.SLD``
   (a raw per-energy diagonal tensor -- see the free-tensor extraction's
   docstring), which has no density parameter at all to pull from, so
   those three keep the DFT model's own fitted density/rotation.
3. Loosen four density bounds that were pinned at their edge in the
   original DFT fit (confirmed directly in the source pickle: Surface and
   Contamination both sat at their upper bound of 1.8): tighten
   Oxide/Substrate density to a +/-0.5 window around their (now
   free-model-informed) starting point, drop Contamination's density
   floor to 0, and raise Surface's density ceiling to 3.
4. Build the ``Objective`` against the DFT extraction's own measured
   dataset and confirm everything is well-formed (finite starting
   log-posterior, sane parameter count) -- but do **not** run the fit.
   The actual ``CurveFitter(...).fit(...)`` call is the last cell, left
   for you to run (a full DE + polish pass takes minutes; see
   ``real_data_repl.py`` for expected timings).

Nevot-Croce note: this script relies on ``Objective(..., nc_constraint=True)``
to reject any ``thick < sqrt(2*pi)*rough/2`` candidate during the fit
itself (via ``nll``/``logp_extra``), rather than trying to keep every
layer's thick/rough BOX bounds entirely outside the forbidden region the
way ``real_data_repl.py`` does. That stricter approach doesn't fit here:
Contamination's free-model geometry sits only ~16% above its own NC floor,
so a rough upper bound with any real headroom would force a thick lower
bound above the current starting value itself. `nc_constraint` is the
correctness mechanism either way -- tight box bounds there are only an
optimization (fewer wasted DE evaluations in the infeasible region), not a
safety requirement.
"""

# %% 0
from __future__ import annotations

import json
import os
from pathlib import Path

from pyparsing.helpers import originalTextFor

os.environ["POLARS_VERBOSE"] = "0"

import numpy as np
import polars as pl
from refnx.analysis import CurveFitter, Transform

from refloxide.data import OpticalConstants, ReflectDataset
from refloxide.model import MaterialSLD, ReflectModel, UniTensorSLD
from refloxide.objective import Objective, thread_workers

# %% Paths

DFT_DIR = Path.home() / "projects/refl-analysis/@models/xrr/znpc/dft"
FREE_DIR = Path.home() / "projects/refl-analysis/@models/xrr/znpc/free"
DFT_SUMMARY_PATH = DFT_DIR / "dft_en_offset_new2_summary.json"
DFT_DATA_PATH = DFT_DIR / "dft_en_offset_new2_data.parquet"
FREE_SUMMARY_PATH = FREE_DIR / "free_en_offset_init_2_summary.json"
for path in (DFT_SUMMARY_PATH, DFT_DATA_PATH, FREE_SUMMARY_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- run the one-time extractions (in the "
            "refl-analysis venv, which has pyref): "
            "scripts/extract_dft_globalobjective.py and "
            "scripts/extract_free_tensor_globalobjective.py"
        )

dft_summary = json.loads(DFT_SUMMARY_PATH.read_text())
free_summary = json.loads(FREE_SUMMARY_PATH.read_text())
dft_layers = {layer["name"]: layer for layer in dft_summary["layers"]}
free_layers = {layer["name"]: layer for layer in free_summary["layers"]}
assert set(dft_layers) == set(free_layers), "DFT/free models disagree on layer names"

energies = sorted({c["energy"] for c in dft_summary["corrections"]})
print(f"{len(energies)} energies, layers: {list(dft_layers)}")

# %% Rebuild the DFT structure with the DFT model's OWN starting values --
# identical to dft_model_comparison_repl.py's build_structure().

znpc_ooc = OpticalConstants.from_file(dft_summary["ooc_csv"])


def build_structure():
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
    structure = slabs[0]
    for slab in slabs[1:]:
        structure = structure | slab
    return structure


model = ReflectModel(build_structure(), energies=energies, parallel=False)
structure = model.structure
print(structure)

# %% Transplant geometry (and, for isotropic layers, density) from the
# free-tensor model's own fit -- thick/rough for every layer; density only
# where the free model actually has a density parameter to give.

ISOTROPIC_LAYERS = ("Vacuum", "Oxide", "Substrate")

print("before transplant:")
for name in dft_layers:
    slab = structure.slab(name)
    print(
        f"  {name:15s} thick={slab.thick.value:9.4f}  rough={slab.rough.value:8.4f}  "
        f"density={slab.sld.density.value:7.4f}"
    )

for name, free_layer in free_layers.items():
    slab = structure.slab(name)
    slab.thick.value = free_layer["thick"]
    slab.rough.value = free_layer["rough"]
    if name in ISOTROPIC_LAYERS:
        slab.sld.density.value = free_layer["density"]

print("after transplant:")
for name in dft_layers:
    slab = structure.slab(name)
    print(
        f"  {name:15s} thick={slab.thick.value:9.4f}  rough={slab.rough.value:8.4f}  "
        f"density={slab.sld.density.value:7.4f}"
    )

# NC compliance check (thick >= sqrt(2*pi)*rough/2) on the new geometry --
# the starting point itself must satisfy nc_constraint, not just have it
# available as a safety net.
for name in dft_layers:
    slab = structure.slab(name)
    if not slab.enforce_nevot_croce:
        continue
    thick = float(slab.thick.value or 0.0)
    rough = float(slab.rough.value or 0.0)
    limit = np.sqrt(2.0 * np.pi) * rough / 2.0
    assert thick >= limit, f"{name}: thick={thick:.3f} below NC limit {limit:.3f}"
print("OK: transplanted geometry satisfies Nevot-Croce at the starting point.")

# %% Load the DFT extraction's own measured dataset (native pol labels
# already resolved during extraction -- no legacy Q-reset/relabeling needed
# here, unlike real_data_repl.py's raw-parquet loading path).
#
# Drops the literal q=0 row (283.7 eV, native "p", r=1.0 -- the direct-beam
# normalization point): theta(q=0) is identically 0, so a theta_offset_p
# candidate of exactly 0 -- squarely inside its (-0.8, 0.8) deg bounds
# below, not an edge case -- maps it straight back to q_eff=0 exactly,
# which is a genuine singularity (true normal incidence, kx=ky=0) for the
# Berreman solver, confirmed directly against the Rust kernel: q_eff
# anywhere else, even 1e-6 away from exactly 0, evaluates fine.

frame = pl.read_parquet(DFT_DATA_PATH).select("energy", "pol", "q", "r", "r_err")
n_before = len(frame)
frame = frame.filter(pl.col("q") > 0.0)
n_dropped = n_before - len(frame)
print(f"dropped {n_dropped} row(s) with q <= 0 (direct-beam normalization point)")

dataset = ReflectDataset.from_polars(frame)
print(f"ReflectDataset: {len(dataset)} rows, {len(list(dataset.groups()))} groups")

# %% Freeze everything, then re-open the same varying set as the original
# DFT fit, with bounds re-centered on the free-model-informed starting
# point (see module docstring for why these are deliberately not also
# NC-safe box bounds).

for param in model.parameters.flattened():
    param.vary = False


def window(value: float, low_frac: float, high_frac: float) -> tuple[float, float]:
    """`(value * low_frac, value * high_frac)`, a reasonable exploration window
    around the just-transplanted starting value -- not a Nevot-Croce safety
    margin (`nc_constraint` handles that; see module docstring)."""
    return value * low_frac, value * high_frac


surface = structure.slab("Surface")
bulk = structure.slab("ZnPc")
interface = structure.slab("Contamination")
oxide = structure.slab("Oxide")
substrate = structure.slab("Substrate")

bulk.thick.setp(vary=True, bounds=window(bulk.thick.value, 0.5, 1.8))
bulk.rough.setp(vary=True, bounds=window(bulk.rough.value, 0.0, 1.5))
bulk.sld.density.setp(vary=True, bounds=window(bulk.sld.density.value, 0.75, 1.15))
bulk.sld.rotation.setp(vary=True, bounds=(0.0, np.pi / 2))

surface.thick.setp(vary=True, bounds=window(surface.thick.value, 0.5, 1.8))
surface.rough.setp(vary=True, bounds=window(surface.rough.value, 0.0, 1.5))
surface.sld.rotation.setp(vary=True, bounds=(0.0, np.pi / 2))

interface.thick.setp(vary=True, bounds=window(interface.thick.value, 0.5, 1.8))
interface.rough.setp(vary=True, bounds=window(interface.rough.value, 0.0, 1.5))
interface.sld.rotation.setp(vary=True, bounds=(0.0, np.pi / 2))

oxide.thick.setp(vary=True, bounds=window(oxide.thick.value, 0.5, 1.8))
oxide.rough.setp(vary=True, bounds=window(oxide.rough.value, 0.0, 1.5))

substrate.rough.setp(vary=True, bounds=window(substrate.rough.value, 0.0, 1.5))

# %% The four explicitly-requested density bound changes.
#
# Surface/Contamination both sat pinned at their original upper bound of
# 1.8 in the source pickle (confirmed directly, not assumed) -- loosen the
# specific edge each one was stuck against. Oxide/Substrate get a tight
# +/-0.5 window around their free-model-informed starting density instead
# of the auto-generated (0, 5*density) default.

oxide_density = oxide.sld.density.value
substrate_density = substrate.sld.density.value
oxide.sld.density.setp(vary=True, bounds=(oxide_density - 0.5, oxide_density + 0.5))
substrate.sld.density.setp(
    vary=True, bounds=(substrate_density - 0.5, substrate_density + 0.5)
)
interface.sld.density.setp(vary=True, bounds=(0.0, interface.sld.density.bounds.ub))
surface.sld.density.setp(vary=True, bounds=(surface.sld.density.bounds.lb, 3.0))

print("density bounds after the explicit overrides:")
density_layers = (
    ("Surface", surface),
    ("Contamination", interface),
    ("Oxide", oxide),
    ("Substrate", substrate),
)
for name, slab in density_layers:
    b = slab.sld.density.bounds
    print(
        f"  {name:15s} value={slab.sld.density.value:.4f}  "
        f"bounds=[{b.lb:.4f}, {b.ub:.4f}]"
    )

# %% Shared + per-energy instrument corrections -- DFT model's own starting
# values (unchanged), same varying set as real_data_repl.py.

energy_offsets = {c["energy_offset"] for c in dft_summary["corrections"]}
assert len(energy_offsets) == 1, "expected one shared energy_offset across all energies"
model.energy_offset.setp(value=energy_offsets.pop(), vary=True, bounds=(-0.5, 0.5))

for c in dft_summary["corrections"]:
    e = c["energy"]
    model.scale_s.at(e).setp(value=c["scale_s"], vary=True, bounds=(0.6, 1.4))
    model.scale_p.at(e).setp(value=c["scale_p"], vary=True, bounds=(0.6, 1.4))
    model.theta_offset_s.at(e).setp(
        value=c["theta_offset_s"], vary=True, bounds=(-0.8, 0.8)
    )
    model.theta_offset_p.at(e).setp(
        value=c["theta_offset_p"], vary=True, bounds=(-0.8, 0.8)
    )
    model.bkg.at(e).value = c["bkg"]

model.parallel = False  # DE thread_workers parallelizes the population instead

# %% Build the Objective and confirm everything is well-formed -- but do
# NOT run the fit here.

objective = Objective(model, dataset, transform=Transform("logY"), nc_constraint=True)
print(f"nc_constraint={objective.nc_constraint}")
print(f"varying parameters: {len(objective.varying_parameters())}")
logp_start = objective.logp()
logl_start = objective.logl()
print(f"logp at start: {logp_start:.3f}")
print(f"logl at start: {logl_start:.3f}")
assert np.isfinite(logp_start), "starting point must satisfy NC + bounds"
assert np.isfinite(logl_start)
print("OK: objective is well-formed and ready to fit.")

# %% Re-run the fit (NOT executed as part of building this file -- run this
# cell yourself; a full DE + polish pass takes minutes, see
# real_data_repl.py for expected timings on this machine).

with thread_workers(8) as workers:
    workers.bind(objective)
    CurveFitter(objective).fit(
        method="differential_evolution",
        popsize=12,
        polish=True,
        seed=1,
        workers=workers,
        updating="deferred",
    )

logl_after = float(objective.logl())
print(f"logl after fit: {logl_after:.3f} (delta = {logl_after - logl_start:.3f})")
assert np.isfinite(objective.logp()), "fit exited outside NC/bounds support"

# %% Plot the fit results and compare to the original DFT fit. 

print(objective.varying_parameters())

fit_structure = objective.model.structure
original_structure = build_structure()

# original_structure.plot.property("orientation|density", roughness=True)
fit_structure.plot.param("orientation|density", roughness=True)
