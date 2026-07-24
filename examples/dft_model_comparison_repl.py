"""Direct comparison: refloxide vs. the legacy DFT-fit GlobalObjective.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/dft_model_comparison_repl.py

``@models/xrr/znpc/dft/dft_en_offset_new2.pkl`` (sibling ``refl-analysis``
checkout) is a 5.4 GB pickled ``refnx.analysis.GlobalObjective`` of 21
``pyref.fitting.fitters.AnisotropyObjective`` terms (one per energy), each
wrapping a ``pyref.fitting.model.ReflectModel`` -- the legacy per-energy
fitting pattern refloxide's own ``Objective``/``ReflectModel`` replace.
Unpickling it needs ``pyref``/``refnx``-legacy, which refloxide's own
environment intentionally does not depend on (see AGENTS.md: "drop pyref
and refloxide.integrations").

So this comparison is two stages:

1. A one-time extraction (**not** part of this script -- run once inside
   the refl-analysis venv, which has pyref installed) pulls out a portable
   summary: the anchor-energy (283.7 eV, the carbon K-edge pi* resonance)
   structure geometry -- every other energy's slab parameters are refnx
   constraints pointing at that one, confirmed by inspecting the pickle
   directly -- the per-energy instrument corrections, and the measured
   data plus the legacy model's own predicted curves at every (energy,
   q) point. Saved as ``dft_en_offset_new2_summary.json`` +
   ``dft_en_offset_new2_data.parquet`` next to the source pickle.
2. **This script** loads only those two lightweight files (no pyref
   needed), rebuilds the identical structure/corrections in refloxide,
   and compares its predictions against the legacy model's own saved
   predictions at the exact same (energy, q) points -- a true
   apples-to-apples parity check on identical parameter values, not a
   re-fit.

Legacy label note: pyref's ``AnisotropyObjective``/``XrayReflectDataset``
concatenate s+p as one array split at the largest ``q`` gap; the FIRST
chunk is native ``R_pp`` (refloxide ``pol="p"``) and the SECOND is native
``R_ss`` (refloxide ``pol="s"``) -- confirmed directly from
``pyref.fitting.model.ReflectModel.model``'s ``pol='s' -> refl[:, 1, 1]``
mapping. The extraction step already resolves this, so the saved parquet's
``pol`` column is the native refloxide label throughout -- no relabeling
needed here.
"""

# %% 0
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["POLARS_VERBOSE"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from refloxide.data import OpticalConstants
from refloxide.model import MaterialSLD, ReflectModel, UniTensorSLD

# %% Paths

DFT_DIR = Path.home() / "projects/refl-analysis/@models/xrr/znpc/dft"
SUMMARY_PATH = DFT_DIR / "dft_en_offset_new2_summary.json"
DATA_PATH = DFT_DIR / "dft_en_offset_new2_data.parquet"
for path in (SUMMARY_PATH, DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- run the one-time extraction (in the "
            "refl-analysis venv, which has pyref) against "
            f"{DFT_DIR / 'dft_en_offset_new2.pkl'} first"
        )

summary = json.loads(SUMMARY_PATH.read_text())
frame = pl.read_parquet(DATA_PATH)
energies = sorted({c["energy"] for c in summary["corrections"]})
print(f"anchor energy: {summary['anchor_energy']} eV, {len(energies)} energies")
print(f"layers: {[layer['name'] for layer in summary['layers']]}")

# %% Rebuild the identical structure in refloxide (same values, not a re-fit)

znpc_ooc = OpticalConstants.from_file(summary["ooc_csv"])


def build_structure():
    slabs = []
    for layer in summary["layers"]:
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
for layer in summary["layers"]:
    slab = structure.slab(layer["name"])
    assert slab.thick.value == layer["thick"]
    assert slab.rough.value == layer["rough"]
print(structure)

# %% Wire up the identical shared + per-energy corrections

energy_offsets = {c["energy_offset"] for c in summary["corrections"]}
assert len(energy_offsets) == 1, "expected one shared energy_offset across all energies"
model.energy_offset.value = energy_offsets.pop()

for c in summary["corrections"]:
    e = c["energy"]
    # scale_s/scale_p do NOT need the legacy pol-label inversion: pyref's
    # own reflectivity() multiplies scale_s * refl[:, 0, 0] (R_ss, native
    # "s") and scale_p * refl[:, 1, 1] (R_pp, native "p") unconditionally,
    # before `.model()`'s pol-selection logic ever runs -- so they map
    # straight across. theta_offset_s/theta_offset_p DO need the inversion:
    # `.model()` picks theta_offset_s to build the q grid whenever
    # `self.pol == "s"` is requested, but that branch's OUTPUT is
    # `refl[:, 1, 1]` (R_pp, native "p") -- so legacy theta_offset_s is
    # actually the shift for refloxide's native p channel, and vice versa.
    # `bkg` is unscoped by channel on both sides, so it needs no relabeling.
    model.scale_s.at(e).value = c["scale_s"]
    model.scale_p.at(e).value = c["scale_p"]
    model.theta_offset_p.at(e).value = c["theta_offset_s"]
    model.theta_offset_s.at(e).value = c["theta_offset_p"]
    model.bkg.at(e).value = c["bkg"]
    assert c["dq"] == 0.0, "smearing not exercised by this comparison"
    assert c["q_offset"] == 0.0, "q_offset not exercised by this comparison"

# %% Evaluate refloxide at the exact (energy, q) points the legacy model was
# evaluated at, and compare against its saved predictions directly.

rows = []
for energy in energies:
    group = frame.filter(pl.col("energy") == energy)
    for pol in ("s", "p"):
        sub = group.filter(pl.col("pol") == pol)
        if sub.is_empty():
            continue
        q = sub["q"].to_numpy()
        r = sub["r"].to_numpy()
        r_err = sub["r_err"].to_numpy()
        legacy_pred = sub["legacy_pred"].to_numpy()

        result = model(q, energy)
        refloxide_pred = result.s if pol == "s" else result.p

        rel_dev = np.abs(refloxide_pred - legacy_pred) / np.abs(legacy_pred)
        rows.append(
            {
                "energy": energy,
                "pol": pol,
                "n": len(q),
                "max_rel_dev_vs_legacy": float(np.max(rel_dev)),
                "max_rel_dev_vs_data": float(
                    np.max(np.abs(refloxide_pred - r) / r_err)
                ),
            }
        )

report = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1):
    print(report)

worst = float(report["max_rel_dev_vs_legacy"].max())  # ty: ignore[invalid-argument-type]
print(f"\nworst refloxide-vs-legacy relative deviation, all energies: {worst:.3e}")
assert worst < 1e-6, "refloxide and the legacy pyref model disagree beyond float noise"
print("OK: refloxide reproduces the legacy DFT-fit GlobalObjective's own predictions.")

# %% Overlay data vs. legacy vs. refloxide at a few representative energies

show_energies = (250.0, 283.7, 285.1)
fig, axes = plt.subplots(len(show_energies), 2, figsize=(11, 3.2 * len(show_energies)))
for row, energy in enumerate(show_energies):
    group = frame.filter(pl.col("energy") == energy)
    for col, pol in enumerate(("s", "p")):
        ax = axes[row, col]
        sub = group.filter(pl.col("pol") == pol)
        q = sub["q"].to_numpy()
        r = sub["r"].to_numpy()
        r_err = sub["r_err"].to_numpy()
        legacy_pred = sub["legacy_pred"].to_numpy()
        result = model(q, energy)
        refloxide_pred = result.s if pol == "s" else result.p

        ax.errorbar(
            q, r, yerr=r_err, fmt=".", ms=2, lw=0.5, color="0.4", label="data"
        )
        ax.plot(q, legacy_pred, color="k", ls="--", lw=1.5, label="legacy (pyref)")
        ax.plot(q, refloxide_pred, color="C0", lw=1.2, label="refloxide")
        ax.set_yscale("log")
        ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        ax.set_title(f"{energy:.1f} eV, {pol}-pol")
        if col == 0:
            ax.set_ylabel("Reflectivity")
        ax.legend(fontsize="x-small")
fig.suptitle(
    "DFT-fit GlobalObjective: legacy pyref vs. refloxide, same parameters", y=1.01
)
fig.tight_layout()
plt.show()

# %% Depth profile of the reproduced structure

structure.plot.oc(summary["anchor_energy"], difference=True)
plt.show()
structure.plot.param("density|orientation", roughness=True)
plt.show()
