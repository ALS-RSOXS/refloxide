"""Direct comparison: refloxide vs. the legacy "free tensor" GlobalObjective.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/free_tensor_model_comparison_repl.py

Same two-stage approach as ``dft_model_comparison_repl.py`` (see that
script's docstring for the general pattern and the confirmed pyref label
conventions this reuses) applied to a second legacy pickle:
``@models/xrr/znpc/free/free_en_offset_init_2.pkl`` -- a "free tensor"
GlobalObjective where the Surface/ZnPc/Contamination layers are not tied to
one tabulated dispersion + rotation (``UniTensorSLD``) but instead let each
measured energy's diagonal tensor float independently
(``pyref.fitting.structure.SLD``), a model-independent check of whether the
DFT-model comparison's assumed uniaxial+rotation model was actually
justified.

Confirmed directly by inspecting the pickle -- not assumed -- this
particular legacy object mixes scatterer classes inconsistently: "ZnPc" is
``UniTensorSLD`` at 250 eV specifically and ``SLD`` (free) at every other
energy, apparently an artifact of how it was assembled rather than an
intentional distinction. The one-time extraction step (run in the
refl-analysis venv; see ``refl-analysis/scripts/
extract_free_tensor_globalobjective.py``) sidesteps that inconsistency
entirely: it reads each layer's own RESOLVED diagonal tensor
(``slab.sld.tensor``) at each energy directly, regardless of which
scatterer class produced it, dropping the lab-frame "yy" entry that
neither engine's uniaxial solver ever reads (confirmed against
``src/uniaxial.rs``'s ``compute_eigenstructure``, which only consults
``eps[(0,0)]``/``eps[(2,2)]``). Every layer -- including the plain
isotropic ones, whose ``MaterialSLD``-style tensor is just `n * I`, so
`delta_o == delta_e` trivially -- is represented here by refloxide's new
``FreeTensorSLD``: one independent (delta_o, beta_o, delta_e, beta_e) per
registered energy, no OOC table or formula connecting them, which is the
right shape for "whatever this layer resolved to at each energy" whether
or not it was genuinely free-fitted.

Writes/reads the same ``dft_model_comparison_repl.py`` label conventions:
``scale_s``/``scale_p`` map straight across to refloxide; ``theta_offset_s``/
``theta_offset_p`` need the legacy pol-label swap.
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

from refloxide.model import FreeTensorSLD, ReflectModel

# %% Paths

FREE_DIR = Path.home() / "projects/refl-analysis/@models/xrr/znpc/free"
SUMMARY_PATH = FREE_DIR / "free_en_offset_init_2_summary.json"
DATA_PATH = FREE_DIR / "free_en_offset_init_2_data.parquet"
for path in (SUMMARY_PATH, DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- run the one-time extraction (in the "
            "refl-analysis venv, which has pyref) against "
            f"{FREE_DIR / 'free_en_offset_init_2.pkl'} first"
        )

summary = json.loads(SUMMARY_PATH.read_text())
frame = pl.read_parquet(DATA_PATH)
energies = sorted({c["energy"] for c in summary["corrections"]})
print(f"anchor energy: {summary['anchor_energy']} eV, {len(energies)} energies")
print(f"layers: {[layer['name'] for layer in summary['layers']]}")
for layer in summary["layers"]:
    types = {v["scatterer_type"] for v in layer["tensor_by_energy"].values()}
    print(f"  {layer['name']} ({layer['kind']}): legacy scatterer types = {types}")

# %% Rebuild every layer as a FreeTensorSLD (same values, not a re-fit)
#
# One independent (delta_o, beta_o, delta_e, beta_e) per registered energy
# per layer -- no OOC table, rotation, or density scaling connects them,
# which is exactly the shape needed here regardless of whether a given
# layer/energy was genuinely free-fitted (Surface, Contamination, ZnPc at
# every energy but 250 eV) or just happens to be isotropic/uniaxial under
# the hood (Vacuum/Oxide/Substrate, ZnPc at 250 eV).


def build_structure():
    slabs = []
    for layer in summary["layers"]:
        sld = FreeTensorSLD(energies, name=layer["name"])
        for energy_str, values in layer["tensor_by_energy"].items():
            channel = sld.channel_at(float(energy_str))
            channel.delta_o.value = values["delta_o"]
            channel.beta_o.value = values["beta_o"]
            channel.delta_e.value = values["delta_e"]
            channel.beta_e.value = values["beta_e"]
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
#
# scale_s/scale_p map straight across to refloxide; theta_offset_s/
# theta_offset_p need the legacy pol-label swap (see module docstring).

energy_offsets = {c["energy_offset"] for c in summary["corrections"]}
assert len(energy_offsets) == 1, "expected one shared energy_offset across all energies"
model.energy_offset.value = energy_offsets.pop()

for c in summary["corrections"]:
    e = c["energy"]
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
print("OK: refloxide reproduces the legacy free-tensor GlobalObjective's predictions.")

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
    "Free-tensor GlobalObjective: legacy pyref vs. refloxide, same parameters", y=1.01
)
fig.tight_layout()
plt.show()

# %% Free-tensor trace vs. energy per layer -- the model-independent
# equivalent of the DFT model's dispersion curve, straight from the fit
# results rather than a tabulated OOC source.

fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True)
for layer in summary["layers"]:
    es = sorted(float(e) for e in layer["tensor_by_energy"])
    delta_o = [layer["tensor_by_energy"][str(e)]["delta_o"] for e in es]
    delta_e = [layer["tensor_by_energy"][str(e)]["delta_e"] for e in es]
    axes[0].plot(es, delta_o, marker=".", label=layer["name"])
    axes[1].plot(es, delta_e, marker=".", label=layer["name"])
axes[0].set_title(r"$\delta_o$ (ordinary) vs. energy")
axes[1].set_title(r"$\delta_e$ (extraordinary) vs. energy")
for ax in axes:
    ax.set_xlabel("Energy (eV)")
    ax.legend(fontsize="x-small")
axes[0].set_ylabel(r"$\delta$")
fig.suptitle("Per-layer free-tensor dispersion recovered by the legacy fit")
fig.tight_layout()
plt.show()
