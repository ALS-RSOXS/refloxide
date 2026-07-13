"""Interactive comparison: refloxide vs. python.model, a graded ZnPc film.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/uni_tensor_znpc_repl.py

Same layout as ``model_objective_repl.py``, but the film is three real,
DFT-derived uniaxial sublayers instead of one isotropic `MaterialSLD`: zinc
phthalocyanine (ZnPc) is a planar molecule whose optical response along the
macrocycle plane (`xx`) differs sharply from the response normal to it
(`zz`), especially at the carbon 1s -> pi* resonance near the carbon K-edge
(~285 eV). The film is modeled as a graded stack -- surface / bulk /
interface -- each its own `UniTensorSLD` with a distinct density and
molecular tilt, which is the realistic picture for an as-deposited organic
film (reoriented near the free surface and near the substrate, uniform in
between) and exercises the optical-constants lookup path three times per
model evaluation instead of once, making any per-scatterer overhead far more
visible in the speed comparison below.

Both refloxide's `UniTensorSLD` and py's `UniTensorSLD` implement the same
density-scaled, rotation-mixed uniaxial tensor
(`refloxide.optics.uniaxial_lab_tensor` matches python.model's `n_o`/`n_e` formula
term for term), so this is a genuine independent-implementation parity check,
not an isotropic-vs-anisotropic strawman.

Data: `dft.csv` from the sibling `refl-analysis` checkout
(`@models/optical/znpc/dft.csv`, columns `energy, n_xx, n_zz, n_ixx, n_izz`)
-- referenced directly from that checkout, not copied into refloxide, since
it is real DFT output owned by that project.

Five things, same physical structure (vacuum / ZnPc surface (30/3 A) / ZnPc
bulk (130/4 A) / ZnPc interface (31/3 A) / SiO2 substrate, 285.1 eV -- the
carbon K-edge pi* resonance, where the DFT table's xx/zz dichroism is
largest):

1. Numeric parity -- refloxide.model.ReflectModel vs
   refloxide.python.model.ReflectModel (``refloxide.python.tmm``).
2. Speed -- same two, timed.
3. Fitting -- refloxide.objective.Objective vs
   refloxide.python.model.AnisotropyObjective, fit against a synthetic noisy s+p
   dataset.
4. NEXAFS-style energy scan at fixed q across the resonance -- refloxide vs
   py, both should reproduce the same resonance peak from the DFT table.
5. Rotation (molecular tilt) sweep of the bulk sublayer at fixed (q, energy)
   -- refloxide vs py, both should shift the same way as orientation
   changes.

Note on s/p labeling: identical convention/inversion as
`model_objective_repl.py` -- python.model's `pol='s'` reads the kernel's
`[:, 1, 1]`, `pol='p'` reads `[:, 0, 0]`; `refloxide.model.Reflectivity` uses
the native, non-inverted labeling (`.s = [:, 0, 0]`, `.p = [:, 1, 1]`). So
"py `pol='s'`" corresponds to "refloxide `.p`", and vice versa --
accounted for explicitly below, not a bug.
"""
# %% 0
from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import refloxide.python.model as py
from refnx.analysis import CurveFitter

from refloxide.data import ReflectDataset
from refloxide.model import MaterialSLD, ReflectModel, UniTensorSLD
from refloxide.objective import Objective

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

ENERGY_EV = 285.1  # carbon K-edge pi* resonance -- largest xx/zz dichroism in dft.csv
Q = np.linspace(0.001, 0.25, 200)

# Graded ZnPc film: three sublayers, each its own density and molecular tilt.
SURFACE = {"thick": 30, "rough": 3, "density": 1.61, "rotation": 0.0}
BULK = {"thick": 130, "rough": 4, "density": 1.61, "rotation": 1.35}
INTERFACE = {"thick": 31, "rough": 3, "density": 1.55, "rotation": 0.5}

# %% Shared physical structure, built two ways


def build_refloxide_structure(bulk_rotation: float = BULK["rotation"]):
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    surface = UniTensorSLD(
        ZNPC_DFT_CSV,
        density=SURFACE["density"],
        rotation=SURFACE["rotation"],
        name="ZnPc_surface",
    )(SURFACE["thick"], SURFACE["rough"])
    bulk = UniTensorSLD(
        ZNPC_DFT_CSV, density=BULK["density"], rotation=bulk_rotation, name="ZnPc_bulk"
    )(BULK["thick"], BULK["rough"])
    interface = UniTensorSLD(
        ZNPC_DFT_CSV,
        density=INTERFACE["density"],
        rotation=INTERFACE["rotation"],
        name="ZnPc_interface",
    )(INTERFACE["thick"], INTERFACE["rough"])
    substrate = MaterialSLD("SiO2", density=2.2, name="substrate")(0, 3)
    return vacuum | surface | bulk | interface | substrate


def build_py_structure(bulk_rotation: float = BULK["rotation"]):
    ooc_pd = pd.read_csv(ZNPC_DFT_CSV)
    vacuum = py.MaterialSLD("", density=0.0, energy=ENERGY_EV, name="vacuum")
    surface = py.UniTensorSLD(
        ooc_pd,
        density=SURFACE["density"],
        rotation=SURFACE["rotation"],
        energy=ENERGY_EV,
        name="ZnPc_surface",
    )
    bulk = py.UniTensorSLD(
        ooc_pd,
        density=BULK["density"],
        rotation=bulk_rotation,
        energy=ENERGY_EV,
        name="ZnPc_bulk",
    )
    interface = py.UniTensorSLD(
        ooc_pd,
        density=INTERFACE["density"],
        rotation=INTERFACE["rotation"],
        energy=ENERGY_EV,
        name="ZnPc_interface",
    )
    substrate = py.MaterialSLD("SiO2", density=2.2, energy=ENERGY_EV, name="substrate")
    return (
        vacuum(0, 0)
        | surface(SURFACE["thick"], SURFACE["rough"])
        | bulk(BULK["thick"], BULK["rough"])
        | interface(INTERFACE["thick"], INTERFACE["rough"])
        | substrate(0, 3)
    )


refloxide_model = ReflectModel(build_refloxide_structure())
py_model = py.ReflectModel(build_py_structure(), energy=ENERGY_EV, pol="sp")

# %% 1. Numeric parity

py_model.pol = "s"
py_s = py_model.model(Q)  # native kernel [:, 1, 1]
py_model.pol = "p"
py_p = py_model.model(Q)  # native kernel [:, 0, 0]

refloxide_r = refloxide_model(Q, ENERGY_EV)

max_err_s = np.max(np.abs(refloxide_r.p - py_s))  # refloxide .p <-> py pol='s'
max_err_p = np.max(np.abs(refloxide_r.s - py_p))  # refloxide .s <-> py pol='p'
print(f"max |refloxide.p - py(pol='s')| = {max_err_s:.3e}")
print(f"max |refloxide.s - py(pol='p')| = {max_err_p:.3e}")
assert max_err_s < 1e-8
assert max_err_p < 1e-8
print("Numeric parity OK: refloxide graded ZnPc film matches python.model exactly.\n")

# %% Plot overlay

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Q, refloxide_r.p, label="refloxide .p (== py pol='s')", lw=2)
ax.plot(Q, py_s, "--", label="py pol='s'", lw=1.5, color="k")
ax.plot(Q, refloxide_r.s, label="refloxide .s (== py pol='p')", lw=2)
ax.plot(Q, py_p, "--", label="py pol='p'", lw=1.5, color="0.4")
ax.set_yscale("log")
ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
ax.set_ylabel("Reflectivity")
ax.legend()
ax.set_title(f"refloxide vs python.model, graded ZnPc film, {ENERGY_EV:.1f} eV")
fig.tight_layout()
plt.show()

# %% Structure visualization -- optical constants, density, and orientation vs depth
#
# `Structure.plot.oc` broadens each interface by an error function with
# sigma = that interface's own roughness (the standard NC-consistent
# SLD-profile convention), plus the xx/zz dichroism on a right-hand twin
# axis (`difference=True`, `inset=True` -- same convention as python.model's
# `Structure.plot`). `Structure.plot.param` walks the graded UniTensorSLD
# sublayers for "density"/"orientation" (surface reoriented flat, bulk
# tilted, interface partially relaxed) -- NaN only for the isotropic
# vacuum/substrate bookends, which have no tilt.

refloxide_structure = refloxide_model.structure
refloxide_structure.plot.oc(ENERGY_EV, difference=True)
plt.show()
refloxide_structure.plot.param("density|orientation")
plt.show()

# %% 2. Speed comparison -- vs. stock, python.model (its own pure-Python kernel)


def time_it(fn, n: int = 200, warmup: int = 5) -> float:
    """Mean seconds per call, after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


t_refloxide = time_it(lambda: refloxide_model(Q, ENERGY_EV))
t_py = time_it(lambda: py_model.model(Q))

print(f"refloxide.model.ReflectModel:                 {t_refloxide * 1e3:.4f} ms/call")
print(f"refloxide.python.model.ReflectModel:  {t_py * 1e3:.4f} ms/call")
print(f"speedup: {t_py / t_refloxide:.1f}x\n")

# %% Memory footprint -- peak Python-heap bytes per call, refloxide vs python.model


def peak_memory_bytes(fn, warmup: int = 5) -> int:
    """Peak Python-heap bytes traced during one call, after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    tracemalloc.start()
    fn()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


mem_refloxide = peak_memory_bytes(lambda: refloxide_model(Q, ENERGY_EV))
mem_py = peak_memory_bytes(lambda: py_model.model(Q))

print(f"refloxide.model.ReflectModel:                 {mem_refloxide:>7,} B/call")
print(f"refloxide.python.model.ReflectModel:  {mem_py:>7,} B/call")
print(f"memory ratio (py/refloxide): {mem_py / mem_refloxide:.2f}x\n")

# %% 3. Fitting comparison -- synthetic noisy s+p dataset, fit both ways

rng = np.random.default_rng(0)
py_model.pol = "s"
r_s_true = py_model.model(Q)
py_model.pol = "p"
r_p_true = py_model.model(Q)
py_model.pol = "sp"

r_s = r_s_true * (1 + rng.normal(0, 0.01, size=Q.shape))
r_p = r_p_true * (1 + rng.normal(0, 0.01, size=Q.shape))
err_s = r_s_true * 0.02
err_p = r_p_true * 0.02

# %% ... refloxide side

new_model = ReflectModel(build_refloxide_structure())
new_data = ReflectDataset(
    q=np.concatenate([Q, Q]),
    energy=np.full(2 * len(Q), ENERGY_EV),
    pol=np.concatenate(
        [np.full(Q.shape, "s", dtype=object), np.full(Q.shape, "p", dtype=object)]
    ),
    r=np.concatenate([r_p, r_s]),  # label swap: refloxide "s" <- py p-channel data
    r_err=np.concatenate([err_p, err_s]),
)
new_objective = Objective(new_model, new_data, anisotropy_weight=0.4)

for p in new_model.structure.parameters.flattened():
    if p.constraint is None:
        p.vary = False
new_bulk = new_model.structure.components[2]
new_bulk.thick.setp(vary=True, bounds=(80, 200))

new_fitter = CurveFitter(new_objective)

# %% ... python.model side

py_dataset = py.XrayReflectDataset(
    (
        np.concatenate([Q, Q]),
        np.concatenate([r_s, r_p]),
        np.concatenate([err_s, err_p]),
    )
)
py_objective = py.AnisotropyObjective(
    py_model, py_dataset, logp_anisotropy_weight=0.4
)

for p in py_model.structure.parameters.flattened():
    if p.constraint is None:
        p.vary = False
py_bulk = py_model.structure[2]
py_bulk.thick.setp(vary=True, bounds=(80, 200))

py_fitter = py.CurveFitter(py_objective)

# %% Compare logl before fitting, then fit both and compare timing/result

print("logl before fit:")
print("  refloxide:", new_objective.logl())
print("  py:    ", py_objective.logl())

t0 = time.perf_counter()
new_fitter.fit(method="differential_evolution", maxiter=40, polish=False, seed=1)
t_new_fit = time.perf_counter() - t0

t0 = time.perf_counter()
py_fitter.fit(method="differential_evolution", maxiter=40, polish=False, seed=1)
t_py_fit = time.perf_counter() - t0

print(
    f"\nrefloxide fit: {t_new_fit:.3f} s, "
    f"recovered bulk thick = {new_bulk.thick.value:.2f}"
)
print(
    f"py fit:     {t_py_fit:.3f} s, "
    f"recovered bulk thick = {py_bulk.thick.value:.2f}"
)
print(f"fit speedup: {t_py_fit / t_new_fit:.1f}x (true bulk thickness was 130.0)\n")

# %% 4. NEXAFS-style energy scan at fixed q across the resonance -- refloxide vs py

Q_FIXED = 0.05
ENERGY_SCAN = np.linspace(270.0, 300.0, 150)

scan_refloxide_model = ReflectModel(build_refloxide_structure())
scan_py_model = py.ReflectModel(build_py_structure(), energy=ENERGY_EV, pol="sp")

s_scan_refloxide = np.array(
    [scan_refloxide_model(Q_FIXED, e).p for e in ENERGY_SCAN]
).ravel()
p_scan_refloxide = np.array(
    [scan_refloxide_model(Q_FIXED, e).s for e in ENERGY_SCAN]
).ravel()

s_scan_py = []
p_scan_py = []
for e in ENERGY_SCAN:
    scan_py_model.energy = e
    scan_py_model.pol = "s"
    s_scan_py.append(scan_py_model.model(np.array([Q_FIXED]))[0])
    scan_py_model.pol = "p"
    p_scan_py.append(scan_py_model.model(np.array([Q_FIXED]))[0])
s_scan_py = np.asarray(s_scan_py)
p_scan_py = np.asarray(p_scan_py)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ENERGY_SCAN, s_scan_refloxide, label="refloxide .p (== py pol='s')", lw=2)
ax.plot(ENERGY_SCAN, s_scan_py, "--", label="py pol='s'", lw=1.5, color="k")
ax.plot(ENERGY_SCAN, p_scan_refloxide, label="refloxide .s (== py pol='p')", lw=2)
ax.plot(ENERGY_SCAN, p_scan_py, "--", label="py pol='p'", lw=1.5, color="0.4")
ax.axvline(ENERGY_EV, color="k", lw=0.8, ls=":", label="C 1s -> pi*")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel(f"Reflectivity at q={Q_FIXED}")
ax.legend(fontsize="small")
ax.set_title("NEXAFS-style energy scan: refloxide vs py, both see the resonance")
fig.tight_layout()
plt.show()

# %% 5. Bulk-sublayer rotation (molecular tilt) sweep -- refloxide vs py

rotations = np.linspace(0.0, np.pi / 2, 5)
fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for rotation, color in zip(rotations, colors, strict=False):
    r_rot_refloxide = ReflectModel(build_refloxide_structure(bulk_rotation=rotation))(
        Q, ENERGY_EV
    )
    rot_py_model = py.ReflectModel(
        build_py_structure(bulk_rotation=rotation), energy=ENERGY_EV, pol="p"
    )
    r_rot_py = rot_py_model.model(Q)  # native [:, 0, 0] == refloxide .s

    ax.plot(
        Q,
        r_rot_refloxide.s,
        color=color,
        label=f"refloxide bulk rotation={rotation:.2f}",
    )
    ax.plot(Q, r_rot_py, "--", color=color, lw=1.2)
ax.set_yscale("log")
ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
ax.set_ylabel("s-pol reflectivity (refloxide .s == py pol='p')")
ax.legend(fontsize="small")
ax.set_title("Bulk molecular tilt sweep: refloxide (solid) vs py (dashed)")
fig.tight_layout()
plt.show()
