"""Interactive comparison: refloxide.model/objective vs. stock pyref.fitting.

Run cell-by-cell in an interactive Python window (each ``# %%`` marker is one
cell — VS Code's "Python Interactive" / Jupytext percent format), or
top-to-bottom with::

    uv run python examples/model_objective_repl.py

Three things, same physical structure (vacuum / SiO2 film / Si substrate,
700 eV):

1. Numeric parity — refloxide.model.ReflectModel vs. stock, UNPATCHED
   pyref.fitting.ReflectModel (pyref's own pure-Python kernel,
   pyref.fitting.uniaxial — no Rust involved unless patched).
2. Speed — same two, timed. This is the actual "core fault" refloxide
   exists to fix: stock pyref never touches Rust unless patch_pyref() is
   called. A third timing (pyref, patched) isolates how much of the
   speedup is "just Rust" vs. the new API/batching on top of it.
3. Fitting — refloxide.objective.Objective vs. stock
   pyref.fitting.AnisotropyObjective, built from the same synthetic noisy
   s+p dataset, compared by log-likelihood and by a short
   differential_evolution fit.

Note on s/p labeling: stock pyref's ``pol='s'``/``pol='p'`` extraction is a
historical inversion kept for legacy dataset compatibility (``pol='s'``
reads the kernel's ``[:, 1, 1]``, ``pol='p'`` reads ``[:, 0, 0]``).
``refloxide.model.Reflectivity`` uses the native, non-inverted kernel
labeling (``.s = [:, 0, 0]``, ``.p = [:, 1, 1]``, matching ``rust.pyi``'s own
docs). So "pyref ``pol='s'``" corresponds to "refloxide ``.p``", and vice
versa — accounted for explicitly below, not a bug. See
``tests/test_legacy_parity.py`` for the same convention pinned as a test.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pyref.fitting as fit
from refnx.analysis import CurveFitter

from refloxide.data import ReflectDataset
from refloxide.model import MaterialSLD, ReflectModel
from refloxide.objective import Objective

ENERGY_EV = 700.0
Q = np.linspace(0.03, 0.2, 150)

# %% Shared physical structure, built two ways


def build_refloxide_structure():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("SiO2", density=2.2, name="film")(50, 3)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    return vacuum | film | substrate


def build_pyref_structure():
    vacuum = fit.MaterialSLD("", density=0.0, energy=ENERGY_EV, name="vacuum")
    film = fit.MaterialSLD("SiO2", density=2.2, energy=ENERGY_EV, name="film")
    substrate = fit.MaterialSLD("Si", density=2.33, energy=ENERGY_EV, name="substrate")
    return vacuum(0, 0) | film(50, 3) | substrate(0, 3)


refloxide_model = ReflectModel(build_refloxide_structure())
pyref_model = fit.ReflectModel(build_pyref_structure(), energy=ENERGY_EV, pol="sp")

# %% 1. Numeric parity

pyref_model.pol = "s"
pyref_s = pyref_model.model(Q)  # native kernel [:, 1, 1]
pyref_model.pol = "p"
pyref_p = pyref_model.model(Q)  # native kernel [:, 0, 0]

refloxide_r = refloxide_model(Q, ENERGY_EV)

max_err_s = np.max(np.abs(refloxide_r.p - pyref_s))  # refloxide .p <-> pyref pol='s'
max_err_p = np.max(np.abs(refloxide_r.s - pyref_p))  # refloxide .s <-> pyref pol='p'
print(f"max |refloxide.p - pyref(pol='s')| = {max_err_s:.3e}")
print(f"max |refloxide.s - pyref(pol='p')| = {max_err_p:.3e}")
assert max_err_s < 1e-8
assert max_err_p < 1e-8
print("Numeric parity OK: refloxide.model matches stock pyref exactly.\n")

# %% Plot overlay

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Q, refloxide_r.p, label="refloxide .p (== pyref pol='s')", lw=2)
ax.plot(Q, pyref_s, "--", label="pyref pol='s'", lw=1.5, color="k")
ax.plot(Q, refloxide_r.s, label="refloxide .s (== pyref pol='p')", lw=2)
ax.plot(Q, pyref_p, "--", label="pyref pol='p'", lw=1.5, color="0.4")
ax.set_yscale("log")
ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
ax.set_ylabel("Reflectivity")
ax.legend()
ax.set_title(f"refloxide.model vs stock pyref, {ENERGY_EV:.0f} eV")
fig.tight_layout()
plt.show()

# %% 2. Speed comparison


def time_it(fn, n: int = 200, warmup: int = 5) -> float:
    """Mean seconds per call, after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


t_refloxide = time_it(lambda: refloxide_model(Q, ENERGY_EV))
t_pyref = time_it(lambda: pyref_model.model(Q))

print(f"refloxide.model.ReflectModel:                 {t_refloxide * 1e3:.4f} ms/call")
print(f"stock pyref.fitting.ReflectModel (unpatched):  {t_pyref * 1e3:.4f} ms/call")
print(f"speedup: {t_pyref / t_refloxide:.1f}x\n")

# %% Isolate "is it just Rust, or the new API too" by patching pyref itself

from refloxide.integrations.pyref import patch_pyref  # noqa: E402

patch_pyref(use_rust=True, parallel=False)
t_pyref_patched = time_it(lambda: pyref_model.model(Q))

print(
    "stock pyref.fitting.ReflectModel (patched, Rust kernel): "
    f"{t_pyref_patched * 1e3:.4f} ms/call"
)
print(
    "refloxide.model speedup over patched pyref (API/batching overhead only): "
    f"{t_pyref_patched / t_refloxide:.2f}x\n"
)

# %% 3. Fitting comparison — synthetic noisy s+p dataset, fit both ways

rng = np.random.default_rng(0)
pyref_model.pol = "s"
r_s_true = pyref_model.model(Q)
pyref_model.pol = "p"
r_p_true = pyref_model.model(Q)
pyref_model.pol = "sp"

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
    r=np.concatenate([r_p, r_s]),  # label swap: refloxide "s" <- pyref p-channel data
    r_err=np.concatenate([err_p, err_s]),
)
new_objective = Objective(new_model, new_data, anisotropy_weight=0.4)

for p in new_model.structure.parameters.flattened():
    if p.constraint is None:
        p.vary = False
new_film = new_model.structure.components[1]
new_film.thick.setp(vary=True, bounds=(20, 100))

new_fitter = CurveFitter(new_objective)

# %% ... stock pyref side

pyref_dataset = fit.XrayReflectDataset(
    (
        np.concatenate([Q, Q]),
        np.concatenate([r_s, r_p]),
        np.concatenate([err_s, err_p]),
    )
)
pyref_objective = fit.AnisotropyObjective(
    pyref_model, pyref_dataset, logp_anisotropy_weight=0.4
)

for p in pyref_model.structure.parameters.flattened():
    if p.constraint is None:
        p.vary = False
pyref_film = pyref_model.structure[1]
pyref_film.thick.setp(vary=True, bounds=(20, 100))

pyref_fitter = fit.CurveFitter(pyref_objective)

# %% Compare logl before fitting, then fit both and compare timing/result

print("logl before fit:")
print("  refloxide:", new_objective.logl())
print("  pyref:    ", pyref_objective.logl())

t0 = time.perf_counter()
new_fitter.fit(method="differential_evolution", maxiter=40, polish=False, seed=1)
t_new_fit = time.perf_counter() - t0

t0 = time.perf_counter()
pyref_fitter.fit(method="differential_evolution", maxiter=40, polish=False, seed=1)
t_pyref_fit = time.perf_counter() - t0

print(
    f"\nrefloxide fit: {t_new_fit:.3f} s, "
    f"recovered thick = {new_film.thick.value:.2f}"
)
print(
    f"pyref fit:     {t_pyref_fit:.3f} s, "
    f"recovered thick = {pyref_film.thick.value:.2f}"
)
print(f"fit speedup: {t_pyref_fit / t_new_fit:.1f}x (true thickness was 50.0)")
