"""Interactive comparison: refloxide.model/objective vs. refloxide.python.model.

Run cell-by-cell in an interactive Python window (each ``# %%`` marker is one
cell — VS Code's "Python Interactive" / Jupytext percent format), or
top-to-bottom with::

    uv run python examples/model_objective_repl.py

Three things, same physical structure (vacuum / SiO2 film / Si substrate,
700 eV):

1. Numeric parity — refloxide.model.ReflectModel vs pure-Python
   refloxide.python.model.ReflectModel (``refloxide.python.tmm``).
2. Speed — same two, timed.
3. Fitting — refloxide.objective.Objective vs
   refloxide.python.model.AnisotropyObjective, built from the same synthetic noisy
   s+p dataset, compared by log-likelihood and by a short
   differential_evolution fit.

Note on s/p labeling: python.model's ``pol='s'``/``pol='p'`` extraction is a
historical inversion kept for legacy dataset compatibility (``pol='s'``
reads the kernel's ``[:, 1, 1]``, ``pol='p'`` reads ``[:, 0, 0]``).
``refloxide.model.Reflectivity`` uses the native, non-inverted kernel
labeling (``.s = [:, 0, 0]``, ``.p = [:, 1, 1]``, matching ``rust.pyi``'s own
docs). So "py ``pol='s'``" corresponds to "refloxide ``.p``", and vice
versa — accounted for explicitly below, not a bug. See
``tests/test_legacy_parity.py`` for the same convention pinned as a test.
"""
# %%
from __future__ import annotations

import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import refloxide.python.model as py
from refnx.analysis import CurveFitter

from refloxide.data import ReflectDataset
from refloxide.model import MaterialSLD, ReflectModel
from refloxide.objective import Objective

ENERGY_EV = 250.0
Q = np.linspace(0.03, 0.2, 150)

# %% Shared physical structure, built two ways


def build_refloxide_structure():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    film = MaterialSLD("SiO2", density=2.2, name="film")(100, 3)
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3)
    return vacuum | film | substrate


def build_py_structure():
    vacuum = py.MaterialSLD("", density=0.0, energy=ENERGY_EV, name="vacuum")
    film = py.MaterialSLD("SiO2", density=2.2, energy=ENERGY_EV, name="film")
    substrate = py.MaterialSLD("Si", density=2.33, energy=ENERGY_EV, name="substrate")
    return vacuum(0, 0) | film(100, 3) | substrate(0, 3)


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
print("Numeric parity OK: refloxide.model matches python.model exactly.\n")

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
ax.set_title(f"refloxide.model vs python.model, {ENERGY_EV:.0f} eV")
fig.tight_layout()
plt.show()

# %% Structure visualization -- depth profile of optical constants + density
#
# `Structure.plot.oc` plots the depth-resolved index of refraction, each
# interface broadened by an error function of width sigma = that
# interface's own Nevot-Croce roughness (see `Structure.sld_profile_at`).
# `Structure.plot.param` plots any depth-resolved quantity matching a
# regex against `Structure.named_profiles_at`'s keys -- "density" here
# (this structure is plain isotropic MaterialSLD, so "orientation" would
# be NaN everywhere; see the UniTensorSLD/BookendedComponent repls for
# structures where it isn't).

refloxide_structure = refloxide_model.structure
refloxide_structure.plot.oc(ENERGY_EV)
plt.show()
refloxide_structure.plot.param("density", roughness=True)
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
print(f"memory ratio (python.model/refloxide): {mem_py / mem_refloxide:.2f}x\n")

# %% 3. Fitting comparison — synthetic noisy s+p dataset, fit both ways

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
new_film = new_model.structure.components[1]
new_film.thick.setp(vary=True, bounds=(20, 100))

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
py_film = py_model.structure[1]
py_film.thick.setp(vary=True, bounds=(20, 100))

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
    f"recovered thick = {new_film.thick.value:.2f}"
)
print(
    f"py fit:     {t_py_fit:.3f} s, "
    f"recovered thick = {py_film.thick.value:.2f}"
)
print(f"fit speedup: {t_py_fit / t_new_fit:.1f}x (true thickness was 50.0)")
# %% Plot py structure
refloxide_structure = new_fitter.objective.model.structure
refloxide_structure.plot.oc(ENERGY_EV)
plt.show()
refloxide_structure.plot.param("density", roughness=True)
plt.show()
