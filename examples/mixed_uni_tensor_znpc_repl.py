"""Interactive comparison: mixed-material uniaxial slabs, refloxide vs. pyref.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/mixed_uni_tensor_znpc_repl.py

A "mixed material" slab blends two (or more) uniaxial optical-constants
sources into one laboratory tensor by volume fraction -- e.g. two molecular
populations with different tilt angles occupying the same physical layer.
This exercises `refloxide.model.MixedUniTensorSLD`, a new addition (this
script is also its introduction) alongside the existing single-material
`UniTensorSLD`.

The two components blended in every mixed layer here are:

1. The real, DFT-derived ZnPc optical constants (`dft.csv`, as in
   `uni_tensor_znpc_repl.py`).
2. A synthetic "variant" of the same table: a random number of extra
   resonance-like features added in the 284-320 eV carbon K-edge NEXAFS
   window (not the whole table). Each feature is one Gaussian added to both
   imaginary columns (`n_ixx`, `n_izz`) at a random center energy, plus a
   Kramers-Kronig-like antisymmetric doublet added to both real columns
   (`n_xx`, `n_zz`): a negative Gaussian at `center - 0.5*width` (half the
   imaginary peak's amplitude) and a positive one at `center + 0.5*width`
   (same magnitude, positive) -- a crude two-Gaussian approximation of the
   dispersive real-part wiggle that straddles a real absorption resonance.
   N imaginary features therefore add `2*N` real features. Not a different
   chemical species, just a plausible second population (e.g. extra
   molecular disorder) to blend with the first. Generated once with a fixed
   seed so both models below mix bit-for-bit identical tables, not
   independently-randomized ones.

Five mixed-material layers are stacked into one compositionally-graded film,
each blending the same two tables with its own volume fractions, tilts, and
densities -- a thin ZnPc-rich surface, a bulk region still mostly the first
population, a genuine ~50/50 mixing layer, a bulk region mostly the second
population, and a substrate-side interface layer that is mostly the second
population again -- before the isotropic SiO2/Si substrate:

    vacuum / surface / bulk_1 / mixing / bulk_2 / interface / SiO2 / Si

`refloxide.model.MixedUniTensorSLD` is compared against
`PyrefMixedUniTensorSLD`, a from-scratch pyref-side reference implementation
defined below (pyref itself ships no mixed-material uniaxial scatterer). It
follows the same density-scaled, rotation-mixed uniaxial formula as pyref's
own single-material `UniTensorSLD` and sums the result across components by
volume fraction -- radians throughout, matching
`pyref.fitting.structure.Scatterer.get_rotation`'s own documented convention.

Three things (the graded film above, 285.1 eV -- the carbon K-edge pi*
resonance):

1. Correctness -- refloxide vs. the local pyref-style reference, same
   blended tables, same volume fractions/tilts/densities.
2. Speed -- same two, timed against stock, UNPATCHED pyref (its own
   pure-Python kernel; pyref's optional Rust patch is being phased out and
   isn't the focus here).
3. Fitting -- recover the mixing layer's ZnPc volume fraction from a
   synthetic noisy dataset, refloxide.objective.Objective vs. stock
   pyref.fitting.AnisotropyObjective built around the pyref-side reference
   model.

Note on s/p labeling: identical convention/inversion as
`model_objective_repl.py` -- stock pyref's `pol='s'` reads the kernel's
`[:, 1, 1]`, `pol='p'` reads `[:, 0, 0]`; `refloxide.model.Reflectivity` uses
the native, non-inverted labeling (`.s = [:, 0, 0]`, `.p = [:, 1, 1]`).
"""

# %% 0
from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyref.fitting as fit
from pyref.fitting.structure import Scatterer as PyrefScatterer
from refnx.analysis import CurveFitter, Parameters, possibly_create_parameter
from scipy.interpolate import interp1d

from refloxide.data import OpticalConstants, ReflectDataset
from refloxide.integrations.pyref import pyref_patched
from refloxide.model import MaterialSLD, MixedUniTensorSLD, ReflectModel
from refloxide.objective import Objective

if pyref_patched():
    msg = (
        "pyref.fitting.ReflectModel is already patched (patch_pyref() ran "
        "earlier in this process/kernel). The 'unpatched' timing below would "
        "silently measure the Rust kernel on both sides. Restart the "
        "interpreter/kernel and rerun this script on its own."
    )
    raise RuntimeError(msg)

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


# %% Generate the two blended optical-constants tables once, with a fixed seed


NEXAFS_WINDOW_EV = (
    284.0,
    289.0,
)  # carbon K-edge NEXAFS window the features are drawn in


def add_resonance_gaussians(
    table: pl.DataFrame,
    rng: np.random.Generator,
    n_gaussians: int,
    *,
    window_ev: tuple[float, float] = NEXAFS_WINDOW_EV,
    width_range_ev: tuple[float, float] = (0.1, 3),
) -> pl.DataFrame:
    """Add `n_gaussians` synthetic resonance features to an OOC table's NEXAFS window.

    Each feature is one Gaussian added to both imaginary columns (`n_ixx`,
    `n_izz`) at a center energy drawn uniformly from `window_ev`, plus a
    Kramers-Kronig-like antisymmetric doublet added to both real columns
    (`n_xx`, `n_zz`): a negative Gaussian at `center - 0.5*width` (half the
    imaginary peak's amplitude) and a positive Gaussian at
    `center + 0.5*width` (same magnitude, positive), both using that
    feature's own width -- a crude two-Gaussian approximation of the
    dispersive real-part wiggle that straddles a real absorption resonance.
    `n_gaussians` imaginary features therefore produce `2 * n_gaussians`
    real features.

    Parameters
    ----------
    table : polars.DataFrame
        Source OOC table with `energy`/`n_xx`/`n_ixx`/`n_zz`/`n_izz` columns.
    rng : numpy.random.Generator
        Caller-owned generator -- pass the same seeded instance everywhere
        this is called so repeated runs (and both models below) see
        bit-for-bit identical features.
    n_gaussians : int
        Number of imaginary-part resonance features to add.
    window_ev : tuple[float, float], optional
        Range each feature's center energy is drawn uniformly from.
    width_range_ev : tuple[float, float], optional
        Range each feature's Gaussian width (sigma, eV) is drawn uniformly
        from.

    Returns
    -------
    polars.DataFrame
        Same shape and columns as `table`.
    """
    energy = table["energy"].to_numpy()
    n_xx = table["n_xx"].to_numpy().copy()
    n_ixx = table["n_ixx"].to_numpy().copy()
    n_zz = table["n_zz"].to_numpy().copy()
    n_izz = table["n_izz"].to_numpy().copy()

    in_window = (energy >= window_ev[0]) & (energy <= window_ev[1])
    beta_scale = float(np.ptp(n_ixx[in_window]) + np.ptp(n_izz[in_window])) / 2.0

    def gaussian(center: float, width: float, amplitude: float) -> np.ndarray:
        return amplitude * np.exp(-0.5 * ((energy - center) / width) ** 2)

    for _ in range(n_gaussians):
        center = rng.uniform(*window_ev)
        width = rng.uniform(*width_range_ev)
        amplitude = rng.uniform(0.5, 1.2) * beta_scale

        imaginary_bump = gaussian(center, width, amplitude)
        n_ixx += imaginary_bump
        n_izz += imaginary_bump

        real_doublet = gaussian(
            center - 0.5 * width, width, -amplitude / 2.0
        ) + gaussian(center + 0.5 * width, width, amplitude / 2.0)
        n_xx += real_doublet
        n_zz += real_doublet

    return table.with_columns(
        pl.Series("n_xx", n_xx),
        pl.Series("n_ixx", n_ixx),
        pl.Series("n_zz", n_zz),
        pl.Series("n_izz", n_izz),
    )


GENERATION_SEED = 20250710  # fixed so both models mix identical tables every run
rng = np.random.default_rng(GENERATION_SEED)
N_GAUSSIANS = int(
    rng.integers(2, 5)
)  # random count of imaginary-part features (here: 3)
znpc_table = pl.read_csv(ZNPC_DFT_CSV)
znpc_variant_table = add_resonance_gaussians(znpc_table, rng, N_GAUSSIANS)
print(
    f"Generated {N_GAUSSIANS} resonance features in [{NEXAFS_WINDOW_EV[0]}, "
    f"{NEXAFS_WINDOW_EV[1]}] eV ({2 * N_GAUSSIANS} real + {N_GAUSSIANS} imaginary) "
    "for the ZnPc variant table.\n"
)

znpc_pd = znpc_table.to_pandas()
znpc_variant_pd = znpc_variant_table.to_pandas()

# %%
ax1 = znpc_pd.plot(x="energy", y=["n_xx", "n_ixx", "n_zz", "n_izz"])
ax2 = znpc_variant_pd.plot(x="energy", y=["n_xx", "n_ixx", "n_zz", "n_izz"])
ax1.set_xlim(250, 320)
ax1.set_ylim(-1e-2, 1e-2)
ax2.set_xlim(250, 320)
ax2.set_ylim(-1e-2, 1e-2)


# %% Five mixed-material layers: same two tables, a graded blend through the film
#
# Composition (ZnPc volume fraction) runs 0.85 -> 0.75 -> 0.5 -> 0.25 -> 0.15
# from the vacuum interface to the substrate interface -- mostly the first
# population at the surface, a genuine mixing layer in the middle, mostly
# the second population approaching the substrate.

SURFACE = {
    "thick": 15,
    "rough": 3,
    "vf": [0.85, 0.15],
    "rotation": [0.0, 0.2],
    "density": [1.61, 1.55],
}
BULK_1 = {
    "thick": 50,
    "rough": 0,
    "vf": [0.75, 0.25],
    "rotation": [0.2, 0.4],
    "density": [1.61, 1.58],
}
MIXING = {
    "thick": 40,
    "rough": 0,
    "vf": [0.5, 0.5],
    "rotation": [0.6, 0.6],
    "density": [1.58, 1.58],
}
BULK_2 = {
    "thick": 50,
    "rough": 0,
    "vf": [0.25, 0.75],
    "rotation": [0.9, 1.1],
    "density": [1.55, 1.61],
}
INTERFACE = {
    "thick": 20,
    "rough": 3,
    "vf": [0.15, 0.85],
    "rotation": [1.2, 1.3],
    "density": [1.5, 1.61],
}
MIXED_LAYERS = {
    "surface": SURFACE,
    "bulk_1": BULK_1,
    "mixing": MIXING,
    "bulk_2": BULK_2,
    "interface": INTERFACE,
}


# %% pyref-side reference: pyref ships no mixed-material uniaxial scatterer


class PyrefMixedUniTensorSLD(PyrefScatterer):
    """pyref-style volume-fraction-weighted mixture of uniaxial materials.

    pyref has no built-in mixed-material uniaxial scatterer; this is a
    from-scratch reference implementation for comparison, following the same
    density-scaled, rotation-mixed formula as pyref's own single-material
    `pyref.fitting.UniTensorSLD.tensor`, applied per component and then
    summed by volume fraction. Rotation is radians throughout, matching
    `Scatterer.get_rotation`'s own documented convention.
    """

    def __init__(
        self,
        oocs,
        vf,
        rotation,
        density,
        energy: float = 250.0,
        energy_offset: float = 0.0,
        name: str = "",
    ):
        super().__init__(name=name)
        n = len(oocs)
        lengths = {len(oocs), len(vf), len(rotation), len(density)}
        if lengths != {n}:
            msg = (
                "PyrefMixedUniTensorSLD component sequences must all be the same length"
            )
            raise ValueError(msg)

        self._parameters = Parameters(name=name)
        self.density = [
            possibly_create_parameter(
                d, name=f"{name}_density_{i}", bounds=(0, 5 * d), vary=True
            )
            for i, d in enumerate(density)
        ]
        self.rotation = [
            possibly_create_parameter(
                r, name=f"{name}_rotation_{i}", bounds=(-np.pi, np.pi), vary=True
            )
            for i, r in enumerate(rotation)
        ]
        self.volfrac = [
            possibly_create_parameter(
                v, name=f"{name}_vol_frac_{i}", bounds=(0, 1.0), vary=True
            )
            for i, v in enumerate(vf)
        ]
        self.energy = energy
        # A single, shared energy_offset (not one per blended component) --
        # `pyref.fitting.Structure.energy_offset`'s setter assumes every
        # component's `sld.energy_offset` is one Parameter it can `.setp(...)`
        # directly (it links every scatterer to one model-level offset), so a
        # per-component list here would break `ReflectModel.__init__` itself.
        # refloxide.model.MixedUniTensorSLD has no such constraint and keeps
        # a genuinely independent offset per component.
        self.energy_offset = possibly_create_parameter(
            energy_offset, name=f"{name}_energy_offset", bounds=(-0.01, 0.01), vary=True
        )
        self.n_xx = [
            interp1d(oc["energy"], oc["n_xx"], bounds_error=False) for oc in oocs
        ]
        self.n_ixx = [
            interp1d(oc["energy"], oc["n_ixx"], bounds_error=False) for oc in oocs
        ]
        self.n_zz = [
            interp1d(oc["energy"], oc["n_zz"], bounds_error=False) for oc in oocs
        ]
        self.n_izz = [
            interp1d(oc["energy"], oc["n_izz"], bounds_error=False) for oc in oocs
        ]

        self._parameters.extend(self.density)
        self._parameters.extend(self.rotation)
        self._parameters.extend(self.volfrac)
        self._parameters.extend([self.energy_offset])

    def n(self, i: int) -> np.ndarray:
        e = self.energy + self.energy_offset.value
        return np.array(
            [
                [self.n_xx[i](e) + self.n_ixx[i](e) * 1j, 0],
                [0, self.n_zz[i](e) + self.n_izz[i](e) * 1j],
            ],
            dtype=np.complex128,
        )

    @property
    def parameters(self) -> Parameters:
        self._parameters.name = self.name
        return self._parameters

    @property
    def tensor(self) -> np.ndarray:
        n_o_sum = complex(0.0)
        n_e_sum = complex(0.0)
        for i, vf in enumerate(self.volfrac):
            nd = self.density[i].value * self.n(i)
            cos_sq = np.cos(self.rotation[i].value) ** 2
            sin_sq = 1.0 - cos_sq
            n_o = (nd[0, 0] * (1 + cos_sq) + nd[1, 1] * sin_sq) / 2
            n_e = nd[0, 0] * sin_sq + nd[1, 1] * cos_sq
            n_o_sum += n_o * vf.value
            n_e_sum += n_e * vf.value
        return np.diag(np.array([n_o_sum, n_o_sum, n_e_sum], dtype=np.complex128))

    def __complex__(self) -> complex:
        t = self.tensor
        return complex((2 * t[0, 0] + t[2, 2]) / 3)

    def __repr__(self) -> str:
        return f"PyrefMixedUniTensorSLD(n={len(self.volfrac)}, name={self.name!r})"


# %% Shared physical structure, built two ways


def build_refloxide_structure():
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    mixed_layers = [
        MixedUniTensorSLD(
            [znpc_table, znpc_variant_table],
            vf=cfg["vf"],
            rotation=cfg["rotation"],
            density=cfg["density"],
            name=name,
        )(cfg["thick"], cfg["rough"])
        for name, cfg in MIXED_LAYERS.items()
    ]
    oxide = MaterialSLD("SiO2", density=2.2, name="oxide")(8, 3)
    si = MaterialSLD("Si", density=2.33, name="si")(0, 3)
    structure = vacuum
    for layer in mixed_layers:
        structure = structure | layer
    return structure | oxide | si


def build_pyref_structure():
    vacuum = fit.MaterialSLD("", density=0.0, energy=ENERGY_EV, name="vacuum")
    mixed_layers = [
        PyrefMixedUniTensorSLD(
            [znpc_pd, znpc_variant_pd],
            vf=cfg["vf"],
            rotation=cfg["rotation"],
            density=cfg["density"],
            energy=ENERGY_EV,
            name=name,
        )(cfg["thick"], cfg["rough"])
        for name, cfg in MIXED_LAYERS.items()
    ]
    oxide = fit.MaterialSLD("SiO2", density=2.2, energy=ENERGY_EV, name="oxide")
    si = fit.MaterialSLD("Si", density=2.33, energy=ENERGY_EV, name="si")
    structure = vacuum(0, 0)
    for layer in mixed_layers:
        structure = structure | layer
    return structure | oxide(8, 3) | si(0, 3)


refloxide_model = ReflectModel(build_refloxide_structure())
pyref_model = fit.ReflectModel(build_pyref_structure(), energy=ENERGY_EV, pol="sp")
pyref_model.structure.plot()

# %% Sharing guarantee: five mixed layers, same two tables, one cache entry each

cache_size = OpticalConstants.cache_size()
mixed_slds = [refloxide_model.structure.components[i].sld for i in range(1, 6)]
for other in mixed_slds[1:]:
    assert other.oocs[0] is mixed_slds[0].oocs[0]
    assert other.oocs[1] is mixed_slds[0].oocs[1]
print(
    f"OpticalConstants.cache_size() = {cache_size}: all 5 mixed layers (surface, "
    "bulk_1, mixing, bulk_2, interface) share the same two loaded tables (ZnPc + "
    "its noisy variant), not ten separate copies.\n"
)

# %% 1. Correctness -- refloxide vs. the local pyref-style reference

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
print(
    "Correctness OK: refloxide.MixedUniTensorSLD matches the pyref-style reference.\n"
)

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
ax.set_title(f"Mixed-material ZnPc slabs, refloxide vs pyref-style, {ENERGY_EV:.1f} eV")
fig.tight_layout()
plt.show()

# %% Structure visualization -- optical constants, density, orientation, and vf vs depth
#
# `Structure.plot.oc` shows the depth profile with the xx/zz dichroism on a
# right-hand twin axis (`difference=True`). `Structure.plot.param` plots
# any depth-resolved quantity matching a regex against
# `Structure.named_profiles_at`'s keys -- `"density"`/`"orientation"` give
# each layer's own volume-fraction-weighted average across its two blended
# ZnPc populations (surface/bulk_1 skew toward population 1, mixing is
# ~50/50, bulk_2/interface skew toward population 2, NaN over the isotropic
# vacuum/oxide/Si bookends). `MixedUniTensorSLD.named_profile_values`
# additionally exposes each layer's own two population fractions as
# `"vf_0"`/`"vf_1"` -- five separately-parameterized `MixedUniTensorSLD`
# layers all define those same two keys, but each occupies its own depth
# range, so `"vf_"` composes them into ONE trace per population spanning
# the whole graded film, showing the composition grade directly.

refloxide_structure = refloxide_model.structure
refloxide_structure.plot.oc(ENERGY_EV, difference=True)
plt.show()
refloxide_structure.plot.param("density|orientation")
plt.show()
refloxide_structure.plot.param("vf_")
plt.show()

# %% 2. Speed comparison -- vs. stock, UNPATCHED pyref (its own pure-Python kernel)


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

# %% Memory footprint -- peak Python-heap bytes per call, refloxide vs stock pyref


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
mem_pyref = peak_memory_bytes(lambda: pyref_model.model(Q))

print(f"refloxide.model.ReflectModel:                 {mem_refloxide:>7,} B/call")
print(f"stock pyref.fitting.ReflectModel (unpatched):  {mem_pyref:>7,} B/call")
print(f"memory ratio (pyref/refloxide): {mem_pyref / mem_refloxide:.2f}x\n")

# %% 3. Fitting comparison -- recover mixed_1's ZnPc volume fraction

rng_fit = np.random.default_rng(0)
pyref_model.pol = "s"
r_s_true = pyref_model.model(Q)
pyref_model.pol = "p"
r_p_true = pyref_model.model(Q)
pyref_model.pol = "sp"

r_s = r_s_true * (1 + rng_fit.normal(0, 0.01, size=Q.shape))
r_p = r_p_true * (1 + rng_fit.normal(0, 0.01, size=Q.shape))
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
new_mixing = new_model.structure.components[3].sld
new_mixing.vf[0].setp(vary=True, bounds=(0.3, 0.7))

new_fitter = CurveFitter(new_objective)

# %% ... pyref side (AnisotropyObjective around the pyref-style reference model)

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
pyref_mixing = pyref_model.structure[3].sld
pyref_mixing.volfrac[0].setp(vary=True, bounds=(0.3, 0.7))

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
    f"recovered mixing-layer ZnPc vf = {new_mixing.vf[0].value:.3f}"
)
print(
    f"pyref fit:     {t_pyref_fit:.3f} s, "
    f"recovered mixing-layer ZnPc vf = {pyref_mixing.volfrac[0].value:.3f}"
)
print(
    f"fit speedup: {t_pyref_fit / t_new_fit:.1f}x "
    f"(true ZnPc vf was {MIXING['vf'][0]})"
)
