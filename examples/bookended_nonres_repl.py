"""Bookended fitting of a nonresonant, off-edge lab XRR dataset.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/bookended_nonres_repl.py

Loads the resonant (283.7 eV, C K-edge) graded book-ended ZnPc fit at
``@models/xrr/znpc/graded/graded_fit.pkl`` (extracted into a portable JSON
summary by ``refl-analysis/scripts/extract_graded_bookended_fit.py`` --
run that once first if ``graded_fit_summary.json`` is missing) and asks: does
the SAME graded density/orientation profile shape, evaluated at a completely
different (hard x-ray, lab-source) energy, describe a nonresonant XRR
dataset -- with only the film's overall thickness allowed to change?

``energy = 8.04e3`` eV matches Cu-K-alpha (a common lab XRR source,
wavelength ~1.542 A, consistent with this dataset's angle range) and sits
nowhere near a C/N/Zn absorption edge, so the film is expected to be
optically isotropic here (confirmed directly: the DFT OOC table's
``n_xx``/``n_zz`` already agree to 5 significant figures at 8 keV) --
"nonresonant" in the sense that molecular orientation should have no
measurable effect on the reflectivity, not that the film has literally lost
its shape.

The transplanted book-ended profile is refit with every parameter frozen at
its resonant-fit value EXCEPT ``total_thick``, ``surface_roughness``, and
the Oxide/SiO2 slab's own roughness (orientation stays fixed, per the
profile's own physical picture, since dichroism is negligible at this
energy anyway), plus the ordinary instrument nuisance parameters
(``scale_s``, ``bkg``) needed to convert the dataset's raw,
footprint-uncorrected detector counts into reflectivity units.

``bkg`` is the one nuisance parameter that needs care: for q > 0.21 the raw
counts are consistently 1-2 detector counts (essentially Poisson noise with
no remaining trend), so the true background floor is directly measurable
from the data itself (~1.5e-6 in normalized units) and NOT something to
leave the optimizer free to discover on its own. An earlier version of this
script bounded ``bkg`` to a wide-open ``(0, 1e-3)``, and the fit walked it
up to ~5e-5 -- above several of the real fringe points between q=0.13 and
q=0.2 -- which buried those fringes under a flat floor (looked like the
film had no structure past q~0.13-0.15). Anchoring ``bkg``'s bounds to the
data's own measured plateau instead keeps the optimizer from using
background as a substitute for getting the film geometry right.
"""
# %% 0
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from refnx.analysis import CurveFitter, Transform

from refloxide.data import ReflectDataset
from refloxide.model import BookendedComponent, MaterialSLD, ReflectModel
from refloxide.objective import Objective
from refloxide.pxr.energy.bookended import BookendedOrientationProfile
from refloxide.pxr.energy.ooc import OocAnchor

# %% Paths

GRADED_DIR = Path.home() / "projects/refl-analysis/@models/xrr/znpc/graded"
SUMMARY_PATH = GRADED_DIR / "graded_fit_summary.json"
CSV_PATH = (
    Path.home()
    / "projects/refl-analysis/notebooks/fitting/HiRes_BigRegion_ExcelMerge_CSV.csv"
)
for path in (SUMMARY_PATH, CSV_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- run scripts/extract_graded_bookended_fit.py in the "
            "refl-analysis venv first (for SUMMARY_PATH), or check the dataset path"
        )

summary = json.loads(SUMMARY_PATH.read_text())
ooc_csv = Path(summary["ooc_csv"])
if not ooc_csv.exists():
    raise FileNotFoundError(f"missing {ooc_csv} referenced by {SUMMARY_PATH}")
ZNPC_OOC = OocAnchor.from_file(ooc_csv)

ENERGY_EV = 8.04e3  # Cu-K-alpha lab source; far off any C/N/Zn/Si/O edge
_HC_EV_ANGSTROM = 12398.42  # matches refloxide.model's own internal constant

print(
    f"resonant reference fit was at {summary['energy']:.1f} eV; refitting at "
    f"{ENERGY_EV:.2f} eV"
)
film_params = summary["film"]["params"]
for name, value in film_params.items():
    print(f"  {name}: {value:.4f}")

# %% Load the CSV dataset: angle (deg) + raw intensity (a.u.) -> q (1/A) + r
#
# No incident-flux normalization is recorded, only raw detector counts, so
# `r` is left in the SAME arbitrary units and `scale_s` (an ordinary
# instrument nuisance parameter, not a film parameter) absorbs the unknown
# overall factor during the fit. The rough starting normalization below
# (divide by the peak count) just gives the optimizer a sane `scale_s ~ 1`
# starting point -- it is not asserted to be the true I0, and the very
# lowest angles are visibly still ramping up (beam footprint not yet fully
# on the sample), not flat at R=1, which is exactly what a free `scale_s`
# is for.

frame = pl.read_csv(CSV_PATH)
angle_deg = frame["IncidentAngle(deg)"].to_numpy()
intensity = frame["Intensity(a.u.)"].to_numpy()
sigma_i = frame["Sigma_I(a.u.)"].to_numpy()

wavelength = _HC_EV_ANGSTROM / ENERGY_EV
q = (4.0 * np.pi / wavelength) * np.sin(np.radians(angle_deg))
norm = float(intensity.max())
r = intensity / norm
r_err = sigma_i / norm
print(f"wavelength = {wavelength:.4f} A, q range = [{q.min():.4f}, {q.max():.4f}] 1/A")
print(f"{len(q)} points, normalized peak r = {r.max():.4f}")

dataset = ReflectDataset(
    q=q,
    energy=np.full_like(q, ENERGY_EV),
    pol=np.full(q.shape, "s", dtype=object),
    r=r,
    r_err=r_err,
)

# %% Estimate the detector background directly from the data's own high-q
# plateau, rather than let `bkg` roam free and find its own
#
# For q > 0.21 the raw counts sit at ~1-2 (single/double-photon Poisson
# noise, no further trend -- confirmed directly: min/median/max over that
# range are 1.01e-6/1.53e-6/2.05e-6, i.e. almost exactly 1/1.5/2 detector
# counts once normalized by the peak). This IS the true background floor,
# measurable independent of any fit.

BKG_ESTIMATE = r.min()

# %% Shared Oxide/Substrate geometry, frozen at the resonant fit's own values
#
# `MaterialSLD(...)(thick, rough)` defaults `thick`/`rough`/`density` to
# `vary=True` (see bookended_repl.py's `freeze_entire_structure` note) --
# explicitly freezing them here is what makes "fit just the thickness" mean
# what it says, rather than silently also re-fitting the substrate stack.
#
# Oxide roughness is the one exception, left free below: it's a native
# SiO2/Si interface, physically independent of the ZnPc film sitting on top
# of it, and there's no reason the resonant (283.7 eV) fit's own value for
# it should still be correct for a completely different sample/dataset at
# 8.04 keV -- freezing it there silently forced the ZnPc film alone to
# absorb any interface mismatch.

layers = {layer["name"]: layer for layer in summary["layers"]}


def build_slab(
    name: str,
    *,
    vary_rough: bool = False,
    rough_bounds: tuple[float, float] = (0.0, 30.0),
):
    layer = layers[name]
    sld = MaterialSLD(layer["formula"], density=layer["density"], name=name)
    slab = sld(layer["thick"], layer["rough"])
    slab.thick.vary = False
    sld.density.vary = False
    if vary_rough:
        slab.rough.setp(vary=True, bounds=rough_bounds)
    else:
        slab.rough.vary = False
    return slab


def freeze_instrumentation(model: ReflectModel) -> None:
    """Open scale_s/theta_offset_s; fix bkg at the data's own measured floor.

    Must run AFTER `model.scale_s`/`model.theta_offset_s` have a real
    per-energy channel to act on -- `InstrumentFieldView.setp()` silently
    does nothing when called before the first `.at(energy)` lazily creates
    that channel (confirmed directly: calling it beforehand left
    `theta_offset_s` frozen at `vary=False` and `scale_s` still at
    whatever THIS function set, not the caller's intended bounds). A
    newly created channel otherwise defaults scale_p/theta_offset_p to
    `vary=True` with unbounded (-inf, inf) bounds -- harmless for
    predictions (the dataset is pol='s' only, so those terms never enter
    the likelihood) but fatal for `differential_evolution`, which requires
    finite bounds on every parameter it is told is free.

    `bkg` is FIXED at `BKG_ESTIMATE`, not fit, even with tight bounds
    around it -- confirmed directly: bounding it to (0.3, 3.0) x
    `BKG_ESTIMATE` still walked it to the top of that window (still ~3x
    the real floor), because a free `bkg` is the cheapest way for the
    optimizer to buy down weighted residuals in the q=0.14-0.2 fringe
    region rather than get the film geometry to actually reproduce them.
    Pinning `bkg` to the true, independently-measured floor forced the
    high-q tail to track the real background much more closely (the
    q > 0.21 model/data ratio dropped from ~5.5x to ~2.5x) instead of
    silently absorbing the mismatch as background.
    """
    model.scale_s.at(ENERGY_EV).setp(vary=True, bounds=(0.8, 1.2))
    model.theta_offset_s.at(ENERGY_EV).setp(vary=True, bounds=(-0.8, 0.8))
    model.bkg.at(ENERGY_EV).value = BKG_ESTIMATE
    model.bkg.at(ENERGY_EV).vary = False
    model.scale_p.at(ENERGY_EV).vary = False
    model.theta_offset_p.at(ENERGY_EV).vary = False


# Oxide roughness's upper bound is the Nevot-Croce ceiling for its own
# FROZEN thickness (`thick >= sqrt(2*pi)*rough/2`), not an arbitrary round
# number: widening it further just lets the optimizer smear the Oxide/Si
# interface away rather than fit anything real (confirmed directly -- (0,
# 20) and (0, 30) each pegged the recovered value right at the new ceiling,
# never converging short of it, the signature of an unconstrained parameter
# chasing missing physics elsewhere, e.g. the dataset's angular resolution
# is not modeled at all here).
_oxide_thick = layers["Oxide"]["thick"]
_oxide_rough_ceiling = float(np.sqrt(2.0 * np.pi) * _oxide_thick / 2.0)

BOUNDS = {
    "surface_roughness": (0.0, 30.0),
    "oxide_rough": (0.0, _oxide_rough_ceiling),
}

# %% Reference geometry: an independent discrete 5-slab (surface / bulk
# ZnPc / interface / SiO2 / Si) fit against this SAME nonresonant dataset,
# used here to seed the graded profile's book-end densities and
# relaxation lengths with informed starting values instead of the
# transplanted resonant-fit (283.7 eV) numbers, which describe a
# different sample geometry entirely. That discrete fit's own
# uncertainties are huge (e.g. interface_thick = 18.3 +/- 111 A) -- this
# dataset barely constrains individual discrete-slab boundaries on their
# own -- but the densities/lengths it settled on are still a better
# starting point than resonant-fit values transplanted across energy AND
# sample. "surface" (vacuum-facing) maps to the profile's density_vac/
# tau_vac book-end; "interface" (substrate-facing) maps to density_si/
# tau_si; "Zinc Phthalocyanine" (bulk) maps to density_bulk.
REFERENCE = {
    "surface_thick": 7.89151,
    "surface_rough": 6.28961,
    "surface_rho": 1.8,
    "bulk_thick": 125.596,
    "bulk_rho": 2.0,
    "interface_thick": 18.2626,
    "interface_rho": 1.60845,
    "oxide_thick": 9.84672,
    "oxide_rho": 2.0692,
    "substrate_rough": 0.5,
    "substrate_rho": 2.21255,
}
REFERENCE_TOTAL_THICK = (
    REFERENCE["surface_thick"] + REFERENCE["bulk_thick"] + REFERENCE["interface_thick"]
)

# %% 1. Graded fit: book-ended profile seeded from REFERENCE, orientation
# frozen, total_thick + surface_roughness + Oxide thick/rough free (plus
# scale_s/theta_offset_s)
#
# `BookendedOrientationProfile.__init__` uses `possibly_create_parameter`
# with its refnx default `vary=False` for every shape parameter, so the
# rebuilt profile already starts fully frozen -- alpha_bulk/alpha_si/
# alpha_vac stay fixed at the resonant-fit value (orientation is
# irrelevant at this nonresonant energy), and density_vac/density_bulk/
# density_si/tau_vac/tau_si are explicitly re-seeded from REFERENCE below,
# still frozen. Only total_thick and surface_roughness (confirmed to
# actually move the predicted reflectivity, not a no-op) are reopened.

graded_film = BookendedOrientationProfile(
    ooc=ZNPC_OOC,
    energy=summary["energy"],
    num_slabs=summary["film"]["num_slabs"],
    mesh_constant=summary["film"]["mesh_constant"],
    name="ZnPc",
    **film_params,
)
graded_structure = (
    build_slab("Vacuum")
    | BookendedComponent(graded_film)
    | build_slab("Oxide", vary_rough=True, rough_bounds=BOUNDS["oxide_rough"])
    | build_slab("Substrate")
)

graded_model = ReflectModel(graded_structure, parallel=False)
graded_film.total_thick.setp(
    value=REFERENCE_TOTAL_THICK, vary=True, bounds=(80.0, 200.0)
)
graded_film.surface_roughness.setp(
    value=REFERENCE["surface_rough"], vary=True, bounds=BOUNDS["surface_roughness"]
)
graded_film.density_vac.setp(value=REFERENCE["surface_rho"], vary=False)
graded_film.density_bulk.setp(value=REFERENCE["bulk_rho"], vary=False)
graded_film.density_si.setp(value=REFERENCE["interface_rho"], vary=False)
graded_film.tau_vac.setp(value=REFERENCE["surface_thick"], vary=False)
graded_film.tau_si.setp(value=REFERENCE["interface_thick"], vary=False)

oxide_slab = graded_structure.slab("Oxide")
oxide_slab.thick.setp(value=REFERENCE["oxide_thick"], vary=True, bounds=(0.0, 12.0))
oxide_slab.sld.density.setp(value=REFERENCE["oxide_rho"], vary=False)

substrate_slab = graded_structure.slab("Substrate")
substrate_slab.rough.value = REFERENCE["substrate_rough"]
substrate_slab.sld.density.value = REFERENCE["substrate_rho"]

freeze_instrumentation(graded_model)

# nc_constraint would fault immediately on the frozen Oxide geometry
# (thick ~9.8 A, rough up to the NC ceiling above) even though nothing
# here varies that slab's thick against its own rough independently, so
# the safety check has nothing to protect against.
graded_objective = Objective(
    graded_model, dataset, transform=Transform("logY"), nc_constraint=False
)
print(f"free parameters: {len(graded_objective.varying_parameters())}")
print(f"logl before fit: {graded_objective.logl():.3f}")

# %% 2. Run the fit and report the recovered geometry
#
# workers=-1 (multiprocess pool) is NOT worth it here: each `nll()` call
# is a few milliseconds (124 points, 5 free parameters), while spawning a
# handful of fresh Python processes -- each re-importing refloxide/refnx/
# scipy/numpy from scratch -- costs seconds. Confirmed directly: a tiny
# popsize=4/maxiter=2 smoke test with workers=-1 was still running after
# 90+ seconds, almost entirely process-startup overhead, not computation.
# Single-process is both simpler and faster at this problem size.

CurveFitter(graded_objective).fit(
    method="differential_evolution", popsize=30, polish=True, seed=1
)
print(f"logl after fit:  {graded_objective.logl():.3f}")
print(f"chisqr after fit: {graded_objective.chisqr():.3f}")
print(f"recovered total_thick        = {graded_film.total_thick.value:.2f} A")
print(f"recovered surface_roughness  = {graded_film.surface_roughness.value:.2f} A")
print(f"recovered oxide thick        = {oxide_slab.thick.value:.2f} A")
print(f"recovered oxide rough        = {oxide_slab.rough.value:.2f} A")
print(f"recovered scale_s            = {graded_model.scale_s.at(ENERGY_EV).value:.4f}")
print(
    f"recovered theta_offset_s     = "
    f"{graded_model.theta_offset_s.at(ENERGY_EV).value:.4f}"
)
print(f"bkg (fixed at measured floor) = {graded_model.bkg.at(ENERGY_EV).value:.3e}")

# %% 3. Data vs. fit overlay -- specifically check the fringes past q=0.15
# survive instead of being buried under an inflated bkg

r_graded = graded_model(q, ENERGY_EV).s

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(q, r, yerr=r_err, fmt=".", ms=3, alpha=0.4, color="0.4", label="data")
ax.plot(q, r_graded, color="C0", lw=1.5, label="graded fit")
ax.axhline(BKG_ESTIMATE, color="0.6", ls=":", lw=1, label="measured background floor")
ax.set_yscale("log")
ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
ax.set_ylabel("Reflectivity")
ax.legend(fontsize="small")
ax.set_title(f"ZnPc nonresonant XRR, {ENERGY_EV / 1000:.2f} keV")
fig.tight_layout()
plt.show()
# %% 4. Recovered orientation/density profile, with roughness broadening
# applied at each interface -- confirms the fitted surface_roughness/Oxide
# rough values are actually shaping the profile, not sitting inert.
#
# Both the vacuum/film and film/Oxide edges must show a real erf-broadened
# transition here, not a hard step -- `_named_depth_walk` (refloxide.model)
# used to treat a BookendedComponent as a break in every roughness run on
# BOTH sides (it's already continuous, so nothing "ran" through it), which
# silently dropped surface_roughness/Oxide's own rough from these two edges
# regardless of their fitted value. Fixed directly in refloxide.model.

graded_model.structure.plot.param("orientation|density", roughness=True, pad=60.0)
plt.savefig("graded_orientation_density.png")
plt.show()

# %% 5. Density gradient on its own -- the standard way to SEE where an
# interface sits and how wide it is: a sharp step shows up as a narrow
# spike in d(density)/dz, a rough one as a broad hump. Confirms the two
# bookended edges (vacuum/film, film/Oxide) are genuinely broadened by
# their own fitted roughness, not just visually smoothed in the profile
# plot above.

z_grad = graded_model.structure.depth_grid(num_points=4000, pad=60.0)
_, density_grad = graded_model.structure.density_profile_at(z_grad, roughness=True)
density_gradient = np.gradient(density_grad, z_grad)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(z_grad, density_gradient, color="C2")
ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
ax.set_ylabel(r"$d(\mathrm{density})/dz$ (g cm$^{-3}$ $\mathrm{\AA}^{-1}$)")
ax.set_title("Density gradient: interface location and width")
fig.tight_layout()
plt.savefig("graded_density_gradient.png")
plt.show()

