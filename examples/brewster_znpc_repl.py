"""Soft X-ray Brewster angle: 20 nm ZnPc on Si, s vs p reflectivity.

Run cell-by-cell (each ``# %%`` marker is one cell) or top-to-bottom with::

    uv run python examples/brewster_znpc_repl.py

Stack: vacuum / ZnPc (200 A, 2.0 g/cm^3) / Si (2.33 g/cm^3), all interfaces
ideally sharp, at 250 eV. Soft X-ray refractive indices sit at n = 1 - delta
with delta << 1, so the vacuum/film Brewster angle from the surface normal is
near 45 deg and the grazing angle is likewise near 45 deg. Only p-pol
(R_pp) develops a deep minimum there; s-pol (R_ss) does not.

Polarization labeling: ``ReflectModel`` maps ``Reflectivity.s <- refl[:,0,0]``
and ``Reflectivity.p <- refl[:,1,1]``. Against Fresnel vacuum/Si at this
energy, ``refl[:,0,0]`` matches physical R_pp (Brewster channel) and
``refl[:,1,1]`` matches physical R_ss — the same pairing as
``python.model`` ``pol='p'`` / ``pol='s'`` (see ``model_objective_repl.py``).
This script therefore plots physical R_pp from ``r.s`` and R_ss from ``r.p``.
"""

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from periodictable.xsf import index_of_refraction

from refloxide.model import MaterialSLD, ReflectModel

ENERGY_EV = 250.0
HC_EV_A = 12398.42
ZNPC_THICK_A = 200.0
ZNPC_DENSITY = 2.0
SI_DENSITY = 2.33
FIGURE_PATH = Path("/tmp/brewster_znpc.png")

# %% Stack: vacuum / ZnPc / Si

vacuum = MaterialSLD("", 0.0, name="vacuum")(0.0, 0.0)
znpc = MaterialSLD("C32H16N8Zn", density=ZNPC_DENSITY, name="ZnPc")(ZNPC_THICK_A, 0.0)
silicon = MaterialSLD("Si", density=SI_DENSITY, name="silicon")(0.0, 0.0)
structure = vacuum | znpc | silicon
model = ReflectModel(structure)

# %% Grazing-angle sweep through the Brewster neighborhood

wavelength_a = HC_EV_A / ENERGY_EV
theta_deg = np.linspace(5.0, 70.0, 1401)
q = (4.0 * np.pi / wavelength_a) * np.sin(np.deg2rad(theta_deg))
r = model(q, ENERGY_EV)
r_pp = r.s
r_ss = r.p

n_znpc = complex(
    index_of_refraction("C32H16N8Zn", density=ZNPC_DENSITY, energy=ENERGY_EV * 1e-3)
)
theta_b_normal = float(np.rad2deg(np.arctan(n_znpc.real)))
theta_b_theory = 90.0 - theta_b_normal

brewster_window = (theta_deg > theta_b_theory - 8.0) & (
    theta_deg < theta_b_theory + 8.0
)
imin_local = int(np.argmin(r_pp[brewster_window]))
theta_b_meas = float(theta_deg[brewster_window][imin_local])
q_b_meas = float(q[brewster_window][imin_local])
r_pp_min = float(r_pp[brewster_window][imin_local])
r_ss_at_b = float(r_ss[brewster_window][imin_local])

print(f"energy = {ENERGY_EV:.0f} eV, lambda = {wavelength_a:.3f} A")
print(f"n_ZnPc = {n_znpc.real:.6f}{n_znpc.imag:+.6e}j")
print(
    f"Brewster (vacuum -> ZnPc): "
    f"theta_normal = {theta_b_normal:.3f} deg, "
    f"theta_grazing = {theta_b_theory:.3f} deg"
)
print(
    f"measured R_pp minimum near theory: "
    f"theta = {theta_b_meas:.3f} deg, q = {q_b_meas:.5f} 1/A, "
    f"R_pp = {r_pp_min:.3e}, R_ss = {r_ss_at_b:.3e}, "
    f"R_pp/R_ss = {r_pp_min / r_ss_at_b:.3e}"
)
assert r_pp_min < 0.1 * r_ss_at_b, "p-pol should be far below s-pol at Brewster"
assert abs(theta_b_meas - theta_b_theory) < 3.0, (
    "R_pp minimum should sit near the vacuum/ZnPc Brewster grazing angle"
)

# %% Reflectivity vs grazing angle and vs q

fig, (ax_th, ax_q) = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)

ax_th.semilogy(theta_deg, r_ss, label=r"$R_{ss}$ (s)", lw=2)
ax_th.semilogy(theta_deg, r_pp, label=r"$R_{pp}$ (p)", lw=2)
ax_th.axvline(
    theta_b_meas,
    color="0.35",
    ls="--",
    lw=1.2,
    label=rf"$\theta_{{\mathrm{{B}}}}$ = {theta_b_meas:.2f}$^\circ$",
)
ax_th.axvline(theta_b_theory, color="0.6", ls=":", lw=1.0, label="vacuum/ZnPc theory")
ax_th.set_xlabel(r"grazing angle $\theta$ (deg)")
ax_th.set_ylabel("Reflectivity")
ax_th.legend(frameon=False)
ax_th.set_title("vs grazing angle")

ax_q.semilogy(q, r_ss, label=r"$R_{ss}$ (s)", lw=2)
ax_q.semilogy(q, r_pp, label=r"$R_{pp}$ (p)", lw=2)
ax_q.axvline(q_b_meas, color="0.35", ls="--", lw=1.2)
ax_q.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
ax_q.legend(frameon=False)
ax_q.set_title("vs momentum transfer")

fig.suptitle(
    rf"vacuum / ZnPc ({ZNPC_THICK_A:.0f} $\mathrm{{\AA}}$, "
    rf"{ZNPC_DENSITY:g} g/cm$^3$) / Si @ {ENERGY_EV:.0f} eV",
    y=1.02,
)
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
print(f"wrote {FIGURE_PATH}")
plt.show()
