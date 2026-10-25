#!/usr/bin/env python3
"""Single-slab film whose optics vary with exponential diffusion at the surface.

One material, the full ZnPc tensor (`PeakSumSLD`), in one `DepthProfile` with no
`Mix`. The film is 250 Å thick on a native SiO2 layer and a silicon substrate:
vacuum | film | oxide | Si. Each varying quantity is an exponential `Diffusion`
from its surface value to its bulk value:

- orientation `gamma`: `MAGIC_ANGLE` at the surface, the measured `gamma_ref` in
  the bulk;
- `density_scale`: 0.60 at the surface, 1.00 in the bulk;
- heights of the four strongest pi* peaks (energy below 286 eV): 0.35 times the
  bulk height at the surface, the fitted (bulk) height in the bulk;
- the TDM tilt of the strongest pi* peak: 35 degrees at the surface, its derived
  bulk tilt in the bulk.

Every `Diffusion` shares one `length` Parameter named ``diffusion_length``
(60 Å), so one fit parameter drives all of them. The values are illustrative.

`build_reference()` is the same film with every diffusion replaced by its bulk
value, a uniform slab, for comparison.

The frozen ZnPc peak set and the helpers come from
``examples/lineshape_materials.py``.

Regenerate the docs figures::

    uv run python examples/lineshape_surface_diffusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from refnx.analysis import Parameter

from refloxide.lineshapes import MAGIC_ANGLE, Peak, PeakSumSLD
from refloxide.model import MaterialSLD, ReflectModel, Structure
from refloxide.profiles import DepthProfile, Diffusion

if TYPE_CHECKING:
    from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lineshape_materials import (
    DENSITY_ZNPC,
    PI_STAR_MAX_EV,
    baseline_table,
    load_materials,
    znpc_iso,
    znpc_tilts,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


# --8<-- [start:construction]
THICKNESS = 250.0  # film thickness, Angstrom (illustrative)
N_SLABS = 60  # adaptive microslab mesh (the default)
OXIDE_THICKNESS = 15.0  # native SiO2, Angstrom (illustrative)
# Illustrative roughness values. The film's own roughness is 0 (sharp); the
# diffusion provides the gradient.
OXIDE_ROUGHNESS = 3.0  # film/oxide interface, Angstrom
SUBSTRATE_ROUGHNESS = 3.0  # oxide/Si interface, Angstrom
DIFFUSION_LENGTH = 60.0  # 1/e depth of every diffusion, Angstrom
SURFACE_DENSITY_SCALE = 0.60  # density_scale at the surface (bulk is 1.0)
SURFACE_HEIGHT_FRACTION = 0.35  # pi* heights at the surface, times the bulk value
SURFACE_TILT = np.deg2rad(35.0)  # TDM tilt of the strongest pi* peak, surface
N_PI_STAR_HEIGHTS = 4  # pi* peaks whose heights diffuse


def _pi_star_order(records: list[dict[str, Any]]) -> list[int]:
    """Indices of the pi* peaks (energy below `PI_STAR_MAX_EV`), strongest first."""
    f_iso = np.array([znpc_iso(rec) for rec in records])
    energy = np.array([float(rec["energy"]) for rec in records])
    candidates = np.flatnonzero(energy < PI_STAR_MAX_EV)
    return [int(i) for i in candidates[np.argsort(-f_iso[candidates])]]


def build_profile(
    *, diffused: bool = True, length: float = DIFFUSION_LENGTH
) -> DepthProfile:
    """The ZnPc film in one `DepthProfile`.

    With `diffused=True` every varying channel is a `Diffusion` whose ``length``
    is one shared `Parameter` named ``diffusion_length``. With `diffused=False`
    each channel is its bulk number instead (the sharp reference).
    """
    materials = load_materials()
    records = materials["znpc"]["peaks"]
    gamma_ref, thetas, _ = znpc_tilts(records)
    f_iso = [znpc_iso(rec) for rec in records]
    strongest = _pi_star_order(records)
    diffusing_heights = set(strongest[:N_PI_STAR_HEIGHTS])
    tilt_index = strongest[0]
    shared_length = Parameter(
        float(length), name="diffusion_length", bounds=(1e-6, None)
    )

    def channel(edge: float, base: float, name: str) -> Any:
        if not diffused:
            return base
        return Diffusion(
            THICKNESS,
            kind="exponential",
            edge=edge,
            base=base,
            length=shared_length,
            name=name,
        )

    peaks = []
    for i, rec in enumerate(records):
        label = f"p{i + 1}"
        height: Any = f_iso[i]
        if i in diffusing_heights:
            height = channel(
                SURFACE_HEIGHT_FRACTION * f_iso[i], f_iso[i], f"znpc_{label}_height"
            )
        tilt: Any = float(thetas[i])
        if i == tilt_index:
            tilt = channel(SURFACE_TILT, float(thetas[i]), f"znpc_{label}_tdm_tilt")
        peaks.append(
            Peak(
                "gaussian",
                label=label,
                energy=float(rec["energy"]),
                width=float(rec["width"]),
                height=height,
                tdm_tilt=tilt,
            )
        )
    znpc = PeakSumSLD(
        peaks,
        baseline=baseline_table(materials["znpc"]),
        density=DENSITY_ZNPC,
        name="znpc",
    )
    return DepthProfile(
        znpc,
        thickness=THICKNESS,
        n_slabs=N_SLABS,
        gamma=channel(MAGIC_ANGLE, float(gamma_ref), "gamma"),
        density_scale=channel(SURFACE_DENSITY_SCALE, 1.0, "density_scale"),
        name="film",
    )


def _stack(film: DepthProfile) -> Structure:
    """Vacuum | film | native SiO2 | Si substrate."""
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    oxide = MaterialSLD("SiO2", density=2.2, name="native_oxide")(
        OXIDE_THICKNESS, OXIDE_ROUGHNESS
    )
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(
        0, SUBSTRATE_ROUGHNESS
    )
    return vacuum | film | oxide | substrate


def build_structure(length: float = DIFFUSION_LENGTH) -> Structure:
    """The diffused film on oxide and substrate.

    `length` is the diffusion length, Angstrom.
    """
    return _stack(build_profile(diffused=True, length=length))


def build_reference() -> Structure:
    """The sharp reference: the same film with every diffusion at its bulk value."""
    return _stack(build_profile(diffused=False))


# --8<-- [end:construction]


Q = np.linspace(0.01, 0.27, 520)  # 1/Angstrom
REFLECTIVITY_ENERGIES = np.array([284.4, 285.1, 285.9, 287.0, 288.7])  # eV
SCAN_ENERGY = 285.1  # eV, for the diffusion-length scan
LENGTH_SCAN = (5.0, 20.0, 60.0, 120.0)  # Angstrom
NEXAFS_FINE = np.linspace(283.5, 289.0, 276)  # eV, 0.02 eV steps
NEXAFS_DEPTHS = np.array([2.0, 30.0, 60.0, 120.0, 240.0])  # Angstrom, requested


def channel_profiles(
    n: int = 401, *, length: float = DIFFUSION_LENGTH
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Every diffused channel and peak address as `(z, values)` on a uniform grid.

    Keys are the `DepthProfile` channel names (``orientation``, ``density_scale``)
    and the peak addresses (``znpc.p<k>.height``, ``znpc.p<k>.tdm_tilt``). Angles
    are in radians.
    """
    z = np.linspace(0.0, THICKNESS, n)
    values = build_profile(length=length).named_values_at(z)
    return {key: (z, value) for key, value in values.items()}


def reflectivity(
    structure: Structure,
    energies: NDArray[np.float64],
    q: NDArray[np.float64] = Q,
) -> NDArray[np.float64]:
    """Reflectivity over `q` at `energies`, shape `(2, len(q), n_E)`.

    Index 0 is R_ss (s-polarized) and index 1 is R_pp (p-polarized).
    """
    result = ReflectModel(structure)(q, energies)
    return np.stack([result.s, result.p])


def fresnel_flattened(
    refl: NDArray[np.float64], q: NDArray[np.float64] = Q
) -> NDArray[np.float64]:
    """log10(R q^4) for each polarization, same shape as `refl`."""
    flat = refl * (q[:, None] ** 4)[None, :, :]
    return np.log10(np.clip(flat, 1e-30, None))


def local_tensors(
    profile: DepthProfile, energies: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Local lab tensors `(n_E, n_depth, 3, 3)` at the microslab nearest each depth.

    The depths are the microslab midpoints (Angstrom) nearest to
    `NEXAFS_DEPTHS`, returned alongside the tensors. The mesh is adaptive, so the
    midpoints are not evenly spaced.
    """
    tensors = []
    for energy in energies:
        rows, tensor = profile.rows_and_tensors_at(float(energy))
        tensors.append(tensor)
    thicknesses = rows[:, 0]
    z_mid = np.cumsum(thicknesses) - 0.5 * thicknesses
    index = np.abs(z_mid[None, :] - NEXAFS_DEPTHS[:, None]).argmin(axis=1)
    return z_mid[index], np.stack(tensors)[:, index]


def _depth_axis(ax: Any, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
    ax.set_xlim(0.0, THICKNESS)
    ax.grid(True, alpha=0.25)
    ax.axvspan(0.0, DIFFUSION_LENGTH, color="0.92", lw=0, zorder=0)
    ax.axvline(DIFFUSION_LENGTH, color="0.3", ls=":", lw=1.0, zorder=1)


def save_profiles(path: Path) -> None:
    """Diffusion profiles versus depth, with the bulk value as a dashed line."""
    import matplotlib.pyplot as plt

    values = channel_profiles()
    materials = load_materials()
    records = materials["znpc"]["peaks"]
    gamma_ref, thetas, _ = znpc_tilts(records)
    f_iso = [znpc_iso(rec) for rec in records]
    strongest = _pi_star_order(records)
    deg = 180.0 / np.pi

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0), constrained_layout=True)
    ax_g, ax_s, ax_h, ax_t = axes.flat

    z, gamma = values["orientation"]
    ax_g.plot(z, deg * gamma, color="C0", lw=1.8)
    ax_g.axhline(deg * gamma_ref, color="C0", ls="--", lw=0.9)
    _depth_axis(ax_g, r"orientation $\gamma$ (deg)")
    ax_g.set_ylabel(r"$\gamma$ (deg)")

    z, scale = values["density_scale"]
    ax_s.plot(z, scale, color="C1", lw=1.8)
    ax_s.axhline(1.0, color="C1", ls="--", lw=0.9)
    _depth_axis(ax_s, "density scale (multiplier, 1 = bulk)")
    ax_s.set_ylabel("density scale (-)")

    for k, idx in enumerate(strongest[:N_PI_STAR_HEIGHTS]):
        key = f"znpc.p{idx + 1}.height"
        z, height = values[key]
        color = f"C{k}"
        ax_h.plot(z, height, color=color, lw=1.6, label=f"p{idx + 1}")
        ax_h.axhline(f_iso[idx], color=color, ls="--", lw=0.8)
    ax_h.plot([], [], color="0.3", ls="--", lw=0.8, label="bulk")
    _depth_axis(ax_h, r"$\pi^*$ peak heights (unit-density $\beta$)")
    ax_h.set_ylabel(r"height (unit-density $\beta$)")
    top = max(f_iso[idx] for idx in strongest[:N_PI_STAR_HEIGHTS])
    ax_h.set_ylim(0.0, 1.35 * top)  # headroom for the legend, one row
    ax_h.legend(frameon=False, fontsize=8, ncol=5, loc="upper center")

    tilt_idx = strongest[0]
    z, tilt = values[f"znpc.p{tilt_idx + 1}.tdm_tilt"]
    ax_t.plot(z, deg * tilt, color="C3", lw=1.8)
    ax_t.axhline(deg * float(thetas[tilt_idx]), color="C3", ls="--", lw=0.9)
    _depth_axis(
        ax_t,
        rf"strongest $\pi^*$ TDM tilt, p{tilt_idx + 1} "
        rf"({float(records[tilt_idx]['energy']):.2f} eV)",
    )
    ax_t.set_ylabel(r"TDM tilt $\theta$ (deg)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_nexafs(path: Path) -> None:
    """Local ordinary and extraordinary beta(E) at several depths, and the bulk."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    z, tensors = local_tensors(build_profile(), NEXAFS_FINE)
    _, reference = local_tensors(build_profile(diffused=False), NEXAFS_FINE)
    bulk = reference[:, 0]  # uniform film: any microslab gives the bulk value
    norm = Normalize(z.min(), z.max())
    colors = plt.get_cmap("viridis")(norm(z))
    fig, axes = plt.subplots(
        1, 2, figsize=(8.8, 3.9), sharey=True, constrained_layout=True
    )
    for j in range(z.size):
        axes[0].plot(NEXAFS_FINE, tensors[:, j, 0, 0].imag, color=colors[j], lw=1.4)
        axes[1].plot(NEXAFS_FINE, tensors[:, j, 2, 2].imag, color=colors[j], lw=1.4)
    axes[0].plot(
        NEXAFS_FINE, bulk[:, 0, 0].imag, color="k", ls="--", lw=1.6, label="sharp bulk"
    )
    axes[1].plot(
        NEXAFS_FINE, bulk[:, 2, 2].imag, color="k", ls="--", lw=1.6, label="sharp bulk"
    )
    axes[0].set_title(r"$\beta_o(E)$, ordinary ($[0,0]$)")
    axes[1].set_title(r"$\beta_e(E)$, extraordinary ($[2,2]$)")
    for ax in axes:
        ax.set_xlim(283.5, 289.0)
        ax.set_xlabel("photon energy (eV)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc=(0.02, 0.6))
    axes[0].set_ylabel(r"local $\beta$ (unit density)")
    mappable = ScalarMappable(norm=norm, cmap="viridis")
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label(r"depth $z$ ($\mathrm{\AA}$), nearest microslab midpoint")
    cbar.set_ticks(z)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_reflectivity(path: Path) -> None:
    """R_ss and R_pp at five energies: diffused (solid) against sharp bulk (dashed)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    flat = fresnel_flattened(reflectivity(build_structure(), REFLECTIVITY_ENERGIES))
    flat_ref = fresnel_flattened(reflectivity(build_reference(), REFLECTIVITY_ENERGIES))
    norm = Normalize(REFLECTIVITY_ENERGIES[0], REFLECTIVITY_ENERGIES[-1])
    colors = plt.get_cmap("viridis")(norm(REFLECTIVITY_ENERGIES))
    n_e = REFLECTIVITY_ENERGIES.size
    fig, axes = plt.subplots(
        n_e, 2, figsize=(8.8, 9.6), sharex=True, constrained_layout=True
    )
    for j in range(n_e):
        for pol, title in (
            (0, r"$R_{ss}$ (s-polarized)"),
            (1, r"$R_{pp}$ (p-polarized)"),
        ):
            ax = axes[j, pol]
            ax.plot(Q, flat[pol][:, j], color=colors[j], lw=1.3, label="diffused")
            ax.plot(
                Q,
                flat_ref[pol][:, j],
                color=colors[j],
                lw=1.0,
                ls="--",
                label="sharp bulk",
            )
            lo = min(flat[pol][:, j].min(), flat_ref[pol][:, j].min())
            hi = max(flat[pol][:, j].max(), flat_ref[pol][:, j].max())
            ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.25 * (hi - lo))
            ax.text(
                0.02,
                0.92,
                f"{REFLECTIVITY_ENERGIES[j]:.1f} eV",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )
            ax.grid(True, alpha=0.25)
            if j == 0:
                ax.set_title(title)
            if j == n_e - 1:
                ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
            if pol == 0:
                ax.set_ylabel(r"$\log_{10}(R\,q^4)$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(r"reflectivity: $\log_{10}(R\,q^4)$, $q$ in $\mathrm{\AA}^{-1}$")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_length_scan(path: Path) -> None:
    """R_pp at 285.1 eV for several diffusion lengths, with the sharp bulk dashed."""
    import matplotlib.pyplot as plt

    energies = np.array([SCAN_ENERGY])
    cmap = plt.get_cmap("Blues")
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for k, length in enumerate(LENGTH_SCAN):
        shade = cmap(0.35 + 0.6 * k / max(len(LENGTH_SCAN) - 1, 1))
        flat = fresnel_flattened(reflectivity(build_structure(length), energies))
        ax.plot(Q, flat[1][:, 0], color=shade, lw=1.6, label=f"L = {length:g} Å")
    flat_ref = fresnel_flattened(reflectivity(build_reference(), energies))
    ax.plot(Q, flat_ref[1][:, 0], color="k", ls="--", lw=1.4, label="sharp bulk")
    ax.set_xlim(Q[0], Q[-1])
    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$\log_{10}(R_{pp}\,q^4)$, $q$ in $\mathrm{\AA}^{-1}$")
    ax.set_title(rf"$R_{{pp}}$ at {SCAN_ENERGY:.1f} eV versus diffusion length")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import time

    start = time.perf_counter()
    ASSETS.mkdir(parents=True, exist_ok=True)
    save_profiles(ASSETS / "lineshape-diffusion-profiles.png")
    save_nexafs(ASSETS / "lineshape-diffusion-nexafs.png")
    save_reflectivity(ASSETS / "lineshape-diffusion-reflectivity.png")
    save_length_scan(ASSETS / "lineshape-diffusion-length-scan.png")
    print(f"wrote 4 figures to {ASSETS} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
