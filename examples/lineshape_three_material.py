#!/usr/bin/env python3
"""Three-material film: frozen Alq3 and ZnPc fits, a scrambled ZnPc, and gradients.

The materials come from the frozen lineshapes fits in
``examples/data/lineshapes_materials.json`` (written by
``examples/data/freeze_lineshapes_fits.py``): an isotropic Alq3 peak set and a
ZnPc set with five-angle film-frame strengths. Three materials fill one 200 A
film on a native SiO2 layer and a silicon substrate: vacuum | film | oxide | Si.

- ``alq3``: isotropic peaks. The first five peaks get the depth gradients of
  the surface-rich outline (heights, TDM tilts); the rest are constant.
- ``znpc``: the full uniaxial ZnPc tensor. Each peak's height is the isotropic
  magic-angle strength, and its molecular TDM tilt reproduces the measured
  film-frame anisotropy at the measured molecular tilt `gamma_ref`.
- ``znpc_scrambled``: the ZnPc peak set with three height swaps, jittered heights
  and small energy shifts, all from one seed. It is a synthetic variant for the
  demo, not a measured material.

One shared depth shape, a natural cubic `SplineShape` through four knots, drives
every gradient: the volume fractions, the total density, the orientation gamma,
and the heights and TDM tilts of the first five Alq3 peaks.

Regenerate the docs figures::

    uv run python examples/lineshape_three_material.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy.optimize import brentq

from refloxide.data import OpticalConstants
from refloxide.fields import LevelField, Shape, SplineShape
from refloxide.lineshapes import MAGIC_ANGLE, Peak, PeakSumSLD, tdm_fractions
from refloxide.model import MaterialSLD, Mix, ReflectModel, Structure
from refloxide.profiles import DepthProfile

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib.pyplot as plt
    from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


# --8<-- [start:construction]
DATA_FILE = Path(__file__).resolve().parent / "data" / "lineshapes_materials.json"
# Illustrative film densities (g/cm^3). The frozen tables are per unit density.
DENSITY_ALQ3 = 1.31
DENSITY_ZNPC = 1.60
# Seeded scramble of the ZnPc peaks (one generator, drawn in this order).
SCRAMBLE_SEED = 2025
SCRAMBLE_PAIRS = 3  # disjoint height swaps
SCRAMBLE_TOP = 8  # swap partners come from this many strongest peaks (by F_iso)
SCRAMBLE_WINDOW = (283.5, 289.5)  # eV, swap candidates lie inside the demo window
JITTER_SIGMA = 0.15  # lognormal sigma of each height factor
JITTER_CLIP = (0.7, 1.4)  # height factors are clipped to this range
SHIFT_SIGMA = 0.05  # eV, Gaussian energy shift of each peak
SHIFT_CLIP = 0.10  # eV, energy shifts are clipped to +/- this

THICKNESS = 200.0  # film thickness, Angstrom
N_SLABS = 40
# Knot positions as fractions of THICKNESS: surface, two interior points,
# film/oxide interface.
FRACTIONS = np.array([0.0, 0.3, 0.6, 1.0])
KNOTS = THICKNESS * FRACTIONS  # spline knots, Angstrom
OXIDE_THICKNESS = 15.0  # native SiO2, Angstrom (illustrative)
# Illustrative values. Roughness of film/oxide is the oxide's `rough`;
# roughness of oxide/Si is the substrate's `rough`.
OXIDE_ROUGHNESS = 3.0  # film/oxide interface, Angstrom
SUBSTRATE_ROUGHNESS = 3.0  # oxide/Si interface, Angstrom
Q = np.linspace(0.01, 0.27, 520)  # 1/Angstrom
# Dense energy set across the carbon K edge and the pi* region, 0.1 eV steps.
ENERGIES = np.arange(283.5, 289.0001, 0.1)  # eV
# Interior weights of the curved 4-level pattern: level_i = h * (1 + (r - 1) * w_i)
# gives a surface value r * h that relaxes to h at z = THICKNESS.
CURVE_WEIGHTS = (1.0, 1.0 / 3.0, 1.0 / 12.0, 0.0)
# Alq3 first-five-peak outline. Surface/substrate height ratios for p1..p3;
# TDM tilts (degrees) at the four anchors for p1, p2, p4, p5.
ALQ3_SURFACE_HEIGHT_RATIO = {"p1": 1.5, "p2": 2.0, "p3": 0.8}
ALQ3_TILTS_DEG = {
    "p1": (0.0, 12.0, 17.0, 20.0),
    "p2": (0.0, 14.0, 18.0, 20.0),
    "p4": (90.0, 85.0, 82.0, 80.0),
    "p5": (90.0, 89.0, 85.0, 80.0),
}
# ZnPc gamma: interior fractions of the way from MAGIC_ANGLE to gamma_ref.
ZNPC_GAMMA_FRACTIONS = (0.25, 0.72)
SCRAMBLED_GAMMA_FRACTIONS = (0.12, 0.55)


def make_shape() -> SplineShape:
    """Shared depth shape: a natural cubic spline through the four knots."""
    return SplineShape(KNOTS, name="depth")


def field(shape: Shape, name: str, levels: Sequence[float]) -> LevelField:
    """Level field with explicit values at the four anchors, z = 0 to z = THICKNESS."""
    return LevelField(shape, list(levels), name=name)


def load_materials() -> dict[str, Any]:
    """Frozen ZnPc and Alq3 peak sets and baselines (see the module docstring)."""
    return json.loads(DATA_FILE.read_text())["materials"]


def baseline_table(material: dict[str, Any]) -> OpticalConstants:
    """Unit-density baseline `delta + i beta` as an isotropic optical-constant table."""
    base = material["baseline"]
    delta = np.asarray(base["delta"], dtype=np.float64)
    beta = np.asarray(base["beta"], dtype=np.float64)
    table = pl.DataFrame(
        {
            "energy": np.asarray(base["energy_ev"], dtype=np.float64),
            "n_xx": delta,
            "n_ixx": beta,
            "n_zz": delta,
            "n_izz": beta,
        }
    )
    return OpticalConstants.from_source(table)


def _film_ratio(u: float, gamma: float) -> float:
    """Film-frame ratio `n_e / n_o` for a TDM with `u = cos^2(theta)`.

    Uses `tdm_fractions`: `f_zz = 3u`, `f_xx = 1.5 (1 - u)`. At `u = 1` (TDM along
    the molecular normal) it is `2 cos^2(gamma) / sin^2(gamma)`.
    """
    f_xx, f_zz = tdm_fractions(np.arccos(np.sqrt(u)))
    c2, s2 = np.cos(gamma) ** 2, np.sin(gamma) ** 2
    n_e = f_zz * c2 + f_xx * s2
    n_o = (f_xx * (1.0 + c2) + f_zz * s2) / 2.0
    return float(n_e / n_o)


PI_STAR_MAX_EV = 286.0  # pi* peaks lie below this energy (eV)
PI_STAR_FRACTION = 0.1  # ... and have F_iso above this fraction of the largest


def znpc_iso(rec: dict[str, Any]) -> float:
    """Isotropic strength `(2 F_xx + F_zz) / 3` of one ZnPc peak record.

    This is the peak height: the tensor solve's isotropic strength.
    """
    return (2.0 * float(rec["F_xx"]) + float(rec["F_zz"])) / 3.0


def _gaussian(
    label: str, rec: dict[str, Any], height: Any, tilt: Any = MAGIC_ANGLE
) -> Peak:
    """Gaussian `Peak` at a frozen record's energy and width."""
    return Peak(
        "gaussian",
        label=label,
        energy=float(rec["energy"]),
        width=float(rec["width"]),
        height=height,
        tdm_tilt=tilt,
    )


def _sorted_alq3(material: dict[str, Any]) -> list[dict[str, Any]]:
    """Alq3 peak records in energy order (p1 is the lowest-energy peak)."""
    return sorted(material["peaks"], key=lambda p: p["energy"])


def znpc_tilts(
    peaks: Sequence[dict[str, Any]] | None = None,
) -> tuple[float, NDArray[np.float64], list[int]]:
    """Molecular ZnPc orientation and per-peak TDM tilts from the five-angle series.

    Parameters
    ----------
    peaks : sequence of dict, optional
        ZnPc peak records from the frozen file (``energy``, ``width``,
        ``F_xx``, ``F_zz``). Defaults to the frozen ZnPc set.

    Returns
    -------
    gamma_ref : float
        Measured molecular tilt (radians) from the strong pi* peaks (energy below
        `PI_STAR_MAX_EV`, F_iso above `PI_STAR_FRACTION` of the largest), assuming
        their TDM lies along the molecular normal. `tan(gamma_ref) = sqrt(2 / r)`
        with `r = sum(F_zz) / sum(F_xx)` over those peaks.
    thetas : NDArray[np.float64]
        TDM polar angle (radians, in [0, pi/2]) per peak, solved so that the film
        ratio `F_zz / F_xx` at `gamma_ref` equals the measured ratio. A measured
        ratio above the maximum reachable ratio (at theta = pi/2) clips to
        pi/2; a ratio below the minimum reachable ratio (at theta = 0), including
        `F_zz = 0`, clips to 0. Clipped peaks are the closest reachable response
        at `gamma_ref`.
    clipped : list of int
        Indices into ``peaks`` of the clipped peaks.
    """
    if peaks is None:
        peaks = load_materials()["znpc"]["peaks"]
    energy = np.array([p["energy"] for p in peaks])
    f_iso = np.array([znpc_iso(p) for p in peaks])
    f_xx = np.array([p["F_xx"] for p in peaks])
    f_zz = np.array([p["F_zz"] for p in peaks])

    pi_star = (energy < PI_STAR_MAX_EV) & (f_iso > PI_STAR_FRACTION * f_iso.max())
    ratio_pi = f_zz[pi_star].sum() / f_xx[pi_star].sum()
    gamma_ref = float(np.arctan(np.sqrt(2.0 / ratio_pi)))

    lo = _film_ratio(1.0, gamma_ref)  # theta = 0
    hi = _film_ratio(0.0, gamma_ref)  # theta = pi/2
    thetas = np.empty(len(peaks))
    clipped: list[int] = []
    for k in range(len(peaks)):
        measured = f_zz[k] / f_xx[k]
        if measured <= lo:
            thetas[k] = 0.0
            clipped.append(k)
        elif measured >= hi:
            thetas[k] = np.pi / 2
            clipped.append(k)
        else:
            # ratio(u) is monotone in u = cos^2(theta); solve on [0, 1].
            root = brentq(lambda u, m=measured: _film_ratio(u, gamma_ref) - m, 0.0, 1.0)
            thetas[k] = float(np.arccos(np.sqrt(root)))
    return gamma_ref, thetas, clipped


def alq3_peaks(shape: Shape, material: dict[str, Any]) -> list[Peak]:
    """Alq3 peaks p1..p14 sorted by energy; the first five carry the outline."""
    peaks = []
    for i, rec in enumerate(_sorted_alq3(material)):
        label = f"p{i + 1}"
        height: Any = float(rec["height"])
        tilt: Any = MAGIC_ANGLE
        if label in ALQ3_SURFACE_HEIGHT_RATIO:
            ratio = ALQ3_SURFACE_HEIGHT_RATIO[label]
            levels = [rec["height"] * (1.0 + (ratio - 1.0) * w) for w in CURVE_WEIGHTS]
            height = field(shape, f"alq3_{label}_height", levels)
        if label in ALQ3_TILTS_DEG:
            tilt = field(
                shape,
                f"alq3_{label}_tilt",
                np.deg2rad(ALQ3_TILTS_DEG[label]).tolist(),
            )
        peaks.append(_gaussian(label, rec, height, tilt))
    return peaks


def znpc_peaks(material: dict[str, Any]) -> list[Peak]:
    """ZnPc peaks: height `F_iso`, TDM tilt from `znpc_tilts`, all constant."""
    records = material["peaks"]
    _, thetas, _ = znpc_tilts(records)
    return [
        _gaussian(f"p{i + 1}", rec, znpc_iso(rec), float(thetas[i]))
        for i, rec in enumerate(records)
    ]


def scramble_draw(
    records: Sequence[dict[str, Any]],
) -> tuple[list[tuple[int, int]], NDArray[np.float64], NDArray[np.float64]]:
    """Seeded scramble of the ZnPc peaks: swap pairs, heights and energies.

    Returns the swap pairs (indices into ``records``), the absolute heights
    after the swaps and the height jitter, and the shifted energies (eV). The
    swap candidates are the `SCRAMBLE_TOP` strongest peaks by `F_iso` whose
    original energy lies in `SCRAMBLE_WINDOW`; the seeded generator draws the
    disjoint pairs among them. Every peak gets a lognormal height factor and a
    clipped Gaussian energy shift.
    """
    rng = np.random.default_rng(SCRAMBLE_SEED)
    f_iso = np.array([znpc_iso(rec) for rec in records])
    energy = np.array([float(rec["energy"]) for rec in records])
    in_window = np.flatnonzero(
        (energy >= SCRAMBLE_WINDOW[0]) & (energy <= SCRAMBLE_WINDOW[1])
    )
    strongest = in_window[np.argsort(-f_iso[in_window])][:SCRAMBLE_TOP]
    chosen = rng.permutation(strongest)[: 2 * SCRAMBLE_PAIRS]
    pairs = [(int(a), int(b)) for a, b in chosen.reshape(SCRAMBLE_PAIRS, 2)]
    heights = f_iso.copy()
    for a, b in pairs:
        heights[[a, b]] = f_iso[[b, a]]
    jitter = np.clip(rng.lognormal(0.0, JITTER_SIGMA, len(records)), *JITTER_CLIP)
    shift = np.clip(rng.normal(0.0, SHIFT_SIGMA, len(records)), -SHIFT_CLIP, SHIFT_CLIP)
    return pairs, heights * jitter, energy + shift


def scrambled_peaks(material: dict[str, Any]) -> list[Peak]:
    """ZnPc peaks of the scrambled variant: absolute heights, tilts from `znpc`.

    Tilts and widths stay with their peaks. Only heights and energies change.
    """
    records = material["peaks"]
    _, thetas, _ = znpc_tilts(records)
    _, heights, energies = scramble_draw(records)
    return [
        Peak(
            "gaussian",
            label=f"p{i + 1}",
            energy=float(energies[i]),
            width=float(rec["width"]),
            height=float(heights[i]),
            tdm_tilt=float(thetas[i]),
        )
        for i, rec in enumerate(records)
    ]


def curved_gamma(gamma_ref: float, fractions: tuple[float, float]) -> list[float]:
    """Four-anchor orientation: MAGIC_ANGLE at the surface, gamma_ref at the bottom."""
    a, b = fractions
    span = gamma_ref - MAGIC_ANGLE
    return [MAGIC_ANGLE, MAGIC_ANGLE + a * span, MAGIC_ANGLE + b * span, gamma_ref]


def build_profile() -> DepthProfile:
    """The film: a linear `Mix` of three peak-sum sources on one shared shape."""
    shape = make_shape()
    materials = load_materials()
    gamma_ref, _, _ = znpc_tilts(materials["znpc"]["peaks"])

    alq3 = PeakSumSLD(
        alq3_peaks(shape, materials["alq3"]),
        baseline=baseline_table(materials["alq3"]),
        density=DENSITY_ALQ3,
        name="alq3",
    )
    znpc = PeakSumSLD(
        znpc_peaks(materials["znpc"]),
        baseline=baseline_table(materials["znpc"]),
        density=DENSITY_ZNPC,
        name="znpc",
    )
    znpc_scrambled = PeakSumSLD(
        scrambled_peaks(materials["znpc"]),
        baseline=baseline_table(materials["znpc"]),
        density=DENSITY_ZNPC,
        name="znpc_scrambled",
    )
    return DepthProfile(
        Mix([alq3, znpc, znpc_scrambled], rule="linear"),
        thickness=THICKNESS,
        n_slabs=N_SLABS,
        mesh="uniform",
        # Volume fractions, at the four anchors. The three columns sum to 1 at
        # every anchor. Alq3 sits mostly at the surface. The two ZnPc layers
        # enrich toward the substrate.
        phi=[
            field(shape, "phi_alq3", (0.85, 0.55, 0.45, 0.40)),
            field(shape, "phi_znpc", (0.10, 0.33, 0.35, 0.35)),
            field(shape, "phi_znpc_scrambled", (0.05, 0.12, 0.20, 0.25)),
        ],
        # Orientation gamma (radians), one field per material. Alq3 has a
        # face-on surface (gamma = 0) and reaches the magic angle at the
        # substrate. ZnPc starts amorphous and orders toward gamma_ref.
        gamma=[
            field(shape, "gamma_alq3", (0.0, 0.10, 0.50, MAGIC_ANGLE)),
            field(
                shape,
                "gamma_znpc",
                curved_gamma(gamma_ref, ZNPC_GAMMA_FRACTIONS),
            ),
            field(
                shape,
                "gamma_znpc_scrambled",
                curved_gamma(gamma_ref, SCRAMBLED_GAMMA_FRACTIONS),
            ),
        ],
        # Total density, after mixing: changes mostly toward z = THICKNESS.
        density_scale=field(shape, "density_scale", (0.85, 0.87, 0.97, 1.15)),
        name="film",
    )


def build_structure() -> Structure:
    """Vacuum | three-material film | native SiO2 | Si substrate.

    Roughness follows the library convention: a slab's `rough` is the roughness
    of the interface directly above it. So the film's own `roughness` (left at
    0, sharp) is the vacuum/film interface, the oxide's `rough` is the
    film/oxide interface, and the substrate's `rough` is the oxide/Si
    interface. The oxide thickness and both roughness values are illustrative.
    """
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    oxide = MaterialSLD("SiO2", density=2.2, name="native_oxide")(
        OXIDE_THICKNESS, OXIDE_ROUGHNESS
    )
    substrate = MaterialSLD("Si", density=2.33, name="substrate")(
        0, SUBSTRATE_ROUGHNESS
    )
    return vacuum | build_profile() | oxide | substrate


# --8<-- [end:construction]


def channel_profiles(
    n: int = 401,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Every channel and depth-field peak parameter as `(z, values)`.

    Keys are the `DepthProfile` channel names (`phi_0`, `density_scale`,
    `orientation_0`, ...) and the peak addresses (`alq3.p1.height`,
    `alq3.p1.tdm_tilt`, ...). Angles are in radians.
    """
    z = np.linspace(0.0, THICKNESS, n)
    values = build_profile().named_values_at(z)
    return {key: (z, value) for key, value in values.items()}


def _depth_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
    ax.grid(True, alpha=0.25)


def save_channels(path: Path) -> None:
    """Every channel versus depth (angles in degrees for display)."""
    import matplotlib.pyplot as plt

    values = channel_profiles()
    deg = 180.0 / np.pi
    panels = [
        (
            "volume fraction $\\phi$",
            1.0,
            [("phi_0", "alq3"), ("phi_1", "znpc"), ("phi_2", "znpc_scrambled")],
        ),
        ("total density scale", 1.0, [("density_scale", "density_scale")]),
        (
            "orientation $\\gamma$ (deg)",
            deg,
            [
                ("orientation_0", "alq3"),
                ("orientation_1", "znpc"),
                ("orientation_2", "znpc_scrambled"),
            ],
        ),
        (
            "alq3 peak heights",
            1.0,
            [
                ("alq3.p1.height", "p1"),
                ("alq3.p2.height", "p2"),
                ("alq3.p3.height", "p3"),
            ],
        ),
        (
            "alq3 TDM tilt $\\theta$ (deg)",
            deg,
            [
                ("alq3.p1.tdm_tilt", "p1"),
                ("alq3.p2.tdm_tilt", "p2"),
                ("alq3.p4.tdm_tilt", "p4"),
                ("alq3.p5.tdm_tilt", "p5"),
            ],
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.6), constrained_layout=True)
    for ax, (title, scale, curves) in zip(axes.flat, panels, strict=False):
        for key, label in curves:
            z, y = values[key]
            ax.plot(z, scale * y, label=label, lw=1.8)
        _depth_axis(ax, title)
        if title.startswith("alq3 peak heights"):
            # Headroom above the curves, with the legend in one row on top.
            ax.set_ylim(0.0, 1.45 * ax.get_ylim()[1])
            ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
        else:
            ax.legend(frameon=False, fontsize=8)
    axes.flat[-1].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


FINE = np.linspace(283.5, 289.0, 276)  # eV, 0.02 eV steps


def _building_block(
    peaks: list[Peak], baseline: OpticalConstants, *, rotation: float = 0.0
) -> PeakSumSLD:
    """Unit-density source with constant peaks, for the building-block panels."""
    return PeakSumSLD(peaks, baseline=baseline, density=1.0, rotation=rotation)


def save_materials(path: Path) -> None:
    """Building blocks: unit-density beta(E) of each material, and the ZnPc scramble."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    materials = load_materials()
    alq3_rec = materials["alq3"]
    znpc_rec = materials["znpc"]
    gamma_ref, _, _ = znpc_tilts(znpc_rec["peaks"])
    _, scr_heights, _ = scramble_draw(znpc_rec["peaks"])
    f_iso = np.array([znpc_iso(rec) for rec in znpc_rec["peaks"]])

    # Alq3: isotropic, fitted (substrate-end) heights, unit density.
    alq3_fitted = _const_alq3(alq3_rec)
    alq3_src = _building_block(alq3_fitted, baseline_table(alq3_rec))
    alq3_tensor = alq3_src.tensor_at_many(FINE)
    alq3_beta = alq3_tensor[:, 0, 0].imag

    # ZnPc: full tensor at gamma_ref (film frame), and the magic-angle average.
    znpc_src = _building_block(
        znpc_peaks(znpc_rec),
        baseline_table(znpc_rec),
        rotation=gamma_ref,
    )
    znpc_t = znpc_src.tensor_at_many(FINE)
    scrambled_src = _building_block(
        scrambled_peaks(znpc_rec),
        baseline_table(znpc_rec),
        rotation=gamma_ref,
    )
    scrambled_t = scrambled_src.tensor_at_many(FINE)

    fig = plt.figure(figsize=(9.2, 5.6), constrained_layout=True)
    grid = GridSpec(2, 3, figure=fig, height_ratios=[3.0, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_z = fig.add_subplot(grid[0, 1], sharey=ax_a)
    ax_s = fig.add_subplot(grid[0, 2], sharey=ax_a)
    ax_strip = fig.add_subplot(grid[1, :])

    ax_a.plot(FINE, alq3_beta, color="C0", lw=1.6, label=r"$\beta(E)$, isotropic")
    ax_a.vlines(
        [
            float(q["energy"])
            for q in sorted(alq3_rec["peaks"], key=lambda q: q["energy"])
        ],
        0.0,
        [
            float(q["height"])
            for q in sorted(alq3_rec["peaks"], key=lambda q: q["energy"])
        ],
        color="0.35",
        lw=0.9,
        label="fitted peak heights",
    )
    ax_a.set_title("alq3, isotropic (unit density)")

    ax_z.plot(
        FINE, znpc_t[:, 0, 0].imag, color="C0", lw=1.6, label=r"$\beta_o$ ($[0,0]$)"
    )
    ax_z.plot(
        FINE, znpc_t[:, 2, 2].imag, color="C3", lw=1.6, label=r"$\beta_e$ ($[2,2]$)"
    )
    ax_z.plot(
        FINE,
        znpc_t.imag.trace(axis1=1, axis2=2) / 3.0,
        color="0.25",
        lw=1.2,
        ls="--",
        label=r"isotropic average",
    )
    ax_z.set_title(
        rf"znpc, rotation $\gamma_{{ref}}$ = {np.rad2deg(gamma_ref):.1f}$^\circ$"
    )

    ax_s.plot(
        FINE,
        znpc_t.imag.trace(axis1=1, axis2=2) / 3.0,
        color="C0",
        lw=1.6,
        label="znpc, isotropic",
    )
    ax_s.plot(
        FINE,
        scrambled_t.imag.trace(axis1=1, axis2=2) / 3.0,
        color="C1",
        lw=1.6,
        ls="--",
        label="znpc_scrambled, isotropic",
    )
    ax_s.set_title("znpc vs znpc_scrambled")

    for ax in (ax_a, ax_z, ax_s):
        ax.set_xlim(283.5, 289.0)
        ax.set_xlabel("photon energy (eV)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax_a.set_ylabel(r"$\beta$ at unit density")
    ax_z.tick_params(labelleft=False)
    ax_s.tick_params(labelleft=False)

    # Only peaks inside the demo window are drawn; the others are left out.
    centers = np.array([float(rec["energy"]) for rec in znpc_rec["peaks"]])
    inside = (centers >= SCRAMBLE_WINDOW[0]) & (centers <= SCRAMBLE_WINDOW[1])
    pct = (100.0 * (scr_heights / f_iso - 1.0))[inside]
    centers = centers[inside]
    ax_strip.bar(centers, pct, width=0.12, color=np.where(pct >= 0, "C2", "C3"))
    ax_strip.axhline(0.0, color="0.4", lw=0.8)
    span = float(np.abs(pct).max())
    ax_strip.set_ylim(-1.25 * span, 1.25 * span)
    ax_strip.set_xlim(*SCRAMBLE_WINDOW)
    ax_strip.set_ylabel("height change (%)")
    ax_strip.set_xlabel(
        "ZnPc peak energy (eV, original), znpc_scrambled relative to znpc; "
        "peaks inside the window only"
    )
    ax_strip.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


NEXAFS_DEPTHS = np.array([5.0, 40.0, 80.0, 120.0, 160.0, 195.0])  # Angstrom, requested


def local_tensors(
    energies: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Local lab tensors `(n_E, n_depth, 3, 3)` at the microslab nearest each depth.

    Returns the midpoints actually used (`z`, Angstrom) alongside the tensors, so
    the depths are reported exactly rather than as the requested values.
    """
    profile = build_profile()
    z_mid = (np.arange(N_SLABS) + 0.5) * THICKNESS / N_SLABS
    index = np.abs(z_mid[None, :] - NEXAFS_DEPTHS[:, None]).argmin(axis=1)
    tensors = np.stack([profile.tensor_rows_at(float(e))[index] for e in energies])
    return z_mid[index], tensors


def save_nexafs(path: Path) -> None:
    """Local ordinary and extraordinary beta(E) at several depths."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    z, tensors = local_tensors(FINE)
    norm = Normalize(z.min(), z.max())
    colors = plt.get_cmap("viridis")(norm(z))  # colour = depth, as the colorbar
    fig, axes = plt.subplots(
        1, 2, figsize=(8.8, 3.9), sharey=True, constrained_layout=True
    )
    for j in range(z.size):
        axes[0].plot(FINE, tensors[:, j, 0, 0].imag, color=colors[j], lw=1.4)
        axes[1].plot(FINE, tensors[:, j, 2, 2].imag, color=colors[j], lw=1.4)
    axes[0].set_title(r"$\beta_o(E)$, ordinary ($[0,0]$)")
    axes[1].set_title(r"$\beta_e(E)$, extraordinary ($[2,2]$)")
    for ax in axes:
        ax.set_xlim(283.5, 289.0)
        ax.set_xlabel("photon energy (eV)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"local $\beta$")
    mappable = ScalarMappable(norm=norm, cmap="viridis")
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label(r"depth $z$ ($\mathrm{\AA}$), microslab midpoint")
    cbar.set_ticks(z)
    fig.savefig(path, dpi=150)
    plt.close(fig)


RIDGE_SPAN = 2.6  # ridge height in row spacings at the top of the window
RIDGE_DECADES = 2.8  # width of each panel's log10(R q^4) window
RIDGE_TOP_PCT = 99.9  # window top: this percentile of the panel's values


def reflectivity(energies: NDArray[np.float64] = ENERGIES) -> NDArray[np.float64]:
    """Reflectivity over `Q` at `energies`, shape `(2, len(Q), n_E)`.

    Index 0 is R_ss (s-polarized) and index 1 is R_pp (p-polarized).
    """
    result = ReflectModel(build_structure())(Q, energies)
    return np.stack([result.s, result.p])


def fresnel_flattened(refl: NDArray[np.float64]) -> NDArray[np.float64]:
    """log10(R q^4) for each polarization, shape `(2, len(Q), n_E)`.

    Multiplying by q^4 removes the steep Fresnel decay, so the structure
    of the curves sits on one scale.
    """
    flat = refl * (Q[:, None] ** 4)[None, :, :]
    return np.log10(np.clip(flat, 1e-30, None))


def _ridge_window(y: NDArray[np.float64]) -> tuple[float, float]:
    """Common `(bottom, top)` log window for one panel of ridges."""
    top = float(np.percentile(y, RIDGE_TOP_PCT))
    return top - RIDGE_DECADES, top


def _ridge_panel(
    ax: plt.Axes,
    y: NDArray[np.float64],
    colors: NDArray[np.float64],
    window: tuple[float, float],
) -> None:
    """Joyplot panel: row j is energy `ENERGIES[j]`, low energies drawn in front."""
    bottom, top = window
    n_e = y.shape[1]
    height = np.clip((y - bottom) / (top - bottom), 0.0, 1.0) * RIDGE_SPAN
    for j in range(n_e):
        front = n_e - j  # the row at the bottom of the page is in front
        ax.fill_between(Q, j, j + height[:, j], facecolor="white", lw=0, zorder=front)
        ax.plot(Q, j + height[:, j], color=colors[j], lw=0.9, zorder=front + 0.5)
    ax.set_xlim(Q[0], Q[-1])
    ax.set_ylim(-0.4, n_e - 1 + RIDGE_SPAN + 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")


def save_ridge(path: Path, flat: NDArray[np.float64]) -> None:
    """Joyplot of R_ss (left) and R_pp (right), both as log10(R q^4)."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    norm = Normalize(ENERGIES[0], ENERGIES[-1])
    colors = plt.get_cmap("viridis")(norm(ENERGIES))
    fig, axes = plt.subplots(
        1, 2, figsize=(8.8, 7.0), sharey=True, constrained_layout=True
    )
    for ax, pol, title in (
        (axes[0], 0, r"$R_{ss}$ (s-polarized)"),
        (axes[1], 1, r"$R_{pp}$ (p-polarized)"),
    ):
        window = _ridge_window(flat[pol])
        _ridge_panel(ax, flat[pol], colors, window)
        ax.set_title(title)
    ticks = np.arange(0, ENERGIES.size, 5)
    axes[0].set_yticks(ticks, [f"{ENERGIES[t]:.1f}" for t in ticks])
    axes[0].set_ylabel("photon energy (eV)")
    axes[1].tick_params(labelleft=False)
    mappable = ScalarMappable(norm=norm, cmap="viridis")
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("photon energy (eV)")
    fig.suptitle(r"reflectivity ridges: $R\,q^4$, log scale")
    fig.savefig(path, dpi=100)  # 56 ridges: a lower dpi keeps the PNG small
    plt.close(fig)


def _const_alq3(material: dict[str, Any]) -> list[Peak]:
    """Alq3 peaks at their fitted heights and magic-angle TDMs (no depth fields)."""
    return [
        _gaussian(f"p{i + 1}", rec, float(rec["height"]))
        for i, rec in enumerate(_sorted_alq3(material))
    ]


def main() -> None:
    import time

    start = time.perf_counter()
    ASSETS.mkdir(parents=True, exist_ok=True)
    refl = reflectivity()
    flat = fresnel_flattened(refl)
    save_channels(ASSETS / "lineshape-gradient-channels.png")
    save_materials(ASSETS / "lineshape-gradient-materials.png")
    save_nexafs(ASSETS / "lineshape-gradient-nexafs.png")
    save_ridge(ASSETS / "lineshape-gradient-ridge.png", flat)
    print(f"wrote 4 figures to {ASSETS} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
