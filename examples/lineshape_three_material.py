#!/usr/bin/env python3
"""Three-material film: frozen Alq3 and ZnPc fits, a shifted ZnPc, and depth gradients.

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
- ``znpc_shifted``: ZnPc with every peak height nudged by a fixed factor.

One shared depth shape drives every gradient: the volume fractions, the total
density, the orientation gamma, and the heights and TDM tilts of the first
five Alq3 peaks. By default the shape is a natural cubic `SplineShape` through
four knots. `build_structure` takes ``shape="polynomial"`` to use a cubic
`PolynomialShape` through the same anchor positions and level values instead.

Regenerate the docs figures::

    uv run python examples/lineshape_three_material.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from scipy.optimize import brentq

from refloxide.data import OpticalConstants
from refloxide.fields import LevelField, PolynomialShape, Shape, SplineShape
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
# Fixed seed for the shifted-ZnPc height factors.
SHIFT_SEED = 12345
SHIFT_RANGE = (0.85, 1.15)

THICKNESS = 200.0  # film thickness, Angstrom
N_SLABS = 40
# Knot and anchor positions as fractions of THICKNESS: surface, two interior
# points, film/oxide interface. Both shapes use these positions, so they
# compare fairly.
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
SHIFTED_GAMMA_FRACTIONS = (0.12, 0.55)


def make_shape(kind: Literal["spline", "polynomial"] = "spline") -> Shape:
    """Shared depth shape: a 4-knot spline, or a cubic on the same anchors."""
    if kind == "spline":
        return SplineShape(KNOTS, name="depth")
    if kind == "polynomial":
        return PolynomialShape(degree=3, positions=FRACTIONS, name="depth")
    msg = f"shape must be 'spline' or 'polynomial', got {kind!r}"
    raise ValueError(msg)


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


def znpc_tilts(
    peaks: Sequence[dict[str, Any]] | None = None,
) -> tuple[float, NDArray[np.float64], list[str]]:
    """Molecular ZnPc orientation and per-peak TDM tilts from the five-angle series.

    Parameters
    ----------
    peaks : sequence of dict, optional
        ZnPc peak records from the frozen file (``energy``, ``height_iso``,
        ``F_xx``, ``F_zz``). Defaults to the frozen ZnPc set.

    Returns
    -------
    gamma_ref : float
        Measured molecular tilt (radians) from the strong pi* peaks (energy below
        286 eV, `height_iso` above 10 % of the largest), assuming their TDM lies
        along the molecular normal. `tan(gamma_ref) = sqrt(2 / r)` with
        `r = sum(F_zz) / sum(F_xx)` over those peaks.
    thetas : NDArray[np.float64]
        TDM polar angle (radians, in [0, pi/2]) per peak, solved so that the film
        ratio `F_zz / F_xx` at `gamma_ref` equals the measured ratio. A measured
        ratio above the maximum reachable ratio (at theta = pi/2) clips to
        pi/2; a ratio below the minimum reachable ratio (at theta = 0), including
        `F_zz = 0`, clips to 0. Clipped peaks are the closest reachable response
        at `gamma_ref`, so they are skipped by the reproduction check.
    clipped : list of str
        Energies (eV, as ``"284.03"``-style strings) of the clipped peaks.
    """
    if peaks is None:
        peaks = load_materials()["znpc"]["peaks"]
    energy = np.array([p["energy"] for p in peaks])
    f_iso = np.array([p["height_iso"] for p in peaks])
    f_xx = np.array([p["F_xx"] for p in peaks])
    f_zz = np.array([p["F_zz"] for p in peaks])

    pi_star = (energy < 286.0) & (f_iso > 0.1 * f_iso.max())
    ratio_pi = f_zz[pi_star].sum() / f_xx[pi_star].sum()
    gamma_ref = float(np.arctan(np.sqrt(2.0 / ratio_pi)))

    lo = _film_ratio(1.0, gamma_ref)  # theta = 0
    hi = _film_ratio(0.0, gamma_ref)  # theta = pi/2
    thetas = np.empty(len(peaks))
    clipped: list[str] = []
    for k in range(len(peaks)):
        measured = f_zz[k] / f_xx[k]
        if measured <= lo:
            thetas[k] = 0.0
        elif measured >= hi:
            thetas[k] = np.pi / 2
        else:
            # ratio(u) is monotone in u = cos^2(theta); solve on [0, 1].
            root = brentq(lambda u, m=measured: _film_ratio(u, gamma_ref) - m, 0.0, 1.0)
            thetas[k] = float(np.arccos(np.sqrt(root)))
            continue
        clipped.append(f"{energy[k]:.2f}")
    return gamma_ref, thetas, clipped


def znpc_reproduction() -> tuple[float, float]:
    """Worst relative error of the film ratio, and of the magic-angle strength.

    Each ZnPc peak is built alone with its solved TDM tilt and rotated to
    `gamma_ref`. The film ratio `n_e / n_o` is compared with the measured
    `F_zz / F_xx`, skipping clipped peaks; the isotropic average is compared with
    `height_iso`.
    """
    records = load_materials()["znpc"]["peaks"]
    gamma_ref, thetas, clipped = znpc_tilts(records)
    worst_ratio = 0.0
    worst_iso = 0.0
    for rec, theta in zip(records, thetas, strict=True):
        peak = Peak(
            "gaussian",
            label="check",
            energy=float(rec["energy"]),
            width=float(rec["width"]),
            height=float(rec["height_iso"]),
            tdm_tilt=float(theta),
        )
        tensor = PeakSumSLD([peak], density=1.0, rotation=gamma_ref).tensor_at(
            float(rec["energy"])
        )
        measured = rec["F_zz"] / rec["F_xx"]
        if f"{rec['energy']:.2f}" not in clipped:
            worst_ratio = max(
                worst_ratio, abs(tensor[2, 2].imag / tensor[0, 0].imag / measured - 1.0)
            )
        iso_peak = Peak(
            "gaussian",
            label="check",
            energy=float(rec["energy"]),
            width=float(rec["width"]),
            height=float(rec["height_iso"]),
            tdm_tilt=float(theta),
        )
        iso = PeakSumSLD([iso_peak], density=1.0, rotation=0.0).tensor_at(
            float(rec["energy"])
        )
        mean = float(np.trace(iso.imag)) / 3.0
        worst_iso = max(worst_iso, abs(mean / rec["height_iso"] - 1.0))
    return worst_ratio, worst_iso


def alq3_peaks(shape: Shape, material: dict[str, Any]) -> list[Peak]:
    """Alq3 peaks p1..p14 sorted by energy; the first five carry the outline."""
    ordered = sorted(material["peaks"], key=lambda p: p["energy"])
    peaks = []
    for i, rec in enumerate(ordered):
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
    return peaks


def znpc_peaks(
    material: dict[str, Any], *, factors: NDArray[np.float64] | None = None
) -> list[Peak]:
    """ZnPc peaks: height `F_iso`, TDM tilt from `znpc_tilts`, all constant.

    ``factors`` multiplies each height (used for the shifted ZnPc).
    """
    records = material["peaks"]
    _, thetas, _ = znpc_tilts(records)
    if factors is None:
        factors = np.ones(len(records))
    return [
        Peak(
            "gaussian",
            label=f"p{i + 1}",
            energy=float(rec["energy"]),
            width=float(rec["width"]),
            height=float(rec["height_iso"] * factors[i]),
            tdm_tilt=float(thetas[i]),
        )
        for i, rec in enumerate(records)
    ]


def shifted_factors(n: int) -> NDArray[np.float64]:
    """Fixed deterministic height factors in [0.85, 1.15] for the shifted ZnPc."""
    return np.random.default_rng(SHIFT_SEED).uniform(*SHIFT_RANGE, n)


def curved_gamma(gamma_ref: float, fractions: tuple[float, float]) -> list[float]:
    """Four-anchor orientation: MAGIC_ANGLE at the surface, gamma_ref at the bottom."""
    a, b = fractions
    span = gamma_ref - MAGIC_ANGLE
    return [MAGIC_ANGLE, MAGIC_ANGLE + a * span, MAGIC_ANGLE + b * span, gamma_ref]


def build_profile(kind: Literal["spline", "polynomial"] = "spline") -> DepthProfile:
    """The film: a linear `Mix` of three peak-sum sources on one shared shape."""
    shape = make_shape(kind)
    materials = load_materials()
    gamma_ref, _, _ = znpc_tilts(materials["znpc"]["peaks"])
    factors = shifted_factors(len(materials["znpc"]["peaks"]))

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
    znpc_shifted = PeakSumSLD(
        znpc_peaks(materials["znpc"], factors=factors),
        baseline=baseline_table(materials["znpc"]),
        density=DENSITY_ZNPC,
        name="znpc_shifted",
    )
    return DepthProfile(
        Mix([alq3, znpc, znpc_shifted], rule="linear"),
        thickness=THICKNESS,
        n_slabs=N_SLABS,
        mesh="uniform",
        # Volume fractions, at the four anchors. The three columns sum to 1 at
        # every anchor. Alq3 sits mostly at the surface. The two ZnPc layers
        # enrich toward the substrate.
        phi=[
            field(shape, "phi_alq3", (0.85, 0.55, 0.45, 0.40)),
            field(shape, "phi_znpc", (0.10, 0.33, 0.35, 0.35)),
            field(shape, "phi_znpc_shifted", (0.05, 0.12, 0.20, 0.25)),
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
                "gamma_znpc_shifted",
                curved_gamma(gamma_ref, SHIFTED_GAMMA_FRACTIONS),
            ),
        ],
        # Total density, after mixing: changes mostly toward z = THICKNESS.
        density_scale=field(shape, "density_scale", (0.85, 0.87, 0.97, 1.15)),
        name="film",
    )


def build_structure(shape: Literal["spline", "polynomial"] = "spline") -> Structure:
    """Vacuum | three-material film | native SiO2 | Si substrate.

    ``shape="polynomial"`` swaps the shared spline for a cubic `PolynomialShape`
    on the same anchors and levels. Only the shape object changes.

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
    return vacuum | build_profile(shape) | oxide | substrate


# --8<-- [end:construction]


def channel_profiles(
    n: int = 401, shape: Literal["spline", "polynomial"] = "spline"
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Every channel and depth-field peak parameter as `(z, values)`.

    Keys are the `DepthProfile` channel names (`phi_0`, `density_scale`,
    `orientation_0`, ...) and the peak addresses (`alq3.p1.height`,
    `alq3.p1.tdm_tilt`, ...). Angles are in radians.
    """
    z = np.linspace(0.0, THICKNESS, n)
    values = build_profile(shape).named_values_at(z)
    return {key: (z, value) for key, value in values.items()}


def _energy_colors(n: int) -> NDArray[np.float64]:
    """Sequential viridis colors, one per energy in ascending order."""
    import matplotlib.pyplot as plt

    return plt.get_cmap("viridis")(np.linspace(0.05, 0.9, n))


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
            [("phi_0", "alq3"), ("phi_1", "znpc"), ("phi_2", "znpc_shifted")],
        ),
        ("total density scale", 1.0, [("density_scale", "density_scale")]),
        (
            "orientation $\\gamma$ (deg)",
            deg,
            [
                ("orientation_0", "alq3"),
                ("orientation_1", "znpc"),
                ("orientation_2", "znpc_shifted"),
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


def save_shapes(path: Path) -> None:
    """Four channels: 4-knot spline (solid) against cubic polynomial (dashed)."""
    import matplotlib.pyplot as plt

    spline = channel_profiles(shape="spline")
    poly = channel_profiles(shape="polynomial")
    at_knots = build_profile("spline").named_values_at(KNOTS)
    deg = 180.0 / np.pi
    panels = [
        ("alq3 volume fraction $\\phi$", 1.0, "phi_0"),
        ("alq3 orientation $\\gamma$ (deg)", deg, "orientation_0"),
        ("alq3 peak p1 height", 1.0, "alq3.p1.height"),
        ("total density scale", 1.0, "density_scale"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), constrained_layout=True)
    for ax, (title, scale, key) in zip(axes.flat, panels, strict=True):
        z, y_spline = spline[key]
        _, y_poly = poly[key]
        for knot in KNOTS:
            ax.axvline(knot, color="0.6", lw=0.8, ls=":")
        ax.plot(z, scale * y_spline, color="C0", lw=1.8, label="spline (4 knots)")
        ax.plot(
            z,
            scale * y_poly,
            color="C1",
            lw=1.8,
            ls="--",
            label="cubic polynomial (4 anchors)",
        )
        ax.plot(KNOTS, scale * at_knots[key], "o", color="C0", ms=4)
        _depth_axis(ax, title)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


FINE = np.linspace(283.5, 289.0, 276)  # eV, 0.02 eV steps


def _building_block(
    peaks: list[Peak], baseline: OpticalConstants, *, rotation: float = 0.0
) -> PeakSumSLD:
    """Unit-density source with constant peaks, for the building-block panels."""
    return PeakSumSLD(peaks, baseline=baseline, density=1.0, rotation=rotation)


def save_materials(path: Path) -> None:
    """Building blocks: unit-density beta(E) of each material, and the ZnPc shift."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    materials = load_materials()
    alq3_rec = materials["alq3"]
    znpc_rec = materials["znpc"]
    gamma_ref, _, _ = znpc_tilts(znpc_rec["peaks"])
    factors = shifted_factors(len(znpc_rec["peaks"]))

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
    shifted_src = _building_block(
        znpc_peaks(znpc_rec, factors=factors),
        baseline_table(znpc_rec),
        rotation=gamma_ref,
    )
    shifted_t = shifted_src.tensor_at_many(FINE)

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
        shifted_t.imag.trace(axis1=1, axis2=2) / 3.0,
        color="C1",
        lw=1.6,
        ls="--",
        label="znpc_shifted, isotropic",
    )
    ax_s.set_title("znpc vs znpc_shifted")

    for ax in (ax_a, ax_z, ax_s):
        ax.set_xlim(283.5, 289.0)
        ax.set_xlabel("photon energy (eV)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax_a.set_ylabel(r"$\beta$ at unit density")
    ax_z.tick_params(labelleft=False)
    ax_s.tick_params(labelleft=False)

    pct = 100.0 * (factors - 1.0)
    centers = np.array([rec["energy"] for rec in znpc_rec["peaks"]])
    ax_strip.bar(centers, pct, width=0.12, color=np.where(pct >= 0, "C2", "C3"))
    ax_strip.axhline(0.0, color="0.4", lw=0.8)
    ax_strip.set_xlim(283.5, 289.2)
    ax_strip.set_ylabel("height change (%)")
    ax_strip.set_xlabel("ZnPc peak energy (eV), znpc_shifted relative to znpc")
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
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.95, z.size))
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
    mappable = ScalarMappable(norm=Normalize(z.min(), z.max()), cmap="viridis")
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


def save_ridge(path: Path) -> None:
    """Joyplot of R_ss (left) and R_pp (right), both as log10(R q^4)."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    flat = fresnel_flattened(reflectivity())
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


DIP_PROMINENCE = 0.1  # decades of log10(R q^4)
TRACK_DQ = 0.012  # 1/Angstrom, largest q jump between neighbouring energies
TRACK_MIN_POINTS = 4
FINE_E = np.linspace(283.5, 289.0, 551)  # eV, 0.01 eV steps for the NEXAFS scan
BRANCH_JUMP = 0.25  # eV, largest energy jump between depths in one branch
PEAK_PROMINENCE = 0.02  # fraction of each channel's full range
NAME_TOL = 0.3  # eV, frozen peaks this close to a branch are named


def _track(
    coords: NDArray[np.float64], values: NDArray[np.float64]
) -> list[dict[str, NDArray[np.float64]]]:
    """Link dips across energy by nearest-q continuity.

    ``coords`` are the q positions of the dips at each energy in `ENERGIES`
    order, as a list of arrays; ``values`` are the matching energies. Returns
    tracks of at least `TRACK_MIN_POINTS` points.
    """
    open_: list[list[tuple[float, float]]] = []
    done: list[list[tuple[float, float]]] = []
    for e, qd in zip(values, coords, strict=True):
        free = list(range(len(qd)))
        still_open = []
        for track in open_:
            last_q = track[-1][1]
            best = None
            for i in free:
                gap = abs(qd[i] - last_q)
                if gap <= TRACK_DQ and (best is None or gap < best[0]):
                    best = (gap, i)
            if best is None:
                done.append(track)
                continue
            free.remove(best[1])
            track.append((float(e), float(qd[best[1]])))
            still_open.append(track)
        for i in free:
            still_open.append([(float(e), float(qd[i]))])
        open_ = still_open
    done.extend(open_)
    result = []
    for track in done:
        if len(track) >= TRACK_MIN_POINTS:
            arr = np.array(track)
            result.append({"E": arr[:, 0], "q": arr[:, 1]})
    return result


def reflectivity_tracks(flat: NDArray[np.float64]) -> dict[str, list[dict]]:
    """Dip tracks of log10(R q^4) versus energy, for R_ss and R_pp."""
    from scipy.signal import find_peaks

    out: dict[str, list[dict]] = {}
    for pol, name in ((0, "ss"), (1, "pp")):
        coords = []
        for j in range(ENERGIES.size):
            index, _ = find_peaks(-flat[pol][:, j], prominence=DIP_PROMINENCE)
            coords.append(Q[index])
        out[name] = _track(coords, ENERGIES)
        for track in out[name]:
            # Slope over 284.5 to 286 eV. Tracks that do not span the window
            # have no slope and get no label.
            near = (track["E"] >= 284.5) & (track["E"] <= 286.0)
            spans = track["E"][0] <= 284.5 and track["E"][-1] >= 286.0
            if spans and near.sum() >= 3:
                slope = np.polyfit(track["E"][near], track["q"][near], 1)[0]
            else:
                slope = np.nan
            track["slope"] = float(slope)
    return out


def nexafs_branches(channel: str) -> list[dict[str, Any]]:
    """Peak branches of the film's local beta across depth (`o` or `e`)."""
    from scipy.signal import find_peaks

    profile = build_profile()
    tensors = np.array([profile.tensor_rows_at(float(e)) for e in FINE_E])
    comp = 0 if channel == "o" else 2
    beta = tensors[:, :, comp, comp].imag  # (n_E, n_slab)
    threshold = PEAK_PROMINENCE * float(beta.max() - beta.min())
    z = (np.arange(N_SLABS) + 0.5) * THICKNESS / N_SLABS

    open_: list[dict[str, list[float]]] = []
    done: list[dict[str, list[float]]] = []
    for s in range(N_SLABS):
        index, _ = find_peaks(beta[:, s], prominence=threshold)
        pairs = sorted(
            (abs(FINE_E[p] - b["E"][-1]), bi, p)
            for bi, b in enumerate(open_)
            for p in range(index.size)
            if abs(FINE_E[index[p]] - b["E"][-1]) <= BRANCH_JUMP
        )
        used_b: set[int] = set()
        used_p: set[int] = set()
        for _, bi, p in pairs:
            if bi in used_b or p in used_p:
                continue
            used_b.add(bi)
            used_p.add(p)
            _extend(open_[bi], z[s], FINE_E[index[p]], beta[index[p], s])
        still_open = []
        for bi, branch in enumerate(open_):
            if bi in used_b:
                still_open.append(branch)
            else:
                done.append(branch)
        for p in range(index.size):
            if p not in used_p:
                new: dict[str, list[float]] = {"z": [], "E": [], "h": []}
                _extend(new, z[s], FINE_E[index[p]], beta[index[p], s])
                still_open.append(new)
        open_ = still_open
    done.extend(open_)
    return [b for b in done if len(b["z"]) >= 3]


def _extend(branch: dict[str, list[float]], z: float, e: float, h: float) -> None:
    branch["z"].append(float(z))
    branch["E"].append(float(e))
    branch["h"].append(float(h))


def nearby_peaks(energy: float) -> str:
    """Frozen alq3 and znpc peaks within `NAME_TOL` of an energy, as text."""
    materials = load_materials()
    parts = []
    for name in ("alq3", "znpc"):
        ordered = sorted(materials[name]["peaks"], key=lambda p: p["energy"])
        index = [
            i + 1
            for i, rec in enumerate(ordered)
            if abs(rec["energy"] - energy) <= NAME_TOL
        ]
        if not index:
            continue
        if len(index) == 1:
            parts.append(f"{name} p{index[0]}")
        else:
            parts.append(f"{name} p{index[0]}-p{index[-1]}")
    return ", ".join(parts) if parts else "none"


def save_migration(
    path: Path,
    branches: dict[str, list[dict[str, Any]]],
    tracks: dict[str, list[dict[str, Any]]],
) -> None:
    """Top: NEXAFS peak branches vs depth (o, e). Bottom: reflectivity dip tracks."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.8, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.12)
    ax_o = fig.add_subplot(grid[0, 0])
    ax_e = fig.add_subplot(grid[0, 1], sharey=ax_o)
    ax_r = fig.add_subplot(grid[1, :])
    colors = plt.get_cmap("tab10")(np.arange(10))
    for ax, channel, title in (
        (ax_o, "o", r"$\beta_o$ peaks"),
        (ax_e, "e", r"$\beta_e$ peaks"),
    ):
        channel_branches = branches[channel]
        heights = np.concatenate([b["h"] for b in channel_branches])
        scale = 60.0 / float(heights.max())
        for k, b in enumerate(channel_branches):
            color = colors[k % 10]
            mean_e = float(np.mean(b["E"]))
            label = f"B{k + 1} {mean_e:.2f} eV: {nearby_peaks(mean_e)}"
            ax.plot(b["z"], b["E"], "-", color=color, lw=1.0, alpha=0.6)
            mid = len(b["z"]) // 2
            ha = "right" if b["z"][mid] > 150.0 else "left"
            above = k % 2 == 0  # alternate sides so neighbours do not collide
            ax.annotate(
                label,
                (b["z"][mid], b["E"][mid]),
                xytext=(0, 6 if above else -9),
                textcoords="offset points",
                ha=ha,
                fontsize=6,
                color=color,
            )
            ax.scatter(
                b["z"],
                b["E"],
                s=np.asarray(b["h"]) * scale,
                color=color,
                edgecolor="none",
                label=label,
            )
        energies = np.concatenate([b["E"] for b in channel_branches])
        ax.set_ylim(float(energies.min()) - 0.2, float(energies.max()) + 0.25)
        ax.set_title(title)
        ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
        ax.grid(True, alpha=0.25)
    ax_o.set_ylabel("peak energy (eV)")
    ax_e.tick_params(labelleft=False)

    cmap = plt.get_cmap("viridis")
    means = [float(np.mean(t["q"])) for t in tracks["pp"]]
    q_lo, q_hi = min(means), max(means)

    def shade(q_mean: float) -> Any:
        """One sequential colour per track, by mean dip q."""
        return cmap(0.05 + 0.85 * (q_mean - q_lo) / max(q_hi - q_lo, 1e-9))

    for track in tracks["ss"]:  # background context, drawn first
        ax_r.plot(
            track["E"],
            track["q"],
            ls="--",
            color="0.72",
            lw=0.8,
            marker="s",
            ms=2.5,
            mfc="none",
            mec="0.6",
            zorder=1,
        )
    labels = []
    for track, q_mean in zip(tracks["pp"], means, strict=True):
        color = shade(q_mean)
        ax_r.plot(track["E"], track["q"], "-", color=color, lw=1.8, zorder=3)
        ax_r.plot(track["E"], track["q"], "o", color=color, ms=3.5, zorder=4)
        if np.isfinite(track["slope"]):
            labels.append([float(track["q"][-1]), float(track["slope"]), color])
    # Margin labels at each track's right-end height, pushed apart so none overlap.
    labels.sort(key=lambda item: item[0])
    gap = 0.0085
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + gap)
    ax_r.text(
        1.02,
        1.0,
        "dq/dE (284.5-286 eV)",
        transform=ax_r.transAxes,
        fontsize=6.5,
        va="bottom",
        ha="left",
    )
    for y_pos, slope, color in labels:
        ax_r.text(
            1.02,
            y_pos,
            f"{slope:+.2g}",
            transform=ax_r.get_yaxis_transform(),
            fontsize=6.5,
            color=color,
            va="center",
            ha="left",
        )
    from matplotlib.lines import Line2D

    proxies = [
        Line2D([], [], ls="--", color="0.72", marker="s", mfc="none", mec="0.6", ms=3),
        Line2D([], [], ls="-", color=cmap(0.5), marker="o", ms=3.5, lw=1.8),
    ]
    ax_r.legend(
        proxies,
        ["$R_{ss}$ (grey dashed)", "$R_{pp}$ (solid, coloured by mean q)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    ax_r.set_xlabel("photon energy (eV)")
    ax_r.set_ylabel(r"dip position $q$ ($\mathrm{\AA}^{-1}$)")
    ax_r.set_title("reflectivity dip tracks")
    ax_r.grid(True, alpha=0.25)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _const_alq3(material: dict[str, Any]) -> list[Peak]:
    """Alq3 peaks at their fitted heights and magic-angle TDMs (no depth fields)."""
    ordered = sorted(material["peaks"], key=lambda p: p["energy"])
    return [
        Peak(
            "gaussian",
            label=f"p{i + 1}",
            energy=float(rec["energy"]),
            width=float(rec["width"]),
            height=float(rec["height"]),
        )
        for i, rec in enumerate(ordered)
    ]


def main() -> None:
    gamma_ref, _, clipped = znpc_tilts()
    print(f"gamma_ref = {np.rad2deg(gamma_ref):.2f} deg")
    print(f"clipped ZnPc peaks ({len(clipped)}): {', '.join(clipped)}")
    worst_ratio, worst_iso = znpc_reproduction()
    print(f"worst film-ratio error (unclipped): {worst_ratio:.2e}")
    print(f"worst isotropic-strength error: {worst_iso:.2e}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    save_channels(ASSETS / "lineshape-gradient-channels.png")
    save_shapes(ASSETS / "lineshape-gradient-shapes.png")
    save_materials(ASSETS / "lineshape-gradient-materials.png")
    save_nexafs(ASSETS / "lineshape-gradient-nexafs.png")
    save_ridge(ASSETS / "lineshape-gradient-ridge.png")
    branches = {"o": nexafs_branches("o"), "e": nexafs_branches("e")}
    tracks = reflectivity_tracks(fresnel_flattened(reflectivity()))
    save_migration(ASSETS / "lineshape-gradient-migration.png", branches, tracks)
    print(f"wrote figures to {ASSETS}")
    for channel, items in branches.items():
        for k, b in enumerate(items):
            e = np.asarray(b["E"])
            print(
                f"branch {channel}{k + 1}: z {b['z'][0]:.1f}-{b['z'][-1]:.1f} "
                f"E mean {e.mean():.3f} first {e[0]:.3f} last {e[-1]:.3f} "
                f"range {e.min():.3f}-{e.max():.3f} | {nearby_peaks(e.mean())}"
            )
    for pol, items in tracks.items():
        for k, t in enumerate(items):
            print(
                f"track {pol}{k + 1}: E {t['E'][0]:.1f}-{t['E'][-1]:.1f} "
                f"q {t['q'].min():.4f}-{t['q'].max():.4f} slope {t['slope']:.4f}"
            )


if __name__ == "__main__":
    main()
