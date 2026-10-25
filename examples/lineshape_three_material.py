#!/usr/bin/env python3
"""Three-material film with depth-varying composition, orientation and peaks.

Three synthetic five-peak materials (mat1, mat2, mat3) fill one 200 A film.
A single spline shape with five knots (surface at z = 0, substrate at
z = THICKNESS) drives every gradient: the volume fractions, the total
density, the orientation gamma, and the heights and TDM angles of some
mat1 peaks. Each gradient is a `LevelField` on that shape, so every effect
follows the same depth profile.

Regenerate the docs figures::

    uv run python examples/lineshape_three_material.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from refloxide.fields import LevelField, SplineShape
from refloxide.lineshapes import MAGIC_ANGLE, Peak, PeakSumSLD
from refloxide.model import MaterialSLD, Mix, ReflectModel, Structure
from refloxide.profiles import DepthProfile

if TYPE_CHECKING:
    from numpy.typing import NDArray

# --8<-- [start:construction]
THICKNESS = 200.0  # film thickness, Angstrom
N_SLABS = 40
KNOTS = np.linspace(0.0, THICKNESS, 5)  # five knots, surface to substrate
Q = np.linspace(0.01, 0.27, 520)  # 1/Angstrom
REFLECT_ENERGIES = np.array([285.9, 288.4])  # eV
TENSOR_ENERGY = 285.9  # eV, for the n_o / n_e figure


def ramp(shape: SplineShape, name: str, surface: float, substrate: float) -> LevelField:
    """Field equal to `surface` at z = 0 and `substrate` at z = THICKNESS.

    The five levels are evenly spaced between the two end values. Through
    collinear knots the natural spline is that straight line.
    """
    levels = np.linspace(surface, substrate, KNOTS.size).tolist()
    return LevelField(shape, levels, name=name)


def mat1_peaks(shape: SplineShape) -> list[Peak]:
    """mat1 is face-on at the surface. Peak heights are absolute values."""
    h1, h2, h3 = 0.012, 0.010, 0.016  # substrate heights of peaks 1 to 3
    return [
        Peak(
            label="p1",
            energy=284.8,
            width=0.5,
            height=ramp(shape, "mat1_p1_height", 1.5 * h1, h1),
            tdm_tilt=ramp(shape, "mat1_p1_tilt", 0.0, np.deg2rad(20.0)),
        ),
        Peak(
            label="p2",
            energy=285.9,
            width=0.5,
            height=ramp(shape, "mat1_p2_height", 2.0 * h2, h2),
            tdm_tilt=ramp(shape, "mat1_p2_tilt", 0.0, np.deg2rad(20.0)),
        ),
        Peak(
            label="p3",
            energy=287.2,
            width=0.5,
            height=ramp(shape, "mat1_p3_height", 0.8 * h3, h3),
            tdm_tilt=MAGIC_ANGLE,
        ),
        Peak(
            label="p4",
            energy=288.4,
            width=0.5,
            height=0.008,
            tdm_tilt=ramp(shape, "mat1_p4_tilt", np.deg2rad(90.0), np.deg2rad(80.0)),
        ),
        Peak(
            label="p5",
            energy=289.6,
            width=0.5,
            height=0.006,
            tdm_tilt=ramp(shape, "mat1_p5_tilt", np.deg2rad(90.0), np.deg2rad(80.0)),
        ),
    ]


def mat2_peaks() -> list[Peak]:
    """Constant peak set (no depth fields); orientation comes from the film."""
    heights = (0.008, 0.011, 0.006, 0.010, 0.005)
    energies = (284.5, 286.0, 287.4, 288.8, 290.0)
    tilts = np.deg2rad((60.0, 50.0, 70.0, 45.0, 75.0))
    return [
        Peak(
            label=f"p{i + 1}",
            energy=energies[i],
            width=0.5,
            height=heights[i],
            tdm_tilt=float(tilts[i]),
        )
        for i in range(5)
    ]


def mat3_peaks() -> list[Peak]:
    """Constant peak set (no depth fields), offset in energy from mat2."""
    heights = (0.010, 0.006, 0.012, 0.004, 0.007)
    energies = (285.2, 286.6, 288.0, 289.2, 290.4)
    tilts = np.deg2rad((40.0, 65.0, 55.0, 80.0, 35.0))
    return [
        Peak(
            label=f"p{i + 1}",
            energy=energies[i],
            width=0.5,
            height=heights[i],
            tdm_tilt=float(tilts[i]),
        )
        for i in range(5)
    ]


def build_profile() -> DepthProfile:
    """The film: a linear `Mix` of three peak-sum sources on one spline shape."""
    shape = SplineShape(KNOTS, name="depth")
    mat1 = PeakSumSLD(mat1_peaks(shape), name="mat1")
    mat2 = PeakSumSLD(mat2_peaks(), name="mat2")
    mat3 = PeakSumSLD(mat3_peaks(), name="mat3")
    return DepthProfile(
        Mix([mat1, mat2, mat3], rule="linear"),
        thickness=THICKNESS,
        n_slabs=N_SLABS,
        mesh="uniform",
        # Volume fractions: mostly mat1 at the surface; mat2 and mat3 enrich
        # toward the substrate. The three ramps sum to 1 at every depth.
        phi=[
            ramp(shape, "phi_mat1", 0.85, 0.40),
            ramp(shape, "phi_mat2", 0.10, 0.35),
            ramp(shape, "phi_mat3", 0.05, 0.25),
        ],
        # Orientation: gamma in radians, one field per material.
        gamma=[
            ramp(shape, "gamma_mat1", 0.0, MAGIC_ANGLE),
            ramp(shape, "gamma_mat2", MAGIC_ANGLE, np.pi / 2),
            ramp(shape, "gamma_mat3", MAGIC_ANGLE, np.pi / 2),
        ],
        # Total density, after mixing: denser at the substrate.
        density_scale=ramp(shape, "density_scale", 0.85, 1.15),
        name="film",
    )


def build_structure() -> Structure:
    """Vacuum | three-material film | silicon substrate."""
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    substrate = MaterialSLD("Si", density=2.33, name="si")(0, 3)
    return vacuum | build_profile() | substrate


# --8<-- [end:construction]

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def channel_profiles(
    n: int = 401,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Every channel and depth-field peak parameter as `(z, values)`.

    Keys are the `DepthProfile` channel names (`phi_0`, `density_scale`,
    `orientation_0`, ...) and the peak addresses (`mat1.p1.height`,
    `mat1.p1.tdm_tilt`, ...). Angles are in radians.
    """
    z = np.linspace(0.0, THICKNESS, n)
    values = build_profile().named_values_at(z)
    return {key: (z, value) for key, value in values.items()}


def tensors_at_energy(
    energy: float,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Microslab midpoints `z` and lab tensors `(n_slabs, 3, 3)` at one energy."""
    z = (np.arange(N_SLABS) + 0.5) * THICKNESS / N_SLABS
    return z, build_profile().tensor_rows_at(energy)


def reflectivity() -> NDArray[np.float64]:
    """Reflectivity over `Q` at `REFLECT_ENERGIES`, shape `(2, len(Q), n_E)`.

    Index 0 is R_ss (s-polarized) and index 1 is R_pp (p-polarized).
    """
    result = ReflectModel(build_structure())(Q, REFLECT_ENERGIES)
    return np.stack([result.s, result.p])


def _curves(
    ax: plt.Axes,
    values: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    curves: list[tuple[str, str]],
    *,
    scale: float = 1.0,
) -> None:
    for key, label in curves:
        z, y = values[key]
        ax.plot(z, scale * y, label=label, lw=1.8)
    ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def save_channels(path: Path) -> None:
    """Every channel versus depth (angles in degrees for display)."""
    values = channel_profiles()
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.4), constrained_layout=True)
    deg = 180.0 / np.pi
    panels = [
        (
            "volume fraction $\\phi$",
            1.0,
            [("phi_0", "mat1"), ("phi_1", "mat2"), ("phi_2", "mat3")],
        ),
        ("total density scale", 1.0, [("density_scale", "density_scale")]),
        (
            "orientation $\\gamma$ (deg)",
            deg,
            [
                ("orientation_0", "mat1"),
                ("orientation_1", "mat2"),
                ("orientation_2", "mat3"),
            ],
        ),
        (
            "mat1 peak heights",
            1.0,
            [
                ("mat1.p1.height", "p1"),
                ("mat1.p2.height", "p2"),
                ("mat1.p3.height", "p3"),
            ],
        ),
        (
            "mat1 TDM tilt $\\theta$ (deg)",
            deg,
            [
                ("mat1.p1.tdm_tilt", "p1"),
                ("mat1.p2.tdm_tilt", "p2"),
                ("mat1.p4.tdm_tilt", "p4"),
                ("mat1.p5.tdm_tilt", "p5"),
            ],
        ),
    ]
    for ax, (title, scale, curves) in zip(axes.flat, panels, strict=False):
        _curves(ax, values, curves, scale=scale)
        ax.set_title(title)
    axes.flat[-1].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_tensors(path: Path) -> None:
    """Ordinary and extraordinary delta and beta versus depth at one energy."""
    z, tensors = tensors_at_energy(TENSOR_ENERGY)
    n_o, n_e = tensors[:, 0, 0], tensors[:, 2, 2]
    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(8.8, 3.4), constrained_layout=True)
    for ax, part, title in ((ax_d, np.real, r"$\delta$"), (ax_b, np.imag, r"$\beta$")):
        ax.plot(z, part(n_o), color="C0", lw=1.8, label=r"$o$ (ordinary)")
        ax.plot(z, part(n_e), color="C1", lw=1.8, ls="--", label=r"$e$ (extraordinary)")
        ax.set_title(f"{title} at {TENSOR_ENERGY:.1f} eV")
        ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_reflectivity(path: Path) -> None:
    """R_ss and R_pp versus q at each reflectivity energy."""
    refl = reflectivity()
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    for j, energy in enumerate(REFLECT_ENERGIES):
        ax.semilogy(
            Q,
            refl[0, :, j],
            color=f"C{j}",
            lw=1.8,
            label=rf"$R_{{ss}}$ {energy:.1f} eV",
        )
        ax.semilogy(
            Q,
            refl[1, :, j],
            color=f"C{j}",
            lw=1.4,
            ls="--",
            label=rf"$R_{{pp}}$ {energy:.1f} eV",
        )
    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel("reflectivity")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    save_channels(ASSETS / "lineshape-gradient-channels.png")
    save_tensors(ASSETS / "lineshape-gradient-tensors.png")
    save_reflectivity(ASSETS / "lineshape-gradient-reflectivity.png")
    print(f"wrote figures to {ASSETS}")


if __name__ == "__main__":
    main()
