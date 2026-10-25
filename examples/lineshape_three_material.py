#!/usr/bin/env python3
"""Three-material film with depth-varying composition, orientation and peaks.

Three synthetic five-peak materials (mat1, mat2, mat3) fill one 200 A film.
One shared depth shape drives every gradient: the volume fractions, the total
density, the orientation gamma, and the heights and TDM tilts of some mat1
peaks. By default the shape is a natural cubic `SplineShape` through four
knots (surface at z = 0, substrate at z = THICKNESS). `build_structure` takes
``shape="polynomial"`` to use a cubic `PolynomialShape` through the same four
anchor positions and the same level values instead. Each gradient is a
`LevelField` on that shape, so every effect follows the same depth profile.

Regenerate the docs figures::

    uv run python examples/lineshape_three_material.py
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from refloxide.fields import LevelField, PolynomialShape, Shape, SplineShape
from refloxide.lineshapes import MAGIC_ANGLE, Peak, PeakSumSLD
from refloxide.model import MaterialSLD, Mix, ReflectModel, Structure
from refloxide.profiles import DepthProfile

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from numpy.typing import NDArray

# --8<-- [start:construction]
THICKNESS = 200.0  # film thickness, Angstrom
N_SLABS = 40
SHAPE_KINDS = ("spline", "polynomial")
# Knot and anchor positions as fractions of THICKNESS: surface, two interior
# points, substrate. Both shapes use these positions, so they compare fairly.
FRACTIONS = np.array([0.0, 0.3, 0.6, 1.0])
KNOTS = THICKNESS * FRACTIONS  # spline knots, Angstrom
Q = np.linspace(0.01, 0.27, 520)  # 1/Angstrom
# Carbon K edge: below the first resonance, on mat1 peaks 1 to 3, and above.
REFLECT_ENERGIES = np.array([283.5, 284.8, 285.9, 287.2, 289.6])  # eV
TENSOR_ENERGIES = np.array([283.5, 284.8, 285.9, 287.2, 289.6])  # eV


def make_shape(kind: str = "spline") -> Shape:
    """The shared depth shape: a 4-knot spline, or a cubic through the same 4 anchors."""
    if kind == "spline":
        return SplineShape(KNOTS, name="depth")
    if kind == "polynomial":
        return PolynomialShape(degree=3, positions=FRACTIONS, name="depth")
    msg = f"shape must be one of {SHAPE_KINDS}, got {kind!r}"
    raise ValueError(msg)


def field(shape: Shape, name: str, levels: Sequence[float]) -> LevelField:
    """Level field with explicit values at the four anchors, surface to substrate."""
    return LevelField(shape, list(levels), name=name)


def mat1_peaks(shape: Shape) -> list[Peak]:
    """mat1 is face-on at the surface. Peak heights are absolute values."""
    return [
        Peak(
            label="p1",
            energy=284.8,
            width=0.5,
            # Falls fast near the surface, then changes little.
            height=field(shape, "mat1_p1_height", (0.018, 0.014, 0.0125, 0.012)),
            tdm_tilt=field(
                shape, "mat1_p1_tilt", np.deg2rad([0.0, 12.0, 17.0, 20.0]).tolist()
            ),
        ),
        Peak(
            label="p2",
            energy=285.9,
            width=0.5,
            height=field(shape, "mat1_p2_height", (0.020, 0.013, 0.0105, 0.010)),
            tdm_tilt=field(
                shape, "mat1_p2_tilt", np.deg2rad([0.0, 14.0, 18.0, 20.0]).tolist()
            ),
        ),
        Peak(
            label="p3",
            energy=287.2,
            width=0.5,
            # Rises near the surface, then flat through the substrate.
            height=field(shape, "mat1_p3_height", (0.0128, 0.0155, 0.0160, 0.0160)),
            tdm_tilt=MAGIC_ANGLE,
        ),
        Peak(
            label="p4",
            energy=288.4,
            width=0.5,
            height=0.008,
            tdm_tilt=field(
                shape, "mat1_p4_tilt", np.deg2rad([90.0, 85.0, 82.0, 80.0]).tolist()
            ),
        ),
        Peak(
            label="p5",
            energy=289.6,
            width=0.5,
            height=0.006,
            # Holds near 90 degrees, then tilts in the lower half.
            tdm_tilt=field(
                shape, "mat1_p5_tilt", np.deg2rad([90.0, 89.0, 85.0, 80.0]).tolist()
            ),
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


def build_profile(kind: str = "spline") -> DepthProfile:
    """The film: a linear `Mix` of three peak-sum sources on one shared shape."""
    shape = make_shape(kind)
    mat1 = PeakSumSLD(mat1_peaks(shape), name="mat1")
    mat2 = PeakSumSLD(mat2_peaks(), name="mat2")
    mat3 = PeakSumSLD(mat3_peaks(), name="mat3")
    return DepthProfile(
        Mix([mat1, mat2, mat3], rule="linear"),
        thickness=THICKNESS,
        n_slabs=N_SLABS,
        mesh="uniform",
        # Volume fractions, at the four anchors. The three columns sum to 1 at
        # every anchor. mat1 falls fast in the top 60 A and then levels off.
        # mat2 is taken up near the surface. mat3 enriches mostly in the lower
        # half.
        phi=[
            field(shape, "phi_mat1", (0.85, 0.55, 0.45, 0.40)),
            field(shape, "phi_mat2", (0.10, 0.33, 0.35, 0.35)),
            field(shape, "phi_mat3", (0.05, 0.12, 0.20, 0.25)),
        ],
        # Orientation: gamma in radians, one field per material. mat1 stays
        # near face-on at the surface and reaches the magic angle at the
        # substrate. mat2 tilts fast near the top and saturates edge-on. mat3
        # stays near amorphous until the lower half, then tilts toward edge-on.
        gamma=[
            field(shape, "gamma_mat1", (0.0, 0.10, 0.50, MAGIC_ANGLE)),
            field(shape, "gamma_mat2", (MAGIC_ANGLE, 1.10, 1.40, np.pi / 2)),
            field(shape, "gamma_mat3", (MAGIC_ANGLE, 1.02, 1.30, np.pi / 2)),
        ],
        # Total density, after mixing: changes mostly toward the substrate.
        density_scale=field(shape, "density_scale", (0.85, 0.87, 0.97, 1.15)),
        name="film",
    )


def build_structure(shape: str = "spline") -> Structure:
    """Vacuum | three-material film | silicon substrate.

    ``shape="polynomial"`` swaps the shared spline for a cubic `PolynomialShape`
    on the same anchors and levels. Only the shape object changes.
    """
    vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
    substrate = MaterialSLD("Si", density=2.33, name="si")(0, 3)
    return vacuum | build_profile(shape) | substrate


# --8<-- [end:construction]

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def channel_profiles(
    n: int = 401, shape: str = "spline"
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Every channel and depth-field peak parameter as `(z, values)`.

    Keys are the `DepthProfile` channel names (`phi_0`, `density_scale`,
    `orientation_0`, ...) and the peak addresses (`mat1.p1.height`,
    `mat1.p1.tdm_tilt`, ...). Angles are in radians.
    """
    z = np.linspace(0.0, THICKNESS, n)
    values = build_profile(shape).named_values_at(z)
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


def _energy_colors(n: int) -> NDArray[np.float64]:
    """Sequential viridis colors, one per energy in ascending order."""
    import matplotlib.pyplot as plt

    return plt.get_cmap("viridis")(np.linspace(0.05, 0.9, n))


def save_channels(path: Path) -> None:
    """Every channel versus depth (angles in degrees for display)."""
    import matplotlib.pyplot as plt

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


def save_shapes(path: Path) -> None:
    """Four channels: 4-knot spline (solid) against cubic polynomial (dashed)."""
    import matplotlib.pyplot as plt

    spline = channel_profiles(shape="spline")
    poly = channel_profiles(shape="polynomial")
    at_knots = build_profile("spline").named_values_at(KNOTS)
    deg = 180.0 / np.pi
    panels = [
        ("mat1 volume fraction $\\phi$", 1.0, "phi_0"),
        ("mat1 orientation $\\gamma$ (deg)", deg, "orientation_0"),
        ("mat1 peak 1 height", 1.0, "mat1.p1.height"),
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
        ax.set_title(title)
        ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_tensors(path: Path) -> None:
    """Ordinary and extraordinary delta and beta versus depth at each energy."""
    import matplotlib.pyplot as plt

    colors = _energy_colors(TENSOR_ENERGIES.size)
    fig, axes = plt.subplots(
        2, 2, figsize=(8.8, 5.6), sharex=True, constrained_layout=True
    )
    for j, energy in enumerate(TENSOR_ENERGIES):
        z, tensors = tensors_at_energy(float(energy))
        n_o, n_e = tensors[:, 0, 0], tensors[:, 2, 2]
        for row, part in enumerate((np.real, np.imag)):
            for col, comp in enumerate((n_o, n_e)):
                axes[row, col].plot(
                    z, part(comp), color=colors[j], lw=1.6, label=f"{energy:.1f} eV"
                )
    titles = (
        (r"$\delta$, ordinary $o$", r"$\delta$, extraordinary $e$"),
        (r"$\beta$, ordinary $o$", r"$\beta$, extraordinary $e$"),
    )
    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            ax.set_title(titles[row][col])
            ax.grid(True, alpha=0.25)
            ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
    axes[0, 0].legend(frameon=False, fontsize=8, title="energy")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_reflectivity(path: Path) -> None:
    """R_ss (solid) and R_pp (dashed) versus q at each reflectivity energy."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    refl = reflectivity()
    colors = _energy_colors(REFLECT_ENERGIES.size)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for j, energy in enumerate(REFLECT_ENERGIES):
        ax.semilogy(Q, refl[0, :, j], color=colors[j], lw=1.8)
        ax.semilogy(Q, refl[1, :, j], color=colors[j], lw=1.4, ls="--")
    handles = [
        Line2D([], [], color=colors[j], lw=1.8, label=f"{energy:.1f} eV")
        for j, energy in enumerate(REFLECT_ENERGIES)
    ]
    handles += [
        Line2D([], [], color="0.25", lw=1.8, label=r"$R_{ss}$"),
        Line2D([], [], color="0.25", lw=1.4, ls="--", label=r"$R_{pp}$"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2)
    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel("reflectivity")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    save_channels(ASSETS / "lineshape-gradient-channels.png")
    save_tensors(ASSETS / "lineshape-gradient-tensors.png")
    save_reflectivity(ASSETS / "lineshape-gradient-reflectivity.png")
    save_shapes(ASSETS / "lineshape-gradient-shapes.png")
    print(f"wrote figures to {ASSETS}")


if __name__ == "__main__":
    main()
