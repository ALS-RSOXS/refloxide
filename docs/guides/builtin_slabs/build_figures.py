#!/usr/bin/env python3
"""Build depth-profile and reflectivity figures for built-in slab cases.

Each case_* function is the implementation reference for the matching
page under ``docs/guides/building_structures/`` and the composable-OC plan.

Regenerate::

    uv run python docs/guides/builtin_slabs/build_figures.py
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

from refloxide.mixing import MixRule, mix_channels
from refloxide.optics import uniaxial_lab_tensor
from refloxide.profiles import (
    Diffusion,
    Polynomial,
    SecondOrderTransition,
    Spline,
)
from refloxide.tmm import general_reflectivity, uniaxial_reflectivity

OUT = Path(__file__).resolve().parent
PAD = 40.0
N_Z = 1600
N_MICRO = 96
ROUGH = 4.0
ENERGY_EV = 284.4
# Full accessible XR window at 284.4 eV (q_max = 4*pi/lambda ~ 0.288 A^-1).
# Dense sampling so thick-film Kiessig fringes resolve.
Q = np.linspace(0.008, 0.275, 520)

N_VAC = 0.0
B_VAC = 0.0
N_SI = 7.5e-4
B_SI = 1.2e-4
N_A_O, N_A_E = 1.1e-3, 1.8e-3
B_A_O, B_A_E = 3.5e-4, 5.5e-4
N_B_O, N_B_E = 0.6e-3, 0.9e-3
B_B_O, B_B_E = 2.0e-4, 3.0e-4
N_C_O, N_C_E = 1.5e-3, 0.7e-3
B_C_O, B_C_E = 4.5e-4, 2.5e-4
N_MOL_XX, N_MOL_ZZ = 1.0e-3, 2.0e-3
B_MOL_XX, B_MOL_ZZ = 3.0e-4, 6.0e-4

# Lab diagonals for N-ary Mix demos (A, B, C).
MAT_O: tuple[complex, ...] = (
    complex(N_A_O, B_A_O),
    complex(N_B_O, B_B_O),
    complex(N_C_O, B_C_O),
)
MAT_E: tuple[complex, ...] = (
    complex(N_A_E, B_A_E),
    complex(N_B_E, B_B_E),
    complex(N_C_E, B_C_E),
)
MAT_FRAC_COLORS: tuple[str, ...] = ("C0", "C1", "C3")
MAT_FRAC_LABELS: tuple[str, ...] = (r"$\phi_A$", r"$\phi_B$", r"$\phi_C$")

ScalarField = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def second_order(
    z: NDArray[np.float64],
    *,
    thickness: float,
    bulk: float,
    top: float,
    bottom: float,
    tau_top: float,
    tau_bottom: float,
) -> NDArray[np.float64]:
    field = SecondOrderTransition(
        thickness=thickness,
        bulk=bulk,
        top=top,
        bottom=bottom,
        tau_top=tau_top,
        tau_bottom=tau_bottom,
    )
    return np.asarray(field.evaluate(z, thickness=thickness), dtype=np.float64)


def diffusion_couple(
    z: NDArray[np.float64],
    *,
    left: float,
    right: float,
    length: float,
    thickness: float,
) -> NDArray[np.float64]:
    field = Diffusion(
        thickness=thickness, kind="couple", left=left, right=right, length=length
    )
    return np.asarray(field.evaluate(z, thickness=thickness), dtype=np.float64)


def diffusion_exponential(
    z: NDArray[np.float64],
    *,
    edge: float,
    base: float,
    length: float,
) -> NDArray[np.float64]:
    field = Diffusion(
        thickness=1.0, kind="exponential", edge=edge, base=base, length=length
    )
    return np.asarray(field.evaluate(z), dtype=np.float64)


def lab_from_gamma(
    gamma: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    c = np.cos(gamma)
    cos2 = c * c
    sin2 = 1.0 - cos2
    n_o = (N_MOL_XX * (1.0 + cos2) + N_MOL_ZZ * sin2) / 2.0
    n_e = N_MOL_XX * sin2 + N_MOL_ZZ * cos2
    b_o = (B_MOL_XX * (1.0 + cos2) + B_MOL_ZZ * sin2) / 2.0
    b_e = B_MOL_XX * sin2 + B_MOL_ZZ * cos2
    return n_o, n_e, b_o, b_e


def lab_from_cos2(
    cos2: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    cos2 = np.clip(cos2, 0.0, 1.0)
    sin2 = 1.0 - cos2
    n_o = (N_MOL_XX * (1.0 + cos2) + N_MOL_ZZ * sin2) / 2.0
    n_e = N_MOL_XX * sin2 + N_MOL_ZZ * cos2
    b_o = (B_MOL_XX * (1.0 + cos2) + B_MOL_ZZ * sin2) / 2.0
    b_e = B_MOL_XX * sin2 + B_MOL_ZZ * cos2
    return n_o, n_e, b_o, b_e


MIX_RULES: tuple[MixRule, ...] = get_args(MixRule)
MIX_LABELS: dict[MixRule, str] = {
    "linear": "linear",
    "maxwell_garnett": "Maxwell-Garnett",
    "bruggeman": "Bruggeman",
}
MIX_LINESTYLES: dict[MixRule, str] = {
    "linear": "-",
    "maxwell_garnett": "--",
    "bruggeman": ":",
}
MIX_NO_COLOR = "C0"
MIX_NE_COLOR = "C1"
MIX_RSS_COLOR = "C0"
MIX_RPP_COLOR = "C1"
MIX_PHI_COLOR = "C2"
if set(MIX_RULES) != set(MIX_LABELS) or set(MIX_RULES) != set(MIX_LINESTYLES):
    msg = "MIX_LABELS / MIX_LINESTYLES must cover every MixRule"
    raise RuntimeError(msg)


def mix_optics(
    phi: NDArray[np.float64],
    *,
    rule: MixRule = "linear",
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Mix lab diagonals of materials A/B with volume fraction ``phi`` of A."""
    phi = np.clip(np.asarray(phi, dtype=np.float64), 0.0, 1.0)
    n_o = mix_channels(
        [complex(N_A_O, B_A_O), complex(N_B_O, B_B_O)],
        [phi, 1.0 - phi],
        rule=rule,
        host_index=1,
    )
    n_e = mix_channels(
        [complex(N_A_E, B_A_E), complex(N_B_E, B_B_E)],
        [phi, 1.0 - phi],
        rule=rule,
        host_index=1,
    )
    return (
        np.asarray(n_o.real, dtype=np.float64),
        np.asarray(n_e.real, dtype=np.float64),
        np.asarray(n_o.imag, dtype=np.float64),
        np.asarray(n_e.imag, dtype=np.float64),
    )


def mix_nary_optics(
    fractions: Sequence[float],
    *,
    rule: MixRule = "linear",
) -> tuple[float, float, float, float]:
    """Homogeneous Mix of the first ``len(fractions)`` demo materials.

    Parameters
    ----------
    fractions
        Volume fractions summing to one; length selects materials A.. from
        ``MAT_O`` / ``MAT_E``.
    rule
        Mixing rule. Only ``linear`` supports three or more components.
    """
    n = len(fractions)
    if n < 1 or n > len(MAT_O):
        msg = f"fractions length must be in 1..{len(MAT_O)}, got {n}"
        raise ValueError(msg)
    total = float(np.sum(fractions))
    if not np.isclose(total, 1.0):
        msg = f"fractions must sum to 1, got {total}"
        raise ValueError(msg)
    n_o = np.asarray(mix_channels(MAT_O[:n], fractions, rule=rule), dtype=np.complex128)
    n_e = np.asarray(mix_channels(MAT_E[:n], fractions, rule=rule), dtype=np.complex128)
    return (
        float(n_o.real.item()),
        float(n_e.real.item()),
        float(n_o.imag.item()),
        float(n_e.imag.item()),
    )


def erf_steps(
    z: NDArray[np.float64],
    boundaries: NDArray[np.float64],
    roughness: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    profile = np.full_like(z, values[0], dtype=np.float64)
    for i, boundary in enumerate(boundaries):
        sigma = max(float(roughness[i]), 1e-6)
        step = values[i + 1] - values[i]
        profile = profile + step * 0.5 * (
            1.0 + erf((z - boundary) / (sigma * np.sqrt(2.0)))
        )
    return profile


def sandwich_continuous(
    z: NDArray[np.float64],
    film: NDArray[np.float64],
    *,
    thickness: float,
    vac: float,
    sub: float,
    rough_top: float,
    rough_bot: float,
    broaden: bool,
) -> NDArray[np.float64]:
    sharp = np.where(z < 0.0, vac, np.where(z > thickness, sub, film))
    if not broaden:
        return sharp
    sigma0 = max(rough_top, 1e-6)
    sigma_t = max(rough_bot, 1e-6)
    w0 = 0.5 * (1.0 + erf(z / (sigma0 * np.sqrt(2.0))))
    wt = 0.5 * (1.0 + erf((z - thickness) / (sigma_t * np.sqrt(2.0))))
    return vac * (1.0 - w0) + film * w0 * (1.0 - wt) + sub * wt


def depth_axis(thickness: float) -> NDArray[np.float64]:
    return np.linspace(-PAD, thickness + PAD, N_Z)


def style_ax(ax: plt.Axes, *, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
    ax.set_ylabel(ylabel)
    ax.axvline(0.0, color="0.7", lw=0.8, ls=":")
    ax.grid(True, alpha=0.25)


def diag_tensor(n_o: complex, n_e: complex) -> NDArray[np.complex128]:
    t = np.zeros((3, 3), dtype=np.complex128)
    t[0, 0] = n_o
    t[1, 1] = n_o
    t[2, 2] = n_e
    return t


def homogeneous_kernel(
    *,
    thickness: float,
    n_o: float,
    n_e: float,
    b_o: float,
    b_e: float,
    rough: float,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    layers = np.array(
        [
            [0.0, N_VAC, B_VAC, 0.0],
            [thickness, n_o, b_o, rough],
            [0.0, N_SI, B_SI, 3.0],
        ],
        dtype=np.float64,
    )
    tensor = np.stack(
        [
            diag_tensor(N_VAC + 1j * B_VAC, N_VAC + 1j * B_VAC),
            diag_tensor(n_o + 1j * b_o, n_e + 1j * b_e),
            diag_tensor(N_SI + 1j * B_SI, N_SI + 1j * B_SI),
        ]
    )
    return layers, tensor


def microslab_kernel(
    *,
    thickness: float,
    n_o: NDArray[np.float64],
    n_e: NDArray[np.float64],
    b_o: NDArray[np.float64],
    b_e: NDArray[np.float64],
    rough_top: float,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    n = n_o.size
    dz = thickness / n
    rows = [[0.0, N_VAC, B_VAC, 0.0]]
    tensors = [diag_tensor(N_VAC + 1j * B_VAC, N_VAC + 1j * B_VAC)]
    for i in range(n):
        rough = rough_top if i == 0 else 0.0
        delta = float(0.5 * (n_o[i] + n_e[i]))
        beta = float(0.5 * (b_o[i] + b_e[i]))
        rows.append([dz, delta, beta, rough])
        tensors.append(diag_tensor(n_o[i] + 1j * b_o[i], n_e[i] + 1j * b_e[i]))
    rows.append([0.0, N_SI, B_SI, 3.0])
    tensors.append(diag_tensor(N_SI + 1j * B_SI, N_SI + 1j * B_SI))
    return np.asarray(rows, dtype=np.float64), np.stack(tensors)


def reflectivity(
    layers: NDArray[np.float64], tensor: NDArray[np.complex128]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    refl, _tran = uniaxial_reflectivity(Q, layers, tensor, ENERGY_EV, parallel=False)
    return np.abs(refl[:, 1, 1]) ** 2, np.abs(refl[:, 0, 0]) ** 2


AnnotateFn = Callable[[plt.Axes, NDArray[np.float64], float], None]


def save_depth(
    path: Path,
    *,
    title: str,
    z: NDArray[np.float64],
    thickness: float,
    n_o_s: NDArray[np.float64],
    n_e_s: NDArray[np.float64],
    n_o_b: NDArray[np.float64],
    n_e_b: NDArray[np.float64],
    aux_name: str | None = None,
    aux_s: NDArray[np.float64] | None = None,
    aux_b: NDArray[np.float64] | None = None,
    annotate_aux: AnnotateFn | None = None,
) -> None:
    nrows = 2 if aux_name is None else 3
    fig, axes = plt.subplots(
        nrows, 2, figsize=(8.8, 2.55 * nrows), sharex=True, constrained_layout=True
    )
    fig.suptitle(f"{title} — depth profile", fontsize=12, fontweight="bold")
    pairs = (
        ("sharp", n_o_s, n_e_s),
        ("broadened (erf interfacial roughness)", n_o_b, n_e_b),
    )
    for col, (tag, n_o, n_e) in enumerate(pairs):
        ax = axes[0, col]
        ax.plot(z, n_o * 1e3, label=r"$n_o$ / $\delta_o$", color="C0", lw=1.7)
        ax.plot(z, n_e * 1e3, label=r"$n_e$ / $\delta_e$", color="C1", lw=1.7)
        style_ax(
            ax,
            title=f"({chr(97 + col)}) optical constants — {tag}",
            ylabel=r"$\delta$ ($\times 10^{-3}$)",
        )
        ax.axvline(thickness, color="0.7", lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=8)

        ax2 = axes[1, col]
        ax2.plot(z, (n_e - n_o) * 1e3, color="C3", lw=1.5)
        style_ax(
            ax2,
            title=f"({chr(99 + col)}) birefringence $n_e-n_o$ — {tag}",
            ylabel=r"$\Delta$ ($\times 10^{-3}$)",
        )
        ax2.axvline(thickness, color="0.7", lw=0.8, ls=":")

    if aux_name is not None and aux_s is not None and aux_b is not None:
        for col, (tag, data) in enumerate((("sharp", aux_s), ("broadened", aux_b))):
            ax = axes[2, col]
            ax.plot(z, data, color="C2", lw=1.7)
            style_ax(
                ax, title=f"({chr(101 + col)}) {aux_name} — {tag}", ylabel=aux_name
            )
            ax.axvline(0.0, color="0.7", lw=0.8, ls=":")
            ax.axvline(thickness, color="0.7", lw=0.8, ls=":")
            if annotate_aux is not None and col == 0:
                annotate_aux(ax, z, thickness)

    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_refl(
    path: Path,
    *,
    title: str,
    r_ss: NDArray[np.float64],
    r_pp: NDArray[np.float64],
    r_sp: NDArray[np.float64] | None = None,
    r_ps: NDArray[np.float64] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.semilogy(Q, r_ss, label=r"$R_{ss}$", color="C0", lw=1.8)
    ax.semilogy(Q, r_pp, label=r"$R_{pp}$", color="C1", lw=1.8)
    if r_sp is not None:
        ax.semilogy(
            Q,
            np.clip(r_sp, 1e-16, None),
            label=r"$R_{sp}$ (s$\leftarrow$p)",
            color="C2",
            lw=1.5,
        )
    if r_ps is not None:
        ax.semilogy(
            Q,
            np.clip(r_ps, 1e-16, None),
            label=r"$R_{ps}$ (p$\leftarrow$s)",
            color="C3",
            lw=1.5,
            ls="--",
        )
    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel("reflectivity")
    ax.set_title(f"{title} — reflectivity @ {ENERGY_EV:.1f} eV")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


VARIANT_LINESTYLE_CYCLE: tuple[str, ...] = ("-", "--", ":", "-.")

OpticsFromField = Callable[
    [NDArray[np.float64]],
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
]


def _variant_linestyles(keys: list[str]) -> dict[str, str]:
    return {
        key: VARIANT_LINESTYLE_CYCLE[i % len(VARIANT_LINESTYLE_CYCLE)]
        for i, key in enumerate(keys)
    }


def _place_variant_legend(
    fig: plt.Figure, *, labels: list[str], linestyles: dict[str, str]
) -> None:
    """Bottom-centered key: one column per named variant (linestyle)."""
    handles = [
        Line2D([0], [0], color="0.2", ls=linestyles[label], lw=1.7, label=label)
        for label in labels
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=8,
        handlelength=2.8,
        columnspacing=1.6,
    )


def _place_method_legend(fig: plt.Figure) -> None:
    """Bottom-centered MixRule key: one column per method."""
    _place_variant_legend(
        fig,
        labels=[MIX_LABELS[rule] for rule in MIX_RULES],
        linestyles={MIX_LABELS[rule]: MIX_LINESTYLES[rule] for rule in MIX_RULES},
    )


def _phi_linear_optics(
    phi: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    return mix_optics(phi, rule="linear")


def save_profile_variant_compare(
    *,
    title: str,
    path_depth: str,
    path_refl: str | None,
    thickness: float,
    variants: dict[str, NDArray[np.float64]],
    optics_from_field: OpticsFromField,
    field_ylabel: str,
    field_title: str,
    annotate_field: AnnotateFn | None = None,
    field_vac: float = 0.0,
    field_sub: float = 0.0,
    field_ylim: tuple[float, float] | None = None,
    linestyles: dict[str, str] | None = None,
    baseline_key: str | None = None,
) -> None:
    r"""Sharp | broadened depth template for named profile variants.

    Layout matches CallableField / mixed-slab figures:
    - columns: sharp | erf-broadened
    - rows: optical constants ($\delta$), birefringence, profile field

    Variants share channel colors ($n_o$ C0, $n_e$ C1) and are separated by
    linestyle. A bottom-centered legend has one column per variant.
    Reflectivity uses $R_{ss}$/$R_{pp}$ colors, the same linestyles, plus
    $\Delta\log_{10} R$ residuals and RMSE vs ``baseline_key``.
    """
    keys = list(variants.keys())
    if not keys:
        msg = "variants must be non-empty"
        raise ValueError(msg)
    styles = linestyles or _variant_linestyles(keys)
    baseline_key = baseline_key or keys[0]
    z = depth_axis(thickness)

    film_optics: dict[str, tuple[NDArray[np.float64], ...]] = {}
    for key in keys:
        field = np.asarray(variants[key], dtype=np.float64)
        film_optics[key] = optics_from_field(field)

    fig, axes = plt.subplots(
        3, 2, figsize=(8.8, 2.55 * 3), sharex=True, constrained_layout=True
    )
    fig.suptitle(f"{title} — depth profile", fontsize=12, fontweight="bold")
    col_tags = ("sharp", "broadened (erf interfacial roughness)")

    for col, (tag, broaden) in enumerate(zip(col_tags, (False, True), strict=True)):
        ax_n = axes[0, col]
        ax_bi = axes[1, col]
        ax_f = axes[2, col]
        for key in keys:
            n_o_f, n_e_f, _bo, _be = film_optics[key]
            field = np.asarray(variants[key], dtype=np.float64)
            ls = styles[key]
            n_o = sandwich_continuous(
                z,
                n_o_f,
                thickness=thickness,
                vac=N_VAC,
                sub=N_SI,
                rough_top=ROUGH,
                rough_bot=ROUGH,
                broaden=broaden,
            )
            n_e = sandwich_continuous(
                z,
                n_e_f,
                thickness=thickness,
                vac=N_VAC,
                sub=N_SI,
                rough_top=ROUGH,
                rough_bot=ROUGH,
                broaden=broaden,
            )
            field_z = sandwich_continuous(
                z,
                field,
                thickness=thickness,
                vac=field_vac,
                sub=field_sub,
                rough_top=ROUGH,
                rough_bot=ROUGH,
                broaden=broaden,
            )
            ax_n.plot(z, n_o * 1e3, color=MIX_NO_COLOR, lw=1.7, ls=ls)
            ax_n.plot(z, n_e * 1e3, color=MIX_NE_COLOR, lw=1.7, ls=ls)
            ax_bi.plot(z, (n_e - n_o) * 1e3, color="C3", lw=1.5, ls=ls)
            ax_f.plot(z, field_z, color=MIX_PHI_COLOR, lw=1.7, ls=ls)

        style_ax(
            ax_n,
            title=f"({chr(97 + col)}) optical constants — {tag}",
            ylabel=r"$\delta$ ($\times 10^{-3}$)",
        )
        ax_n.axvline(thickness, color="0.7", lw=0.8, ls=":")
        if col == 0:
            ax_n.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=MIX_NO_COLOR,
                        lw=1.7,
                        label=r"$n_o$ / $\delta_o$",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=MIX_NE_COLOR,
                        lw=1.7,
                        label=r"$n_e$ / $\delta_e$",
                    ),
                ],
                frameon=False,
                fontsize=8,
            )

        style_ax(
            ax_bi,
            title=f"({chr(99 + col)}) birefringence $n_e-n_o$ — {tag}",
            ylabel=r"$\Delta$ ($\times 10^{-3}$)",
        )
        ax_bi.axvline(thickness, color="0.7", lw=0.8, ls=":")

        style_ax(
            ax_f,
            title=f"({chr(101 + col)}) {field_title} — {tag}",
            ylabel=field_ylabel,
        )
        ax_f.axvline(0.0, color="0.7", lw=0.8, ls=":")
        ax_f.axvline(thickness, color="0.7", lw=0.8, ls=":")
        if field_ylim is not None:
            ax_f.set_ylim(*field_ylim)
        if annotate_field is not None and col == 0:
            annotate_field(ax_f, z, thickness)

    _place_variant_legend(fig, labels=keys, linestyles=styles)
    fig.savefig(OUT / path_depth, dpi=200)
    plt.close(fig)

    if path_refl is None:
        return

    z_m = (np.arange(N_MICRO) + 0.5) * (thickness / N_MICRO)
    curves: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for key in keys:
        n_o_f, n_e_f, b_o_f, b_e_f = film_optics[key]
        n_o_m = np.interp(z_m, z, n_o_f)
        n_e_m = np.interp(z_m, z, n_e_f)
        b_o_m = np.interp(z_m, z, b_o_f)
        b_e_m = np.interp(z_m, z, b_e_f)
        layers, tensor = microslab_kernel(
            thickness=thickness,
            n_o=n_o_m,
            n_e=n_e_m,
            b_o=b_o_m,
            b_e=b_e_m,
            rough_top=2.0,
        )
        curves[key] = reflectivity(layers, tensor)

    r_ss_ref, r_pp_ref = curves[baseline_key]
    log_ss_ref = np.log10(np.clip(r_ss_ref, 1e-30, None))
    log_pp_ref = np.log10(np.clip(r_pp_ref, 1e-30, None))
    show_residual = len(keys) > 1

    if show_residual:
        fig_r, (ax_r, ax_res) = plt.subplots(
            2,
            1,
            figsize=(7.2, 5.2),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [2.2, 1.0]},
        )
    else:
        fig_r, ax_r = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
        ax_res = None

    for key in keys:
        r_ss, r_pp = curves[key]
        ls = styles[key]
        ax_r.semilogy(Q, r_ss, color=MIX_RSS_COLOR, lw=1.8, ls=ls)
        ax_r.semilogy(Q, r_pp, color=MIX_RPP_COLOR, lw=1.8, ls=ls)
        if ax_res is not None and key != baseline_key:
            ax_res.plot(
                Q,
                np.log10(np.clip(r_ss, 1e-30, None)) - log_ss_ref,
                color=MIX_RSS_COLOR,
                lw=1.4,
                ls=ls,
            )
            ax_res.plot(
                Q,
                np.log10(np.clip(r_pp, 1e-30, None)) - log_pp_ref,
                color=MIX_RPP_COLOR,
                lw=1.4,
                ls=ls,
            )

    ax_r.set_ylabel("reflectivity")
    ax_r.set_title(f"{title} — reflectivity @ {ENERGY_EV:.1f} eV")
    ax_r.grid(True, which="both", alpha=0.25)
    ax_r.legend(
        handles=[
            Line2D([0], [0], color=MIX_RSS_COLOR, lw=1.8, label=r"$R_{ss}$"),
            Line2D([0], [0], color=MIX_RPP_COLOR, lw=1.8, label=r"$R_{pp}$"),
        ],
        frameon=False,
    )
    if show_residual:
        rmse_lines = [rf"RMSE vs {baseline_key} ($\log_{{10}} R$)"]
        for key in keys:
            if key == baseline_key:
                continue
            r_ss, r_pp = curves[key]
            rmse_ss = float(
                np.sqrt(
                    np.mean((np.log10(np.clip(r_ss, 1e-30, None)) - log_ss_ref) ** 2)
                )
            )
            rmse_pp = float(
                np.sqrt(
                    np.mean((np.log10(np.clip(r_pp, 1e-30, None)) - log_pp_ref) ** 2)
                )
            )
            rmse_lines.append(
                rf"{key}: $R_{{ss}}={rmse_ss:.2e}$, $R_{{pp}}={rmse_pp:.2e}$"
            )
        ax_r.text(
            0.02,
            0.04,
            "\n".join(rmse_lines),
            transform=ax_r.transAxes,
            fontsize=7.5,
            va="bottom",
            ha="left",
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.92,
            },
        )
        assert ax_res is not None
        ax_res.axhline(0.0, color="0.5", lw=0.8)
        ax_res.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        ax_res.set_ylabel(r"$\Delta\log_{10} R$")
        ax_res.set_title(rf"residual vs {baseline_key}")
        ax_res.grid(True, alpha=0.25)
    else:
        ax_r.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")

    _place_variant_legend(fig_r, labels=keys, linestyles=styles)
    fig_r.savefig(OUT / path_refl, dpi=200)
    plt.close(fig_r)


def save_volfrac_mix_compare(
    *,
    title: str,
    path_depth: str,
    path_refl: str | None,
    thickness: float,
    phi_on_z: NDArray[np.float64],
    annotate_phi: AnnotateFn | None = None,
    phi_vac: float = 1.0,
    phi_sub: float = 0.0,
) -> None:
    r"""Mix-rule depth + reflectivity in the CallableField / ``save_depth`` layout.

    Depth figure is 3x2 like callable:
    - columns: sharp | broadened (erf interfacial roughness)
    - rows: optical constants ($\delta$), birefringence, volume fraction $\phi$

    Channel keys match the homogeneous / callable notation. Mix rules overlay
    by linestyle; a bottom-centered legend has one column per method. When
    ``path_refl`` is ``None``, only the depth figure is written.
    """
    z = depth_axis(thickness)
    phi_film = np.asarray(phi_on_z, dtype=np.float64)
    phi_s = sandwich_continuous(
        z,
        phi_film,
        thickness=thickness,
        vac=phi_vac,
        sub=phi_sub,
        rough_top=ROUGH,
        rough_bot=ROUGH,
        broaden=False,
    )
    phi_b = sandwich_continuous(
        z,
        phi_film,
        thickness=thickness,
        vac=phi_vac,
        sub=phi_sub,
        rough_top=ROUGH,
        rough_bot=ROUGH,
        broaden=True,
    )

    optics_s: dict[MixRule, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    optics_b: dict[MixRule, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for rule in MIX_RULES:
        n_o_f, n_e_f, _bo, _be = mix_optics(phi_film, rule=rule)
        optics_s[rule] = (
            sandwich_continuous(
                z,
                n_o_f,
                thickness=thickness,
                vac=N_VAC,
                sub=N_SI,
                rough_top=ROUGH,
                rough_bot=ROUGH,
                broaden=False,
            ),
            sandwich_continuous(
                z,
                n_e_f,
                thickness=thickness,
                vac=N_VAC,
                sub=N_SI,
                rough_top=ROUGH,
                rough_bot=ROUGH,
                broaden=False,
            ),
        )
        optics_b[rule] = (
            sandwich_continuous(
                z,
                n_o_f,
                thickness=thickness,
                vac=N_VAC,
                sub=N_SI,
                rough_top=ROUGH,
                rough_bot=ROUGH,
                broaden=True,
            ),
            sandwich_continuous(
                z,
                n_e_f,
                thickness=thickness,
                vac=N_VAC,
                sub=N_SI,
                rough_top=ROUGH,
                rough_bot=ROUGH,
                broaden=True,
            ),
        )

    fig, axes = plt.subplots(
        3, 2, figsize=(8.8, 2.55 * 3), sharex=True, constrained_layout=True
    )
    fig.suptitle(f"{title} — depth profile", fontsize=12, fontweight="bold")
    col_tags = ("sharp", "broadened (erf interfacial roughness)")
    col_optics = (optics_s, optics_b)
    col_phi = (phi_s, phi_b)

    for col, (tag, optics, phi) in enumerate(
        zip(col_tags, col_optics, col_phi, strict=True)
    ):
        ax_n = axes[0, col]
        ax_bi = axes[1, col]
        ax_phi = axes[2, col]
        for rule in MIX_RULES:
            n_o, n_e = optics[rule]
            ls = MIX_LINESTYLES[rule]
            ax_n.plot(z, n_o * 1e3, color=MIX_NO_COLOR, lw=1.7, ls=ls)
            ax_n.plot(z, n_e * 1e3, color=MIX_NE_COLOR, lw=1.7, ls=ls)
            ax_bi.plot(z, (n_e - n_o) * 1e3, color="C3", lw=1.5, ls=ls)
        ax_phi.plot(z, phi, color=MIX_PHI_COLOR, lw=1.7)

        style_ax(
            ax_n,
            title=f"({chr(97 + col)}) optical constants — {tag}",
            ylabel=r"$\delta$ ($\times 10^{-3}$)",
        )
        ax_n.axvline(thickness, color="0.7", lw=0.8, ls=":")
        if col == 0:
            ax_n.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=MIX_NO_COLOR,
                        lw=1.7,
                        label=r"$n_o$ / $\delta_o$",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=MIX_NE_COLOR,
                        lw=1.7,
                        label=r"$n_e$ / $\delta_e$",
                    ),
                ],
                frameon=False,
                fontsize=8,
            )

        style_ax(
            ax_bi,
            title=f"({chr(99 + col)}) birefringence $n_e-n_o$ — {tag}",
            ylabel=r"$\Delta$ ($\times 10^{-3}$)",
        )
        ax_bi.axvline(thickness, color="0.7", lw=0.8, ls=":")

        style_ax(
            ax_phi,
            title=f"({chr(101 + col)}) $\\phi(z)$ — {tag}",
            ylabel=r"$\phi$",
        )
        ax_phi.axvline(0.0, color="0.7", lw=0.8, ls=":")
        ax_phi.axvline(thickness, color="0.7", lw=0.8, ls=":")
        ax_phi.set_ylim(-0.05, 1.05)
        if annotate_phi is not None and col == 0:
            annotate_phi(ax_phi, z, thickness)

    _place_method_legend(fig)
    fig.savefig(OUT / path_depth, dpi=200)
    plt.close(fig)

    z_m = (np.arange(N_MICRO) + 0.5) * (thickness / N_MICRO)
    phi_m = np.interp(z_m, z, phi_film)

    if path_refl is None:
        return

    curves: dict[MixRule, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for rule in MIX_RULES:
        n_o_m, n_e_m, b_o_m, b_e_m = mix_optics(phi_m, rule=rule)
        layers, tensor = microslab_kernel(
            thickness=thickness,
            n_o=n_o_m,
            n_e=n_e_m,
            b_o=b_o_m,
            b_e=b_e_m,
            rough_top=2.0,
        )
        curves[rule] = reflectivity(layers, tensor)

    r_ss_lin, r_pp_lin = curves["linear"]
    log_ss_lin = np.log10(np.clip(r_ss_lin, 1e-30, None))
    log_pp_lin = np.log10(np.clip(r_pp_lin, 1e-30, None))

    fig_r, (ax_r, ax_res) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for rule in MIX_RULES:
        r_ss, r_pp = curves[rule]
        ls = MIX_LINESTYLES[rule]
        ax_r.semilogy(Q, r_ss, color=MIX_RSS_COLOR, lw=1.8, ls=ls)
        ax_r.semilogy(Q, r_pp, color=MIX_RPP_COLOR, lw=1.8, ls=ls)
        if rule == "linear":
            continue
        d_ss = np.log10(np.clip(r_ss, 1e-30, None)) - log_ss_lin
        d_pp = np.log10(np.clip(r_pp, 1e-30, None)) - log_pp_lin
        ax_res.plot(Q, d_ss, color=MIX_RSS_COLOR, lw=1.4, ls=ls)
        ax_res.plot(Q, d_pp, color=MIX_RPP_COLOR, lw=1.4, ls=ls)

    ax_r.set_ylabel("reflectivity")
    ax_r.set_title(f"{title} — reflectivity @ {ENERGY_EV:.1f} eV")
    ax_r.grid(True, which="both", alpha=0.25)
    ax_r.legend(
        handles=[
            Line2D([0], [0], color=MIX_RSS_COLOR, lw=1.8, label=r"$R_{ss}$"),
            Line2D([0], [0], color=MIX_RPP_COLOR, lw=1.8, label=r"$R_{pp}$"),
        ],
        frameon=False,
    )

    rmse_lines = [r"RMSE vs linear ($\log_{10} R$)"]
    for rule in MIX_RULES:
        if rule == "linear":
            continue
        r_ss, r_pp = curves[rule]
        rmse_ss = float(
            np.sqrt(np.mean((np.log10(np.clip(r_ss, 1e-30, None)) - log_ss_lin) ** 2))
        )
        rmse_pp = float(
            np.sqrt(np.mean((np.log10(np.clip(r_pp, 1e-30, None)) - log_pp_lin) ** 2))
        )
        rmse_lines.append(
            rf"{MIX_LABELS[rule]}: "
            rf"$R_{{ss}}={rmse_ss:.2e}$, $R_{{pp}}={rmse_pp:.2e}$"
        )
    ax_r.text(
        0.02,
        0.04,
        "\n".join(rmse_lines),
        transform=ax_r.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="left",
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    ax_res.axhline(0.0, color="0.5", lw=0.8)
    ax_res.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax_res.set_ylabel(r"$\Delta\log_{10} R$")
    ax_res.set_title(r"residual vs linear mix")
    ax_res.grid(True, alpha=0.25)

    _place_method_legend(fig_r)
    fig_r.savefig(OUT / path_refl, dpi=200)
    plt.close(fig_r)


def _annotate_hline(
    ax: plt.Axes, y: float, label: str, *, color: str = "0.35", x: float = 0.02
) -> None:
    ax.axhline(y, color=color, lw=0.9, ls="--", alpha=0.85)
    xmin, xmax = ax.get_xlim()
    ax.text(
        xmin + x * (xmax - xmin),
        y,
        label,
        color=color,
        fontsize=7.5,
        va="bottom",
        ha="left",
    )


def _annotate_vspan(
    ax: plt.Axes, x0: float, x1: float, label: str, *, color: str = "C3"
) -> None:
    ax.axvspan(x0, x1, color=color, alpha=0.12, lw=0)
    ax.annotate(
        label,
        xy=(0.5 * (x0 + x1), ax.get_ylim()[1]),
        xytext=(0, -4),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=7.5,
        color=color,
    )


def case_homogeneous_slab() -> None:
    """Free uniaxial tensor — four free components ``delta/beta_{o,e}``."""
    t = 280.0
    z = depth_axis(t)
    n_o_film, n_e_film = 1.2e-3, 1.8e-3
    b_o_film, b_e_film = 4e-4, 6e-4
    bounds = np.array([0.0, t])
    rough = np.array([ROUGH, ROUGH])
    n_o_s = erf_steps(
        z, bounds, np.array([1e-6, 1e-6]), np.array([N_VAC, n_o_film, N_SI])
    )
    n_e_s = erf_steps(
        z, bounds, np.array([1e-6, 1e-6]), np.array([N_VAC, n_e_film, N_SI])
    )
    n_o_b = erf_steps(z, bounds, rough, np.array([N_VAC, n_o_film, N_SI]))
    n_e_b = erf_steps(z, bounds, rough, np.array([N_VAC, n_e_film, N_SI]))
    b_o_s = erf_steps(
        z, bounds, np.array([1e-6, 1e-6]), np.array([B_VAC, b_o_film, B_SI])
    )
    b_e_s = erf_steps(
        z, bounds, np.array([1e-6, 1e-6]), np.array([B_VAC, b_e_film, B_SI])
    )
    b_o_b = erf_steps(z, bounds, rough, np.array([B_VAC, b_o_film, B_SI]))
    b_e_b = erf_steps(z, bounds, rough, np.array([B_VAC, b_e_film, B_SI]))

    fig, axes = plt.subplots(
        3, 2, figsize=(8.8, 7.6), sharex=True, constrained_layout=True
    )
    fig.suptitle(
        r"Free uniaxial tensor — 4 components "
        r"($\delta_o,\beta_o,\delta_e,\beta_e$)",
        fontsize=12,
        fontweight="bold",
    )
    pairs = (
        ("sharp", n_o_s, n_e_s, b_o_s, b_e_s),
        ("broadened (erf interfacial roughness)", n_o_b, n_e_b, b_o_b, b_e_b),
    )
    for col, (tag, n_o, n_e, b_o, b_e) in enumerate(pairs):
        ax = axes[0, col]
        ax.plot(z, n_o * 1e3, label=r"$\delta_o$", color="C0", lw=1.7)
        ax.plot(z, n_e * 1e3, label=r"$\delta_e$", color="C1", lw=1.7)
        style_ax(
            ax,
            title=f"({chr(97 + col)}) dispersion — {tag}",
            ylabel=r"$\delta$ ($\times 10^{-3}$)",
        )
        ax.axvline(t, color="0.7", lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=8)

        ax2 = axes[1, col]
        ax2.plot(z, b_o * 1e3, label=r"$\beta_o$", color="C0", lw=1.7, ls="--")
        ax2.plot(z, b_e * 1e3, label=r"$\beta_e$", color="C1", lw=1.7, ls="--")
        style_ax(
            ax2,
            title=f"({chr(99 + col)}) absorption — {tag}",
            ylabel=r"$\beta$ ($\times 10^{-3}$)",
        )
        ax2.axvline(t, color="0.7", lw=0.8, ls=":")
        ax2.legend(frameon=False, fontsize=8)

        ax3 = axes[2, col]
        ax3.plot(z, (n_e - n_o) * 1e3, color="C3", lw=1.5, label=r"$\delta_e-\delta_o$")
        ax3.plot(
            z,
            (b_e - b_o) * 1e3,
            color="C2",
            lw=1.5,
            ls="--",
            label=r"$\beta_e-\beta_o$",
        )
        style_ax(
            ax3,
            title=f"({chr(101 + col)}) anisotropy — {tag}",
            ylabel=r"$\Delta$ ($\times 10^{-3}$)",
        )
        ax3.axvline(t, color="0.7", lw=0.8, ls=":")
        if col == 0:
            ax3.legend(frameon=False, fontsize=8)
            ax3.text(
                0.5 * t,
                0.12,
                r"$\mathrm{diag}=(n_o,\,n_o,\,n_e)$"
                "\n"
                rf"$\delta_o={n_o_film * 1e3:.2f}$, "
                rf"$\beta_o={b_o_film * 1e3:.2f}$,"
                "\n"
                rf"$\delta_e={n_e_film * 1e3:.2f}$, "
                rf"$\beta_e={b_e_film * 1e3:.2f}$"
                r" ($\times 10^{-3}$)",
                fontsize=7.5,
                color="0.25",
                ha="center",
                transform=ax3.get_xaxis_transform(),
                va="bottom",
            )
    fig.savefig(OUT / "homogeneous.png", dpi=200)
    plt.close(fig)

    layers, tensor = homogeneous_kernel(
        thickness=t, n_o=n_o_film, n_e=n_e_film, b_o=b_o_film, b_e=b_e_film, rough=2.0
    )
    r_ss, r_pp = reflectivity(layers, tensor)
    save_refl(
        OUT / "homogeneous-R.png",
        title=r"Free uniaxial tensor (4 components)",
        r_ss=r_ss,
        r_pp=r_pp,
    )


def case_homogeneous_material_tensor() -> None:
    """Material uniaxial — OOC diagonals scaled by density, rotated by gamma."""
    t = 280.0
    z = depth_axis(t)
    # Unit-density molecular indices (like a tiny OOC table at 284.4 eV).
    n_xx0, n_zz0 = 1.0e-3 + 3.0e-4j, 2.0e-3 + 6.0e-4j
    density = 1.45
    gamma = 0.65  # UniTensorSLD.rotation (polar angle)
    n_xx = n_xx0 * density
    n_zz = n_zz0 * density
    lab = np.asarray(uniaxial_lab_tensor(n_xx, n_zz, gamma), dtype=np.complex128)
    n_o_film, n_e_film = float(lab[0, 0].real), float(lab[2, 2].real)
    b_o_film, b_e_film = float(lab[0, 0].imag), float(lab[2, 2].imag)

    # Contrast: same OOC, different (density, gamma) — linestyle variants.
    dens_b, gamma_b = 1.10, 0.20
    lab_b = np.asarray(
        uniaxial_lab_tensor(n_xx0 * dens_b, n_zz0 * dens_b, gamma_b),
        dtype=np.complex128,
    )
    variants: list[tuple[str, str, float, float, float, float, float, float]] = [
        (
            "A",
            "-",
            density,
            gamma,
            n_o_film,
            n_e_film,
            b_o_film,
            b_e_film,
        ),
        (
            "B",
            "--",
            dens_b,
            gamma_b,
            float(lab_b[0, 0].real),
            float(lab_b[2, 2].real),
            float(lab_b[0, 0].imag),
            float(lab_b[2, 2].imag),
        ),
    ]

    fig, axes = plt.subplots(
        3, 2, figsize=(8.8, 7.6), sharex=True, constrained_layout=True
    )
    fig.suptitle(
        r"Material uniaxial tensor — $\rho$ and $\gamma$ (API: rotation)",
        fontsize=12,
        fontweight="bold",
    )
    bounds = np.array([0.0, t])
    rough = np.array([ROUGH, ROUGH])
    col_tags = ("sharp", "broadened (erf interfacial roughness)")
    for col, (tag, broaden) in enumerate(zip(col_tags, (False, True), strict=True)):
        sig = np.array([1e-6, 1e-6]) if not broaden else rough
        ax_n = axes[0, col]
        ax_bi = axes[1, col]
        ax_g = axes[2, col]
        for _name, ls, rho, gam, no, ne, _bo, _be in variants:
            n_o = erf_steps(z, bounds, sig, np.array([N_VAC, no, N_SI]))
            n_e = erf_steps(z, bounds, sig, np.array([N_VAC, ne, N_SI]))
            ax_n.plot(z, n_o * 1e3, color=MIX_NO_COLOR, lw=1.6, ls=ls)
            ax_n.plot(z, n_e * 1e3, color=MIX_NE_COLOR, lw=1.6, ls=ls)
            ax_bi.plot(z, (n_e - n_o) * 1e3, color="C3", lw=1.5, ls=ls)
            rho_z = erf_steps(z, bounds, sig, np.array([0.0, rho, 0.0]))
            gam_z = erf_steps(z, bounds, sig, np.array([0.0, gam, 0.0]))
            ax_g.plot(z, rho_z, color="C2", lw=1.5, ls=ls)
            ax_g.plot(z, gam_z, color="C4", lw=1.5, ls=ls)
        style_ax(
            ax_n,
            title=f"({chr(97 + col)}) lab $\\delta_o,\\delta_e$ — {tag}",
            ylabel=r"$\delta$ ($\times 10^{-3}$)",
        )
        ax_n.axvline(t, color="0.7", lw=0.8, ls=":")
        if col == 0:
            ax_n.legend(
                handles=[
                    Line2D([0], [0], color=MIX_NO_COLOR, lw=1.6, label=r"$\delta_o$"),
                    Line2D([0], [0], color=MIX_NE_COLOR, lw=1.6, label=r"$\delta_e$"),
                    Line2D(
                        [0],
                        [0],
                        color="0.3",
                        lw=1.6,
                        ls="-",
                        label=rf"A: $\rho={density:.2f},\gamma={gamma:.2f}$",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="0.3",
                        lw=1.6,
                        ls="--",
                        label=rf"B: $\rho={dens_b:.2f},\gamma={gamma_b:.2f}$",
                    ),
                ],
                frameon=False,
                fontsize=7.5,
            )
        style_ax(
            ax_bi,
            title=f"({chr(99 + col)}) birefringence — {tag}",
            ylabel=r"$\Delta$ ($\times 10^{-3}$)",
        )
        ax_bi.axvline(t, color="0.7", lw=0.8, ls=":")
        style_ax(
            ax_g,
            title=f"({chr(101 + col)}) $\\rho$ (green), $\\gamma$ (purple) — {tag}",
            ylabel=r"$\rho$ / $\gamma$",
        )
        ax_g.axvline(t, color="0.7", lw=0.8, ls=":")
        if col == 0:
            ax_g.text(
                0.5 * t,
                0.88,
                r"$\mathrm{lab}=R(\gamma)\,n_\mathrm{mol}(\rho)$"
                "\n"
                r"$n_\mathrm{mol}\propto\rho\times\mathrm{OOC}$",
                fontsize=7.5,
                color="0.25",
                ha="center",
                transform=ax_g.get_xaxis_transform(),
                va="top",
            )
    fig.savefig(OUT / "homogeneous-material.png", dpi=200)
    plt.close(fig)

    layers, tensor = homogeneous_kernel(
        thickness=t, n_o=n_o_film, n_e=n_e_film, b_o=b_o_film, b_e=b_e_film, rough=2.0
    )
    r_ss, r_pp = reflectivity(layers, tensor)
    save_refl(
        OUT / "homogeneous-material-R.png",
        title=rf"Material uniaxial ($\rho={density:.2f}$, $\gamma={gamma:.2f}$)",
        r_ss=r_ss,
        r_pp=r_pp,
    )


def case_homogeneous_uni_vs_bi() -> None:
    """Uniaxial (4 free scalars) vs biaxial (6 free scalars) diagonal packing."""
    t = 240.0
    z = depth_axis(t)
    bounds = np.array([0.0, t])
    rough = np.array([ROUGH, ROUGH])

    # Uniaxial: xx = yy (4 free: Re/Im o and e).
    uni_xx = 1.15e-3 + 3.5e-4j
    uni_zz = 1.85e-3 + 5.5e-4j
    # Biaxial principal diagonals (6 free); R uses an in-plane rotation.
    bi_xx = 0.80e-3 + 2.0e-4j
    bi_yy = 2.20e-3 + 7.0e-4j
    bi_zz = 1.50e-3 + 4.0e-4j
    bi_phi = np.deg2rad(45.0)

    def step_c(
        val: complex, *, broaden: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sig = rough if broaden else np.array([1e-6, 1e-6])
        d = erf_steps(z, bounds, sig, np.array([N_VAC, val.real, N_SI]))
        b = erf_steps(z, bounds, sig, np.array([B_VAC, val.imag, B_SI]))
        return d, b

    fig, axes = plt.subplots(
        2, 2, figsize=(9.2, 6.2), sharex=True, constrained_layout=True
    )
    fig.suptitle(
        r"Homogeneous diagonals — uniaxial (4 free) vs biaxial (6 free)",
        fontsize=12,
        fontweight="bold",
    )

    # Column 0: uniaxial; column 1: biaxial. Rows: delta, beta.
    for col, (title, components) in enumerate(
        (
            (
                r"uniaxial: $n_{xx}=n_{yy}\neq n_{zz}$ (4 free)",
                {
                    r"$\delta_{xx}=\delta_{yy}$": uni_xx,
                    r"$\delta_{zz}$": uni_zz,
                },
            ),
            (
                r"biaxial: $n_{xx}\neq n_{yy}\neq n_{zz}$ (6 free)",
                {
                    r"$\delta_{xx}$": bi_xx,
                    r"$\delta_{yy}$": bi_yy,
                    r"$\delta_{zz}$": bi_zz,
                },
            ),
        )
    ):
        colors = ("C0", "C1", "C3")
        ax_d = axes[0, col]
        ax_b = axes[1, col]
        for i, (lab, val) in enumerate(components.items()):
            d_s, b_s = step_c(val, broaden=False)
            d_w, b_w = step_c(val, broaden=True)
            c = colors[i % len(colors)]
            ax_d.plot(z, d_s * 1e3, color=c, lw=1.7, label=lab)
            ax_d.plot(z, d_w * 1e3, color=c, lw=1.2, ls=":", alpha=0.85)
            beta_lab = lab.replace(r"\delta", r"\beta")
            ax_b.plot(z, b_s * 1e3, color=c, lw=1.7, ls="--", label=beta_lab)
            ax_b.plot(z, b_w * 1e3, color=c, lw=1.2, ls=":", alpha=0.85)
        style_ax(
            ax_d,
            title=f"({chr(97 + col)}) {title}",
            ylabel=r"$\delta$ ($\times 10^{-3}$)",
        )
        ax_d.axvline(t, color="0.7", lw=0.8, ls=":")
        ax_d.legend(frameon=False, fontsize=7.5)
        style_ax(
            ax_b,
            title=f"({chr(99 + col)}) absorption diagonals — "
            + ("uni" if col == 0 else "bi"),
            ylabel=r"$\beta$ ($\times 10^{-3}$)",
        )
        ax_b.axvline(t, color="0.7", lw=0.8, ls=":")
        ax_b.legend(frameon=False, fontsize=7.5)
        if col == 0:
            ax_b.text(
                0.5 * t,
                0.12,
                r"solid=sharp, dotted=broadened"
                "\n"
                r"uni free: $\delta_o,\beta_o,\delta_e,\beta_e$",
                fontsize=7.5,
                color="0.3",
                ha="center",
                transform=ax_b.get_xaxis_transform(),
                va="bottom",
            )
        else:
            ax_b.text(
                0.5 * t,
                0.12,
                r"bi free: $\delta_{xx},\beta_{xx}$,"
                "\n"
                r"$\delta_{yy},\beta_{yy},\delta_{zz},\beta_{zz}$",
                fontsize=7.5,
                color="0.3",
                ha="center",
                transform=ax_b.get_xaxis_transform(),
                va="bottom",
            )
    fig.savefig(OUT / "homogeneous-uni-bi.png", dpi=200)
    plt.close(fig)

    # Biaxial stack via general kernel packing [[R_pp, R_sp], [R_ps, R_ss]]:
    # R_sp = s reflected from p incident ([:,0,1]); R_ps = p from s ([:,1,0]).
    # Lab-frame diagonal biaxial keeps R_sp=R_ps=0 under xz incidence; rotate
    # about z so the in-plane anisotropy mixes s and p.
    def full_diag(nx: complex, ny: complex, nz: complex) -> NDArray[np.complex128]:
        m = np.zeros((3, 3), dtype=np.complex128)
        m[0, 0], m[1, 1], m[2, 2] = nx, ny, nz
        return m

    c_phi, s_phi = float(np.cos(bi_phi)), float(np.sin(bi_phi))
    rot_z = np.array(
        [[c_phi, -s_phi, 0.0], [s_phi, c_phi, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    t_bi_lab = rot_z @ full_diag(bi_xx, bi_yy, bi_zz) @ rot_z.T
    layers_bi = np.array(
        [
            [0.0, N_VAC, B_VAC, 0.0],
            [
                t,
                float(0.5 * (t_bi_lab[0, 0].real + t_bi_lab[1, 1].real)),
                float(0.5 * (t_bi_lab[0, 0].imag + t_bi_lab[1, 1].imag)),
                2.0,
            ],
            [0.0, N_SI, B_SI, 3.0],
        ],
        dtype=np.float64,
    )
    tensor_bi = np.stack(
        [
            full_diag(N_VAC + 1j * B_VAC, N_VAC + 1j * B_VAC, N_VAC + 1j * B_VAC),
            t_bi_lab,
            full_diag(N_SI + 1j * B_SI, N_SI + 1j * B_SI, N_SI + 1j * B_SI),
        ]
    )
    refl_bi = general_reflectivity(Q, layers_bi, tensor_bi, ENERGY_EV, parallel=False)
    save_refl(
        OUT / "homogeneous-uni-bi-R.png",
        title=r"Biaxial homogeneous ($\phi_z=45^\circ$) — "
        r"$R_{ss},R_{sp},R_{ps},R_{pp}$",
        r_ss=refl_bi[:, 1, 1],
        r_pp=refl_bi[:, 0, 0],
        r_sp=refl_bi[:, 0, 1],
        r_ps=refl_bi[:, 1, 0],
    )


def _continuous_case(
    *,
    title: str,
    thickness: float,
    film_optics: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    aux_name: str | None,
    aux_film: NDArray[np.float64] | None,
    aux_vac: float = 0.0,
    aux_sub: float = 0.0,
    path_depth: str,
    path_refl: str,
    annotate_aux: AnnotateFn | None = None,
) -> None:
    z = depth_axis(thickness)
    n_o_f, n_e_f, b_o_f, b_e_f = film_optics
    n_o_s = sandwich_continuous(
        z,
        n_o_f,
        thickness=thickness,
        vac=N_VAC,
        sub=N_SI,
        rough_top=ROUGH,
        rough_bot=ROUGH,
        broaden=False,
    )
    n_e_s = sandwich_continuous(
        z,
        n_e_f,
        thickness=thickness,
        vac=N_VAC,
        sub=N_SI,
        rough_top=ROUGH,
        rough_bot=ROUGH,
        broaden=False,
    )
    n_o_b = sandwich_continuous(
        z,
        n_o_f,
        thickness=thickness,
        vac=N_VAC,
        sub=N_SI,
        rough_top=ROUGH,
        rough_bot=ROUGH,
        broaden=True,
    )
    n_e_b = sandwich_continuous(
        z,
        n_e_f,
        thickness=thickness,
        vac=N_VAC,
        sub=N_SI,
        rough_top=ROUGH,
        rough_bot=ROUGH,
        broaden=True,
    )
    aux_s = aux_b = None
    if aux_name is not None and aux_film is not None:
        aux_s = sandwich_continuous(
            z,
            aux_film,
            thickness=thickness,
            vac=aux_vac,
            sub=aux_sub,
            rough_top=ROUGH,
            rough_bot=ROUGH,
            broaden=False,
        )
        aux_b = sandwich_continuous(
            z,
            aux_film,
            thickness=thickness,
            vac=aux_vac,
            sub=aux_sub,
            rough_top=ROUGH,
            rough_bot=ROUGH,
            broaden=True,
        )
    save_depth(
        OUT / path_depth,
        title=title,
        z=z,
        thickness=thickness,
        n_o_s=n_o_s,
        n_e_s=n_e_s,
        n_o_b=n_o_b,
        n_e_b=n_e_b,
        aux_name=aux_name,
        aux_s=aux_s,
        aux_b=aux_b,
        annotate_aux=annotate_aux,
    )
    z_m = (np.arange(N_MICRO) + 0.5) * (thickness / N_MICRO)
    n_o_m = np.interp(z_m, z, n_o_f)
    n_e_m = np.interp(z_m, z, n_e_f)
    b_o_m = np.interp(z_m, z, b_o_f)
    b_e_m = np.interp(z_m, z, b_e_f)
    layers, tensor = microslab_kernel(
        thickness=thickness, n_o=n_o_m, n_e=n_e_m, b_o=b_o_m, b_e=b_e_m, rough_top=2.0
    )
    r_ss, r_pp = reflectivity(layers, tensor)
    save_refl(OUT / path_refl, title=title, r_ss=r_ss, r_pp=r_pp)


def case_mixed_homogeneous() -> None:
    """Single slab - Mix at fixed volume fraction; all MixRule variants."""
    t = 200.0
    phi_val = 0.3
    z = depth_axis(t)
    phi = np.full_like(z, phi_val, dtype=np.float64)

    def annotate(ax: plt.Axes, _z: NDArray[np.float64], _thickness: float) -> None:
        _annotate_hline(ax, phi_val, rf"$\phi={phi_val}$", color=MIX_PHI_COLOR)

    save_volfrac_mix_compare(
        title=rf"Mixed homogeneous ($\phi={phi_val}$)",
        path_depth="mixed-homogeneous.png",
        path_refl="mixed-homogeneous-R.png",
        thickness=t,
        phi_on_z=phi,
        annotate_phi=annotate,
        phi_vac=0.0,
        phi_sub=0.0,
    )


def save_nary_homogeneous_mix_compare(
    *,
    title: str,
    path_depth: str,
    path_refl: str,
    thickness: float,
    variants: dict[str, Sequence[float]],
    rule: MixRule = "linear",
    baseline_key: str | None = None,
) -> None:
    r"""Compare homogeneous Mix slabs with 1 / 2 / 3 materials.

    Layout matches the mixed-homogeneous template (sharp | broadened; rows
    $\delta$, birefringence, volume fractions). Each variant is a fraction
    vector over materials A..; linestyle separates $N$. Bottom row draws
    horizontal lines at each listed $\phi_i$ (color by material).
    Reflectivity overlays $R_{ss}$/$R_{pp}$ with residuals vs ``baseline_key``.
    """
    keys = list(variants.keys())
    if not keys:
        msg = "variants must be non-empty"
        raise ValueError(msg)
    styles = _variant_linestyles(keys)
    baseline_key = baseline_key or keys[0]
    z = depth_axis(thickness)

    film_optics: dict[str, tuple[float, float, float, float]] = {}
    for key in keys:
        fracs = list(variants[key])
        film_optics[key] = mix_nary_optics(fracs, rule=rule)

    fig, axes = plt.subplots(
        3, 2, figsize=(8.8, 2.55 * 3), sharex=True, constrained_layout=True
    )
    fig.suptitle(f"{title} — depth profile", fontsize=12, fontweight="bold")
    col_tags = ("sharp", "broadened (erf interfacial roughness)")

    for col, (tag, broaden) in enumerate(zip(col_tags, (False, True), strict=True)):
        ax_n = axes[0, col]
        ax_bi = axes[1, col]
        ax_f = axes[2, col]
        for key in keys:
            n_o_f, n_e_f, _bo, _be = film_optics[key]
            ls = styles[key]
            n_o_film = np.full_like(z, n_o_f, dtype=np.float64)
            n_e_film = np.full_like(z, n_e_f, dtype=np.float64)
            n_o = sandwich_continuous(
                z,
                n_o_film,
                thickness=thickness,
                vac=N_VAC,
                sub=N_SI,
                rough_top=ROUGH,
                rough_bot=ROUGH,
                broaden=broaden,
            )
            n_e = sandwich_continuous(
                z,
                n_e_film,
                thickness=thickness,
                vac=N_VAC,
                sub=N_SI,
                rough_top=ROUGH,
                rough_bot=ROUGH,
                broaden=broaden,
            )
            ax_n.plot(z, n_o * 1e3, color=MIX_NO_COLOR, lw=1.7, ls=ls)
            ax_n.plot(z, n_e * 1e3, color=MIX_NE_COLOR, lw=1.7, ls=ls)
            ax_bi.plot(z, (n_e - n_o) * 1e3, color="C3", lw=1.5, ls=ls)

            fracs = list(variants[key])
            for i, frac in enumerate(fracs):
                ax_f.axhline(
                    float(frac),
                    color=MAT_FRAC_COLORS[i],
                    lw=1.6,
                    ls=ls,
                    xmin=0.08,
                    xmax=0.92,
                )

        style_ax(
            ax_n,
            title=f"({chr(97 + col)}) optical constants — {tag}",
            ylabel=r"$\delta$ ($\times 10^{-3}$)",
        )
        ax_n.axvline(thickness, color="0.7", lw=0.8, ls=":")
        if col == 0:
            ax_n.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=MIX_NO_COLOR,
                        lw=1.7,
                        label=r"$n_o$ / $\delta_o$",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=MIX_NE_COLOR,
                        lw=1.7,
                        label=r"$n_e$ / $\delta_e$",
                    ),
                ],
                frameon=False,
                fontsize=8,
            )

        style_ax(
            ax_bi,
            title=f"({chr(99 + col)}) birefringence $n_e-n_o$ — {tag}",
            ylabel=r"$\Delta$ ($\times 10^{-3}$)",
        )
        ax_bi.axvline(thickness, color="0.7", lw=0.8, ls=":")

        style_ax(
            ax_f,
            title=f"({chr(101 + col)}) volume fractions — {tag}",
            ylabel=r"$\phi_i$",
        )
        ax_f.set_ylim(-0.05, 1.05)
        ax_f.axvline(0.0, color="0.7", lw=0.8, ls=":")
        ax_f.axvline(thickness, color="0.7", lw=0.8, ls=":")
        if col == 0:
            ax_f.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=MAT_FRAC_COLORS[i],
                        lw=1.6,
                        label=MAT_FRAC_LABELS[i],
                    )
                    for i in range(len(MAT_FRAC_LABELS))
                ],
                frameon=False,
                fontsize=8,
                loc="upper right",
            )
            ax_f.text(
                0.5 * thickness,
                0.08,
                rf"rule={rule!r}",
                fontsize=8,
                color="0.35",
                ha="center",
            )

    _place_variant_legend(fig, labels=keys, linestyles=styles)
    fig.savefig(OUT / path_depth, dpi=200)
    plt.close(fig)

    curves: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for key in keys:
        n_o_f, n_e_f, b_o_f, b_e_f = film_optics[key]
        layers, tensor = homogeneous_kernel(
            thickness=thickness,
            n_o=n_o_f,
            n_e=n_e_f,
            b_o=b_o_f,
            b_e=b_e_f,
            rough=2.0,
        )
        curves[key] = reflectivity(layers, tensor)

    r_ss_ref, r_pp_ref = curves[baseline_key]
    log_ss_ref = np.log10(np.clip(r_ss_ref, 1e-30, None))
    log_pp_ref = np.log10(np.clip(r_pp_ref, 1e-30, None))

    fig_r, (ax_r, ax_res) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for key in keys:
        r_ss, r_pp = curves[key]
        ls = styles[key]
        ax_r.semilogy(Q, r_ss, color=MIX_RSS_COLOR, lw=1.8, ls=ls)
        ax_r.semilogy(Q, r_pp, color=MIX_RPP_COLOR, lw=1.8, ls=ls)
        if key != baseline_key:
            ax_res.plot(
                Q,
                np.log10(np.clip(r_ss, 1e-30, None)) - log_ss_ref,
                color=MIX_RSS_COLOR,
                lw=1.4,
                ls=ls,
            )
            ax_res.plot(
                Q,
                np.log10(np.clip(r_pp, 1e-30, None)) - log_pp_ref,
                color=MIX_RPP_COLOR,
                lw=1.4,
                ls=ls,
            )

    ax_r.set_ylabel("reflectivity")
    ax_r.set_title(f"{title} — reflectivity @ {ENERGY_EV:.1f} eV")
    ax_r.grid(True, which="both", alpha=0.25)
    ax_r.legend(
        handles=[
            Line2D([0], [0], color=MIX_RSS_COLOR, lw=1.8, label=r"$R_{ss}$"),
            Line2D([0], [0], color=MIX_RPP_COLOR, lw=1.8, label=r"$R_{pp}$"),
        ],
        frameon=False,
    )
    rmse_lines = [rf"RMSE vs {baseline_key} ($\log_{{10}} R$)"]
    for key in keys:
        if key == baseline_key:
            continue
        r_ss, r_pp = curves[key]
        rmse_ss = float(
            np.sqrt(np.mean((np.log10(np.clip(r_ss, 1e-30, None)) - log_ss_ref) ** 2))
        )
        rmse_pp = float(
            np.sqrt(np.mean((np.log10(np.clip(r_pp, 1e-30, None)) - log_pp_ref) ** 2))
        )
        rmse_lines.append(rf"{key}: $R_{{ss}}={rmse_ss:.2e}$, $R_{{pp}}={rmse_pp:.2e}$")
    ax_r.text(
        0.02,
        0.04,
        "\n".join(rmse_lines),
        transform=ax_r.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="left",
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )
    ax_res.axhline(0.0, color="0.5", lw=0.8)
    ax_res.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax_res.set_ylabel(r"$\Delta\log_{10} R$")
    ax_res.set_title(rf"residual vs {baseline_key}")
    ax_res.grid(True, alpha=0.25)
    _place_variant_legend(fig_r, labels=keys, linestyles=styles)
    fig_r.savefig(OUT / path_refl, dpi=200)
    plt.close(fig_r)


def case_mixed_homogeneous_nary() -> None:
    """Single slab - linear Mix with 1 / 2 / 3 materials and listed fractions."""
    t = 200.0
    save_nary_homogeneous_mix_compare(
        title=r"Mixed homogeneous $N=1,2,3$ (linear)",
        path_depth="mixed-homogeneous-nary.png",
        path_refl="mixed-homogeneous-nary-R.png",
        thickness=t,
        variants={
            r"$N=1$: $[1]$": [1.0],
            r"$N=2$: $[0.3,\,0.7]$": [0.3, 0.7],
            r"$N=3$: $[0.2,\,0.7,\,0.1]$": [0.2, 0.7, 0.1],
        },
        rule="linear",
        baseline_key=r"$N=1$: $[1]$",
    )


def _cosine_free_optics(
    z_film: NDArray[np.float64],
    *,
    period: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Oscillatory free diagonals: mean, birefringence, and dichroism vs cos/sin."""
    c = np.cos(2.0 * np.pi * z_film / period)
    s = np.sin(2.0 * np.pi * z_film / period)
    mid = 1.35e-3 + 0.85e-3 * c
    biref = 0.15e-3 + 1.05e-3 * c
    delta_o = mid - 0.5 * biref
    delta_e = mid + 0.5 * biref
    b_mid = 4.5e-4 + 2.5e-4 * s
    dich = 0.5e-4 + 3.5e-4 * s
    beta_o = np.clip(b_mid - 0.5 * dich, 5e-5, None)
    beta_e = np.clip(b_mid + 0.5 * dich, 5e-5, None)
    return delta_o, delta_e, beta_o, beta_e, c


def case_callable_field() -> None:
    """User function — CallableField free optics via cos(2 pi z / Lambda)."""
    t = 320.0
    period = 48.0
    z = depth_axis(t)
    z_film = np.clip(z, 0.0, t)
    delta_o, delta_e, beta_o, beta_e, _carrier = _cosine_free_optics(
        z_film, period=period
    )
    dich = beta_e - beta_o

    def annotate(ax: plt.Axes, _z: NDArray[np.float64], thickness: float) -> None:
        ax.axvline(period, color="C3", lw=1.0, ls="--")
        ax.annotate(
            r"$\Lambda$",
            xy=(period, float(np.max(dich))),
            xytext=(period + 0.04 * thickness, float(np.max(dich))),
            fontsize=8,
            color="C3",
        )
        ax.text(
            0.55 * thickness,
            0.88,
            r"$\cos(2\pi z/\Lambda)$ drives"
            "\n"
            r"mean $\delta$, birefringence,"
            "\n"
            r"and dichroism (phase-shifted)",
            fontsize=7.5,
            color="0.25",
            transform=ax.get_xaxis_transform(),
            va="top",
        )

    _continuous_case(
        title=r"CallableField: free optics $\propto\cos(2\pi z/\Lambda)$",
        thickness=t,
        film_optics=(delta_o, delta_e, beta_o, beta_e),
        aux_name=r"dichroism $\beta_e-\beta_o$",
        aux_film=dich,
        path_depth="callable-field.png",
        path_refl="callable-field-R.png",
        annotate_aux=annotate,
    )


def case_callable_tensor() -> None:
    """User function — CallableDepthProfile with material diagonals."""
    t = 360.0
    g0, lam = 0.5, 50.0
    z = depth_axis(t)
    z_film = np.clip(z, 0.0, t)
    cos2 = np.cos(g0 * np.exp(-z_film / lam)) ** 2

    def annotate(ax: plt.Axes, _z: NDArray[np.float64], thickness: float) -> None:
        ax.axvline(lam, color="C3", lw=1.0, ls="--")
        ax.annotate(
            r"$\lambda$",
            xy=(lam, ax.get_ylim()[1]),
            xytext=(0, -6),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="C3",
        )
        ax.text(
            0.55 * thickness,
            0.2,
            r"$\cos^2(g_0 e^{-z/\lambda})$",
            fontsize=8,
            color="0.25",
            transform=ax.get_xaxis_transform(),
        )

    _continuous_case(
        title="CallableDepthProfile (material lab diagonals)",
        thickness=t,
        film_optics=lab_from_cos2(cos2),
        aux_name=r"$\langle\cos^2\gamma\rangle$",
        aux_film=cos2,
        path_depth="callable-tensor.png",
        path_refl="callable-tensor-R.png",
        annotate_aux=annotate,
    )


def case_diffusion() -> None:
    """Functional form — Diffusion kinds compared (couple vs exponential)."""
    t = 160.0
    length = 20.0
    left, right = 1.0, 0.0
    edge, base = 1.0, 0.0
    z = depth_axis(t)
    z_film = np.clip(z, 0.0, t)
    phi_couple = diffusion_couple(
        z_film, left=left, right=right, length=length, thickness=t
    )
    phi_expo = diffusion_exponential(z_film, edge=edge, base=base, length=length)

    def annotate(ax: plt.Axes, _z: NDArray[np.float64], thickness: float) -> None:
        mid = 0.5 * thickness
        _annotate_hline(ax, left, r"left / edge", color="0.35")
        _annotate_hline(ax, right, r"right / base", color="0.45")
        _annotate_vspan(ax, mid - length, mid + length, r"couple: $\pm$ length")
        ax.axvline(length, color="0.4", lw=1.0, ls="-.", alpha=0.9)
        ax.annotate(
            r"exp: length",
            xy=(length, np.exp(-1.0)),
            xytext=(length + 0.08 * thickness, np.exp(-1.0) + 0.12),
            fontsize=7.5,
            color="0.35",
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.8},
        )

    save_profile_variant_compare(
        title="Diffusion kinds",
        path_depth="diffusion.png",
        path_refl="diffusion-R.png",
        thickness=t,
        variants={
            r'kind="couple"': phi_couple,
            r'kind="exponential"': phi_expo,
        },
        optics_from_field=_phi_linear_optics,
        field_ylabel=r"$\phi$",
        field_title=r"$\phi(z)$",
        annotate_field=annotate,
        field_ylim=(-0.05, 1.05),
        baseline_key=r'kind="couple"',
    )


def _annotate_second_order(
    ax: plt.Axes,
    thickness: float,
    *,
    bulk: float,
    top: float,
    bottom: float,
    tau_top: float,
    tau_bottom: float,
) -> None:
    _annotate_hline(ax, bulk, r"bulk", color="0.35")
    _annotate_hline(ax, top, r"top", color="C0")
    _annotate_hline(ax, bottom, r"bottom", color="C1")
    y_top = bulk + 0.55 * (top - bulk)
    ax.annotate(
        "",
        xy=(tau_top, y_top),
        xytext=(0.0, y_top),
        arrowprops={"arrowstyle": "<->", "color": "C3", "lw": 1.1},
    )
    ax.text(
        0.5 * tau_top,
        y_top,
        r"$\tau_\mathrm{top}$",
        color="C3",
        fontsize=8,
        ha="center",
        va="bottom",
    )
    y_bot = bulk + 0.55 * (bottom - bulk)
    ax.annotate(
        "",
        xy=(thickness - tau_bottom, y_bot),
        xytext=(thickness, y_bot),
        arrowprops={"arrowstyle": "<->", "color": "C3", "lw": 1.1},
    )
    ax.text(
        thickness - 0.5 * tau_bottom,
        y_bot,
        r"$\tau_\mathrm{bottom}$",
        color="C3",
        fontsize=8,
        ha="center",
        va="bottom",
    )


def case_second_order_gamma() -> None:
    """Functional form — SecondOrderTransition on gamma."""
    t = 320.0
    bulk, top, bottom = 0.40, 0.15, 0.90
    tau_top, tau_bottom = 28.0, 55.0
    z = depth_axis(t)
    z_film = np.clip(z, 0.0, t)
    gamma = second_order(
        z_film,
        thickness=t,
        bulk=bulk,
        top=top,
        bottom=bottom,
        tau_top=tau_top,
        tau_bottom=tau_bottom,
    )

    def annotate(ax: plt.Axes, _z: NDArray[np.float64], thickness: float) -> None:
        _annotate_second_order(
            ax,
            thickness,
            bulk=bulk,
            top=top,
            bottom=bottom,
            tau_top=tau_top,
            tau_bottom=tau_bottom,
        )

    save_profile_variant_compare(
        title=r"SecondOrderTransition on $\gamma$",
        path_depth="second-order-gamma.png",
        path_refl="second-order-gamma-R.png",
        thickness=t,
        variants={r"$\gamma(z)$": gamma},
        optics_from_field=lab_from_gamma,
        field_ylabel=r"$\gamma$ (rad)",
        field_title=r"$\gamma(z)$",
        annotate_field=annotate,
    )


def case_second_order_phi() -> None:
    """Functional form — SecondOrderTransition on phi (linear Mix optics)."""
    t = 220.0
    bulk, top, bottom = 0.5, 0.9, 0.1
    tau_top, tau_bottom = 20.0, 40.0
    z = depth_axis(t)
    z_film = np.clip(z, 0.0, t)
    phi = second_order(
        z_film,
        thickness=t,
        bulk=bulk,
        top=top,
        bottom=bottom,
        tau_top=tau_top,
        tau_bottom=tau_bottom,
    )

    def annotate(ax: plt.Axes, _z: NDArray[np.float64], thickness: float) -> None:
        _annotate_second_order(
            ax,
            thickness,
            bulk=bulk,
            top=top,
            bottom=bottom,
            tau_top=tau_top,
            tau_bottom=tau_bottom,
        )

    save_profile_variant_compare(
        title=r"SecondOrderTransition on $\phi$",
        path_depth="second-order-phi.png",
        path_refl="second-order-phi-R.png",
        thickness=t,
        variants={r"$\phi(z)$": phi},
        optics_from_field=_phi_linear_optics,
        field_ylabel=r"$\phi$",
        field_title=r"$\phi(z)$",
        annotate_field=annotate,
        field_ylim=(-0.05, 1.05),
    )


def case_second_order_cos2() -> None:
    """Functional form — SecondOrderTransition on cos2_gamma."""
    t = 320.0
    bulk, top, bottom = 0.70, 0.95, 0.20
    tau_top, tau_bottom = 28.0, 55.0
    z = depth_axis(t)
    z_film = np.clip(z, 0.0, t)
    cos2 = second_order(
        z_film,
        thickness=t,
        bulk=bulk,
        top=top,
        bottom=bottom,
        tau_top=tau_top,
        tau_bottom=tau_bottom,
    )

    def annotate(ax: plt.Axes, _z: NDArray[np.float64], thickness: float) -> None:
        _annotate_second_order(
            ax,
            thickness,
            bulk=bulk,
            top=top,
            bottom=bottom,
            tau_top=tau_top,
            tau_bottom=tau_bottom,
        )

    save_profile_variant_compare(
        title=r"SecondOrderTransition on $\langle\cos^2\gamma\rangle$",
        path_depth="second-order-cos2.png",
        path_refl="second-order-cos2-R.png",
        thickness=t,
        variants={r"$\langle\cos^2\gamma\rangle$": cos2},
        optics_from_field=lab_from_cos2,
        field_ylabel=r"$\langle\cos^2\gamma\rangle$",
        field_title=r"$\langle\cos^2\gamma\rangle$",
        annotate_field=annotate,
        field_ylim=(-0.05, 1.05),
    )


def _spline_target_phi(z: NDArray[np.float64], thickness: float) -> NDArray[np.float64]:
    """High-contrast target in (0, 1) so sparse knots miss mid-film swings."""
    x = np.asarray(z, dtype=np.float64) / thickness
    phi = (
        0.50
        + 0.45 * np.sin(3.0 * np.pi * x)
        + 0.30 * np.sin(7.0 * np.pi * x)
        + 0.18 * np.cos(5.0 * np.pi * x)
    )
    return np.clip(phi, 0.02, 0.98)


def case_volfrac_spline() -> None:
    """Functional form — spline order contrast, then cubic knot-count contrast."""
    t = 200.0
    # Fixed non-monotonic polygon for order (k) comparison.
    knots_k = np.array([0.0, 30.0, 70.0, 110.0, 150.0, 200.0], dtype=np.float64)
    values_k = np.array([0.20, 0.95, 0.25, 0.80, 0.15, 0.55], dtype=np.float64)
    z = depth_axis(t)
    z_film = np.clip(z, 0.0, t)

    def phi_order(k: int) -> NDArray[np.float64]:
        return np.clip(
            np.asarray(
                make_interp_spline(knots_k, values_k, k=k)(z_film), dtype=np.float64
            ),
            0.0,
            1.0,
        )

    def annotate_orders(
        ax: plt.Axes, _z: NDArray[np.float64], _thickness: float
    ) -> None:
        ax.plot(
            knots_k,
            values_k,
            "o",
            color="0.2",
            ms=6,
            zorder=5,
            label="control points",
        )
        for i, (kpt, v) in enumerate(zip(knots_k, values_k, strict=True)):
            ax.axvline(kpt, color="0.5", lw=0.6, ls=":", alpha=0.7)
            dy = 8 if i % 2 == 0 else -14
            ax.annotate(
                rf"$({kpt:.0f},\,{v:.2f})$",
                xy=(kpt, v),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color="0.25",
            )
        ax.legend(frameon=False, fontsize=7, loc="upper right")

    save_profile_variant_compare(
        title=r"Spline $\phi(z)$ orders",
        path_depth="volfrac-spline.png",
        path_refl="volfrac-spline-R.png",
        thickness=t,
        variants={
            r"$k=1$ (linear)": phi_order(1),
            r"$k=2$ (quadratic)": phi_order(2),
            r"$k=3$ (cubic)": phi_order(3),
        },
        optics_from_field=_phi_linear_optics,
        field_ylabel=r"$\phi$",
        field_title=r"$\phi(z)$",
        annotate_field=annotate_orders,
        field_ylim=(-0.05, 1.05),
        baseline_key=r"$k=3$ (cubic)",
    )

    # Cubic (k=3) only: high-frequency target; sparse knots under-resolve.
    knot_counts = (4, 6, 10)
    knot_sets: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    variants_n: dict[str, NDArray[np.float64]] = {}
    for n in knot_counts:
        kn = np.linspace(0.0, t, n, dtype=np.float64)
        va = _spline_target_phi(kn, t)
        key = rf"$N={n}$ knots"
        knot_sets[key] = (kn, va)
        variants_n[key] = np.clip(
            np.asarray(make_interp_spline(kn, va, k=3)(z_film), dtype=np.float64),
            0.0,
            1.0,
        )

    def annotate_knots(
        ax: plt.Axes, _z: NDArray[np.float64], _thickness: float
    ) -> None:
        markers = ("o", "s", "D")
        for i, (key, (kn, va)) in enumerate(knot_sets.items()):
            ax.plot(
                kn,
                va,
                linestyle="None",
                marker=markers[i % len(markers)],
                ms=5.5,
                color=f"C{i}",
                markeredgecolor="0.15",
                markeredgewidth=0.8,
                zorder=5,
                label=key,
            )
        ax.text(
            0.5 * t,
            0.08,
            r"cubic $k=3$; $N=4$ flattens the high-frequency target",
            fontsize=7.5,
            color="0.35",
            ha="center",
            transform=ax.get_xaxis_transform(),
        )
        ax.legend(frameon=False, fontsize=7, loc="upper right")

    save_profile_variant_compare(
        title=r"Cubic spline $\phi(z)$ knot count",
        path_depth="volfrac-spline-knots.png",
        path_refl="volfrac-spline-knots-R.png",
        thickness=t,
        variants=variants_n,
        optics_from_field=_phi_linear_optics,
        field_ylabel=r"$\phi$",
        field_title=r"$\phi(z)$",
        annotate_field=annotate_knots,
        field_ylim=(-0.05, 1.05),
        baseline_key=r"$N=10$ knots",
    )


def case_volfrac_polynomial() -> None:
    """Functional form - polynomial degrees 1-3 compared on φ."""
    t = 180.0
    z = depth_axis(t)
    z_film = np.clip(z, 0.0, t)
    # Endpoints ~0.6 -> 0.4; higher degrees add a mid-film bump then dip.
    coeffs1 = [0.6, -0.2 / t]
    coeffs2 = [0.6, 1.4 / t, -1.6 / t**2]
    coeffs3 = [0.6, 5.0 / t, -15.0 / t**2, 9.8 / t**3]

    def eval_poly(coeffs: list[float]) -> NDArray[np.float64]:
        return np.clip(
            np.asarray(Polynomial(coeffs=coeffs).evaluate(z_film), dtype=np.float64),
            0.0,
            1.0,
        )

    def annotate(ax: plt.Axes, _z: NDArray[np.float64], thickness: float) -> None:
        _annotate_hline(ax, 0.6, r"$\phi(0)\approx 0.6$", color="0.35")
        _annotate_hline(ax, 0.4, r"$\phi(T)\approx 0.4$", color="0.45")
        ax.text(
            0.55 * thickness,
            0.92,
            r"$\phi=\sum_n c_n z^n$"
            "\n"
            r"deg 1 / 2 / 3"
            "\n"
            rf"$T={thickness:.0f}\,\mathrm{{\AA}}$",
            fontsize=8,
            color="0.25",
            transform=ax.get_xaxis_transform(),
            va="top",
        )

    save_profile_variant_compare(
        title=r"Polynomial $\phi(z)$ degrees",
        path_depth="volfrac-polynomial.png",
        path_refl="volfrac-polynomial-R.png",
        thickness=t,
        variants={
            "degree 1": eval_poly(coeffs1),
            "degree 2": eval_poly(coeffs2),
            "degree 3": eval_poly(coeffs3),
        },
        optics_from_field=_phi_linear_optics,
        field_ylabel=r"$\phi$",
        field_title=r"$\phi(z)$",
        annotate_field=annotate,
        field_ylim=(-0.05, 1.05),
        baseline_key="degree 2",
    )


def case_free_optical_polynomial() -> None:
    """Functional form — extreme free optical Polynomial on delta / beta."""
    t = 300.0
    z = depth_axis(t)
    z_film = np.clip(z, 0.0, t)
    # Strong cubic / quartic swings (not a near-constant film).
    c_o = [0.55e-3, 1.2e-5, -7.5e-8, 1.1e-10]
    c_e = [2.40e-3, -1.5e-5, 9.0e-8, -1.3e-10]
    c_bo = [2.0e-4, 3.0e-6, -1.5e-8]
    c_be = [7.5e-4, -4.0e-6, 2.0e-8]
    delta_o = np.asarray(Polynomial(coeffs=c_o).evaluate(z_film), dtype=np.float64)
    delta_e = np.asarray(Polynomial(coeffs=c_e).evaluate(z_film), dtype=np.float64)
    beta_o = np.clip(
        np.asarray(Polynomial(coeffs=c_bo).evaluate(z_film), dtype=np.float64),
        5e-5,
        None,
    )
    beta_e = np.clip(
        np.asarray(Polynomial(coeffs=c_be).evaluate(z_film), dtype=np.float64),
        5e-5,
        None,
    )
    dich = beta_e - beta_o

    def annotate(ax: plt.Axes, _z: NDArray[np.float64], th: float) -> None:
        ax.text(
            0.5 * th,
            0.88,
            r"extreme Poly on $\delta_{o,e}$ and $\beta_{o,e}$"
            "\n"
            r"(aux = dichroism)",
            fontsize=7.5,
            color="0.25",
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="center",
        )

    _continuous_case(
        title=r"Free optical Polynomial (extreme $\delta$, $\beta$)",
        thickness=t,
        film_optics=(delta_o, delta_e, beta_o, beta_e),
        aux_name=r"dichroism $\beta_e-\beta_o$",
        aux_film=dich,
        path_depth="free-optical-polynomial.png",
        path_refl="free-optical-polynomial-R.png",
        annotate_aux=annotate,
    )


def case_free_optical_spline() -> None:
    """Functional form — extreme free optical Spline on delta / beta."""
    t = 280.0
    knots = np.array([0.0, 35.0, 70.0, 110.0, 160.0, 210.0, 280.0], dtype=np.float64)
    values_o = np.array(
        [0.40e-3, 2.20e-3, 0.35e-3, 2.00e-3, 0.50e-3, 1.80e-3, 0.70e-3],
        dtype=np.float64,
    )
    values_e = np.array(
        [2.30e-3, 0.45e-3, 2.40e-3, 0.55e-3, 2.10e-3, 0.60e-3, 1.90e-3],
        dtype=np.float64,
    )
    values_bo = np.array(
        [1.0e-4, 6.0e-4, 1.5e-4, 7.0e-4, 1.2e-4, 5.5e-4, 2.0e-4], dtype=np.float64
    )
    values_be = np.array(
        [7.0e-4, 1.5e-4, 8.0e-4, 1.0e-4, 7.5e-4, 1.8e-4, 6.5e-4], dtype=np.float64
    )
    z = depth_axis(t)
    z_film = np.clip(z, 0.0, t)
    delta_o = np.asarray(
        make_interp_spline(knots, values_o, k=3)(z_film), dtype=np.float64
    )
    delta_e = np.asarray(
        make_interp_spline(knots, values_e, k=3)(z_film), dtype=np.float64
    )
    beta_o = np.clip(
        np.asarray(make_interp_spline(knots, values_bo, k=3)(z_film), dtype=np.float64),
        5e-5,
        None,
    )
    beta_e = np.clip(
        np.asarray(make_interp_spline(knots, values_be, k=3)(z_film), dtype=np.float64),
        5e-5,
        None,
    )
    dich = beta_e - beta_o

    def annotate(ax: plt.Axes, _z: NDArray[np.float64], th: float) -> None:
        kn_d = make_interp_spline(knots, values_be - values_bo, k=3)(knots)
        ax.plot(knots, kn_d, "D", color="C3", ms=5.0, zorder=5, label="dich. knots")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.text(
            0.45 * th,
            0.12,
            r"extreme cubic Spline on $\delta$ and $\beta$",
            fontsize=7.5,
            color="0.25",
            transform=ax.get_xaxis_transform(),
        )

    _continuous_case(
        title=r"Free optical Spline (extreme $\delta$, $\beta$)",
        thickness=t,
        film_optics=(delta_o, delta_e, beta_o, beta_e),
        aux_name=r"dichroism $\beta_e-\beta_o$",
        aux_film=dich,
        path_depth="free-optical-spline.png",
        path_refl="free-optical-spline-R.png",
        annotate_aux=annotate,
    )


def case_free_optical_osc_stack() -> None:
    """Stack extreme free-optical Poly | Spline | cos CallableField."""
    lengths = np.array([220.0, 240.0, 280.0], dtype=np.float64)
    labels = ["FreePoly", "FreeSpline", "CosCallable"]
    edges = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(edges[-1])
    z = np.linspace(-PAD, total + PAD, 5000)

    def local(z_abs: NDArray[np.float64], i: int) -> NDArray[np.float64]:
        return np.clip(z_abs - edges[i], 0.0, lengths[i])

    n_o = np.full_like(z, N_VAC)
    n_e = np.full_like(z, N_VAC)
    b_o = np.full_like(z, B_VAC)
    b_e = np.full_like(z, B_VAC)
    for i, lab in enumerate(labels):
        mask = (z >= edges[i]) & (z < edges[i + 1])
        zl = local(z[mask], i)
        t = lengths[i]
        if lab == "FreePoly":
            o = np.asarray(
                Polynomial(coeffs=[0.55e-3, 1.2e-5, -7.5e-8, 1.1e-10]).evaluate(zl),
                dtype=np.float64,
            )
            e = np.asarray(
                Polynomial(coeffs=[2.40e-3, -1.5e-5, 9.0e-8, -1.3e-10]).evaluate(zl),
                dtype=np.float64,
            )
            bo = np.clip(
                np.asarray(
                    Polynomial(coeffs=[2.0e-4, 3.0e-6, -1.5e-8]).evaluate(zl),
                    dtype=np.float64,
                ),
                5e-5,
                None,
            )
            be = np.clip(
                np.asarray(
                    Polynomial(coeffs=[7.5e-4, -4.0e-6, 2.0e-8]).evaluate(zl),
                    dtype=np.float64,
                ),
                5e-5,
                None,
            )
        elif lab == "FreeSpline":
            kn = np.linspace(0.0, t, 7)
            vals_o = [
                0.40e-3,
                2.20e-3,
                0.35e-3,
                2.00e-3,
                0.50e-3,
                1.80e-3,
                0.70e-3,
            ]
            vals_e = [
                2.30e-3,
                0.45e-3,
                2.40e-3,
                0.55e-3,
                2.10e-3,
                0.60e-3,
                1.90e-3,
            ]
            o = np.asarray(
                Spline(knots=kn.tolist(), values=vals_o).evaluate(zl),
                dtype=np.float64,
            )
            e = np.asarray(
                Spline(knots=kn.tolist(), values=vals_e).evaluate(zl),
                dtype=np.float64,
            )
            bo = np.clip(
                np.asarray(
                    Spline(
                        knots=kn.tolist(),
                        values=[1e-4, 6e-4, 1.5e-4, 7e-4, 1.2e-4, 5.5e-4, 2e-4],
                    ).evaluate(zl),
                    dtype=np.float64,
                ),
                5e-5,
                None,
            )
            be = np.clip(
                np.asarray(
                    Spline(
                        knots=kn.tolist(),
                        values=[7e-4, 1.5e-4, 8e-4, 1e-4, 7.5e-4, 1.8e-4, 6.5e-4],
                    ).evaluate(zl),
                    dtype=np.float64,
                ),
                5e-5,
                None,
            )
        else:
            o, e, bo, be, _c = _cosine_free_optics(zl, period=48.0)
        n_o[mask] = o
        n_e[mask] = e
        b_o[mask] = bo
        b_e[mask] = be
    n_o[z >= total] = N_SI
    n_e[z >= total] = N_SI
    b_o[z >= total] = B_SI
    b_e[z >= total] = B_SI

    sigma_iface = 3.0
    n_o_b = _gaussian_interface_broaden(z, n_o, sigma_angstrom=sigma_iface)
    n_e_b = _gaussian_interface_broaden(z, n_e, sigma_angstrom=sigma_iface)
    b_o_b = _gaussian_interface_broaden(z, b_o, sigma_angstrom=sigma_iface)
    b_e_b = _gaussian_interface_broaden(z, b_e, sigma_angstrom=sigma_iface)

    fig, axes = plt.subplots(
        3, 2, figsize=(10.5, 7.4), sharex=True, constrained_layout=True
    )
    fig.suptitle(
        r"Free-optical oscillatory stack — Poly | Spline | $\cos$ Callable",
        fontsize=12,
        fontweight="bold",
    )
    col_tags = (
        "sharp",
        rf"interface-softened ($\sigma={sigma_iface:.0f}\,\mathrm{{\AA}}$)",
    )
    for col, (tag, oo, ee, boo, bee) in enumerate(
        zip(
            col_tags,
            (n_o, n_o_b),
            (n_e, n_e_b),
            (b_o, b_o_b),
            (b_e, b_e_b),
            strict=True,
        )
    ):
        ax_n = axes[0, col]
        ax_bi = axes[1, col]
        ax_di = axes[2, col]
        ax_n.plot(z, oo * 1e3, color=MIX_NO_COLOR, lw=1.5, label=r"$n_o$ / $\delta_o$")
        ax_n.plot(z, ee * 1e3, color=MIX_NE_COLOR, lw=1.5, label=r"$n_e$ / $\delta_e$")
        ax_bi.plot(z, (ee - oo) * 1e3, color="C3", lw=1.4)
        ax_di.plot(z, (bee - boo) * 1e3, color="C2", lw=1.4)
        for edge in edges:
            for ax in (ax_n, ax_bi, ax_di):
                ax.axvline(edge, color="0.75", lw=0.7, ls=":")
        style_ax(
            ax_n,
            title=f"({chr(97 + col)}) optical constants — {tag}",
            ylabel=r"$\delta$ ($\times 10^{-3}$)",
        )
        if col == 0:
            ax_n.legend(frameon=False, fontsize=8)
        style_ax(
            ax_bi,
            title=f"({chr(99 + col)}) birefringence $n_e-n_o$ — {tag}",
            ylabel=r"$\Delta$ ($\times 10^{-3}$)",
        )
        style_ax(
            ax_di,
            title=f"({chr(101 + col)}) dichroism $\\beta_e-\\beta_o$ — {tag}",
            ylabel=r"$\Delta\beta$ ($\times 10^{-3}$)",
        )
        if col == 0:
            ymin, ymax = ax_di.get_ylim()
            for i, lab in enumerate(labels):
                ax_di.text(
                    0.5 * (edges[i] + edges[i + 1]),
                    ymin + 0.1 * (ymax - ymin),
                    lab,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="0.35",
                )
    fig.savefig(OUT / "free-optical-osc-stack.png", dpi=200)
    plt.close(fig)

    n_sink = 280
    z_m = np.linspace(0.0, total, n_sink, endpoint=False) + 0.5 * (total / n_sink)
    layers, tensor = microslab_kernel(
        thickness=total,
        n_o=np.interp(z_m, z, n_o),
        n_e=np.interp(z_m, z, n_e),
        b_o=np.interp(z_m, z, b_o),
        b_e=np.interp(z_m, z, b_e),
        rough_top=2.0,
    )
    r_ss, r_pp = reflectivity(layers, tensor)
    save_refl(
        OUT / "free-optical-osc-stack-R.png",
        title=r"Free-optical oscillatory stack (Poly | Spline | $\cos$)",
        r_ss=r_ss,
        r_pp=r_pp,
    )


def _gaussian_interface_broaden(
    z: NDArray[np.float64],
    profile: NDArray[np.float64],
    *,
    sigma_angstrom: float,
) -> NDArray[np.float64]:
    """Mild depth-axis Gaussian blur; preserves intra-layer structure.

    Unlike ``erf_steps`` on mid-slab constants, this only softens jumps near
    interfaces (and slightly within layers) without replacing each slab by a
    single midpoint value.
    """
    dz = float(np.median(np.diff(z)))
    if dz <= 0.0:
        msg = "z must be strictly increasing"
        raise ValueError(msg)
    sigma_bins = max(sigma_angstrom / dz, 0.5)
    return np.asarray(
        gaussian_filter1d(profile, sigma=sigma_bins, mode="nearest"),
        dtype=np.float64,
    )


HOMO_NO, HOMO_NE = 1.2e-3, 1.8e-3
HOMO_BO, HOMO_BE = 4e-4, 6e-4

OpticsLocal = Callable[
    [NDArray[np.float64]],
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
]
AuxLocal = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def save_segment_stack(
    *,
    title: str,
    path_depth: str,
    path_refl: str,
    lengths: Sequence[float],
    labels: Sequence[str],
    optics_locals: Sequence[OpticsLocal],
    aux_name: str | None = None,
    aux_locals: Sequence[AuxLocal | None] | None = None,
    sigma_iface: float = 3.0,
) -> None:
    """Multi-layer depth + reflectivity figure under ``|``.

    Depth layout: sharp | interface-softened columns; rows optical constants,
    birefringence, and optional aux (phi / gamma / dichroism).
    """
    if not (len(lengths) == len(labels) == len(optics_locals)):
        msg = "lengths, labels, and optics_locals must have equal length"
        raise ValueError(msg)
    if aux_locals is not None and len(aux_locals) != len(lengths):
        msg = "aux_locals must match lengths when provided"
        raise ValueError(msg)

    length_arr = np.asarray(lengths, dtype=np.float64)
    edges = np.concatenate([[0.0], np.cumsum(length_arr)])
    total = float(edges[-1])
    z = np.linspace(-PAD, total + PAD, 5000)

    n_o = np.full_like(z, N_VAC)
    n_e = np.full_like(z, N_VAC)
    b_o = np.full_like(z, B_VAC)
    b_e = np.full_like(z, B_VAC)
    want_aux = aux_name is not None and aux_locals is not None
    aux = np.zeros_like(z) if want_aux else None

    for i, optics_fn in enumerate(optics_locals):
        mask = (z >= edges[i]) & (z < edges[i + 1])
        zl = np.clip(z[mask] - edges[i], 0.0, length_arr[i])
        o, e, bo, be = optics_fn(zl)
        n_o[mask] = o
        n_e[mask] = e
        b_o[mask] = bo
        b_e[mask] = be
        if aux is not None and aux_locals is not None:
            aux_fn = aux_locals[i]
            if aux_fn is not None:
                aux[mask] = aux_fn(zl)

    n_o[z >= total] = N_SI
    n_e[z >= total] = N_SI
    b_o[z >= total] = B_SI
    b_e[z >= total] = B_SI

    n_o_b = _gaussian_interface_broaden(z, n_o, sigma_angstrom=sigma_iface)
    n_e_b = _gaussian_interface_broaden(z, n_e, sigma_angstrom=sigma_iface)
    b_o_b = _gaussian_interface_broaden(z, b_o, sigma_angstrom=sigma_iface)
    b_e_b = _gaussian_interface_broaden(z, b_e, sigma_angstrom=sigma_iface)
    aux_b = (
        _gaussian_interface_broaden(z, aux, sigma_angstrom=sigma_iface)
        if aux is not None
        else None
    )

    nrows = 3 if aux is not None else 2
    fig, axes = plt.subplots(
        nrows, 2, figsize=(10.5, 2.45 * nrows), sharex=True, constrained_layout=True
    )
    if nrows == 2:
        axes = np.asarray(axes, dtype=object)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    col_tags = (
        "sharp",
        rf"interface-softened ($\sigma={sigma_iface:.0f}\,\mathrm{{\AA}}$)",
    )
    for col, (tag, oo, ee, _boo, _bee, aa) in enumerate(
        zip(
            col_tags,
            (n_o, n_o_b),
            (n_e, n_e_b),
            (b_o, b_o_b),
            (b_e, b_e_b),
            (aux, aux_b),
            strict=True,
        )
    ):
        ax_n = axes[0, col]
        ax_bi = axes[1, col]
        ax_n.plot(z, oo * 1e3, color=MIX_NO_COLOR, lw=1.5, label=r"$n_o$ / $\delta_o$")
        ax_n.plot(z, ee * 1e3, color=MIX_NE_COLOR, lw=1.5, label=r"$n_e$ / $\delta_e$")
        ax_bi.plot(z, (ee - oo) * 1e3, color="C3", lw=1.4)
        for edge in edges:
            ax_n.axvline(edge, color="0.75", lw=0.7, ls=":")
            ax_bi.axvline(edge, color="0.75", lw=0.7, ls=":")
        style_ax(
            ax_n,
            title=f"({chr(97 + col)}) optical constants — {tag}",
            ylabel=r"$\delta$ ($\times 10^{-3}$)",
        )
        if col == 0:
            ax_n.legend(frameon=False, fontsize=8)
        style_ax(
            ax_bi,
            title=f"({chr(99 + col)}) birefringence $n_e-n_o$ — {tag}",
            ylabel=r"$\Delta$ ($\times 10^{-3}$)",
        )
        if aa is not None and aux_name is not None:
            ax_a = axes[2, col]
            ax_a.plot(z, aa, color=MIX_PHI_COLOR, lw=1.5)
            for edge in edges:
                ax_a.axvline(edge, color="0.75", lw=0.7, ls=":")
            style_ax(
                ax_a,
                title=f"({chr(101 + col)}) {aux_name} — {tag}",
                ylabel=aux_name,
            )
            if col == 0:
                ymin, ymax = ax_a.get_ylim()
                for i, lab in enumerate(labels):
                    ax_a.text(
                        0.5 * (edges[i] + edges[i + 1]),
                        ymin + 0.1 * (ymax - ymin),
                        lab,
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color="0.35",
                    )
        elif col == 0:
            ymin, ymax = ax_bi.get_ylim()
            for i, lab in enumerate(labels):
                ax_bi.text(
                    0.5 * (edges[i] + edges[i + 1]),
                    ymin + 0.1 * (ymax - ymin),
                    lab,
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="0.35",
                )
    fig.savefig(OUT / path_depth, dpi=200)
    plt.close(fig)

    n_micro = max(160, int(total / 2.0))
    z_m = np.linspace(0.0, total, n_micro, endpoint=False) + 0.5 * (total / n_micro)
    layers, tensor = microslab_kernel(
        thickness=total,
        n_o=np.interp(z_m, z, n_o),
        n_e=np.interp(z_m, z, n_e),
        b_o=np.interp(z_m, z, b_o),
        b_e=np.interp(z_m, z, b_e),
        rough_top=2.0,
    )
    r_ss, r_pp = reflectivity(layers, tensor)
    save_refl(OUT / path_refl, title=title, r_ss=r_ss, r_pp=r_pp)


def save_short_stack(
    *,
    title: str,
    path_depth: str,
    path_refl: str,
    featured_length: float,
    homo_length: float,
    featured_optics_local: OpticsLocal,
    featured_label: str,
    homo_label: str = "homo",
    aux_name: str | None = None,
    aux_local: AuxLocal | None = None,
    aux_homo: float = 0.0,
    sigma_iface: float = 3.0,
) -> None:
    """Featured DepthProfile plus one homogeneous film under ``|``."""

    def homo_optics(
        zl: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        n = zl.shape[0]
        return (
            np.full(n, HOMO_NO),
            np.full(n, HOMO_NE),
            np.full(n, HOMO_BO),
            np.full(n, HOMO_BE),
        )

    def homo_aux(zl: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(zl.shape[0], aux_homo, dtype=np.float64)

    aux_locals: Sequence[AuxLocal | None] | None = None
    if aux_name is not None:
        aux_locals = [aux_local, homo_aux if aux_local is not None else None]

    save_segment_stack(
        title=title,
        path_depth=path_depth,
        path_refl=path_refl,
        lengths=[featured_length, homo_length],
        labels=[featured_label, homo_label],
        optics_locals=[featured_optics_local, homo_optics],
        aux_name=aux_name,
        aux_locals=aux_locals,
        sigma_iface=sigma_iface,
    )


def case_diffusion_short_stack() -> None:
    """Oxidation exponential, then couple, then exponential into bulk."""
    t_ox, t_couple, t_expo = 80.0, 160.0, 120.0

    def phi_oxide(zl: NDArray[np.float64]) -> NDArray[np.float64]:
        return diffusion_exponential(zl, edge=1.0, base=0.05, length=18.0)

    def phi_couple(zl: NDArray[np.float64]) -> NDArray[np.float64]:
        return diffusion_couple(
            zl, left=0.05, right=0.95, length=22.0, thickness=t_couple
        )

    def phi_expo(zl: NDArray[np.float64]) -> NDArray[np.float64]:
        return diffusion_exponential(zl, edge=0.95, base=0.10, length=35.0)

    def optics_from_phi(
        phi_fn: AuxLocal,
    ) -> OpticsLocal:
        def optics(
            zl: NDArray[np.float64],
        ) -> tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]:
            return _phi_linear_optics(phi_fn(zl))

        return optics

    save_segment_stack(
        title=r"Diffusion stack — oxide exp | couple | exp into bulk",
        path_depth="diffusion-stack.png",
        path_refl="diffusion-stack-R.png",
        lengths=[t_ox, t_couple, t_expo],
        labels=["oxide exp", "couple", "exp"],
        optics_locals=[
            optics_from_phi(phi_oxide),
            optics_from_phi(phi_couple),
            optics_from_phi(phi_expo),
        ],
        aux_name=r"$\phi$",
        aux_locals=[phi_oxide, phi_couple, phi_expo],
    )


def case_second_order_short_stack() -> None:
    """SecondOrder gamma into homogeneous pinned at bottom gamma."""
    t_feat, t_homo = 220.0, 120.0
    bulk, top, bottom = 0.40, 0.15, 0.90
    tau_top, tau_bottom = 28.0, 55.0
    pin = bottom

    def gamma_feat(zl: NDArray[np.float64]) -> NDArray[np.float64]:
        return second_order(
            zl,
            thickness=t_feat,
            bulk=bulk,
            top=top,
            bottom=bottom,
            tau_top=tau_top,
            tau_bottom=tau_bottom,
        )

    def optics_feat(
        zl: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        return lab_from_gamma(gamma_feat(zl))

    def optics_homo(
        zl: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        return lab_from_gamma(np.full(zl.shape[0], pin, dtype=np.float64))

    def aux_homo(zl: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(zl.shape[0], pin, dtype=np.float64)

    save_segment_stack(
        title=(
            r"Short stack — SecondOrder $\gamma$ | "
            rf"homo at $\gamma_\mathrm{{bottom}}={pin:.2f}$"
        ),
        path_depth="second-order-gamma-stack.png",
        path_refl="second-order-gamma-stack-R.png",
        lengths=[t_feat, t_homo],
        labels=[r"SO $\gamma$", rf"homo ($\gamma={pin:.2f}$)"],
        optics_locals=[optics_feat, optics_homo],
        aux_name=r"$\gamma$",
        aux_locals=[gamma_feat, aux_homo],
    )


def _orientation_only_free_optics(
    z_film: NDArray[np.float64],
    *,
    period: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Fixed mean refractive index; only birefringence / dichroism oscillate."""
    c = np.cos(2.0 * np.pi * z_film / period)
    s = np.sin(2.0 * np.pi * z_film / period)
    mid = 1.35e-3
    biref = 0.15e-3 + 1.05e-3 * c
    delta_o = mid - 0.5 * biref
    delta_e = mid + 0.5 * biref
    b_mid = 4.5e-4
    dich = 0.5e-4 + 3.5e-4 * s
    beta_o = np.clip(b_mid - 0.5 * dich, 5e-5, None)
    beta_e = np.clip(b_mid + 0.5 * dich, 5e-5, None)
    return delta_o, delta_e, beta_o, beta_e


def case_free_optical_polynomial_short_stack() -> None:
    """Quadratic | cubic | linear Mix phi with C0 matching at interfaces."""
    t_quad, t_cub, t_lin = 140.0, 160.0, 120.0
    # Quadratic: phi(0)=0.92, phi(T)=0.35
    c0_q, c1_q, c2_q = 0.92, -0.0020, -1.5e-5
    phi_q_end = float(
        Polynomial(coeffs=[c0_q, c1_q, c2_q]).evaluate(np.array([t_quad]))[0]
    )
    # Cubic: match value at local 0; end at 0.78 with a mid bump
    # phi = a + b z + c z^2 + d z^3; a = phi_q_end
    # phi(T)=0.78, phi'(0)=c1_q (C1 at first interface), phi(T/2)=0.25
    a = phi_q_end
    b = c1_q + 2.0 * c2_q * t_quad
    t2 = t_cub
    phi_mid = 0.22
    phi_end = 0.78
    # Solve for c, d from mid and end with fixed a,b:
    # a + b(T/2) + c(T/2)^2 + d(T/2)^3 = phi_mid
    # a + b T + c T^2 + d T^3 = phi_end
    half = 0.5 * t2
    m11, m12 = half**2, half**3
    m21, m22 = t2**2, t2**3
    rhs1 = phi_mid - a - b * half
    rhs2 = phi_end - a - b * t2
    det = m11 * m22 - m12 * m21
    c = (rhs1 * m22 - rhs2 * m12) / det
    d = (m11 * rhs2 - m21 * rhs1) / det
    # Linear: match value at local 0 (= phi_end); end at 0.15
    a_lin = phi_end
    b_lin = (0.15 - a_lin) / t_lin

    def phi_quad(zl: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(
            np.asarray(
                Polynomial(coeffs=[c0_q, c1_q, c2_q]).evaluate(zl), dtype=np.float64
            ),
            0.0,
            1.0,
        )

    def phi_cub(zl: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(
            np.asarray(Polynomial(coeffs=[a, b, c, d]).evaluate(zl), dtype=np.float64),
            0.0,
            1.0,
        )

    def phi_lin(zl: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(
            np.asarray(
                Polynomial(coeffs=[a_lin, b_lin]).evaluate(zl), dtype=np.float64
            ),
            0.0,
            1.0,
        )

    def optics_from_phi(phi_fn: AuxLocal) -> OpticsLocal:
        def optics(
            zl: NDArray[np.float64],
        ) -> tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]:
            return _phi_linear_optics(phi_fn(zl))

        return optics

    save_segment_stack(
        title=(
            r"Polynomial stack — quadratic | cubic | linear"
            r" ($\phi$, C0; C1 at first join)"
        ),
        path_depth="free-optical-polynomial-stack.png",
        path_refl="free-optical-polynomial-stack-R.png",
        lengths=[t_quad, t_cub, t_lin],
        labels=["quadratic", "cubic", "linear"],
        optics_locals=[
            optics_from_phi(phi_quad),
            optics_from_phi(phi_cub),
            optics_from_phi(phi_lin),
        ],
        aux_name=r"$\phi$",
        aux_locals=[phi_quad, phi_cub, phi_lin],
    )


def case_callable_field_short_stack() -> None:
    """Magnitude-oscillating free optics over orientation-only oscillator."""
    t_mag, t_ori = 280.0, 280.0
    period = 48.0

    def optics_mag(
        zl: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        o, e, bo, be, _c = _cosine_free_optics(zl, period=period)
        return o, e, bo, be

    def optics_ori(
        zl: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        return _orientation_only_free_optics(zl, period=period)

    def aux_mag(zl: NDArray[np.float64]) -> NDArray[np.float64]:
        o, e, _bo, _be = optics_mag(zl)
        return 0.5 * (o + e) * 1e3

    def aux_ori(zl: NDArray[np.float64]) -> NDArray[np.float64]:
        o, e, _bo, _be = optics_ori(zl)
        return 0.5 * (o + e) * 1e3

    save_segment_stack(
        title=r"Callable stack — magnitude $\cos$ | orientation-only $\cos$",
        path_depth="callable-field-stack.png",
        path_refl="callable-field-stack-R.png",
        lengths=[t_mag, t_ori],
        labels=[r"mag $\propto\cos$", r"ori only"],
        optics_locals=[optics_mag, optics_ori],
        aux_name=r"mean $\delta$ ($\times 10^{-3}$)",
        aux_locals=[aux_mag, aux_ori],
    )


def case_kitchen_sink() -> None:
    """Composed stack — kitchen-sink multi-form (incl. free optical forms)."""
    lengths = np.array(
        [100.0, 140.0, 160.0, 100.0, 180.0, 160.0, 200.0, 160.0, 140.0],
        dtype=np.float64,
    )
    labels = [
        "Diff",
        "SO-ori",
        "SO-phi",
        "Mix",
        "Spline",
        "Poly",
        "FreeSpline",
        "FreePoly",
        "User",
    ]
    edges = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(edges[-1])
    z = np.linspace(-PAD, total + PAD, 5000)

    def local(z_abs: NDArray[np.float64], i: int) -> NDArray[np.float64]:
        return np.clip(z_abs - edges[i], 0.0, lengths[i])

    n_o = np.full_like(z, N_VAC)
    n_e = np.full_like(z, N_VAC)
    b_o = np.full_like(z, B_VAC)
    b_e = np.full_like(z, B_VAC)
    for i, lab in enumerate(labels):
        mask = (z >= edges[i]) & (z < edges[i + 1])
        zl = local(z[mask], i)
        t = lengths[i]
        if lab == "Diff":
            phi = diffusion_couple(zl, left=1.0, right=0.0, length=22.0, thickness=t)
            o, e, bo, be = mix_optics(phi)
        elif lab == "SO-ori":
            g = second_order(
                zl,
                thickness=t,
                bulk=0.4,
                top=0.15,
                bottom=0.9,
                tau_top=24.0,
                tau_bottom=45.0,
            )
            o, e, bo, be = lab_from_gamma(g)
        elif lab == "SO-phi":
            phi = second_order(
                zl,
                thickness=t,
                bulk=0.5,
                top=0.9,
                bottom=0.1,
                tau_top=20.0,
                tau_bottom=40.0,
            )
            o, e, bo, be = mix_optics(phi)
        elif lab == "Mix":
            o = np.full_like(zl, 0.3 * N_A_O + 0.7 * N_B_O)
            e = np.full_like(zl, 0.3 * N_A_E + 0.7 * N_B_E)
            bo = np.full_like(zl, 0.3 * B_A_O + 0.7 * B_B_O)
            be = np.full_like(zl, 0.3 * B_A_E + 0.7 * B_B_E)
        elif lab == "Spline":
            kn = [0.0, 0.15 * t, 0.35 * t, 0.55 * t, 0.75 * t, t]
            va = [0.20, 0.95, 0.25, 0.80, 0.15, 0.55]
            phi = np.clip(
                np.asarray(Spline(knots=kn, values=va).evaluate(zl), dtype=np.float64),
                0.0,
                1.0,
            )
            o, e, bo, be = mix_optics(phi)
        elif lab == "Poly":
            phi = np.clip(
                np.asarray(
                    Polynomial(coeffs=[0.6, 1.4 / t, -1.6 / t**2]).evaluate(zl),
                    dtype=np.float64,
                ),
                0.0,
                1.0,
            )
            o, e, bo, be = mix_optics(phi)
        elif lab == "FreeSpline":
            kn = [0.0, 0.2 * t, 0.4 * t, 0.6 * t, 0.8 * t, t]
            o = np.asarray(
                Spline(
                    knots=kn,
                    values=[1.05e-3, 1.70e-3, 0.85e-3, 1.55e-3, 0.95e-3, 1.25e-3],
                ).evaluate(zl),
                dtype=np.float64,
            )
            e = np.asarray(
                Spline(
                    knots=kn,
                    values=[1.90e-3, 1.25e-3, 2.10e-3, 1.05e-3, 1.85e-3, 1.40e-3],
                ).evaluate(zl),
                dtype=np.float64,
            )
            bo = np.full_like(zl, 4e-4)
            be = np.full_like(zl, 6e-4)
        elif lab == "FreePoly":
            o = np.asarray(
                Polynomial(coeffs=[1.15e-3, 2.5e-6, -1.2e-8]).evaluate(zl),
                dtype=np.float64,
            )
            e = np.asarray(
                Polynomial(coeffs=[1.75e-3, -4e-6, 1.5e-8]).evaluate(zl),
                dtype=np.float64,
            )
            bo = np.full_like(zl, 4e-4)
            be = np.full_like(zl, 6e-4)
        else:
            g = 0.55 * np.exp(-zl / 45.0)
            o, e, bo, be = lab_from_gamma(g)
        n_o[mask] = o
        n_e[mask] = e
        b_o[mask] = bo
        b_e[mask] = be
    n_o[z >= total] = N_SI
    n_e[z >= total] = N_SI
    b_o[z >= total] = B_SI
    b_e[z >= total] = B_SI

    # Soften interfaces only; keep intra-layer spline / poly structure.
    sigma_iface = 3.0
    n_o_b = _gaussian_interface_broaden(z, n_o, sigma_angstrom=sigma_iface)
    n_e_b = _gaussian_interface_broaden(z, n_e, sigma_angstrom=sigma_iface)

    fig, axes = plt.subplots(
        2, 2, figsize=(11.0, 6.2), sharex=True, constrained_layout=True
    )
    fig.suptitle(
        "Kitchen-sink stack — sharp vs interface-softened optical depth",
        fontsize=12,
        fontweight="bold",
    )
    col_tags = (
        "sharp",
        rf"interface-softened ($\sigma={sigma_iface:.0f}\,\mathrm{{\AA}}$)",
    )
    for col, (tag, oo, ee) in enumerate(
        zip(col_tags, (n_o, n_o_b), (n_e, n_e_b), strict=True)
    ):
        ax = axes[0, col]
        ax.plot(z, oo * 1e3, label=r"$n_o$", color="C0", lw=1.5)
        ax.plot(z, ee * 1e3, label=r"$n_e$", color="C1", lw=1.5)
        for edge in edges:
            ax.axvline(edge, color="0.75", lw=0.7, ls=":")
        style_ax(
            ax,
            title=f"({chr(97 + col)}) optical constants — {tag}",
            ylabel=r"$\delta$ ($\times 10^{-3}$)",
        )
        ax.legend(frameon=False, fontsize=8)
        ax2 = axes[1, col]
        ax2.plot(z, (ee - oo) * 1e3, color="C3", lw=1.4)
        for edge in edges:
            ax2.axvline(edge, color="0.75", lw=0.7, ls=":")
        if col == 0:
            ymin, ymax = ax2.get_ylim()
            for i, lab in enumerate(labels):
                ax2.text(
                    0.5 * (edges[i] + edges[i + 1]),
                    ymin + 0.08 * (ymax - ymin),
                    lab,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                    color="0.35",
                )
        style_ax(
            ax2,
            title=f"({chr(99 + col)}) birefringence — {tag}",
            ylabel=r"$\Delta$ ($\times 10^{-3}$)",
        )
    fig.savefig(OUT / "kitchen-sink.png", dpi=200)
    plt.close(fig)

    n_sink = 240
    z_m = np.linspace(0.0, total, n_sink, endpoint=False) + 0.5 * (total / n_sink)
    layers, tensor = microslab_kernel(
        thickness=total,
        n_o=np.interp(z_m, z, n_o),
        n_e=np.interp(z_m, z, n_e),
        b_o=np.interp(z_m, z, b_o),
        b_e=np.interp(z_m, z, b_e),
        rough_top=2.0,
    )
    r_ss, r_pp = reflectivity(layers, tensor)
    save_refl(
        OUT / "kitchen-sink-R.png",
        title="Kitchen-sink multi-form stack",
        r_ss=r_ss,
        r_pp=r_pp,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )
    builders = [
        case_homogeneous_slab,
        case_homogeneous_material_tensor,
        case_homogeneous_uni_vs_bi,
        case_mixed_homogeneous,
        case_mixed_homogeneous_nary,
        case_callable_field,
        case_callable_tensor,
        case_callable_field_short_stack,
        case_diffusion,
        case_diffusion_short_stack,
        case_second_order_gamma,
        case_second_order_phi,
        case_second_order_cos2,
        case_second_order_short_stack,
        case_volfrac_spline,
        case_free_optical_spline,
        case_volfrac_polynomial,
        case_free_optical_polynomial,
        case_free_optical_polynomial_short_stack,
        case_free_optical_osc_stack,
        case_kitchen_sink,
    ]
    for build in builders:
        build()
        print(f"wrote {build.__name__}")


if __name__ == "__main__":
    main()
