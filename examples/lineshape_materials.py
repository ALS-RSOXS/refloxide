"""Shared material helpers for the lineshapes examples.

The frozen ZnPc and Alq3 peak sets (``examples/data/lineshapes_materials.json``,
written by ``examples/data/freeze_lineshapes_fits.py``) enter here as plain
records. This module holds the film-independent pieces: the frozen-file loader,
the unit-density baseline table, the ZnPc molecular tilt derivation, and the
seeded ``hc_contaminant`` record builder. Film layout (shapes, level fields,
depth profiles) stays in each example.

Used by ``lineshape_three_material.py`` and the other lineshapes examples. The
examples directory is not a package, so an example puts its own directory on
``sys.path`` before importing this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy.optimize import brentq

from refloxide.data import OpticalConstants
from refloxide.lineshapes import tdm_fractions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

DATA_FILE = Path(__file__).resolve().parent / "data" / "lineshapes_materials.json"
# Illustrative film densities (g/cm^3). The frozen tables are per unit density.
DENSITY_ALQ3 = 1.31
DENSITY_ZNPC = 1.60
# Seeded contaminant of the ZnPc peaks (one generator, drawn in this order).
CONTAMINANT_SEED = 2025
CONTAMINANT_PAIRS = 3  # disjoint height swaps
CONTAMINANT_TOP = 8  # swap partners come from this many strongest peaks (by F_iso)
CONTAMINANT_WINDOW = (283.5, 289.5)  # eV, swap candidates lie inside the demo window
JITTER_SIGMA = 0.15  # lognormal sigma of each height factor
JITTER_CLIP = (0.7, 1.4)  # height factors are clipped to this range
SHIFT_SIGMA = 0.05  # eV, Gaussian energy shift of each peak
SHIFT_CLIP = 0.10  # eV, energy shifts are clipped to +/- this


def load_materials() -> dict[str, Any]:
    """Frozen ZnPc and Alq3 peak sets and baselines, read from `DATA_FILE`."""
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


def sorted_peak_records(material: dict[str, Any]) -> list[dict[str, Any]]:
    """Peak records of one material in energy order (lowest energy first)."""
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


def contaminant_draw(
    records: Sequence[dict[str, Any]],
) -> tuple[list[tuple[int, int]], NDArray[np.float64], NDArray[np.float64]]:
    """Seeded contaminant of the ZnPc peaks: swap pairs, heights and energies.

    Returns the swap pairs (indices into ``records``), the absolute heights
    after the swaps and the height jitter, and the shifted energies (eV). The
    swap candidates are the `CONTAMINANT_TOP` strongest peaks by `F_iso` whose
    original energy lies in `CONTAMINANT_WINDOW`; the seeded generator draws the
    disjoint pairs among them. Every peak gets a lognormal height factor and a
    clipped Gaussian energy shift.
    """
    rng = np.random.default_rng(CONTAMINANT_SEED)
    f_iso = np.array([znpc_iso(rec) for rec in records])
    energy = np.array([float(rec["energy"]) for rec in records])
    in_window = np.flatnonzero(
        (energy >= CONTAMINANT_WINDOW[0]) & (energy <= CONTAMINANT_WINDOW[1])
    )
    strongest = in_window[np.argsort(-f_iso[in_window])][:CONTAMINANT_TOP]
    chosen = rng.permutation(strongest)[: 2 * CONTAMINANT_PAIRS]
    pairs = [(int(a), int(b)) for a, b in chosen.reshape(CONTAMINANT_PAIRS, 2)]
    heights = f_iso.copy()
    for a, b in pairs:
        heights[[a, b]] = f_iso[[b, a]]
    jitter = np.clip(rng.lognormal(0.0, JITTER_SIGMA, len(records)), *JITTER_CLIP)
    shift = np.clip(rng.normal(0.0, SHIFT_SIGMA, len(records)), -SHIFT_CLIP, SHIFT_CLIP)
    return pairs, heights * jitter, energy + shift
