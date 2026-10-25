"""Freeze the ZnPc and Alq3 optical-constant fits of lineshapes into a JSON file.

Run from the refloxide repository root::

    uv run python examples/data/freeze_lineshapes_fits.py

The lineshapes repository is read from the environment variable ``LINESHAPES_REPO``
(default ``/Users/hduva/projects/lineshapes``); its ``python`` and ``examples``
directories go on ``sys.path`` so the package and the example helpers import without
installing. Re-running the script rewrites ``lineshapes_materials.json`` next to this
file; on the same day the output is byte-for-byte identical.

Peak fits follow the lineshapes examples (``examples/_common.fit_peaks``): the fixed
continuum, DFT-cluster seeds, NNLS seed amplitudes, then ``least_squares`` refinement
of every peak and the global density ``scale``. The ZnPc uniaxial tensor is a joint
non-negative least-squares solve over the five NEXAFS angles, with the magic-angle
peak energies and widths held fixed.

JSON layout (stable)::

    {"provenance": {...},
     "materials": {
       "alq3": {"peaks": [{"kind", "energy", "width", "height"}, ...],
                "baseline": {"energy_ev", "delta", "beta"},
                "rel_rms": ...},
       "znpc": {"peaks": [{"kind", "energy", "width", "height_iso",
                           "F_xx", "F_zz"}, ...],
                "baseline": {...},
                "rel_rms_by_angle": {"20": ..., ...},
                "rel_rms_magic": ...}}}

Units and conventions: energies in eV; ``width`` is the Gaussian 1/e half-width
(sigma = width / sqrt(2)); peak heights are the height of beta at unit density; the
model value is ``delta + i*beta``. ``baseline`` is the fixed continuum at unit density
on a uniform grid from 282.0 to 291.0 eV (0.02 eV steps), values rounded to 8
significant digits. Scale by the material density in the film.

The fit model is ``scale * (continuum + peaks)`` with a fitted global density
``scale``. For Alq3 the scale is folded into the frozen heights and baseline
(provenance ``fit_details.alq3.scale_folded_in``), so the frozen model needs no extra
factor. For ZnPc the scale is not folded: the joint tensor solve is against the
unscaled continuum and absorbs normalisation into F_xx and F_zz, and the magic-angle
``height_iso`` agrees with F_iso better unscaled (provenance
``fit_details.znpc.density_scale_magic_not_folded``).

For ZnPc the anisotropic tensor obeys ``I(theta) = F_xx sin^2(theta) +
F_zz cos^2(theta)`` for incidence angle theta (radians in code); the isotropic
magic-angle value is ``F_iso = (2 F_xx + F_zz) / 3``.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

REPO = Path(os.environ.get("LINESHAPES_REPO", "/Users/hduva/projects/lineshapes"))
for _p in (REPO / "python", REPO / "examples"):
    sys.path.insert(0, str(_p))

import _common as c  # noqa: E402
from lineshapes import Model, datasets, mu_to_beta  # noqa: E402

OUT = Path(__file__).resolve().parent / "lineshapes_materials.json"
ANGLES = (20.0, 40.0, 54.7, 70.0, 90.0)
MAGIC_ANGLE = 54.7
BASELINE_GRID = np.linspace(282.0, 291.0, 451)
DATA_FILES = [
    "alq3_oc_tey.csv",
    "alq3_dft_clusters.csv",
    "alq3_continuum.json",
    "alq3_continuum_delta.csv",
    "znpc_nexafs.csv",
    "znpc_dft_clusters.csv",
    "znpc_continuum.json",
    "znpc_continuum_delta.csv",
]


def sig8(x) -> float:
    """Round to 8 significant digits (keeps the JSON small and reproducible)."""
    return float(f"{float(x):.8g}")


def peaks_of(fit) -> list[dict]:
    """Fitted Gaussian peaks of a `_common.PeakFit`, sorted by energy."""
    rows = []
    for k in range(fit.n_peaks):
        e0, w, a = (
            float(fit.theta[fit.names.index(f"gaussian[{k}].{p}")])
            for p in ("e0", "width", "amp")
        )
        rows.append({"energy": e0, "width": w, "height": a})
    return sorted(rows, key=lambda r: r["energy"])


def scale_of(fit) -> float:
    return float(fit.theta[fit.names.index("scale")])


def baseline_of(material: str, scale: float = 1.0) -> dict:
    """Fixed continuum at unit density on the common grid, times ``scale``."""
    m = scale * datasets.continuum_model(material)(BASELINE_GRID)
    return {
        "energy_ev": [sig8(e) for e in BASELINE_GRID],
        "delta": [sig8(v) for v in m.real],
        "beta": [sig8(v) for v in m.imag],
    }


def spectrum(theta_deg: float):
    """Energy and beta (unit density) of the ZnPc NEXAFS taken at ``theta_deg``."""
    t = datasets.load_table("znpc_nexafs")
    m = np.isclose(t["theta_deg"], theta_deg)
    return t["energy_ev"][m], mu_to_beta(t["energy_ev"][m], t["mu_cm2_per_g"][m])


def fit_alq3() -> tuple[dict, dict]:
    t = datasets.load_table("alq3_oc_tey")
    fit = c.fit_peaks(
        t["energy_ev"], t["beta"], "alq3", datasets.load_table("alq3_dft_clusters")
    )
    # The fit model is scale * (continuum + peaks), so the fitted density scale is
    # folded into the unit-density tables: heights and the baseline are multiplied.
    scale = scale_of(fit)
    peaks = [
        {"kind": "gaussian", **p, "height": p["height"] * scale} for p in peaks_of(fit)
    ]
    info = {
        "fit_window_ev": [float(t["energy_ev"].min()), float(t["energy_ev"].max())],
        "rel_rms": float(fit.rel_rms),
        "scale_folded_in": scale,
        "n_evaluations": int(fit.n_evals),
    }
    material = {
        "peaks": peaks,
        "baseline": baseline_of("alq3", scale),
        "rel_rms": float(fit.rel_rms),
    }
    return material, info


def fit_znpc() -> tuple[dict, dict]:
    seeds = datasets.load_table("znpc_dft_clusters")
    e_magic, b_magic = spectrum(MAGIC_ANGLE)
    magic = c.fit_peaks(e_magic, b_magic, "znpc", seeds)
    base = peaks_of(magic)
    cont = datasets.continuum_model("znpc")

    # Joint NNLS over all angles:
    # beta_theta = continuum + sum_k [Fxx_k sin^2 + Fzz_k cos^2] g_k.
    per_angle = {}
    blocks, rhs = [], []
    for theta in ANGLES:
        e, b = spectrum(theta)
        g = np.column_stack(
            [Model().gaussian(p["energy"], p["width"], 1.0)(e).imag for p in base]
        )
        s2, c2 = np.sin(np.deg2rad(theta)) ** 2, np.cos(np.deg2rad(theta)) ** 2
        blocks.append(np.hstack([s2 * g, c2 * g]))
        rhs.append(b - cont(e).imag)
        per_angle[theta] = (e, b, g, s2, c2)
    x, _ = nnls(np.vstack(blocks), np.concatenate(rhs))
    k = len(base)
    fxx, fzz = x[:k], x[k:]

    rel_by_angle = {}
    for theta, (e, b, g, s2, c2) in per_angle.items():
        pred = cont(e).imag + g @ (s2 * fxx + c2 * fzz)
        rel_by_angle[theta] = float(np.sqrt(np.mean((pred - b) ** 2)) / np.mean(b))

    peaks = []
    for p, xx, zz in zip(base, fxx, fzz, strict=True):
        peaks.append(
            {
                "kind": "gaussian",
                "energy": p["energy"],
                "width": p["width"],
                "height_iso": p["height"],
                "F_xx": float(xx),
                "F_zz": float(zz),
            }
        )
    material = {
        "peaks": peaks,
        "baseline": baseline_of("znpc"),
        "rel_rms_by_angle": {f"{a:g}": rel_by_angle[a] for a in ANGLES},
        "rel_rms_magic": float(magic.rel_rms),
    }

    f_iso = (2.0 * fxx + fzz) / 3.0
    heights = np.array([p["height"] for p in base])
    strong = heights > 0.05 * heights.max()
    dev = np.abs(f_iso - heights) / heights
    worst = int(np.argmax(np.where(strong, dev, -1.0)))
    info = {
        "fit_window_ev": [float(e_magic.min()), float(e_magic.max())],
        "density_scale_magic_not_folded": scale_of(magic),
        "rel_rms_magic": float(magic.rel_rms),
        "rel_rms_by_angle": rel_by_angle,
        "nnls_pinned_zero_at_ev": {
            "F_xx": [
                round(p["energy"], 2)
                for p, v in zip(peaks, fxx, strict=True)
                if v == 0.0
            ],
            "F_zz": [
                round(p["energy"], 2)
                for p, v in zip(peaks, fzz, strict=True)
                if v == 0.0
            ],
        },
        "F_iso_vs_height_iso_worst_rel_dev_over_5pct_peaks": {
            "energy_ev": peaks[worst]["energy"],
            "rel_dev": float(dev[worst]),
            "n_peaks_checked": int(strong.sum()),
        },
    }
    return material, info


def lineshapes_commit() -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def lineshapes_version() -> str:
    text = (REPO / "Cargo.toml").read_text()
    return re.search(r'^version = "([^"]+)"', text, flags=re.MULTILINE).group(1)


def freeze() -> dict:
    alq3, alq3_info = fit_alq3()
    znpc, znpc_info = fit_znpc()
    all_energies = [p["energy"] for p in alq3["peaks"] + znpc["peaks"]]
    provenance = {
        "lineshapes_repo": str(REPO),
        "lineshapes_commit": lineshapes_commit(),
        "lineshapes_version": lineshapes_version(),
        "freeze_date": datetime.date.today().isoformat(),
        "data_files": DATA_FILES,
        "fit_method": (
            "examples/_common.fit_peaks: continuum fixed, DFT seeds, NNLS then "
            "least_squares on peaks and scale"
        ),
        "znpc_tensor_method": (
            "joint NNLS over five NEXAFS angles; magic-angle energies and widths "
            "fixed; continuum fixed"
        ),
        "density_convention": "unit density; scale by material density in the film",
        "baseline_grid_ev": [282.0, 291.0, 0.02],
        "peak_energy_range_ev": [min(all_energies), max(all_energies)],
        "peaks_above_291_ev": {
            "alq3": [p["energy"] for p in alq3["peaks"] if p["energy"] > 291.0],
            "znpc": [p["energy"] for p in znpc["peaks"] if p["energy"] > 291.0],
        },
        "fit_details": {"alq3": alq3_info, "znpc": znpc_info},
    }
    return {"provenance": provenance, "materials": {"alq3": alq3, "znpc": znpc}}


def main() -> None:
    result = freeze()
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
