"""Parity of `PeakSumSLD` against a tabulated source built from the same peaks."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.special import dawsn

from refloxide.lineshapes import Peak, PeakSumSLD, tdm_fractions
from refloxide.model import UniTensorSLD


def _gaussian_reference(
    energy: np.ndarray, e0: float, width: float, height: float
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic `(delta, beta)` of a Gaussian peak with 1/e half-width `width`."""
    sigma = width / np.sqrt(2.0)
    x = energy - e0
    beta = height * np.exp(-0.5 * (x / sigma) ** 2)
    delta = height * (2.0 / np.sqrt(np.pi)) * dawsn(x / (np.sqrt(2.0) * sigma))
    return delta, beta


def test_peak_sum_matches_tabulated_reference() -> None:
    """Peak-sum `tensor_at_many` matches a dense table built from the same peaks.

    The table is a 0.002 eV grid over 280-290 eV holding the analytic Gaussian
    delta and beta of two peaks (heights 2.0 and 1.2, TDM tilts 0.3 and 1.2 rad).
    The query energies sit at half-grid offsets, so the only difference is the
    table's linear interpolation. The Gaussian interpolation error is about
    h*(step)**2/8 * f''/f ~ 1e-5 relative to the peak maximum. `rtol=1e-4` covers
    that with a factor of about 10 margin, and `atol=1e-6*max|reference|` guards
    the near-zero tails where the relative error is meaningless. Both were chosen
    from that estimate and were not tuned to pass.
    """
    grid = np.linspace(280.0, 290.0, 5001)
    peaks_spec = [
        {"e0": 284.5, "width": 0.5, "height": 2.0, "tilt": 0.3},
        {"e0": 286.8, "width": 0.7, "height": 1.2, "tilt": 1.2},
    ]
    delta_xx = np.zeros_like(grid)
    beta_xx = np.zeros_like(grid)
    delta_zz = np.zeros_like(grid)
    beta_zz = np.zeros_like(grid)
    peaks = []
    for k, spec in enumerate(peaks_spec, start=1):
        f_xx, f_zz = tdm_fractions(spec["tilt"])
        delta, beta = _gaussian_reference(
            grid, spec["e0"], spec["width"], spec["height"]
        )
        delta_xx += float(f_xx) * delta
        beta_xx += float(f_xx) * beta
        delta_zz += float(f_zz) * delta
        beta_zz += float(f_zz) * beta
        peaks.append(
            Peak(
                "gaussian",
                label=f"p{k}",
                height=spec["height"],
                tdm_tilt=spec["tilt"],
                energy=spec["e0"],
                width=spec["width"],
            )
        )
    table = pl.DataFrame(
        {
            "energy": grid,
            "n_xx": delta_xx,
            "n_ixx": beta_xx,
            "n_zz": delta_zz,
            "n_izz": beta_zz,
        }
    )
    query = 280.0 + 0.002 * (np.arange(80) * 62.5 + 0.25)

    actual = PeakSumSLD(peaks, density=1.1, rotation=0.7).tensor_at_many(query)
    reference = UniTensorSLD(table, density=1.1, rotation=0.7).tensor_at_many(query)

    np.testing.assert_allclose(
        actual,
        reference,
        rtol=1e-4,
        atol=1e-6 * np.abs(reference).max(),
    )
