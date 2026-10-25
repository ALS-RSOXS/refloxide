"""End-to-end run of the surface-diffusion example (docs page)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from refloxide.lineshapes import MAGIC_ANGLE

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "lineshape_surface_diffusion.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "lineshape_surface_diffusion", EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_surface_diffusion_example_runs_and_diffusion_holds():
    example = _load_example()
    q = np.linspace(0.01, 0.27, 8)
    energies = np.array([285.1, 287.0])

    diffused = example.build_structure()
    reference = example.build_reference()
    assert len(diffused) == 4  # vacuum | film | native oxide | Si substrate
    for structure in (diffused, reference):
        refl = example.reflectivity(structure, energies, q=q)
        assert np.all(np.isfinite(refl))
        assert np.all(refl >= 0.0)

    film = diffused[1]
    rows, _ = film.rows_and_tensors_at(285.1)
    thicknesses = rows[:, 0]
    z_mid = np.cumsum(thicknesses) - 0.5 * thicknesses
    shallow, deep = z_mid[0], z_mid[-1]
    gamma_ref = float(example.znpc_tilts()[0])

    # Gamma runs from the magic angle at the surface toward gamma_ref in the bulk.
    gamma_at = film.named_values_at(np.array([shallow, deep]))["orientation"]
    assert abs(gamma_at[0] - MAGIC_ANGLE) < np.deg2rad(2.0)
    assert abs(gamma_at[1] - gamma_ref) < np.deg2rad(1.0)

    # One shared diffusion-length Parameter drives every diffusion.
    flat = film.parameters.flattened(unique=True)
    length_params = [p for p in flat if p.name == "diffusion_length"]
    assert len(length_params) == 1
    length_param = length_params[0]
    mid = np.array([60.0])
    before = film.named_values_at(mid)["orientation"][0]
    length_param.value = 20.0
    after = film.named_values_at(mid)["orientation"][0]
    length_param.value = 60.0
    assert abs(after - gamma_ref) < abs(before - gamma_ref)

    # Surface heights are 0.35 times the bulk height of each diffused pi* peak.
    records = example.load_materials()["znpc"]["peaks"]
    heights = film.named_values_at(np.array([0.0]))
    height_keys = [
        k for k in heights if k.startswith("znpc.p") and k.endswith(".height")
    ]
    assert len(height_keys) == example.N_PI_STAR_HEIGHTS
    for key in height_keys:
        index = int(key.split(".")[1][1:]) - 1
        bulk = example.znpc_iso(records[index])
        assert abs(heights[key][0] - 0.35 * bulk) < 0.05 * 0.35 * bulk
