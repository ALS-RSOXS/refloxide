"""End-to-end run of the three-material peak-gradient example (docs page)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from refloxide.lineshapes import MAGIC_ANGLE

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "lineshape_three_material.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("lineshape_three_material", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_material_example_runs_and_gradients_hold():
    example = _load_example()

    structure = example.build_structure()
    assert len(structure) == 3  # vacuum | film | substrate
    refl = example.reflectivity()
    assert np.all(np.isfinite(refl))
    assert np.all(refl >= 0.0)

    profiles = example.channel_profiles()
    _, gamma = profiles["orientation_0"]
    assert abs(gamma[0]) < 1e-6
    assert abs(gamma[-1] - MAGIC_ANGLE) < 1e-6

    _, tilt = profiles["mat1.p1.tdm_tilt"]
    assert abs(tilt[0]) < 1e-6
    assert abs(tilt[-1] - np.deg2rad(20.0)) < 1e-6
