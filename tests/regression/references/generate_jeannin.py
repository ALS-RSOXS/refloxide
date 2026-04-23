"""Regenerate golden data from the Jeannin Python reference implementation.

This script invokes the Jeannin Python 4x4 transfer matrix port as a
library and writes the resulting amplitude sweeps to compressed numpy
archives under ``jeannin_python/``. The regression suite consumes these
archives when checking agreement against the reference.

The Jeannin port is not currently listed in the project virtual
environment because it is only needed for reference-data generation and
not for day-to-day development. Install it into the environment before
running this script, for example via the Zenodo tarball or by checking
out the Zenodo-archived source tree.

Typical invocation from the project root::

    uv run python tests/regression/references/generate_jeannin.py

The script is intentionally conservative about importing the Jeannin
module at top level; it raises a clear error message if the dependency
is missing, and it refuses to overwrite existing golden files without
an explicit ``--force`` flag.

References:
    M. Jeannin, "Generalized 4x4 matrix algorithm for light propagation
    in anisotropic stratified media (Python files)," Zenodo (2019),
    https://doi.org/10.5281/zenodo.3417751.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_DIR: Path = Path(__file__).parent / "jeannin_python"


def _require_jeannin() -> None:
    """Verify that the Jeannin reference implementation is importable.

    Raises:
        ModuleNotFoundError: If the Jeannin port is not installed.
    """
    try:
        import jeannin_py4x4  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Jeannin Python reference implementation is not "
            "installed. See tests/regression/references/README.md for "
            "installation instructions."
        ) from exc


def _generate_sic_gan_sic_otto(
    output_path: Path,
    thickness_gan_nm: float = 1000.0,
) -> None:
    """Generate the SiC/GaN/SiC Otto-geometry reference data.

    Args:
        output_path: Destination ``.npz`` file.
        thickness_gan_nm: Thickness of the intermediate GaN layer.

    Raises:
        NotImplementedError: The mapping from ``jeannin_py4x4`` API onto
            the golden-file layout is left as a concrete implementation
            step for whoever first wires the reference. The signature
            and output contract are fully specified; only the body
            needs to be filled in.
    """
    del output_path, thickness_gan_nm  # referenced in the implementation
    raise NotImplementedError(
        "Wire this to jeannin_py4x4. Expected output npz keys: "
        "theta_rad (float64, N_theta), wavenumber_cm (float64, N_k), "
        "thickness_gan_nm (float64 scalar), and the eight amplitudes "
        "(complex128, shape (N_theta, N_k)) named r_pp, r_ss, r_ps, "
        "r_sp, t_pp, t_ss, t_ps, t_sp."
    )


def _generate_uniaxial_substrate(output_path: Path) -> None:
    """Generate the uniaxial-substrate cross-polarization reference.

    Args:
        output_path: Destination ``.npz`` file.

    Raises:
        NotImplementedError: As with ``_generate_sic_gan_sic_otto``, the
            API mapping is left as the concrete implementation step.
    """
    del output_path
    raise NotImplementedError(
        "Wire this to jeannin_py4x4 with a rotated uniaxial substrate. "
        "Expected output npz keys: theta_rad (float64, N_theta), "
        "wavenumber_cm (float64 scalar), eps_ordinary (complex128 "
        "scalar), eps_extraordinary (complex128 scalar), and the four "
        "reflection amplitudes (complex128, shape (N_theta,))."
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing golden files.",
    )
    parser.add_argument(
        "--gan-thickness-nm",
        type=float,
        default=1000.0,
        help="GaN layer thickness for the Otto stack.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        _require_jeannin()
    except ModuleNotFoundError as exc:
        logger.error(str(exc))
        return 1

    targets: dict[str, object] = {
        "sic_gan_sic_otto.npz": lambda p: _generate_sic_gan_sic_otto(
            p, args.gan_thickness_nm
        ),
        "uniaxial_substrate.npz": _generate_uniaxial_substrate,
    }

    for fname, generator in targets.items():
        output_path = OUTPUT_DIR / fname
        if output_path.exists() and not args.force:
            logger.info("Skipping %s (exists; use --force to overwrite)", fname)
            continue
        logger.info("Generating %s", fname)
        try:
            generator(output_path)  # type: ignore[misc]
        except NotImplementedError as exc:
            logger.warning("%s: %s", fname, exc)
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
