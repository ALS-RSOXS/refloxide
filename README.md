# refloxide

[CI](https://github.com/HarlanHeilman/refloxide/actions/workflows/ci.yml)
[PyPI version](https://badge.fury.io/py/refloxide)
[codecov](https://codecov.io/gh/HarlanHeilman/refloxide)
[Python 3.13+](https://www.python.org/downloads/)
[uv](https://github.com/astral-sh/uv)
[Ruff](https://github.com/astral-sh/ruff)
[ty](https://github.com/astral-sh/ty)
[License: GPL-3.0](https://github.com/HarlanHeilman/refloxide/blob/main/LICENSE)
[Renovate](https://renovateapp.com/)

A blazingly fast 4x4 transfer matrix method for simulating reflection though stratified media.

This project is designed to be a fast core backend for simulating reflectivity using the 
4x4 transfer matrix method. We are basing the code on the following pre existing codebases, but 
this is a ground up rewrite.
- [refnx](https://github.com/refnx/refnx) - this library is great for simulation
reflectivity though scalar media, but fails at tensor indeces.
- [refl1d](https://github.com/refl1d/refl1d) - again, this works great for scal indeces,
but is hard to work with in tensor materials, and with buidling complex structures. 

We are basing this library on this paper:
- https://opg.optica.org/viewmedia.cfm?r=1&rwjcode=josab&uri=josab-34-10-2128&html=true

## Core philosophy
This library is designed as a pure functional simulation core. A caller defines a structure
as fronting medium, backing medium, and a sequence of interior layers with thickness,
interfacial roughness, and refractive-index parameterization. The solver then evaluates
reflectivity for a requested set of experimental coordinates (for example angle, energy,
or wavelength), and returns deterministic numerical arrays with no hidden mutable state.

The design intent is:
- Explicit physical inputs with validation at construction time.
- Predictable outputs for scalar and vectorized evaluation paths.
- Single-threaded execution as the default, so this engine composes cleanly inside
  larger fitting and orchestration systems that already manage parallelism.
- Strict separation between geometry (`Structure`), optical parameterization (`n`),
  and propagation numerics (4x4 transfer-matrix kernels).

In short, `refloxide` should behave like a reliable computational kernel: give it a fully
specified layered system and a measurement grid, and it returns reproducible reflectivity
without side effects.

## Features

- Fast and modern Python toolchain using Astral's tools (uv, ruff, ty)
- Type-safe with full type annotations
- Comprehensive documentation with MkDocs — [View Docs](https://HarlanHeilman.github.io/refloxide/)

## Installation

```bash
pip install refloxide
```

Or using uv (recommended):

```bash
uv add refloxide
```

## Quick Start

```python
import refloxide

print(refloxide.__version__)
```

## Development

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for package management

### Setup

```bash
git clone https://github.com/HarlanHeilman/refloxide.git
cd refloxide
make install
```

### Running Tests

```bash
make test

# With coverage
make test-cov

# Across all Python versions
make test-matrix
```

### Code Quality

```bash
# Run all checks (lint, format, type-check)
make verify

# Auto-fix lint and format issues
make fix
```

### Prek

```bash
prek install
prek run --all-files
```

### Documentation

```bash
make docs-serve
```

## Dependency Updates

This project uses [Renovate](https://renovateapp.com/) to keep dependencies up to date automatically. Renovate will open pull requests when new versions of dependencies are available.

To enable it, install the [Renovate GitHub App](https://github.com/apps/renovate) and grant it access to this repository.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.