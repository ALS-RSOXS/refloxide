# refloxide

A blazingly fast 4x4 transfer matrix method for simulating reflection though stratified media.

This project is designed to be a fast core backend for simulating reflectivity using the 
4x4 transfer matrix method. We are basing the code on the following pre existing codebases, but 
this is a ground up rewrite.
- [refnx](https://github.com/refnx/refnx) - this library is great for simulation
reflectivity though scalar media, but fails at tensor indeces.
- [refl1d](https://github.com/refl1d/refl1d) - again, this works great for scal indeces,
but is hard to work with in tensor materials, and with buidling complex structures. 

# What is refloxide?

This library started as just the pure transfer matrix simulation core
but has grown to a full-featured optimized library for polarized
energy dependent reflectivity simulation and optimization. We maintain
a set of low memory footprint structures used to describe stratified media
that are optimized for fast simulation and memory efficiency. 

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