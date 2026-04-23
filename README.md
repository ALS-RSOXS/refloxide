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

A blaxingly fast 4x4 transfer matrix method for simulating reflection though stratified media

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