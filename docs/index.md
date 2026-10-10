# refloxide

A blaxingly fast 4x4 transfer matrix method for simulating reflection though stratified media

## Installation

CPU and GPU are **separate wheels** (same import path; install one):

| Install | Purpose |
| --- | --- |
| `refloxide` / `refloxide[core]` | Fast Rust TMM on CPU |
| `refloxide[plugin]` | CPU core plus refnx modeling / fitting |
| `refloxide-gpu` | GPU wheel with wgpu (`device="gpu"`; see [GPU backend](guides/gpu.md)) |
| `refloxide[all]` | Alias for `refloxide[plugin]` |

```bash
pip install 'refloxide[core]'
pip install 'refloxide[plugin]'
pip install refloxide-gpu
```

With uv (recommended):

```bash
uv add 'refloxide[core]'
uv add 'refloxide[plugin]'
uv add refloxide-gpu
```

Bare `pip install refloxide` matches `[core]`.

## Quick Start

```python
import refloxide

print(refloxide.__version__)
```

## Team and support

- ALS-REIXS team: [RIXS Program at ALS](https://als.lbl.gov/science/photon-science-programs/rixs-program/)
- Group GitHub: [ALS-RSOXS organization](https://github.com/ALS-RSOXS)
- Submit issues: [refloxide issue tracker](https://github.com/ALS-RSOXS/refloxide/issues)

## Development

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for package management

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/HarlanHeilman/refloxide.git
cd refloxide
uv sync --group dev --extra plugin
```

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check
```

### Prek Hooks

Install prek hooks:

```bash
prek install
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](https://github.com/HarlanHeilman/refloxide/blob/main/LICENSE) file for details.
