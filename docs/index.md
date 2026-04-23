# refloxide

A blaxingly fast 4x4 transfer matrix method for simulating reflection though stratified media

## Installation

Install using pip:

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

Clone the repository and install dependencies:

```bash
git clone https://github.com/HarlanHeilman/refloxide.git
cd refloxide
uv sync --group dev
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
