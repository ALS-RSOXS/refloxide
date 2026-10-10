# Contributing to refloxide

Thank you for your interest in contributing to refloxide!

## Development Setup

1. Fork and clone the repository:
  ```bash
   git clone https://github.com/HarlanHeilman/refloxide.git
   cd refloxide
  ```
2. Install dependencies using uv (dev tooling + `[plugin]` modeling stack):
  ```bash
   uv sync --group dev --extra plugin
   ```
3. Install prek hooks:
  ```bash
   prek install
  ```

## Making Changes

1. Create a new branch for your feature or bugfix:
  ```bash
   git checkout -b feature/your-feature-name
  ```
2. Make your changes and ensure tests pass:
  ```bash
   uv run pytest
  ```
3. Ensure code quality:
  ```bash
   uv run ruff check .
   uv run ruff format .
   uv run ty check
  ```
4. Commit your changes using [conventional commits](https://www.conventionalcommits.org/):
  ```bash
   git commit -m "feat: add new feature"
  ```

## Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/). Here are some examples:

- `feat: add new feature` - A new feature
- `fix: resolve bug in X` - A bug fix
- `docs: update README` - Documentation changes
- `refactor: simplify code` - Code refactoring
- `test: add tests for X` - Adding tests
- `chore: update dependencies` - Maintenance tasks

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Submit a pull request with a clear description

### CI and the merge queue

`main` is protected by a ruleset that **requires the merge queue**.

| When | What runs |
| --- | --- |
| Every PR push | **CI** — lint, types, tests, Rust check, wheels |
| PR enters the merge queue | **CI again** on the prospective merge commit, plus **Pre-merge** — benchmarks (`github-action-benchmark`), pysentry, semgrep |

Land with **Merge when ready** / add to the merge queue (do not push straight to `main`). Optional early Pre-merge: apply the `pre-merge` label, mark Ready for review, or `gh workflow run pre-merge.yml --ref <branch>`.

## Releasing

Releases are a two-step, manual Actions flow (no auto-publish from a version bump alone).

1. **Prepare Release** (`workflow_dispatch`)
   - Computes the next semver from conventional commits since the latest `v*`
     tag (git-cliff / Commitizen rules), or an explicit `patch` / `minor` /
     `major` override.
   - Updates `pyproject.toml`, `Cargo.toml`, `src/refloxide/__init__.py`,
     lockfiles, and `CHANGELOG.md`.
   - Opens a `release/vX.Y.Z` pull request for review and optional Pre-merge
     checks.
2. Merge the release PR into `main` after review.
3. **Finalize Release** (`workflow_dispatch`)
   - Set `confirm` to `publish`.
   - Cuts annotated tag `vX.Y.Z` on `main`, which triggers the existing
     **Release** workflow (wheels, PyPI, GitHub Release notes).

Dry-run Prepare with `dry_run=true` to print the next version without opening
a PR.

## Dependency Updates

This project uses [Renovate](https://renovateapp.com/) for automated dependency updates. Renovate will automatically open pull requests when new versions are available for:

- GitHub Actions
- Python dependencies (via `pyproject.toml`)
- `uv.lock` lockfile

To activate it:

1. Go to [github.com/apps/renovate](https://github.com/apps/renovate) and click **Install**
2. Choose your GitHub account or organization
3. Under **Repository access**, select this repository (or all repositories)
4. Click **Install & Authorize**
5. Renovate will open an onboarding pull request titled `Configure Renovate` — merge it to activate
6. Renovate will now open PRs automatically when new dependency versions are available

The `renovate.json` at the root of this project is pre-configured to manage:

- GitHub Actions workflow dependencies
- Python dependencies (via `pyproject.toml`)

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- We use [ty](https://docs.astral.sh/ty/) for type checking
- All code should be properly typed
- Write docstrings for public functions and classes
