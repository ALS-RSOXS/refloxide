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
4. Run the benchmark comparison (below) and paste its table into the PR description
5. Submit a pull request with a clear description

### Benchmarks are a PR requirement

Every pull request that touches Rust code, the build profile, or the Python
hot path must include a before/after benchmark from at least one machine.
Shared CI runners are too noisy to measure performance, so the benchmark is
not an Actions job: the author runs it locally when opening the PR.

```bash
make bench-compare        # or: uv run python scripts/bench_compare.py
```

The script builds a release wheel of the merge-base with `main` (in a
temporary git worktree) and of your working tree, runs
`examples/gpu_backend_bench.py` against each in an isolated environment, and
writes a markdown table (also saved to `bench-compare.md`, which is
gitignored) with the machine, toolchain, and per-benchmark change. Paste the
whole block into the PR description. Pass `--base <ref>` to compare against a
tag or another commit, and `--repeats N` for tighter medians.

Reviewers expect no unexplained slowdown. A regression is acceptable only with
a stated reason in the PR. Docs-only changes may state "not applicable".

### CI

| Workflow | When | What |
| --- | --- | --- |
| **CI** | every PR push, push to `main` | `rust` (fmt, clippy, tests, shader check), `python` (lint, format, types, pytest; extension built once), `package` (wheel and sdist build plus import; Linux on PRs, Linux/macOS/Windows on `main`) |
| **Security** | weekly, manual, PRs that change a lockfile or manifest | dependency scan; semgrep on the schedule only |
| **Documentation** | push to `main` | build and deploy the site |
| **Release** | push to `main` that changes the package, manual dry run | see Releasing |

Performance is not checked in CI; see "Benchmarks are a PR requirement".
Reproduce the CI checks locally with `cargo fmt --all --check`,
`cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test --workspace --all-targets`, and `make verify test`.

## Releasing

A release is a version bump merged to `main`. Nothing is published from a
branch or a manual step.

1. Run `make release-prep`. It derives the next semver from conventional
   commits since the latest `v*` tag (or take `--bump patch|minor|major`) and
   updates `pyproject.toml`, `Cargo.toml` (`[workspace.package]`),
   `python/refloxide/__init__.py`, lockfiles, and `CHANGELOG.md`.
2. Commit the result and open a normal pull request.
3. Merge it. The **Release** workflow checks that the three versions agree and
   that tag `vX.Y.Z` does not exist yet. If so, it runs CI, builds wheels for
   Linux (x86_64, aarch64), macOS (x86_64, arm64) and Windows (x64) plus the
   sdist, publishes to PyPI by trusted publishing, then creates the tag and a
   GitHub release with git-cliff notes. If the tag exists, it does nothing.

Run **Release** manually with `dry_run` to build everything without
publishing or tagging. The Rust crates are not published to crates.io.

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
