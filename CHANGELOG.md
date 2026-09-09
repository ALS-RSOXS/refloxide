# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6] - 2026-09-09

### Added

- Per-energy experiment correction channels (`refloxide.instrument`)
- `FreeTensorSLD`, batch materialize paths, and instrument wiring on `ReflectModel`
- Nevot-Croce prior and `thread_workers` on `Objective`
- Examples: Brewster ZnPc/Si REPL, bookended and real-data fitting workflows,
  DFT / free-tensor comparison REPLs

### Changed

- Dropped `pyref` integration; pure-Python modeling lives under
  `refloxide.python.model` / `refloxide.python.tmm`
- Cached `OpticalConstants` OOC lookups per energy
- Raised transitive `gitpython` / `tornado` / `pymdown-extensions` floors for
  pysentry

### Fixed

- `Reflectivity.s` / `.p` map to physical `R_ss` / `R_pp` (Fresnel-validated
  kernel layout); fused bookended path separates wavelength from OC query
  energy

## [0.1.5] - 2026-07-13

### Added

- Initial project structure
- Basic package setup with refloxide
- Documentation with MkDocs
- GitHub Actions CI/CD pipelines
- Prek hooks for code quality
