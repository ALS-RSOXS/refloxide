
<!-- DOTAGENT MANAGED - DO NOT EDIT THIS section>
# General

## General Structure

This codebase is maintained by contributors with physics PhDs and extensive backgrounds in scientific and engineering software, including numerical computing, data analysis, instrumentation, simulation, and research-grade reproducibility. Maintainers are highly mathematically literate, comfortable with linear algebra and statistics, and expect rigorous numerics with explicit type handling—silent coercion and imprecise computations are not acceptable.
## Operating principles

- Prefer the smallest coherent change set that satisfies the stated specification. Avoid drive-by refactors, unrelated formatting sweeps, and scope expansion.
- Treat the repository’s existing patterns as the default contract. Match naming, module boundaries, error-handling style, and test layout unless the user explicitly requests a migration.
- Default to production-grade output: complete, runnable, and reviewable. Do not ship placeholder text such as ellipses, “the rest of the implementation here”, “TODO: implement”, or “fill in later” inside code or patches unless the user explicitly authorizes a stub.
- Remain non-lazy: if a command fails, diagnose, adjust, and retry with a different approach when reasonable. Do not stop after the first error without analysis.
- Do not use emoji in code, comments, documentation strings, commit messages, or user-facing text unless the user explicitly requests emoji.

## Communication and documentation outside code

- Avoid standalone documentation files or long narrative write-ups unless the user asks for them or the repository already uses them for the same purpose.
- Prefer editing the code and tests that enforce behavior over adding parallel prose that can drift out of date.
- When the user asks for explanation, keep it precise and tied to the change set.

## Public API documentation (language-agnostic)

- Every **public** function, method, or type exported from a library module carries documentation appropriate to the language (for example Python docstrings, Rust `///` on public items, TSDoc/JSDoc on exported symbols).
- Documentation states the **surface**: name, purpose, parameters, return value, and thrown or returned error shapes when that is part of the contract.
- Prefer **prescriptive** voice that states what the symbol **does** and **means** for callers. Prefer “Maximum `foo` grouped by `bar` using a stable sort on `bar`.” over “Returns the max of foo by bar.” or "Computes the maximum `foo` grouped by `bar` using a stable sort on `bar`." or "Returns the maximum `foo` grouped by `bar` using a stable sort on `bar`."
- For each parameter: name, type as used in the project, allowed ranges or invariants when non-obvious, and interaction with other parameters.
- For results: type, semantics, units when relevant, ordering guarantees, and stability promises when they matter for science or reproducibility.
- Describe **what** the function does at the abstraction level of the API, **how** only when algorithmic choices affect correctness, performance contracts, or numerical stability, and **why** that approach is chosen when trade-offs are non-obvious (for example streaming vs materializing, online vs batch statistics).
- **Internal** helpers omit the long-form contract unless complexity warrants a short note. **Private** helpers keep documentation minimal (a phrase or single sentence at most).

## Module and package documentation

- Each library module (or the closest equivalent in the language’s module system) includes a short module-level description of responsibility: what problems it solves, what it explicitly does not handle, and which invariants callers should respect.
- Module docs are prescriptive about intent and boundaries so new contributors and agents do not duplicate concerns across modules.

## Tooling, skills, and continued learning

- Use project rules, agent skills, and MCP documentation tools when they apply to the task. Prefer authoritative library and framework documentation over memory when behavior, defaults, or breaking changes matter.
- When touching unfamiliar APIs, verify signatures, deprecations, and error modes against current docs or source in the dependency before guessing.
- Prefer file-scoped or package-scoped commands when the repository documents them (typecheck, lint, format, test on a single path) to shorten feedback loops.
- State permission-sensitive actions clearly (dependency installs, destructive commands, credential access) and follow the user’s safety expectations for the workspace.

## Task shape (goal, context, constraints, completion)

- Restate the goal in terms of observable outcomes: behavior, tests, or interfaces that change.
- Ground work in the relevant files, modules, and existing tests named by the user or discovered through search.
- Honor explicit constraints (performance, numerics, compatibility, style) before proposing alternatives.
- Stop when the completion criteria are met: tests pass where applicable, edge cases called out by the user are handled, and no placeholder implementation remains.

## Quality bar for agent output

- Do not substitute templates, pseudocode, or abbreviated implementations when the user asked for working code.
- If scope is too large for one pass, propose a staged plan and complete the first stage fully rather than leaving partial files full of omissions.

## Git Commit Messages

- Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. First word of the commit message should be the type of commit, followed by a colon and the scope of the commit. The scope should be the file or module that is being changed. The commit message should be in the present tense. The commit message should be no more than 72 characters.

# Python

## Python

The following applies to **Python** work in this repository: scientific and general-purpose code, with emphasis on clear structure, reproducible tooling, and documentation that matches how the team uses Cursor (skills, subagents, and editor rules).

### Conventions

- Follow **PEP 8** surface style; treat **Ruff** configuration in `pyproject.toml` as the enforced interpretation of those conventions.
- Prefer **Python 3.12+** unless the repo pins an older interpreter.
- Prefer **readability** over micro-optimizations; vectorize numerics when a library primitive exists instead of tight Python loops over large data.
- **NumPy-style docstrings** on **public** APIs (parameters, returns, and examples where they clarify behavior). Keep implementation bodies clear **without** long narrative **inline comments**—use names, small functions, and docstrings instead.
- **Tables and time series**: explicit **index/column** semantics when using **pandas**; **lazy** queries when standardizing on **Polars** for heavy pipelines.
- **Lab / instruments**: separate **resource lifecycle** (open, configure, close) from **command strings** and parsing (e.g. **pyvisa** patterns).

### Tooling

Use **[uv](https://docs.astral.sh/uv/latest/)** for environments, runs, and dependency changes. Pair it with the **[Astral](https://astral.sh/)** stack as configured in this project.

- **Dependencies**: add, upgrade, and remove packages with `**uv add`**, `**uv add … --upgrade**`, `**uv remove**`—do **not** hand-edit version pins in `pyproject.toml`.
- **Environment**: `**uv sync`** after cloning or when the lockfile changes; `**uv run …**` to execute Python, tools, and tests.
- **Dev tools**: keep `**ruff`** and `**ty**` in the development (or project) dependency group; run `**ruff check**` (and project formatting if applicable) plus `**ty check**` on changed code. `**uvx**` remains an option for one-off tool runs.
- **pytest**: install via `**uv add --dev pytest`** (or the project’s dev group); run with `**uv run pytest**`.

If a `**uv**` subcommand differs by version, use `**uv --help**` or the [uv docs](https://docs.astral.sh/uv/latest/).

### Testing

- Prefer **fast, deterministic** unit tests; isolate I/O and timing-sensitive checks when the team uses markers or separate jobs.
- **Regression tests** for fixed bugs; for numerics, assert **shapes**, **dtypes**, and stability expectations when science or reproducibility requires it.

### Cursor: skills

Load these **skills** by **name** when the task matches (each skill’s own `SKILL.md` and references hold the full detail). Installed skills usually live under `.cursor/skills/` (or your editor’s equivalent).


| Skill                     | Use it for                                                                                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **general-python**        | Hub: **uv** / **ruff** / **ty** workflow, builtins and collections, functions and classes, **dataclasses**, typing boundaries, **pytest**, scientific defaults, and pointers to the other skills. |
| **numpy-scientific**      | **NumPy**: dtypes, views vs copies, broadcasting, ufuncs and reductions, **linalg** / **einsum**, `**Generator`**, I/O, interop with tables and plotting.                                         |
| **dataframes**            | **pandas** and **Polars**: when to use which, indexing, joins, lazy execution, I/O, nulls, Arrow interop.                                                                                         |
| **numpy-docstrings**      | **Numpydoc**-style docstrings: section order, semantics (what belongs in docstrings vs types vs tests), anti-patterns, **Parameters** / **Returns** / **Examples** / classes / modules.           |
| **matplotlib-scientific** | Publication-style **Matplotlib**: OO API, axes and legends, layout, export, journal widths, optional **SciencePlots**.                                                                            |
| **lab-instrumentation**   | **PyVISA** / VISA sessions, **sockets** vs VISA, **hardware abstraction**, **input validation** before I/O, **testing** without hardware, **PDF** extraction for datasheets and manuals.          |


### Cursor: subagents

Delegate by **subagent name** when a focused pass is better than inline editing. Subagents usually live under `.cursor/agents/` (or your editor’s equivalent).


| Subagent            | Use it for                                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **python-reviewer** | Reviewing changes: **uv** hygiene, typing, numerics footguns, tests, docstring quality.                                           |
| **python-types**    | Deep **typing** for **ty**: annotations, PEP 695-style generics, exhaustive `**match`**, fixing checker output.                   |
| **python-refactor** | **Structure**: unclear multi-value returns, composition vs inheritance, oversized functions or classes, deterministic boundaries. |


### Cursor: rules

- A **Python** Cursor **rule** applies to Python sources (typically `**/*.py` when the rule is configured for those globs). It restates **interpreter preference**, **uv** usage, **ruff** / **ty** expectations, numerics and docstring defaults, and points to **general-python**, domain skills such as **lab-instrumentation** when editing drivers or lab I/O, and the subagents above.
- **Rule text is authoritative for “always on” editor hints**; **skills** carry the long-form patterns and examples. When the two differ on a detail, follow **this spec** and `**pyproject.toml`**, then the **rule**, then skill nuance.

### External references

- [numpydoc format](https://numpydoc.readthedocs.io/en/latest/format.html)
- [uv](https://docs.astral.sh/uv/latest/), [Ruff](https://docs.astral.sh/ruff/), [ty](https://docs.astral.sh/ty/)

# Rust

This workspace uses the **dotagent Rust** stack.

- Prefer `cargo` as the interface for builds, tests, and dependency changes; add crates with `cargo add`.
- Match edition and MSRV documented in the repository; do not bump them casually.
- Prefer `clippy` clean builds when the project already enforces clippy in CI.
- Error handling should be structured (`Result`) at boundaries; avoid `unwrap` in library code unless explicitly documented.

# Python - PyO3

This workspace extends Python with **PyO3 / Maturin** integration expectations.

### General Guidelines

We are using the [pyo3](https://pyo3.rs/) library to create a python extension for our project. The ONLY reason we are doing this is to speed up the execution of critical code. As such, we should only use pyo3 for code that is critical to the performance of the project, and should avoid using it for code that is not performance critical. Generally, iO operations are easier to implement in python, while multi-threaded operations are faster in rust.

The python GIL is the global interpreter lock, limiting the ability for python to be truly multi-threaded. However, rust executions can bypass the GIL leading to a significant speedup in performance. As such, we should not multi-thread in python, but rather pass tasks into rust for parallel processing, and then return the results back to python.

### Tooling

`uv` is still king within the project, and should be used for all python needs. However, we will use the `maturin` tool as the build system for the project. See the [maturin documentation](https://www.maturin.rs/) for more information on how to use it. Ensure that `maturin` is installed in the dev group. See the [uv documentation](https://docs.astral.sh/uv/concepts/projects/init/#projects-with-extension-modules) for more information on how to use maturin as a tool. In general, we prefer a structure native to maturin following the following format:

```
.
├── pyproject.toml
├── Cargo.toml
├── python/
│   ├── <module_name>/
│   │   ├── __init__.py
│   │   └── ...
├── src/
│   ├── lib.rs
│   └── bindings.rs
```

Ensure that the `python/` directory is a valid python package, and that the `src/` directory is a valid rust crate.

# Rust - PyO3

This workspace extends Rust with **PyO3 / Maturin** extension expectations.

- Design the Rust crate as an extension first: minimal Python surface, maximal safety around lifetimes and exceptions.
- Use Maturin's layout and metadata conventions; keep `pyproject.toml` and `Cargo.toml` agreeing on names and versions.
- Document unsafe blocks with the project's standard (even if you avoid new unsafe code).
- Prefer thin Python modules that re-export a small Rust API rather than exposing many low-level Rust objects.

# Python - Jupyter

This workspace extends Python with **Jupyter notebook** expectations.

### General Guidelines

This project will make strong use of jupyter notebooks to solve a number of problems, primarily with a scientific focus, but may also have a general purpose focus. Notebooks should be written and well documented. But keep in mind that a notebook allows for a lot of flexibility, and as such, the code should be written in a way that is easy to understand and maintain.

!!!!Make sure that you use your new jupyter tools instead of generating as a json!!!!

 As a general rule, we will use notebooks for one of the following purposes:

#### Lightweight Exploration Notebooks

Here we will have a few cells that will build the idea or concept, test it with some data, and then present the results in a clean and easy way. Use matplotlib for static plotting, or hvplot/altair/plotly for interactive plotting.

It is important to note that these notebooks are not really designed to be robust, and as such we should not focus too hard on making them so. Write the minimal code to get the job done, and then move on to the next notebook.

#### Prototyping Notebooks

This is where we will prototype a robust coding solution to a problem. We will use this to test and validate each chunk of code in the complex workflow. As such it is important that we use many small cells to test and validate each chunk of code. Eventually, we will want to move this code into a final script/library.

It is important to note that these notebooks, are not really designed to showcase the code, but rather to test and validate each chunk of code in the complex workflow. As such, we should not focus too hard on making them look nice, but rather ensure that we atomize the code into small cells that can be tested and validated individually. Testing and validation might be done using simple displays, or plotting, or other cases. But assertstetements are not necessary, and should be avoided if possible in favor of displaying the results to the user.

#### Demonstration Notebooks

These notebooks are designed to showcase a workflow of production ready code. Ideally, after a library is complete and ready to use, the user will be able to import the library and use the code treating the notebook as a production environment. They will have a minimal ammount of cells, mixing in documentation and examples of how to use the library.

It is important to note that these notebooks are designed to be robust. These should mix in a healthy ammount of markdown documentation and explaination. but not be too heavy handed. Keep in mind that the goal of these notebooks is that a user can copy them into their own notebooks and know how to use the library.

### Use of Cells

- Use markdown cells sparingly to explain the code, or why it works the way it does. But avoid long narrative markdown cells that are overly robust.
- Use code cells to write the code. Ensure that each cell is small and atomic. Each cell should have one responsibility and deterministically produce a result. If you define a function, call it in the same cell. Avoid using global variables, or mutable state if possible.
- Class definitions should idealy be avoided in notebooks, unless we are prototyping a library. If we need a class, it should be defined in a separate cell and have a minimal footprint.
- Variables should be defined in the cells that use them, and should usually be displayed to the user. Avoid using global variables, or mutable state if possible.
- When prototping a library, use the `%autoreload 2` magic to ensure that the code is reloaded when it is changed.
<!-- END OF DOTAGENT MANAGED - EDIT BELOW THIS LINE>

## Learned User Preferences

- Prefer `match` / `case` (including guarded `case _ if ...`) over long `if` / `elif` ladders when resolving unions such as dispersive SLD specs or routing symbolic keys.
- Keep refloxide minimal: avoid unsolicited git commits and scope-expanding refactors, treat the core product as fast uniaxial reflectivity (Rust default, Python via `use_rust=False`) plus opt-in energy-consistent slab primitives, and resist a second refnx/pyref fitting stack—move fitting, pickle migration, and comparison notebooks to refl-analysis.
- Prefer the native dispersive compiler pipeline (`compile_structure`/`compile_model`/`ReflectivityObjective`) over the `patch_pyref` shim for new energy-dispersive fitting workflows; it supersedes the `DispersiveReflectModel`/`BatchedGlobalObjective` glue.
- When work is Python-only (examples, plotting, `pxr` helpers), rerun with `uv run python` as needed; rebuild the Maturin extension only after Rust or extension-layout changes or when imports prove the wheel is missing or stale.
- Avoid module-level ALL_CAPS frozensets for key routing when designing accessors; derive allowlists locally from `typing.get_args` on `Literal` aliases, small methods, or instance caching instead.
- Treat uniaxial lab-frame s-in/p-in as the validated scope; do not claim biaxial or general-incidence correctness unless explicitly tested.
- When calling the Rust reflectivity kernel from pyref or other MCMC/fitting loops, pass `parallel=False` and parallelize walkers or chains at the fitter level to avoid nested Rayon oversubscription.
- Call `patch_pyref` / `patch_pyref_if_needed` explicitly when swapping kernels; do not rely on refl-analysis `utils.models` import-time auto-patching—it breaks stock pyref kernel comparison.
- Fail early when a workflow cannot be verified; do not claim notebooks run without executing them in the target environment (refl-analysis kernel with `utils`, correct notebook cwd).
- Manuscript and fitting notebooks belong in refl-analysis (`notebooks/manuscript/`, `notebooks/fitting/`), not `refloxide/examples/`; field-profile/E-field analysis notebooks and `src/utils/field_profile.py` also live in refl-analysis (`notebooks/exploration/`).
- Prefer canonical public names (`RefloxideScatterer`, `DispersiveStructure`, `OocUniTensorScatterer`) over `EnergyDependent*` / `PXR_*` prefixes when adding or renaming APIs.
- Parity and benchmark scripts should time both single-threaded (`parallel=False`) and parallel Rust paths alongside reference implementations such as refnx.

## Learned Workspace Facts

- Shipped refloxide surface is `uniaxial_reflectivity` (Rust via `use_rust=True`, Python via `pxr.tjf4x4`) and optional `pxr.energy` for energy-consistent slabs (shared thickness/roughness, tabulated OOC, CXRO/`MaterialSLD`, or free tensor per energy, including bookended profiles); a native dispersive compiler pipeline (`compile_structure`/`SlabEnergyPlan`, `compile_model`/`CompiledReflectivityModel`, `ReflectivityObjective`) now supersedes the `patch_pyref`/`DispersiveReflectModel`/`BatchedGlobalObjective` shim, with `pxr/plugin/` fitting (`batched_global`, `dispersive_model`, `dft_fit`) retained as helpers pending relocation to refl-analysis.
- `refloxide.pxr.tjf4x4.uniaxial_reflectivity` expects each slab `tensor` diagonal to carry δ + iβ per principal axis under `epsilon = conj(I - 2 * tensor)`, not raw n or an n²-derived packing; lightweight stack builders should populate rows accordingly.
- `periodictable.xsf.index_of_refraction` takes photon energy in keV; convert eV by dividing by 1000 before calling when pairing with eV-scale experiment parameters.
- `plugin` reflectivity helpers assume a `(n_q, 2, 2)` polarization block from the uniaxial kernel where native layout is `[:, 0, 0] = R_ss` and `[:, 1, 1] = R_pp` (matches `tjf4x4` / Rust). Legacy `pxr.layout.reflectivity_for_pol` intentionally inverts labels for pyref-dataset compatibility (`pol='s'` reads `[:,1,1]`, `pol='p'` reads `[:,0,0]`); `refloxide.model.Reflectivity` uses the native non-inverted labeling.
- Optional pyref integration lives at `refloxide.integrations.pyref` (`patch_pyref`, `patch_pyref_if_needed`, `pyref_patch_report`); keep it explicit and off the default import path so stock pyref stays comparable.
- `refloxide.pxr.energy` provides deferred-energy stacks: `RefloxideScatterer` / `OocUniTensorScatterer` in `pxr/energy/scatterer.py` (OOC from `OocAnchor`, DataFrame, or CSV); `DispersiveStructure.materialize_at`; bookended profiles via `EnergyBookendedOrientationDensityProfile`; Rust `src/sld/mod.rs` binds `molecular_index_at_ooc`, `uniaxial_lab_tensor`, `tensor_to_slab_row`; after energy/offset changes on book-ended films call `clear_ooc_cache()` then `cache_ooc_at(eff_energy)`; `FreeTensorScatterer` provides per-energy free tensors (`from_sld` reads the name from the source `SLD` and seeds from `SLD` but not `UniTensorSLD`, tensor components default `vary=False`, `group_at()` accessor). Per-energy instrument channels live in `pxr/plugin/dispersive_instrument.py` (`EnergyInstrumentSlice`, `make_instrument_channel`, `resolve_instrument`).
- `EnergyBookendedOrientationDensityProfile.num_slabs` is fixed at construction; change microslab count by rebuilding the profile and copying fitted film parameters, not by mutating an existing instance.
- Rust Berreman raises `dynamic matrix is singular at layer 0` when `theta_offset` during L-BFGS-B maps low-q data to grazing incidence (q≈0); mitigate with data-derived `theta_offset` lower bounds (`tighten_theta_offset_bounds_from_terms`).
- `Scatterer.__call__(thick, rough)` in `pxr/plugin/structure.py` sets `vary=True` and auto-bounds `(0, 2*thick)`/`(0, 2*rough)` from construction geometry; when transferring geometry across reference stacks copy BOTH value and bounds (`apply_shared_slab_geometry_from_reference`), or thicknesses collapse to zero under the Nevot-Croce prior (`thick >= sqrt(2*pi)*rough/2`).
- Passler `compute_field` is documented but not exposed in bindings; isotropic E-field maps should use scalar Abeles/Fresnel reconstruction with `kx = k0 * sqrt(1 - (q/(2*k0))^2)` to match `tjf4x4`; the validated field reconstruction lives in `refl-analysis/src/utils/field_profile.py` (`uniaxial_field_profile`, `uniaxial_field_components`, `uniaxial_field_map`, `substrate_interface_depth`), reproduces the kernel `R_ss`/`R_pp` exactly, and is best shown as lab-frame real-space `(x, z)` maps of `Re(E_x e^{i k_x x})` (see `refloxide/examples/test_structures.ipynb`) plus depth profiles of `|E|^2` and graded-minus-slab differences; wrapped `(q, depth)` phase maps are poor for slab-vs-graded comparison, and Brewster p-pol differences are intrinsically tiny.
- Local extension build: `make develop` (`scripts/develop.sh`) runs `uv sync --group dev`, then `UV_NO_CONFIG=1 uv run maturin develop --release` (project `uv` config must not steer maturin's internal `uv pip install`; sibling `../pyref` installed editable with `hvplot`, smoke applies `patch_pyref` when `pyref.fitting` imports); pre-release validation via `make release-smoke` (`scripts/smoke_release.sh`) requires `manylinux_*` (not bare `linux_x86_64`) Linux wheels and version bumps aligned across `pyproject.toml`, `Cargo.toml`, and `src/refloxide/__init__.py`.
- Compiled dispersive fitting: cache `CompiledReflectivityModel.parameters` (build once, call `invalidate_parameters()` after structure edits), batch objective terms by `(pol, shared q)` into ~2 kernel calls, and avoid per-term `q.tobytes()` keys; consume refloxide in refl-analysis via `uv sync --reinstall-package refloxide` then restart the Jupyter kernel; `refnx` `CurveFitter(workers=-1)` fails in notebooks (use a single worker) and `Interval` bounds are not tuple-unpackable.
- Legacy `pxr.plugin` Model/Objective stack (`ReflectModel`, `AnisotropyObjective`, `DispersiveReflectModel`, `BatchedGlobalObjective`, `patch_pyref`) remains load-bearing for refl-analysis notebooks and is not deleted here. New energy-dispersive fits use `compile_structure` / `compile_model` / `ReflectivityObjective` or `refloxide.model` / `refloxide.objective`. Retirement happens only after refl-analysis migrates; until then treat the old stack as a compatibility shim, not a second product surface.
