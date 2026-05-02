

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

## Learned User Preferences

- Prefer no Typer and no user-facing CLI or `[project.scripts]` entrypoints unless the user explicitly requests them.
- When the user supplies an explicit list of paths to stage or commit, restrict `git add` and the commit to that list only; avoid staging unrelated changes.
- Treat the core library as a **pure functional** reflectivity generator: callers supply stratified layers and query reflected intensity (or related outputs) for incident angle and wavelength; support **vectorized** inputs and outputs while keeping execution **single-threaded** so the stack composes cleanly with other libraries.
- When changing MkDocs layout or CSS, align documentation styling with **ALSComputing** org dev-standard references (for example the shared ALS dev-standards docs) rather than one-off ad hoc rules.

## Learned Workspace Facts

- Stratified film stacks are modeled with explicit semi-infinite fronting and backing media, interior film layers, and per-interface roughness with one value per interface boundary (`layers.len() + 1`).
- Optical constants are intended to support **full 3x3 complex** dielectric or index-of-refraction tensors (not scalar-only layer fields); conversions among common parameterizations (for example n+ik, 1-delta+i*beta, f0+f'+if'', scattering-length density, susceptibility chi, permittivity epsilon) belong in a dedicated optics/index module rather than being hard-coded as a single `f64` per layer.
- The docs site is MkDocs with the **mkdocs-rsoxs** theme (`theme.name: rsoxs`). Math uses **`pymdownx.arithmatex`** with **`generic: true`** so the theme injects KaTeX; set **`theme.katex_options`** to a non-empty dict (for example `throwOnError: false`) because an empty dict is falsy in the theme’s KaTeX template and breaks `renderMathInElement` options. Code fences use **`pymdownx.highlight`** with **`css_class: codehilite`** to match the theme’s Pygments CSS. Optional branding: **`theme.icon`** / **`theme.icon_light`** (paths under `docs/` if overridden); defaults ship in the theme package. Any **site-specific CSS** files must be listed under **`extra_css`** in `mkdocs.yml` or they are not loaded (the theme does not automatically import everything under `docs/stylesheets/`).
- `docs/theory` is organized in the left nav in grouped subsections aligned with `docs/theory/overview.md` (Narrative context, Core pipeline, Roughness models).
