# Workspace split design (stage 1)

Status: draft for user approval. Not committed. Resume notes live in `tmp/workspace-split/HANDOFF.md`.

## Intent

Refloxide is a single Cargo package whose Rust code, PyO3 bindings, and Python package share one tree. The next features (a Rust optics-source trait, tabulated and peak-sum sources built on the `lineshapes` crate) need a standalone place for the optics layer, which does not exist. This stage restructures the repository into a Cargo workspace of coherent crates with a single PyO3 bindings crate, so later stages have clean seams. This stage changes no behavior, no numerics, and no public Python API.

## Goals

- A Cargo workspace with five crates: `refloxide-core`, `refloxide-optics`, `refloxide-tmm`, `refloxide-gpu`, `refloxide-py`.
- Only `refloxide-py` depends on PyO3 and numpy. All other crates build and test as plain Rust.
- The Python package moves from `src/refloxide/` to `python/refloxide/`, built by maturin from `crates/refloxide-py`.
- The Python import surface (`refloxide`, `refloxide.tmm`, `refloxide.optics`, `refloxide.model`, `refloxide.objective`, `refloxide.data`, `refloxide.python.*`, deprecated `refloxide.pxr`) and the compiled module name `refloxide.rust` are unchanged.
- Numerical results are bit-identical to the pre-split build on the existing test suite, and benchmark times do not regress beyond noise.
- The kernel dependency cycle (`Real` defined in `kernel` but imported back by `c4x4` and `math`) is removed.

## Non-goals and anti-goals

- No new features: no `OpticalSource` trait, no table type in Rust, no `lineshapes` dependency (stages 2 and 3), no step-edge fitting.
- No algorithm changes, no f32/f64 generic expansion, no renames of public Rust or Python symbols, no pxr cleanup, no model.py refactor.
- Anti-goal: a split that compiles but is slower, grows the wheel, or changes a result. That is a failure even if tests pass.
- Anti-goal: configuration creep. No new Cargo features beyond what exists, and no feature may alter numerics or data layout.

## Target layout

```
Cargo.toml              [workspace] resolver = "2" (edition 2021 is retained), [workspace.package], [workspace.dependencies], [profile.*]
pyproject.toml          maturin: manifest-path = "crates/refloxide-py/Cargo.toml", python-source = "python", module-name = "refloxide.rust"
crates/
  refloxide-core/
  refloxide-optics/
  refloxide-tmm/
  refloxide-gpu/
  refloxide-py/
python/refloxide/       moved verbatim from src/refloxide/
```

Dependency direction: `core` <- `optics`, `core` <- `tmm`, `core`+`tmm` <- `gpu`, all <- `py`. `optics` and `tmm` do not depend on each other except where `bookended` needs `optics` (see below), so `tmm` depends on `optics`.

## Crate contents (current source to destination)

- `refloxide-core`: `error.rs` (`RefloxideError`, `Result`, without the PyO3 conversion), the `Real` trait and its f32/f64 impls, `LayerCoeffs<T>`, `c4x4`, `math` (`exact_inv_4x4`, `exact_inv_4x4_generic`), and the items currently in `uniaxial.rs` that other modules reach into: `Layer`, `HC_EV_ANGSTROM`, `wavenumber`, `layer_coeffs`, `jvp_coeffs`, `validate_points`, `validate_batch`, and the output structs (`UniaxialOutput`, `UniaxialBatchOutput`, `PointsJvpOutput`, `PointsTangents`). Items needed across crates become `pub` with documentation.
- `refloxide-optics`: `optics.rs` and `sld/` unchanged in behavior (`interpolate_ooc_linear`, `lab_diagonal_uniaxial`, `lab_diagonal_uniaxial_batch`, `pack_diagonal_tensors`, `isotropic_tensor`, `molecular_index`, `molecular_index_at_ooc`, `uniaxial_lab_tensor`, `isotropic_lab_tensor`, `tensor_to_slab_row`). Depends only on `num-complex` (and core for shared types if any). This is the crate stage 2 grows.
- `refloxide-tmm`: `kernel.rs` and `kernel/{rmatrix,general,tangent}.rs`, the CPU drivers in `uniaxial.rs` (`uniaxial_reflectivity`, `uniaxial_reflectance`, `uniaxial_reflectivity_batch`, `uniaxial_reflectivity_points`, `uniaxial_reflectivity_points_jvp`), `general_stack.rs`, and `bookended.rs`. Owns the `parallel` rayon logic.
- `refloxide-gpu`: `gpu/` including WGSL shaders, the `uniaxial_reflectivity_batch/_points/_points_jvp` GPU functions, and the examples that need it. wgpu, pollster, and bytemuck are dependencies of this crate only.
- `refloxide-py`: `python.rs` as the `#[pymodule] rust`, pyo3 and numpy dependencies, numpy marshalling, GIL release, device selection, and a local conversion from `RefloxideError` to `PyErr` (a function or newtype, because the orphan rule forbids the `From` impl outside `core`).

## Decisions made by the user

- Split first, before the optics trait and peak-sum work.
- GPU is always part of the Python wheel. `refloxide-gpu` is a normal (non-optional) dependency of `refloxide-py`; wgpu stays out of the core, optics, and tmm builds.
- `lineshapes` (later stage) is a direct crates.io Rust dependency (0.1.0 released), not reached through PyO3.
- Performance is the priority: no regression is acceptable.
- No unsolicited commits.

## Constraints

- Preserve exactly: module name `refloxide.rust`, the 12 registered PyO3 function names and signatures (including `parallel` defaults and `device="cpu"`), kernel packing conventions (`tensor` holds delta + i*beta, `epsilon = conj(I - 2*tensor)`, `[:,0,0]=R_pp`, `[:,1,1]=R_ss`), and edition 2021 and Python >= 3.12.
- Profiles move to the workspace root unchanged: `[profile.release]` has only `strip = "none"` and `[profile.perf]` inherits release with `opt-level = 3`, `codegen-units = 1`. Neither sets `lto`. Crate boundaries can block cross-crate inlining under the default release profile, so small hot functions that cross crates (kernel entry points, `wavenumber`, `layer_coeffs`) are marked `#[inline]` where the benchmark shows a loss. The user decided to add `lto = "fat"` and `codegen-units = 1` to `[profile.release]` as a separate final step (step 7), after the split is verified bit-identical under the original profile.
- Public Rust items moved across crates keep `///` documentation; each new crate has module-level docs stating responsibility and non-responsibilities.
- `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` stay clean.
- Static dispatch only. No `dyn` is introduced.

## Verification and completion criteria

Baseline before touching anything (record in `tmp/workspace-split/HANDOFF.md`): `uv run pytest`, `cargo test --all-targets`, the benchmark gate (`tests/test_bench_gates.py` and `examples/gpu_backend_bench.py`) with timings for single-threaded and parallel paths, wheel size, and release build time.

Done when, on the post-split tree:
- `cargo check --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` pass; the Rust-only crates build without a Python interpreter.
- `make develop` builds and smoke-imports; `uv run pytest` matches the baseline pass set.
- Benchmark timings are within measurement noise of baseline; wheel size is not larger than baseline beyond noise.
- CI workflows (`ci.yml`, `release.yml`, `pre-merge.yml`) are updated for the workspace (`--workspace`, example paths, the naga dev-dependency for `gpu_general_check`, maturin manifest path, sdist inclusion of all crates) and validated by inspection, plus a local `maturin build` and `maturin sdist` check.
- No behavior change: no Python test edits other than path updates.

## Staging inside this stage

Each step leaves the tree compiling and tests passing; stop only at a step boundary.

1. Convert the root to a workspace with a single member that is today's crate unchanged; verify.
2. Extract `refloxide-core` (breaking the `Real`/`c4x4`/`math` cycle, moving the `uniaxial.rs` shared items); verify.
3. Extract `refloxide-optics`; verify.
4. Extract `refloxide-tmm`; verify.
5. Extract `refloxide-gpu`; verify.
6. Extract `refloxide-py`, move the Python package to `python/`, update maturin, scripts, Makefile, and CI; verify wheel and sdist.
7. Enable `lto = "fat"` and `codegen-units = 1` in `[profile.release]`. Rerun the golden-output comparison against the step-6 outputs, which are bit-identical to the pre-split baseline. For every output array report the fraction of elements that differ bitwise and the distribution of deviation in ULPs and relative error (max, 99th percentile, mean), separately for CPU single-thread, CPU parallel, and each kernel (uniaxial, general, bookended, points, JVP). Acceptance: deviations are confined to rounding-level (a few ULP, relative error at or below the f64 kernel's existing test tolerances), show no systematic bias (mean signed error indistinguishable from zero), and no physical observable moves beyond existing tolerances. Anything beyond that is reported to the user and the profile is not kept without their decision. Also record build time, wheel size, and benchmark timings before and after.

## Left to the builder

Exact module file organization within each crate; which items need `#[inline]`; naming of the `PyErr` conversion helper; whether `sld` merges into `optics` as a module; removal of the apparently unused `physical_constants` dependency (grep first); the doc comment in `kernel.rs` that wrongly claims no threading (fix as part of the move); whether `rust.cpython-*.so` artifacts in the source tree are cleaned or gitignored; workspace-level lint configuration.

## Open items and calls I made

- Decided by user: fat LTO and `codegen-units = 1` in release, validated by step 7 below.
- Call (mine): bit-identity is checked with a throwaway golden-output script (not a committed test) that captures baseline CPU outputs (single-thread and parallel, uniaxial, general, bookended, points, JVP) before step 1 and compares bitwise after each step; GPU outputs are compared to the existing tolerances. "Within noise" for benchmarks means the thresholds already in `tests/test_bench_gates.py` and `examples/gpu_backend_bench.py`; wheel size must not grow by more than 1 percent.
- Call (mine): `refloxide-py` drops the `python` and `gpu` Cargo features (python is its reason to exist, gpu is always linked). Examples with `required-features = ["gpu"]` move to `refloxide-gpu/examples` with no feature gate; `gpu_general_check` and its naga dev-dependency move there too. Rust-only examples (`general_reference`, `kernel_precision`) move to `refloxide-tmm/examples`. No facade crate named `refloxide`.
- Call (mine): `gpu` depends on `tmm` for `GeneralLayer` and the kernel; `GeneralLayer` and `PointResult` stay in `tmm`.
- Call (mine): unit tests move with their code; intra-doc links are rewritten; items made `pub` get docs.
- Call (mine): sdist is verified with `maturin sdist` and a build from the unpacked sdist; `include`, ruff, ty, and pytest paths that reference `src/refloxide` are updated to `python/refloxide`. refl-analysis consumption is unaffected (it installs the wheel) and is not edited here.

- Call (mine): `tmm` depends on `optics` because `bookended` uses the OOC interpolation helpers.
- Call (mine): resolver "2" with edition 2021 retained; no MSRV bump.
- Call (mine): benchmark baseline recorded before step 1 rather than inferred from CI.
