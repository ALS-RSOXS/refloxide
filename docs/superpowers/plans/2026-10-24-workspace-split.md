# Workspace Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Resume state lives in `tmp/workspace-split/HANDOFF.md`; update its Progress log at the end of every task.

**Goal:** Restructure refloxide into a Cargo workspace of five crates with one PyO3 bindings crate and a `python/` package directory, with no change to behavior, numerics, or the public Python API, then enable fat LTO with validated numerics.

**Architecture:** Extract crates bottom-up (`core`, `optics`, `tmm`, `gpu`, `py`), keeping the tree compiling and the golden-output comparison bit-identical after each extraction. LTO is a separate final task so its effect is isolated.

**Tech Stack:** Rust 2021, PyO3 0.28.3, numpy 0.28, nalgebra 0.33, rayon 1.10, wgpu 30.0.1, maturin, uv, pytest.

**Spec:** `docs/superpowers/specs/2026-10-24-workspace-split-design.md`

## Global Constraints

- Public Python API and compiled module name `refloxide.rust` unchanged; the 12 registered PyO3 function names, signatures, `parallel` defaults, and `device="cpu"` unchanged.
- Kernel packing unchanged: tensor holds delta + i*beta, `epsilon = conj(I - 2*tensor)`, `[:,0,0]=R_pp`, `[:,1,1]=R_ss`.
- Edition 2021, resolver "2", Python >= 3.12, no MSRV bump, no new Cargo features; `refloxide-py` has no `python`/`gpu` features and always links `refloxide-gpu`.
- Only `refloxide-py` depends on pyo3/numpy; wgpu/pollster/bytemuck only in `refloxide-gpu`.
- Static dispatch only; no `dyn`; no new f32/f64 generics.
- Items made `pub` across crates carry `///` docs; each crate has module-level docs on responsibility and non-responsibilities; no `unwrap` in library code; `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` clean.
- Profiles move to the workspace root unchanged until Task 8: release `strip = "none"`; perf inherits release with `opt-level = 3`, `codegen-units = 1`.
- No unsolicited commits and no new committed tests (user policy: verify by golden-output script and existing suite). The golden script lives in `tmp/workspace-split/` (gitignored). Verification checkpoints replace commit steps.
- Rebuild the extension with `make develop` after Rust or layout changes.

## Review Focus

- Rust-only crates build with no Python interpreter present (`cargo check -p refloxide-core -p refloxide-optics -p refloxide-tmm -p refloxide-gpu`, with `PYO3_PYTHON` unset).
- `parallel=False` and `parallel=True` outputs remain identical to baseline for each kernel.
- Error mapping: each `RefloxideError` variant still raises the same Python exception type and message after the `From<RefloxideError> for PyErr` impl is replaced.
- Wheel contents: `maturin build` wheel contains `refloxide/rust*.so`, WGSL shaders, `py.typed`, and `rust.pyi`; the sdist builds from an unpacked tree.
- Deprecated `refloxide.pxr` and `refloxide.python.*` still import after the directory move.

---

### Task 1: Baseline and golden outputs

**Files:**
- Create: `tmp/workspace-split/golden.py`, `tmp/workspace-split/golden_baseline.npz`, `tmp/workspace-split/BASELINE.md`

**Interfaces:**
- Produces: `golden.py capture <out.npz>` and `golden.py compare <a.npz> <b.npz>`. `compare` reports, per array, the fraction of bitwise-differing elements and ULP/relative-error statistics (max, p99, mean, mean signed error). Exit code 1 if any element differs when run with `--exact`.

- [ ] **Step 1:** Write `golden.py` using a fixed `numpy.random.default_rng(seed)` stack covering `uniaxial_reflectivity`, `_batch`, `_points`, `_points_jvp`, `general_reflectivity`, and `bookended_uniaxial_reflectivity`, each with `parallel=False` and `parallel=True` (and `device="cpu"` only). Arrays saved as float64/complex128.
- [ ] **Step 2:** Run `make develop && uv run python tmp/workspace-split/golden.py capture tmp/workspace-split/golden_baseline.npz`. Run it twice into two files and confirm `compare --exact` reports zero differences (proves the capture is deterministic, including the parallel path).
- [ ] **Step 3:** Record in `BASELINE.md`: `uv run pytest` pass/fail summary, `cargo test --all-targets` summary, `tests/test_bench_gates.py` result, `examples/gpu_backend_bench.py` timings (single-thread and parallel), wheel size from `uv run maturin build --release`, release build time.
- [ ] **Step 4:** Update HANDOFF.md Progress log. Expected: baseline recorded, tree untouched.

### Task 2: Workspace root with the existing crate as sole member

**Files:**
- Modify: `Cargo.toml`
- Create: none (the crate stays at the repository root as `.` plus `[workspace]`)

**Interfaces:**
- Produces: a `[workspace]` with `resolver = "2"`, `[workspace.package]` (version, edition, license, repository), and `[workspace.dependencies]` for nalgebra, num-complex, num-traits, thiserror, rayon, pyo3, numpy, wgpu, pollster, bytemuck, approx, naga. Later tasks reference these as `{ workspace = true }`.

- [ ] **Step 1:** Add `[workspace] members = ["."]` and the shared tables; keep the package definition and profiles unchanged.
- [ ] **Step 2:** `cargo check --all-targets` and `cargo check --all-targets --features gpu` pass.
- [ ] **Step 3:** `make develop` then `golden.py capture` to `golden_t2.npz`; `compare --exact` against baseline reports zero differences.

### Task 3: Extract `refloxide-core`

**Files:**
- Create: `crates/refloxide-core/Cargo.toml`, `crates/refloxide-core/src/{lib.rs,error.rs,real.rs,c4x4.rs,math/mod.rs,math/inv4.rs,layer.rs,validate.rs,output.rs}`
- Modify: root `Cargo.toml` (members), `src/lib.rs`, `src/kernel.rs`, `src/kernel/*.rs`, `src/uniaxial.rs`, `src/general_stack.rs`, `src/bookended.rs`, `src/gpu/mod.rs`, `src/python.rs`, `src/error.rs` (deleted, re-exported)

**Interfaces:**
- Produces (all `pub`, in `refloxide_core`): `Real` (+ f32/f64 impls), `LayerCoeffs<T>`, `Layer {thickness, sld_re, sld_im, sigma}`, `HC_EV_ANGSTROM: f64`, `wavenumber(energy_ev: f64) -> f64`, `layer_coeffs`, `jvp_coeffs`, `validate_points`, `validate_batch` (signatures unchanged from `uniaxial.rs`), `UniaxialOutput`, `UniaxialBatchOutput`, `PointsJvpOutput`, `PointsTangents`, `RefloxideError`, `Result<T>`, `exact_inv_4x4`, `exact_inv_4x4_generic`, and the `c4x4` helpers (`Mat4`, `mul`, `hadamard`, `propagation_diag`, `fill_w`, `fused_interface_kernel`).
- `RefloxideError` has no PyO3 impl here. Add `pub fn to_pyerr(e: RefloxideError) -> PyErr` in the bindings code (still `src/python.rs` until Task 7) and replace the `From`-based `?` conversions with `.map_err(to_pyerr)`.
- Consumes: nothing.

- [ ] **Step 1:** Create the crate with dependencies nalgebra, num-complex, num-traits, thiserror (workspace). Move `Real` out of `kernel.rs` into `real.rs` first, so `c4x4` and `math` import from within the crate; this removes the cycle.
- [ ] **Step 2:** Move the `uniaxial.rs` shared items per the interface list; mark `pub`, add `///` docs and a module-level doc on every file. Update intra-doc links.
- [ ] **Step 3:** In the root crate replace local definitions with `use refloxide_core::...` and re-export where public paths existed (`pub use`), so no path changes for external callers. Delete `impl From<RefloxideError> for PyErr` and add `to_pyerr`.
- [ ] **Step 4:** `cargo check -p refloxide-core` with no Python; `cargo clippy --workspace --all-targets -- -D warnings`; `cargo test --workspace`.
- [ ] **Step 5:** `make develop`; golden capture to `golden_t3.npz`; `compare --exact` vs baseline: zero differences. Run `uv run pytest tests/test_rust_batch.py -q` and a small script triggering each `RefloxideError` variant to confirm exception type and message match the baseline (see Review Focus).
- [ ] **Step 6:** Update HANDOFF.md.

### Task 4: Extract `refloxide-optics`

**Files:**
- Create: `crates/refloxide-optics/Cargo.toml`, `crates/refloxide-optics/src/{lib.rs,optics.rs,sld.rs}`
- Modify: root `Cargo.toml`, `src/lib.rs`, `src/python.rs`, `src/bookended.rs`

**Interfaces:**
- Produces (`pub`, in `refloxide_optics`): `interpolate_ooc_linear`, `lab_diagonal_uniaxial`, `lab_diagonal_uniaxial_batch`, `pack_diagonal_tensors`, `isotropic_tensor`, `molecular_index`, `molecular_index_at_ooc`, `uniaxial_lab_tensor`, `isotropic_lab_tensor`, `tensor_to_slab_row`; signatures unchanged. `sld` stays a module of this crate.
- Consumes: `refloxide_core` only if shared types are needed (expected: none; depends only on num-complex).

- [ ] **Step 1:** Move `src/optics.rs` and `src/sld/mod.rs` with their `#[cfg(test)]` modules; add crate and module docs.
- [ ] **Step 2:** Repoint `bookended.rs` and `python.rs` to `refloxide_optics`.
- [ ] **Step 3:** Run the Task 3 verification set (clippy, `cargo test --workspace`, `make develop`, golden `compare --exact`, `tests/test_optical_constants_cache.py`, `tests/test_composable_oc.py`).
- [ ] **Step 4:** Update HANDOFF.md.

### Task 5: Extract `refloxide-tmm`

**Files:**
- Create: `crates/refloxide-tmm/Cargo.toml`, `crates/refloxide-tmm/src/{lib.rs,kernel.rs,kernel/*.rs,uniaxial.rs,general_stack.rs,bookended.rs}`, `crates/refloxide-tmm/examples/{general_reference.rs,kernel_precision.rs}`
- Modify: root `Cargo.toml`, `src/lib.rs`, `src/python.rs`, `src/gpu/mod.rs`

**Interfaces:**
- Produces (`pub`, in `refloxide_tmm`): everything currently public in `kernel`, `uniaxial` (the five driver functions), `general_stack::general_reflectivity`, and the `bookended` public items; `GeneralLayer` and `PointResult` stay here.
- Consumes: `refloxide_core`, `refloxide_optics` (for `bookended`); rayon (workspace). Fix the `kernel.rs:11` doc comment to say the flat solvers use rayon.

- [ ] **Step 1:** Move modules and their unit tests; add crate docs. Apply `#[inline]` to small hot functions crossing the crate boundary only if Step 3 shows a regression (do not add speculatively).
- [ ] **Step 2:** Repoint `gpu` and `python.rs`. Move the two Rust-only examples and fix their `use` paths.
- [ ] **Step 3:** Verification set plus the benchmark gate: run `tests/test_bench_gates.py` and `examples/gpu_backend_bench.py`; compare timings to `BASELINE.md`. A regression beyond the existing gate thresholds means add `#[inline]` to the offending functions and re-measure; report if unresolved.
- [ ] **Step 4:** Update HANDOFF.md.

### Task 6: Extract `refloxide-gpu`

**Files:**
- Create: `crates/refloxide-gpu/Cargo.toml`, `crates/refloxide-gpu/src/lib.rs` (from `src/gpu/mod.rs`), shader files alongside, `crates/refloxide-gpu/examples/{gpu_recursive,gpu_df64,gpu_df64_bench,gpu_general,gpu_general_check}.rs`
- Modify: root `Cargo.toml`, `src/lib.rs`, `src/python.rs`

**Interfaces:**
- Produces: `refloxide_gpu::{uniaxial_reflectivity_batch, uniaxial_reflectivity_points, uniaxial_reflectivity_points_jvp}` with unchanged signatures; the `include_str!` WGSL paths remain valid relative to the new file.
- Consumes: `refloxide_core`, `refloxide_tmm`; wgpu, pollster, bytemuck (workspace). naga is a dev-dependency here for `gpu_general_check`. No `gpu` feature or `required-features` remain.

- [ ] **Step 1:** Move the module, shaders, and examples; fix `include_str!` paths.
- [ ] **Step 2:** `cargo check -p refloxide-gpu` with no Python; `cargo run -p refloxide-gpu --example gpu_general_check` passes (naga validation).
- [ ] **Step 3:** Verification set including `tests/test_gpu_precision.py`; GPU golden outputs compared to baseline within the existing tolerances (not bitwise).
- [ ] **Step 4:** Update HANDOFF.md.

### Task 7: Extract `refloxide-py` and move the Python package

**Files:**
- Create: `crates/refloxide-py/Cargo.toml`, `crates/refloxide-py/src/lib.rs` (from `src/python.rs`, `[lib] name = "rust"`, `crate-type = ["cdylib"]`)
- Move: `src/refloxide/` to `python/refloxide/` (use `git mv`; do not commit; delete or gitignore `rust.cpython-*.so` artifacts)
- Modify: root `Cargo.toml` (remove the root package, leave a virtual workspace; move `[profile.*]` and example entries), `pyproject.toml` (`manifest-path = "crates/refloxide-py/Cargo.toml"`, `python-source = "python"`, `include`, ruff, ty, pytest paths, drop `features`), `scripts/develop.sh`, `Makefile`, `.github/workflows/{ci,release,pre-merge}.yml`, `scripts/smoke_release.sh`, `hooks.py`/docs references to `src/refloxide`
- Delete: `src/lib.rs` and the now-empty `src/` Rust tree

**Interfaces:**
- Produces: module `refloxide.rust` with the same 12 functions and signatures; `to_pyerr` helper retained; `rust.pyi` stub unchanged and shipped.
- Consumes: all four lower crates; pyo3 and numpy (workspace).

- [ ] **Step 1:** Create the crate; remove the `cfg(not(feature="gpu"))` stubs and `cfg(feature = ...)` gates since gpu is always linked.
- [ ] **Step 2:** Move the Python package and update every path listed above. CI: `cargo ... --workspace`, example paths, naga check job points at `refloxide-gpu`, maturin invocations use the new manifest path, sdist job asserts all crates are included.
- [ ] **Step 3:** `make develop`; `golden.py capture`; `compare --exact` vs baseline (CPU, bitwise); full `uv run pytest` matches the baseline pass set; `uv run python -W error::DeprecationWarning -c "import refloxide.python.tmm"` and `import refloxide.pxr` still work (the latter emits the expected DeprecationWarning).
- [ ] **Step 4:** `uv run maturin build --release` and `uv run maturin sdist`; unpack the sdist into a fresh empty directory, build it there, and import the result. Check wheel contents (Review Focus) and size versus baseline (limit +1 percent).
- [ ] **Step 5:** `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `uv run ruff check .`, `uv run ty check`. Update HANDOFF.md.

### Task 8: Fat LTO validation

**Files:**
- Modify: root `Cargo.toml` (`[profile.release]` add `lto = "fat"`, `codegen-units = 1`)
- Create: `tmp/workspace-split/LTO_REPORT.md`

**Interfaces:**
- Consumes: `golden.py compare` (statistics mode) and the Task 7 golden output as reference.

- [ ] **Step 1:** Rebuild with `make develop`, capture `golden_lto.npz`, run `golden.py compare golden_t7.npz golden_lto.npz` (non-exact) per kernel and per `parallel` setting.
- [ ] **Step 2:** Accept only if deviations are rounding-level (a few ULP, relative error at or below existing test tolerances), mean signed error is indistinguishable from zero (state the test used and its result), and `uv run pytest` matches baseline. Otherwise stop, set the profile back, report to the user.
- [ ] **Step 3:** Record build time, wheel size, and benchmark timings before and after in `LTO_REPORT.md`; update HANDOFF.md and report the numbers to the user.

---

## Self-review notes

- Spec coverage: five crates (Tasks 3-7), cycle removal (Task 3), PyErr orphan fix (Task 3), `python/` move (Task 7), CI/maturin/sdist (Task 7), baseline and bit-identity (Tasks 1-7), LTO validation (Task 8), benchmark gate (Tasks 5, 8), `physical_constants` check: add to Task 3 Step 1 as "grep; drop from the workspace dependencies if unused".
- Untouched by design: pxr cleanup, optics trait, lineshapes (later stages).
