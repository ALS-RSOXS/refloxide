# Lineshape sources and composable depth channels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a peak-sum optical source (`lineshapes` peaks) whose parameters can be scalars or depth fields, plus shared-shape level fields, derived fields and mix-level density on `DepthProfile`, and document the three-material gradient as a worked example.

**Architecture:** Rust `refloxide-optics` gains a feature-gated `peak_sum_molecular` that evaluates unit-amplitude `lineshapes` peaks, once for scalar parameters and once per slice for depth-varying ones, and accumulates molecular `n_xx`, `n_zz`. A Python `PeakSumSLD` scatterer (same contract as `UniTensorSLD`) calls it through `refloxide.optics`. New field shapes with per-channel levels, a `Derived` field and a `density_scale` keyword extend the existing `profiles.py`/`DepthProfile`; existing keywords and fields are untouched.

**Tech Stack:** Rust 2021 workspace (`lineshapes` 0.1.0 from crates.io, PyO3 0.28, num-complex), Python >= 3.12, refnx `Parameter`/`Parameters`, numpy/scipy, pytest, ruff, ty, mkdocs.

**Spec:** `docs/superpowers/specs/2026-10-24-lineshape-channels-design.md`

## Global Constraints

- Additive only: no removal or rename of current public behavior; existing inputs give unchanged results, to rounding, on the existing suite. Report any deviation instead of accepting it.
- `lineshapes` stays unchanged and is a direct crates.io dependency (`0.1.0`) of `refloxide-optics` behind feature `lineshapes`; no PyO3 hop through `lineshapes._core`. The feature may not change numerics or struct layout.
- Kernel packing convention: tensor diagonals hold `delta + i*beta`; `epsilon = conj(I - 2*tensor)`.
- Angles are radians everywhere in code (as `UniTensorSLD.rotation` already is); the docs example converts from degrees with `np.deg2rad`.
- Orientation channel is `gamma` (tilt of the molecular normal from the surface normal): 0 face-on, pi/2 edge-on, magic angle `arccos(1/sqrt(3))` (about 54.7 degrees) amorphous. Peak TDM tilt `theta` is measured from the molecular normal. Peak height is the height itself (no multipliers).
- Peak parameter that is a single number (frozen or fit) is constant over its slab: evaluated once per energy, never per microslab. Only depth-field-bound parameters are evaluated per microslab. No Python loop over slices in any new code.
- Static dispatch in Rust hot paths (no `dyn` in the energy loop); `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` stay clean; `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check` stay clean.
- Public items get documentation (numpy-style docstrings in Python, `///` in Rust).
- Testing policy: no new committed tests except (a) the peak-sum versus tabulated parity test, (b) Rust unit tests that compare two code paths, (c) one end-to-end run of the docs example. Everything else is verified by a scratch run (`uv run python -c ...`), observed and discarded.
- Conventional commits on the feature branch. End every commit message with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

Each line is verified by a scratch run inside the owning task, not a committed test (testing policy).

1. A shared shape used by several channels contributes its shape parameters once to the fit (Task 4); expected: one varying parameter per tau, not one per channel.
2. A peak with an invalid parameter (width <= 0, eta outside [0, 1], non-finite) raises `ValueError` naming the parameter instead of returning NaN (Task 1, 2).
3. A peak parameter bound to a depth field, evaluated outside a `DepthProfile` (`tensor_at`), uses the field's bulk pin rather than failing (Task 3).
4. A `PeakSumSLD` with no peaks returns the baseline alone, and TDM tilt of exactly 0 and pi/2 and gamma of exactly 0 and pi/2 give finite tensors (Task 3).
5. A `DepthProfile` with `n_slabs=1`, and a thickness-0 level field evaluation, give finite shapes and values (Task 4, 5).

## File Structure

- `Cargo.toml` (workspace deps), `crates/refloxide-core/src/error.rs` (new error variant)
- `crates/refloxide-optics/Cargo.toml`, `src/lib.rs`, new `src/peaks.rs`: Rust peak evaluation and accumulation
- `crates/refloxide-py/Cargo.toml`, `src/lib.rs`: binding `peak_sum_molecular`
- `python/refloxide/rust.pyi`, `python/refloxide/optics.py`: stub and re-export
- new `python/refloxide/lineshapes.py`: `Peak`, `PeakSumSLD`, `tdm_fractions`, `MAGIC_ANGLE`
- new `python/refloxide/fields.py`: `Shape` and families, `LevelField`, `Derived`
- `python/refloxide/profiles.py`: `DepthProfile` extensions
- new `examples/lineshape_three_material.py`, `docs/guides/building_structures/peak_gradients.md`, `mkdocs.yml`
- tests: new `tests/test_peak_sum_source.py`, `tests/test_lineshape_example.py`

---

### Task 0: Worktree and baseline

**Files:** none in the product; creates `tmp/lineshape-channels/BASELINE.md` (gitignored).

**Interfaces:** Produces the isolated worktree used by every later task.

- [ ] **Step 1:** Invoke `superpowers:using-git-worktrees`; create worktree and branch `feat/lineshape-channels` from `main` (not from `ci/consolidate-workflows`).
- [ ] **Step 2:** The spec and this plan are untracked in the original checkout. Copy both into the same paths in the worktree and commit them: `docs: add lineshape-channels design and plan`.
- [ ] **Step 3:** In the worktree run `make develop`, then `uv run pytest -q` and `cargo test --workspace`. Record pass counts and wall times in `tmp/lineshape-channels/BASELINE.md`. Expected: the same pass set as `main`; if anything fails on a clean `main` worktree, stop and report.
- [ ] **Step 4:** Run `uv run pytest tests/test_bench_gates.py -q` and record its numbers in the same file (the before-numbers for Task 7).

### Task 1: Rust peak evaluation (`peak_sum_molecular`)

**Files:**
- Modify: `Cargo.toml` (add `lineshapes = { version = "0.1.0", default-features = false }` to `[workspace.dependencies]`), `crates/refloxide-optics/Cargo.toml`, `crates/refloxide-optics/src/lib.rs`, `crates/refloxide-core/src/error.rs`
- Create: `crates/refloxide-optics/src/peaks.rs`

**Interfaces:**
- Consumes: `lineshapes::{Model, Component}` (`Model::new().try_add(Component::Gaussian { e0, width, amp })`, `Model::eval_into(&[f64], &mut [Complex64])`; amplitude 1 gives the unit line, result is `delta + i*beta`).
- Produces (feature `lineshapes`, exported from `refloxide_optics`):

```rust
pub enum LineKind { Gaussian, Lorentzian, Voigt, PseudoVoigt }
impl LineKind {
    pub fn parse(name: &str) -> Option<Self>;   // "gaussian" | "lorentzian" | "voigt" | "pseudo_voigt"
    pub fn n_shape(self) -> usize;              // 2, 2, 3, 3
}
pub struct PeakSpec<'a> {
    pub kind: LineKind,
    pub shape: &'a [f64],   // [e0, width] | [e0, gamma] | [e0, sigma, gamma] | [e0, fwhm, eta]; len n_shape (broadcast) or n_slices*n_shape (row-major slice, param)
    pub height: &'a [f64],  // len 1 (broadcast) or n_slices
    pub f_xx: &'a [f64],    // len 1 or n_slices
    pub f_zz: &'a [f64],    // len 1 or n_slices
}
pub fn peak_sum_molecular(
    peaks: &[PeakSpec], n_slices: usize, energies: &[f64],
    out_xx: &mut [Complex<f64>], out_zz: &mut [Complex<f64>],   // len n_slices * energies.len(), row-major (slice, energy)
) -> refloxide_core::Result<()>;
```

and a new core variant `RefloxideError::InvalidParameter(String)` ("invalid peak parameter: ...").

- [ ] **Step 1: Write failing Rust unit tests in `peaks.rs`** (testing-policy exception: two independent paths). `gaussian_matches_lineshapes_model`: one Gaussian (`e0=284.5, width=0.4`, height 2.0, `f_xx=1.5, f_zz=0.0`) over 50 energies in `[283, 286]`; `out_xx` equals `Model::new().gaussian(284.5, 0.4, 3.0).eval(&energies)` to `1e-12` relative, `out_zz` is exactly zero. `broadcast_equals_per_slice`: the same peak with `shape` of length `n_shape` versus the row repeated for 4 slices gives bit-identical output.
- [ ] **Step 2:** Run `cargo test -p refloxide-optics --features lineshapes peaks` and confirm FAIL (items undefined).
- [ ] **Step 3:** Implement per the Interfaces. Zero the outputs, then per peak: validate by building the `Component` with amp 1 and calling `Component::validate`, mapping the message to `InvalidParameter`; for broadcast shape evaluate the unit line once and reuse it for every slice, otherwise evaluate per slice; accumulate `height[s] * f_xx[s] * line` into `out_xx` and likewise `f_zz` into `out_zz`; length mismatches give `InvalidShape`. Add the optional `lineshapes` dependency and the feature in the optics `Cargo.toml`, add `refloxide-core` as a dependency of optics, and gate `mod peaks` plus the re-exports on the feature.
- [ ] **Step 4:** Run `cargo test -p refloxide-optics --features lineshapes` and `cargo test -p refloxide-optics` (no feature, must still pass). Expected: PASS both.
- [ ] **Step 5:** Review Focus 2 (Rust half): in a scratch `#[test]` run (not committed) confirm `width = 0.0` returns `Err(InvalidParameter)`.
- [ ] **Step 6:** `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`; commit `feat(optics): add feature-gated peak_sum_molecular`.

### Task 2: Python binding

**Files:**
- Modify: `crates/refloxide-py/Cargo.toml` (enable `refloxide-optics` feature `lineshapes`), `crates/refloxide-py/src/lib.rs`, `python/refloxide/rust.pyi`, `python/refloxide/optics.py`

**Interfaces:**
- Consumes: Task 1's `peak_sum_molecular`, `PeakSpec`, `LineKind::parse`.
- Produces (importable as `refloxide.optics.peak_sum_molecular`):

```python
def peak_sum_molecular(
    kinds: Sequence[str],
    shapes: Sequence[NDArray[np.float64]],   # 1-D, row-major (slice, param), or length n_shape for broadcast
    heights: Sequence[NDArray[np.float64]],  # 1-D, length 1 or n_slices
    f_xx: Sequence[NDArray[np.float64]],
    f_zz: Sequence[NDArray[np.float64]],
    energies_ev: NDArray[np.float64],
    n_slices: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:  # each (n_slices, n_E)
```

Unknown kind and any `RefloxideError` raise `ValueError` (follow the existing `to_pyerr` mapping; confirm it maps to `ValueError`, otherwise map `InvalidParameter` and unknown kinds to `ValueError`).

- [ ] **Step 1:** Add the `#[pyfunction]` in `lib.rs` next to `molecular_index_at_ooc`, mirroring its numpy handling and GIL conventions, register it in `rust()`.
- [ ] **Step 2:** Add the stub to `rust.pyi` and the re-export plus `__all__` entry in `optics.py` (keep alphabetical order and the docstring style).
- [ ] **Step 3:** `make develop`. Scratch check: call it for one Gaussian over 5 energies and confirm shape `(1, 5)`; call with `width=0` and confirm `ValueError` naming the width (Review Focus 2).
- [ ] **Step 4:** Run `cargo clippy --workspace --all-targets -- -D warnings`, `uv run ruff check .`, `uv run ty check`, `uv run pytest -q` (unchanged pass set); commit `feat(py): expose peak_sum_molecular`.

### Task 3: `PeakSumSLD`

**Files:**
- Create: `python/refloxide/lineshapes.py`, `tests/test_peak_sum_source.py`

**Interfaces:**
- Consumes: `refloxide.optics.peak_sum_molecular`; `refloxide.model.Scatterer`, `refloxide.model._depth_field_scalar`, `refloxide.model._is_depth_field`; `refloxide.sources.lab_diagonal_from_cos2`; `refloxide.mixing.pack_lab_diagonal`; `refloxide.data.OpticalConstants`.
- Produces:

```python
MAGIC_ANGLE: float                                  # arccos(1/sqrt(3))
def tdm_fractions(theta: ArrayLike) -> tuple[NDArray, NDArray]:
    """(f_xx, f_zz) = (1.5*sin(theta)**2, 3*cos(theta)**2); (2*f_xx + f_zz)/3 == 1."""

class Peak:
    def __init__(self, kind: str = "gaussian", *, label: str, height, tdm_tilt=MAGIC_ANGLE, **shape_params) -> None
    # shape_params by kind: gaussian energy,width; lorentzian energy,gamma; voigt energy,sigma,gamma; pseudo_voigt energy,fwhm,eta
    # each value: float | Parameter | depth field (_FieldBase); numbers become Parameter(vary=False)
    label: str
    def parameter_items(self) -> dict[str, Parameter | _FieldBase]   # name -> stored value, ordered

class PeakSumSLD(Scatterer):
    def __init__(self, peaks: Sequence[Peak], *, baseline: OpticalConstants | None = None,
                 density=1.0, rotation=0.0, energy_offset=0.0, name: str = "") -> None
    peaks: list[Peak]
    density, rotation, energy_offset   # same storage and defaults as UniTensorSLD (Parameter or depth field)
    def tensor_at(self, energy_ev: float) -> NDArray[np.complex128]
    def tensor_at_many(self, energies_ev: NDArray[np.float64]) -> NDArray[np.complex128]
    def lab_channels_at_depth(self, energy_ev: float, z: NDArray[np.float64], thickness: float, *,
                              density: NDArray | None, gamma: NDArray | None, cos2: NDArray | None
                              ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]  # (n_o, n_e), each (n_z,)
    def depth_fields(self) -> dict[str, _FieldBase]   # address f"{name}.{label}.{param}" (and ".height", ".tdm_tilt") for peak params that are depth fields
    parameters (property), effective_density(), effective_rotation()
```

- [ ] **Step 1: Write the failing parity test** `test_peak_sum_matches_tabulated_reference` in `tests/test_peak_sum_source.py` (testing-policy exception: two independent paths). Reference: build a dense table (grid spacing 0.002 eV over 280-290 eV) from two Gaussian peaks using scipy only: `sigma = width/sqrt(2)`, `beta = h*exp(-0.5*((E-E0)/sigma)**2)`, `delta = h*(2/sqrt(pi))*dawsn((E-E0)/(sqrt(2)*sigma))`, with molecular `n_xx = sum(h_k*f_xx_k*(delta_k + i beta_k))` and `n_zz` likewise using `tdm_fractions` (peak 1 `tdm_tilt=0.3`, peak 2 `1.2`); wrap as `OpticalConstants.from_source(pl.DataFrame(energy, n_xx=delta_xx, n_ixx=beta_xx, n_zz=..., n_izz=...))`. Assert `PeakSumSLD(peaks, rotation=0.7, density=1.1).tensor_at_many(E)` matches `UniTensorSLD(table, density=1.1, rotation=0.7).tensor_at_many(E)` at 80 query energies off the grid points, tolerance interpolation-limited (start at `rtol=1e-4`, `atol=1e-6*abs(reference).max()`, record the final choice and why in the docstring).
- [ ] **Step 2:** Run `uv run pytest tests/test_peak_sum_source.py -v`; confirm FAIL (module missing).
- [ ] **Step 3:** Implement `lineshapes.py`. Resolution rule: per peak parameter, a `Parameter` gives a length-1 array (broadcast, evaluated once), a depth field is evaluated at `z` (`_eval_field`-style, thickness passed) giving length `n_z`. `tensor_at*` run with `n_slices=1` and replace depth fields by `_depth_field_scalar`. Molecular indices are `density * (baseline_molecular + peaks)` where baseline comes from `OpticalConstants.molecular_index_at_many(eff_ev, 1.0)` (zeros when absent) and `eff_ev = energy + energy_offset`; lab diagonals come from `lab_diagonal_from_cos2` with `cos2 = cos(gamma)**2` (or the supplied `cos2`); pack with `pack_lab_diagonal`. Parameter names `f"{name}_{label}_{param}"`; `parameters` includes peak parameters, density, rotation, energy_offset, and parameters of carried depth fields. Frozen by default for peaks, same defaults as `UniTensorSLD` for the rest.
- [ ] **Step 4:** Run the parity test; Expected: PASS.
- [ ] **Step 5:** Scratch checks (Review Focus 3, 4): a peak `height` given a field, called through `tensor_at`, returns finite values equal to the field's bulk pin; `PeakSumSLD([], baseline=table)` equals the table-only tensor; `tdm_tilt` in `{0, pi/2}` and `rotation` in `{0, pi/2}` all finite.
- [ ] **Step 6:** `uv run ruff check . && uv run ruff format --check . && uv run ty check`; commit `feat: add PeakSumSLD and Peak`.

### Task 4: Level fields, shared shapes, derived fields

**Files:**
- Create: `python/refloxide/fields.py`

**Interfaces:**
- Consumes: `refloxide.profiles._FieldBase`, `_param`, `_eval_field`; refnx `Parameter`, `Parameters`; scipy `CubicSpline`.
- Produces:

```python
class Shape(ABC):
    name: str
    level_names: tuple[str, ...]
    @property
    def parameters(self) -> Parameters                  # shape parameters only
    def basis(self, z: NDArray, *, thickness: float) -> NDArray | None
        # (n_levels, n_z) weights with value(z) = levels @ basis, or None when not linear in levels; cached on (shape parameter values, thickness, z)
    def evaluate(self, z: NDArray, levels: NDArray, *, thickness: float) -> NDArray
        # default: levels @ basis(z); nonlinear shapes override

class LevelField(_FieldBase):
    def __init__(self, shape: Shape, levels: Sequence[float | Parameter], *, name: str) -> None
    # evaluate(z, thickness=...) -> shape.evaluate(...); parameters = this field's level Parameters + shape.parameters
class Derived(_FieldBase):
    def __init__(self, fn: Callable[..., NDArray], /, *, name: str = "derived", **inputs: _FieldBase | float | Parameter) -> None
    # evaluate: each input evaluated like a channel (field, number, Parameter), then fn(**arrays); parameters = union of inputs' parameters

def cos2_of(gamma: _FieldBase | float | Parameter) -> Derived     # Derived(lambda gamma: np.cos(gamma)**2, gamma=gamma)

class TransitionShape(Shape):    # levels ("top", "bulk", "bottom"); shape params tau_top, tau_bottom, thickness-aware: top*e_top + bulk*(1-e_top-e_bot) + bottom*e_bot
class SplineShape(Shape):        # __init__(knots); levels are knot values; natural cubic spline basis
class PolynomialShape(Shape):    # __init__(degree, positions=None); levels are values at `positions` (fractions of thickness, default linspace(0, 1, degree+1))
class SinusoidShape(Shape):      # shape params wavelength, phase; levels ("offset", "amplitude")
class LocalizedShape(Shape):     # shape params center, width (Gaussian bump; center=0 is a free-surface layer); levels ("base", "amplitude")
```

Existing fields (`Constant`, `SecondOrderTransition`, `Diffusion`, `Polynomial`, `Spline`, `CallableField`) are unchanged and remain valid on every channel; `CallableField` is the user-function route.

- [ ] **Step 1:** Implement the classes as specified. `SplineShape.basis` is `CubicSpline(knots, np.eye(k), axis=0, bc_type="natural")(z).T`; `PolynomialShape.basis` is the Vandermonde of `z` times the inverse Vandermonde of the anchor depths.
- [ ] **Step 2:** Scratch parity checks, observed not committed: `LevelField(TransitionShape(...), [top, bulk, bottom])` equals `SecondOrderTransition` for the same values and taus to 1e-12; `LevelField(SplineShape(knots), values)` equals `Spline(knots, values)`; `PolynomialShape` reproduces its anchor values exactly at the anchors.
- [ ] **Step 3:** Review Focus 1 and 5: two `LevelField`s sharing one `TransitionShape`, put in a `DepthProfile` (Task 5) or collected directly with `Parameters.flattened(unique=True)`, show each tau once; confirm `basis` at `z` of one point and `thickness=0` is finite (shape-dependent clamps allowed, document them). If refnx does not deduplicate shared `Parameter`s where `DepthProfile.parameters` collects them, dedupe by identity in `Task 5`'s `parameters` property.
- [ ] **Step 4:** Ruff, ty; commit `feat: add level fields, shared shapes and derived fields`.

### Task 5: `DepthProfile` extensions

**Files:**
- Modify: `python/refloxide/profiles.py` (`DepthProfile.__init__`, `_tensors_at_z`, `_component_lab_channels`, `_components_carry_depth_fields`, `named_values_at`, `parameters`)

**Interfaces:**
- Consumes: `PeakSumSLD.lab_channels_at_depth` and `.depth_fields()` (Task 3); `LevelField`, `Derived` (Task 4, no import needed in `profiles.py`: they are `_FieldBase` subclasses).
- Produces: `DepthProfile(..., density_scale: _FieldBase | float | Parameter | None = None)` multiplying the final tensors per slice (after mixing, for every code path); a scatterer hook protocol: any material with `lab_channels_at_depth` is called with the already-evaluated `density`, `gamma`, `cos2` arrays (or `None` for its own scalars) for both single-material and `Mix` components, and any material with `depth_fields()` marks the profile as carrying depth fields and contributes its entries to `named_values_at`.

- [ ] **Step 1:** Add `density_scale` (stored like `density`; included in `parameters`, `named_values_at` under `"density_scale"`). Rename the current `_tensors_at_z` body to `_tensors_at_z_unscaled` and make `_tensors_at_z` multiply by `density_scale(z)[:, None, None]` only when it is set, so the unset path is byte-identical.
- [ ] **Step 2:** In `_component_lab_channels` and in `_tensors_at_z` (before the `UniTensorSLD` branch) call the scatterer hook when present. Extend `_components_carry_depth_fields` so a `Mix` of `PeakSumSLD`s with depth-field peak parameters takes the per-component path instead of the binary `mix_fn` path. Merge each material's `depth_fields()` into `named_values_at`.
- [ ] **Step 3:** Run the existing suite for this area: `uv run pytest tests/test_depth_profiles.py tests/test_composable_oc.py tests/test_scatterers.py -q`. Expected: unchanged pass set (Global Constraints: existing inputs unchanged).
- [ ] **Step 4:** Scratch checks: (a) a single-material `DepthProfile(PeakSumSLD(...))` with a `LevelField` height reproduces `PeakSumSLD.tensor_at` at the profile's bulk anchor and varies with depth; (b) a three-material `Mix` with `phi=[f1, f2, f3]`, `gamma=[g1, g2, g3]`, and `density_scale=` runs and returns `(n_slabs, 3, 3)` finite tensors; (c) `n_slabs=1` finite (Review Focus 5); (d) a uniform `PeakSumSLD(...)(thick, rough)` slab inside a `Structure` evaluates in `ReflectModel` with only scalar peak parameters and its peak heights fit as `Parameter`s.
- [ ] **Step 5:** Ruff, ty; commit `feat(profiles): density_scale, scatterer depth hooks`.

### Task 6: Worked example, docs and end-to-end test

**Files:**
- Create: `examples/lineshape_three_material.py`, `docs/guides/building_structures/peak_gradients.md`, `tests/test_lineshape_example.py`
- Modify: `mkdocs.yml` (nav entry under "Building structures" after "Mixed slabs")

**Interfaces:**
- Consumes: everything above plus `Structure`, `ReflectModel`, `Mix`, `DepthProfile`, `MaterialSLD`.
- Produces: `build_structure() -> Structure`, `channel_profiles() -> dict[str, tuple[NDArray, NDArray]]` (address to (z, values)), `reflectivity() -> NDArray`, and `main()` that writes three figures to `docs/assets/` (channels versus depth; `n_o`, `n_e` versus depth at one energy; reflectivity over a q-energy range).

- [ ] **Step 1:** Write the example with synthetic five-peak sets for mat1, mat2, mat3 (illustrative energies near the carbon K edge, linear mix rule, `n_slabs=40`), one shared `SplineShape` with 5 knots across the film, and every channel a `LevelField` on it with 5 levels (intermediate knots interpolate the surface and substrate values linearly). Pin these values from the spec: volume fractions mostly mat1 at the surface with little mat2 and mat3, mat2 and mat3 enriching toward the substrate; `density_scale` larger at the substrate; mat1 `gamma` from 0 at the surface to `MAGIC_ANGLE` at the substrate; mat2 and mat3 `gamma` from `MAGIC_ANGLE` to pi/2; mat1 peaks 1 and 2 heights at the surface 1.5 and 2.0 times their substrate values, peak 3 at the surface 0.8 times its substrate value (all written as absolute heights); mat1 TDM tilt of peaks 1 and 2 from 0 at the surface to `deg2rad(20)` at the substrate, of peaks 4 and 5 from `deg2rad(90)` to `deg2rad(80)`.
- [ ] **Step 2:** Write `peak_gradients.md`: the construction (included from the example file via the snippets extension), the three figures, a note on how each channel works in a single-material film, a second short snippet with a `TransitionShape` and a `LocalizedShape` surface layer on one peak height with constant orientation and density, a user-function channel via `CallableField`, and the `gamma` and radians conventions. Add the nav entry.
- [ ] **Step 3: Write the end-to-end test** `tests/test_lineshape_example.py` (testing-policy exception): import the example by path, call `build_structure()` and `reflectivity()`; assert all finite and non-negative; from `channel_profiles()` assert mat1 `gamma` is 0 at the surface and `MAGIC_ANGLE` at the substrate to 1e-6, and mat1 peak-1 `tdm_tilt` is 0 at the surface and `deg2rad(20)` at the substrate.
- [ ] **Step 4:** Run `uv run pytest tests/test_lineshape_example.py -v`; Expected: PASS. Run `uv run python examples/lineshape_three_material.py` and open the three figures; check by eye that each trace matches the spec's description (direction of every gradient). `uv run mkdocs build --strict` (if the docs toolchain is installed) passes.
- [ ] **Step 5:** Commit `docs: add three-material peak-gradient example`.

### Task 7: Performance record and final verification

**Files:** none in the product; writes `tmp/lineshape-channels/BENCH.md`.

- [ ] **Step 1:** Scratch timing script in `tmp/lineshape-channels/` (not committed): median of 200 runs for (a) a uniform slab `UniTensorSLD(table).tensor_at_many` versus `PeakSumSLD` with 5 scalar peaks, 200 energies; (b) a 40-slab `DepthProfile` with all-scalar peaks, with one depth-varying height, with one depth-varying width. Record in `BENCH.md` beside the Task 0 baseline. The spec's anti-goal says the all-scalar path must not cost more per energy than the tabulated path; a table lookup may be intrinsically cheaper than evaluating Dawson and Faddeeva lines. If the measured ratio is above 1, report the numbers to the user rather than optimizing further or silently accepting.
- [ ] **Step 2:** Re-run `uv run pytest tests/test_bench_gates.py -q` and compare against Task 0; the tabulated path must be within noise.
- [ ] **Step 3:** Run the full gate: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`, `uv run pytest -q`, `make verify`. Expected: all green, pass set equal to baseline plus the three new tests.
- [ ] **Step 4:** Report to the user: the benchmark table, any deviation from existing results, and the calls the plan made beyond the spec (below). Do not open a PR unless asked.

## Calls made while planning (mine; for the user to check)

- No Rust `OpticalSource` trait or `TableSource` here (HANDOFF stage 2): the peak evaluation is a concrete function. The Python contract (`Scatterer`) is already the common seam. The trait remains a separate follow-up.
- The channel "address" (`mat1.p1.height`) is a naming and plotting convention (`depth_fields()`, `named_values_at`, parameter names), not a string-binding mini-language: peak parameters bind by passing a field where a number would go, exactly as `UniTensorSLD.density` already accepts a depth field.
- `PeakSumSLD` takes `rotation` (not `gamma`) to match `UniTensorSLD`; the `DepthProfile` keyword stays `gamma`.
- The baseline (continuum and step) in v1 is an isotropic or anisotropic `OpticalConstants` table added to the molecular diagonals before density scaling.
- The mix-level density multiplies tensors per slice after mixing.
