# Lineshape sources and composable depth channels (design)

Status: draft for user approval. Not committed. Builds on `2026-10-24-workspace-split-design.md` (stage 1) and `tmp/workspace-split/HANDOFF.md` (stages 2 and 3).

## Intent

Fit ZnPc and Alq3 (see `/Users/hduva/projects/lr-optical-models`) with optical models whose NEXAFS is a sum of `lineshapes` peaks instead of a fixed table, and let the peak parameters take part in the same structural model as everything else. A peak height, a peak's transition-dipole (TDM) tilt, a material's density, tilt or volume fraction are all just quantities that may vary with depth. Any of them can be held fixed, fitted as a single number, or driven along any depth profile we have built, in single-material films as well as mixtures. Spline-driven three-material mixing is one use case, not the design center.

This is additive. The tabulated `OpticalConstants` source, `UniTensorSLD`, `DepthProfile`, `Mix`, the existing fields and every current fit keep working unchanged.

## Goals

1. A **peak-sum optical source**: same contract as the tabulated source (energy in, optical tensor out), built from `lineshapes` peaks, offered as an additional option beside tabulated sources. It keeps the existing uniaxial knobs (density, tilt `gamma`, energy offset) and in addition exposes each peak parameter (energy, width, height, TDM polar angle) as a parameter that is a scalar (frozen or fit) or bound to a depth field.
2. **Addressable channels on `DepthProfile`** (no new layer name): every tunable quantity has a stable address (for example `mat1.density`, `mat1.vf`, `mat1.gamma`, `mat1.p1.height`, `mat1.p4.tdm_tilt`, `mat1.p3.width`) and can be driven over depth by a field.
3. **Derived channels**: a function of channels is itself a channel (for example `<cos^2(tilt)>` from tilt and order, `f(p1.height)`). It can be plotted along depth and used as an input.
4. **Fields with shared shape and per-channel levels** (next section).
5. **Per-channel independence**: a channel may have its own field. Example: orientation and density constant through the film while one peak parameter has a thin field localized at the free surface (hydrogen bonding or polar effects).
6. **N-material mixing** where volume fractions, per-material density, a mix-level total density, per-material orientation and per-material peak parameters are all channels. The existing `Mix` rules (linear, Maxwell-Garnett, Bruggeman) are reused.
7. **A documented worked example** in the docs: the three-material gradient described below, runnable, with plots of the channels versus depth and the resulting reflectivity.

## Non-goals and anti-goals

- No removal or rename of current public behavior. No change to the `lineshapes` crate (it stays general-purpose; the optics source lives in `refloxide-optics`, per HANDOFF).
- No step-edge fitting (HANDOFF stage 4). Delta for the continuum and step comes from a table in v1; peak delta is the analytic Dawson partner from `lineshapes`. No Kramers-Kronig per depth slice.
- No Rust implementation of the parameter set now. The field and channel specification must be plain declarative data so a later Rust port needs no redesign.
- No change to the TMM kernels, the fitters, or the objective.
- Anti-goal: an API in which spline plus volume fraction is the only thing that works. Every field family must work on every channel, and every channel must work in a single-material film.
- Anti-goal: slowdown. With all peak parameters scalar, the peak-sum path must not cost more per energy than the tabulated path at the same slice count. No Python loop over slices in any new hot path (the current `_tensors_at_z` loops over slices for `UniTensorSLD`; the new path must be vectorized over slices).
- Anti-goal: configuration creep that changes data sizes or numerics.

## Fields: shape parameters plus per-channel levels

A **field** is a shape family. Its parameters come in two kinds.

- **Shape parameters** control where and how fast values change: `tau_top`, `tau_bottom` for the second-order transition; wavelength and phase for a sinusoid; knot positions for a spline; center and width for a localized surface layer. Several channels may share one shape object (hence one set of shape parameters), or a channel may have its own.
- **Level parameters** belong to each channel and set its value at the family's anchors. A family exposes its **natural number of anchors**: 3 for the second-order transition (`top`, `bulk`, `bottom`), k for a k-knot spline, 2 for a linear ramp, offset plus amplitude for a sinusoid, one for a constant. Levels are ordinary refnx `Parameter`s, so bounds, constraints and `vary` work as usual.

For polynomial and spline families the user may instead supply raw coefficients (A, B, C). Anchor values and coefficients describe the same function and are converted internally.

General contract (decided by the user): a field is any function `value(z) = f(z; shape parameters, levels)`. Linearity in the levels is an optional fast path for the families that have it (linear ramp, polynomial, spline, and the second-order transition in its `top`, `bulk`, `bottom`), not a requirement. Families that are not linear in their levels, and user-defined Python functions, are first-class: a field that does not declare a basis is simply evaluated as `f(z, levels)` on each call, with no shared-basis caching. The second-order transition is served by both paths: its taus change the weights (recomputed when a tau changes) while its three levels stay linear. A user callable receives `z`, its shape parameters and its levels (as in today's `CallableField`), may be bound to any channel, and may carry its own parameters. Nothing in the channel, derived-channel or source machinery may assume linearity.

Fast-path note (mine, an implementation detail with one user-visible consequence): once a shape's parameters are fixed, each built-in family's value at depth `z` is a weighted sum of that channel's levels, `value(z) = sum_a level_a * phi_a(z)`, where the weights `phi_a(z)` depend only on `z` and the shape (never on the levels). This is not about fields composing with each other. It is about how one field depends on its own levels. Examples: `SecondOrderTransition` is `bulk*(1 - e_top - e_bot) + top*e_top + bottom*e_bot`, with `e_top = exp(-z/tau_top)` and `e_bot = exp(-(T-z)/tau_bottom)` (linear in `top`, `bulk`, `bottom`, nonlinear in the taus, which live inside the weights); a cubic spline through fixed knots is linear in its knot values; a polynomial is linear in its coefficients. Consequences: (1) the weights are computed once per shape and mesh, so many channels sharing a shape cost one small matrix product each, instead of rebuilding an interpolator per channel as `Spline.evaluate` does today; (2) anchor values and coefficients are two bases for the same function, which is why both are offered; (3) the shape parameters (taus, knot positions, wavelength, phase) are the only nonlinear parameters, and they are shared when channels share a shape, which is the intended meaning of "same profile". A callable field is the escape hatch and does not need to be linear.

Existing fields (`Constant`, `SecondOrderTransition`, `Diffusion`, `Polynomial`, `Spline`, `CallableField`) keep their present constructors and behavior. New channels accept them, and the builder decides how they map onto shape plus levels (for example by an adapter) without breaking existing call sites. A sinusoid and a localized-layer family are new.

## Channels and derived channels

A channel is an address plus a binding: a number, a `Parameter`, or (field shape, levels). Resolution order for any quantity is: explicit channel binding, else the owning object's own scalar `Parameter`. The existing `DepthProfile` keyword channels (`density`, `gamma`, `cos2_gamma`, `phi`, `delta_*`, `beta_*`) remain and become sugar over the same machinery.

A derived channel is a pure function of other channels evaluated at each slice. Built-ins cover the common ones (`cos2` from tilt, tilt from `cos2`, the TDM-angle to molecular-fraction map). A user callable is allowed.

Orientation convention (decided by the user): the orientation channel is `gamma`, the tilt angle of the molecule (its normal against the surface normal). Face-on is `gamma = 0`, edge-on is `gamma = 90` degrees, and amorphous is the magic angle, about 54.7 degrees, where `<cos^2 gamma> = 1/3` and the azimuthally averaged lab tensor is isotropic (`n_o = n_e`). `cos2_gamma` and `<cos^2 gamma>` remain available as derived channels.

Naming (decided by the user): this is a `DepthProfile`, not a new component. The channel and field machinery extends `DepthProfile` (and `Mix`/`UniTensorSLD` where they carry channels) in place. "Channel" is already the vocabulary in the `DepthProfile` docstring and keywords. No new top-level class or layer name is introduced for it; only genuinely new things (a peak-sum source, new field families) get names.

## Peak-sum source

One source object per material holds an ordered, labeled peak set (`p1`, `p2`, ...), the energy offset, density, and an optional continuum plus step whose delta comes from a table. Per peak: line-shape kind (from `lineshapes`: Gaussian, Lorentzian, Voigt, pseudo-Voigt), energy, width, height, and TDM polar angle `theta_k` measured from the molecular normal (90 degrees lies in the molecular plane).

Tensor construction follows `lr-optical-models/src/optical_model/bookended_oriented.py` (cone parameterization): each peak contributes `h_k * line_k(E)` to the molecular tensor with molecular fractions `(f_xx, f_zz)` set by `theta_k` and normalized so `(2 f_xx + f_zz)/3 = 1`. The molecular diagonals are summed over peaks and then projected to the laboratory frame with the material's orientation channel (`n_o = (n_xx(1 + cos^2) + n_zz sin^2)/2`, `n_e = n_xx sin^2 + n_zz cos^2`), then scaled by density.

Scalar versus depth-varying (decided by the user, refining my earlier open/closed split): the cost distinction is not fitted versus fixed. A peak parameter that is a single number, whether frozen or varied by the fitter, is constant over its slab. A uniform slab (a plain slab with a peak-sum source, for instance measured-table-like optical constants whose peak heights are fit) evaluates its peaks once per energy and produces one tensor; it is never expanded into microslabs and never evaluates peaks per slice. Only a parameter bound to a depth field in a `DepthProfile` is evaluated per microslab. Peaks whose parameters are all scalar inside a `DepthProfile` are evaluated once per energy and shared across microslabs, and one source object used by several slabs with identical parameters is evaluated once per energy. Height and TDM angle are linear in the tensor, so a depth-varying height or TDM angle costs a multiply per slice per peak on top of the cached per-energy line shape; a depth-varying energy or width requires the line shape per slice and energy and is the expensive case. The Rust source takes per-slice parameter arrays only for depth-varying parameters, in one batched call per energy grid, with static dispatch and no `dyn` in the energy loop (HANDOFF guardrails).

Placement: Rust `PeakSumSource` in `refloxide-optics` behind the `lineshapes` feature, depending on the crates.io `lineshapes` crate directly, exposed through `refloxide-py` (HANDOFF stage 3). The `DepthProfile` channel and field machinery is Python in v1.

## Composition with the existing stack

Per slice, per material: resolve channels, build the lab tensor from the source, then mix with the existing `mix_channels` (volume fractions from `matK.vf` channels, clipped and renormalized as `DepthProfile` does today for sequence `phi`). A mix-level channel scales the mixed result for total density. All peak and structure parameters sit in the one refnx `Parameters` set, so a fit opens peak and structure parameters together.

Call (mine): a material's own `density` channel scales that material's tabulated or peak-sum constants; an optional mix-level `density_scale` channel multiplies the mixed tensor. The example uses the mix-level one for "denser at the substrate".

## The worked example (becomes the docs page)

Three materials, mat1, mat2, mat3, as peak-sum sources (synthetic or ZnPc/Alq3-like peak sets; five peaks per material for illustration). One shared spline shape with k knots, with k chosen by the example author (a few knots across the film). Every channel below takes its own k levels on that one shared shape, so every effect follows the same depth profile.

- Volume fractions: at the surface mostly mat1 with little mat2 and mat3; mat2 and mat3 both enrich toward the substrate.
- Total density (`density_scale`): denser at the substrate than at the surface.
- Orientation `gamma`: mat1 face-on at the surface (`gamma` near 0), amorphous (magic angle, about 54.7 degrees) in the mixed region at the substrate; mat2 and mat3 near the magic angle at the surface and mostly edge-on (`gamma` toward 90 degrees) at the substrate.
- Electronic effects in mat1, tied to its surface packing: the heights of peaks 1 and 2 are larger at the surface than at the substrate (the example uses surface values 1.5 and 2.0 times the substrate values, written out as absolute heights), and the height of peak 3 is smaller at the surface (0.8 times the substrate value, also written out). TDM polar angle of peaks 1 and 2 goes from 0 degrees (along the molecular normal) at the surface to 20 degrees at the substrate; peaks 4 and 5 go from 90 degrees (in the molecular plane) at the surface to 80 degrees at the substrate.

The page shows the construction, a plot of every channel versus depth, the resulting tensor components versus depth at one energy, a reflectivity over a q-energy range, and a short note on how the same channels work on a single material and with other field families (second-order transition, a localized surface layer, a sinusoid). Interpretation in the example text (confirmed by the user): TDM angles are measured from the molecular normal. Heights are always the heights themselves, not multipliers (decided by the user); there is no multiplier mechanism.

## Constraints

- Python >= 3.12, edition 2021 Rust, refnx `Parameter`/`Parameters` throughout (the fitters and `.setp` idioms stay valid).
- Existing inputs give unchanged results, to rounding, on the existing test suite (see the call on existing paths below).
- Kernel packing convention: tensor diagonals hold `delta + i*beta`; `epsilon = conj(I - 2*tensor)`.
- Public Rust and Python items get documentation (numpy-style docstrings in Python, `///` in Rust). Follow the repo testing policy: no new tests by default.

## Verification

Per the testing policy, the two exceptions that justify committed tests apply here: (a) a parity check that a peak-sum source with all-scalar peak parameters reproduces the tabulated source built from the same peaks (two independent code paths), and (b) one end-to-end run of the documented example. Everything else is verified by inspection and a scratch run. Existing `uv run pytest` and `cargo test --workspace` must pass unchanged. A before/after benchmark (existing gates) shows no regression on the tabulated path, and a measurement shows the all-scalar peak-sum path (including a uniform slab) is not slower than the tabulated path at equal slice count, plus a record of the per-slice cost of making each parameter kind depth-varying.

## Staging (my call)

1. Extend `DepthProfile` in Python (with the existing `UniTensorSLD` and `Mix`): shared shapes with levels, N-material `vf`, mix-level density, per-material `gamma`, derived channels, localized and sinusoid families. Vectorized over slices. This stage has value on its own and is checkable with the tabulated source.
2. `OpticalSource` trait and `TableSource` in `refloxide-optics` (HANDOFF stage 2), if not already done; the builder confirms the current state of the tree.
3. `PeakSumSource` with scalar peak parameters, including the uniform-slab case (HANDOFF stage 3), with the parity check.
4. Depth-varying peak parameters (peak channels bound to fields, batched per-slice evaluation).
5. The worked-example docs page and its end-to-end run.

## Calls made after the builder check (mine)

- Height versus TDM angle: height scales a peak's isotropic strength; the TDM angle only redistributes that strength between xx and zz at fixed trace `(2 f_xx + f_zz)/3 = 1`. The two are independent channels.
- Parity reference: the tabulated source in the parity check is built by evaluating the same peaks' beta and their analytic Dawson delta on a dense energy grid (no Kramers-Kronig), so the only difference is table interpolation. The builder picks the grid and a tolerance consistent with the interpolation error and records both.
- Existing paths (revised by the user): there is no separate "new layer". `DepthProfile` itself is extended, and its existing keywords and fields keep their meaning. Existing inputs must give unchanged results, to rounding, on the existing test suite; the builder reports any deviation rather than accepting it silently.
- Volume fractions: all N materials get independent non-negative weights, clipped and renormalized to sum to 1 (as sequence `phi` does today). A fit that wants N-1 free fractions fixes one weight's levels or constrains them.
- Density order: per-material `density` scales that material's constants before mixing; `density_scale` multiplies the mixed tensor after mixing. This matches how `Mix` and `DepthProfile` order things today.
- Depth-varying parameters in the example: height and TDM angle. Depth-varying energy and width are supported and measured in the benchmark, not required by the example.
- Line-shape kinds with extra shape parameters (Voigt gamma, pseudo-Voigt eta) expose them as additional named per-peak channels (for example `p1.gamma`, `p1.eta`) under the same scalar versus depth-varying rule.
- Tree state: the builder checks whether the stage 2 trait and `TableSource` exist before sizing the Rust work.

## Left to the builder

Class and module names; the exact Python spelling of channel binding and addresses (the sketches in this document are illustrative); how existing fields adapt to shape plus levels; the Rust trait signature for per-slice depth-varying parameters; how derived channels are declared; peak-line-shape selection per peak; anchor placement defaults for polynomials and splines; serialization format for the declarative spec; whether the docs example uses synthetic or ZnPc/Alq3-derived peak sets.
