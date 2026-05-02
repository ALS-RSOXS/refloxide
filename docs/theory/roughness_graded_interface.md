# Roughness model, graded interface

## Scope of this page

This page documents the graded-interface approach to roughness,
in which each rough boundary is replaced before stage 1 of the
4x4 pipeline by a stack of thin sublayers whose dielectric
tensors interpolate between the two adjacent media along the
longitudinal direction. The treatment follows Esashi
[[1](#references), Sec. 4] adapted from scalar isotropic Parratt
to the tensor anisotropic 4x4 setting. The graded approach has
no small-roughness assumption, accommodates arbitrary roughness
profile shapes, and is the only model in `refloxide` that
admits chemical interdiffusion as physically distinct from
morphological roughness. It is the most expensive of the three
roughness models because the layer count grows by a factor of
order 30 to 100 per rough interface, and it is the recommended
fallback whenever the multiplicative models of
[Nevot-Croce](roughness_nevot_croce.md) and
[Debye-Waller](roughness_debye_waller.md) sit
outside their validity regions.

## Two implementation strategies

Esashi [[1](#references), Sec. 4] distinguishes two
implementations of the graded approach. Both produce a
discretized refractive-index profile that the downstream
transfer-matrix machinery consumes without modification.

### Convolution implementation

The first implementation is the convolution of the sharp-stack
index profile with a single distribution function whose
characteristic width sets the roughness amplitude. Given the
sharp profile $n_{\text{sharp}}(z)$ as a piecewise-constant
function across the interfaces and a normalized distribution
function $g(z)$ with unit area and characteristic width
$\sigma$, the smoothed profile is

$$
n_{\text{graded}}(z) = \int n_{\text{sharp}}(z - z')\,g(z')\,\mathrm{d}z',
$$

discretized onto a uniform $z$-grid that the kernel then
converts back into a sublayer stack. The convolution is the
algorithmically simpler of the two implementations and runs in
$O(N_{\text{grid}}\log N_{\text{grid}})$ time when implemented
through the convolution theorem. Its restriction is that the
single $g(z)$ is shared by every interface in the stack, so
distinct roughness amplitudes or distinct profile shapes at
distinct interfaces cannot be modeled. This restriction is
acceptable for symmetric multilayer optics where every
interface carries the same morphology, and is unacceptable for
heterostructures where buried interfaces are systematically
different from the surface.

The kernel implements the convolution variant in
`core::roughness::graded::convolution` and exposes it through
`Roughness.graded(sigma_nm=..., profile=...)` when the user
supplies a single $\sigma$ and the stack is uniform. A
diagnostic flag in the audit log records when the convolution
implementation was applied.

### Per-interface profile-function implementation

The second implementation operates per interface. For the
boundary between layers $k$ and $k+1$, a profile function
$f_{k,k+1}(z)$ that monotonically transitions from zero to one
across the rough region is supplied, with characteristic width
$\sigma_{k,k+1}$ centered on the nominal interface position.
The Gaussian-distributed roughness corresponds to an error
function $f_{k,k+1}(z) = \tfrac{1}{2}[1 + \operatorname{erf}((z - z_k)/\sigma_{k,k+1}\sqrt{2})]$,
the linear distribution corresponds to a piecewise-linear
profile, and the $\operatorname{sech}^2$ distribution
corresponds to a tanh profile [[1](#references), Fig. 2(a)].

The effective refractive index at depth $z$ is the weighted sum
of the indices of all materials with non-zero presence at $z$
[[1](#references), Eq. (3)],

$$
n(z) = \sum_{k} r_k\,n_k,
$$

with weights

$$
r_k = \begin{cases}
1 - f_{k,k+1}(z), & k = 1, \\
f_{k-1,k}(z) - f_{k,k+1}(z), & 1 < k < N, \\
f_{k,k+1}(z), & k = N.
\end{cases}
$$

The weights sum to unity by construction and the formula
collapses correctly when only two adjacent profiles are
non-trivial at a given $z$. For a depth where only the boundary
between layers 1 and 2 is non-trivial, the formula reduces to
$n(z) = n_1\,(1 - f_{1,2}(z)) + n_2\,f_{1,2}(z)$ as expected.

The profile-function implementation is more expensive per stack
because it evaluates $N - 1$ profile functions at every
discretization point, but it accommodates per-interface
$\sigma_{k,k+1}$ and per-interface $f_{k,k+1}$ shape, which the
convolution implementation does not. The kernel implements the
profile-function variant in
`core::roughness::graded::profile_function` and exposes it
through `Roughness.graded(per_interface=...)` when the user
supplies a list of `RoughnessSpec` instead of a single one.
This is the default for fits to layered heterostructures where
the burial depth correlates with the roughness amplitude.

## Generalization to the dielectric tensor

The Esashi treatment is scalar, $n(z) \in \mathbb{C}$, and the
weighted-sum formula carries directly to the diagonal-tensor
case where each principal component is interpolated
independently,

$$
\bar{\varepsilon}_{\text{principal}}(z) = \sum_{k} r_k\,\bar{\varepsilon}_{\text{principal},k}.
$$

For tilted optic axes the interpolation is more delicate
because the rotation matrices are not vector-space objects and
linear interpolation produces non-unitary transforms in
general. The kernel adopts spherical linear interpolation
between the two Euler angle triples on either side of the
rough interface,

$$
\Omega_{\text{interp}}(s) = \Omega_{\text{above}}^{1 - s}\,
\Omega_{\text{below}}^{s},
\quad s = f_{k,k+1}(z),
$$

which is well-defined because the rotation group is a
geodesically complete Riemannian manifold. The numerical
implementation uses quaternion slerp on the $z, x', z''$ Euler
angles converted to quaternion representation, with the
quaternion inversion check at each step to enforce shortest-path
interpolation across the antipodal map.

We hypothesize that linear interpolation of the rotation
matrices in component form is acceptable when the angular
separation between the two adjacent layers is small, with
"small" defined as below $5\,\text{deg}$, and that slerp is
necessary above this threshold. The regression scaffold should
test this hypothesis with a uniaxial bilayer at $1\,\text{deg}$,
$10\,\text{deg}$, and $45\,\text{deg}$ relative tilt and assert
that the slerp result agrees with a fully resolved DWBA
calculation while the linear-interpolation result diverges
beyond the small-angle threshold.

## Numerical considerations

Four numerical knobs govern the convergence and cost of the
graded approach.

The first is the support extent of the distribution function.
Gaussian distributions have infinite support and require an
artificial cutoff in the discretization. Esashi
[[1](#references), text after Fig. 1] reports good convergence
when the Gaussian is truncated to $\pm 3\sigma$, with wider
truncations producing negligible additional change. The kernel
adopts the $\pm 3\sigma$ default and exposes a `cutoff_sigmas`
parameter for users who want to tighten or loosen it. The
truncated distribution must be renormalized so that $\int g(z)\,\mathrm{d}z = 1$
holds after the truncation, otherwise the index profile picks
up a spurious offset.

The second knob is the discretization thickness. The sublayer
thickness must be small relative to the smallest roughness
$\sigma$ in the stack. Esashi reports that the discretization
thickness affects the reflected phase more strongly than the
reflected intensity because the discretization controls the
optical path length. The kernel exposes the discretization
thickness as either an absolute value in nanometers or as a
relative factor of $\sigma$, and the default is
$\Delta z = \sigma / 30$, which sits in the converged region
for the Esashi benchmark stacks.

The third knob is the merge step. After the discretized stack
is generated, neighboring sublayers with effectively identical
dielectric tensors can be merged before stage 1 of the pipeline
without affecting the result. Esashi [[1](#references), Fig.
1(c)] reports near-linear runtime scaling in the merged layer
count, so the merge is essential when the stack has long
homogeneous regions between rough interfaces. The kernel
implements the merge with an $L^{\infty}$ tolerance of
$10^{-12}$ on the dielectric tensor entries, exposed through a
`merge_tolerance` parameter.

The fourth knob is the convergence acceptance criterion. The
graded approach has no closed-form correctness condition, and
the only way to certify a result is to refine the
discretization and verify that the amplitudes converge. The
kernel exposes a `convergence_check` mode that doubles the
sublayer count, recomputes the amplitudes, and asserts pointwise
convergence to a user-supplied tolerance. The default is
$10^{-6}$ on the absolute amplitude difference, and the kernel
emits a `KernelError::EigenSolveFailure` with the discrepancy
magnitude when the bound is violated.

## Composition with the 4x4 pipeline

The graded discretization runs as a pre-processing pass before
stage 1 of `core::pipeline::Pipeline::new`. The dispatcher
expands each rough interface into its sublayer stack, attaches
the resulting layers to the parent stack, and passes the
expanded stack into the unmodified pipeline. No stage of the
pipeline is aware that the layers came from a discretization
rather than from user input, which is the architectural
property that lets the graded approach inherit the
erratum-corrected amplitudes and the field-reconstruction
machinery without modification.

For mixed stacks where some interfaces use multiplicative
models and others use the graded approach, the dispatcher walks
the stack twice. The first pass expands graded interfaces. The
second pass runs the pipeline and applies multiplicative
corrections at the remaining sharp interfaces. The two passes
do not interfere because the multiplicative models attach to
amplitudes after stage 5 and the graded discretization attaches
before stage 1.

## Where the code lives

In `refloxide`, the graded interface implementations live under
`core::roughness::graded::convolution` and
`core::roughness::graded::profile_function`. The shared profile
function evaluators (Gaussian, linear, sine, tanh) live in
`core::roughness::graded::profiles`. The slerp interpolation of
Euler angles lives in `material::rotation::slerp`. The merge
pass lives in `core::roughness::graded::merge`.

The Python wrapper exposes the model through
`Roughness.graded(sigma_nm, profile, sublayer_count)` for the
convolution variant and
`Roughness.graded_per_interface([RoughnessSpec, ...])` for the
profile-function variant. The convergence check is exposed
through `Roughness.graded(..., convergence_check=True)` and
the merge tolerance through `Roughness.graded(..., merge_tolerance=...)`.

## Validation

Three regression tests gate the graded implementation in
addition to the sharp-interface recovery shared with the other
two models.

The first is convergence under sublayer refinement, where the
amplitudes computed at $30$, $60$, and $120$ sublayers per
interface must form a monotonically converging sequence with
the absolute differences decreasing by at least a factor of two
at each refinement. This test catches discretization errors and
sets the empirical value of the default $\Delta z / \sigma$
ratio. It lives in `tests/regression/test_roughness_graded_convergence.py`.

The second is profile-function consistency, where the same
roughness specified through the convolution implementation and
through the profile-function implementation must produce
amplitudes that agree to $10^{-9}$ when the profile shape and
$\sigma$ match. This test catches inconsistencies between the
two implementation paths. It lives in
`tests/regression/test_roughness_graded_consistency.py`.

The third is Esashi-benchmark agreement, where the Si/Mo
bilayer at 13.5 nm and the W/B$_4$C multilayer at 0.154 nm
reflectivity sweeps from [[1](#references), Figs. 7 and 11]
must agree with the published reference data to $10^{-3}$
relative tolerance across the full angular sweep. This is the
external-reference test for the graded model and lives in
`tests/regression/test_roughness_against_esashi.py`.

## Limitations

The graded approach is the most general of the three roughness
models in `refloxide`, but it is not unconditionally correct.
Three limitations apply.

The first is the implicit assumption that roughness can be
described by a one-dimensional profile $f_{k,k+1}(z)$. The
formalism averages transversely over the in-plane spatial
distribution and does not capture in-plane correlations. For
in-plane structured roughness (e.g. periodic gratings or
structured magnetic domains), a proper distorted-wave Born
approximation in the full three-dimensional roughness spectrum
is required, which is outside the scope of this kernel.

The second is the assumption that the chemical species in
adjacent layers are linearly miscible, so that the weighted-sum
formula for $n(z)$ has physical meaning. For interfaces where
the species form a third compound through chemical reaction
(e.g. silicide formation at metal-silicon interfaces), the
correct profile is not a smooth interpolation of the two
end-member indices and the user must supply the profile
explicitly through the per-interface implementation.

The third is the cost in stage 4 of the pipeline. A typical
graded discretization with 10 rough interfaces in a stack
inflates the layer count by a factor of $30 \times 10 = 300$,
which translates to a $300\times$ runtime overhead in the
amplitude assembly. The merge pass typically reclaims most of
this, but the worst case remains $30\times$ to $100\times$
slower than the multiplicative models. For large parameter
sweeps the user should consider Nevot-Croce or Debye-Waller
where applicable, escalating to graded only for the regions
where the multiplicative models are out of validity.

## References

1. Y. Esashi, M. Tanksalvala, Z. Zhang, N. W. Jenkins, H. C.
   Kapteyn, and M. M. Murnane, "Influence of surface and
   interface roughness on X-ray and extreme ultraviolet
   reflectance: A comparative numerical study," OSA Continuum
   **4**, 1497 (2021).
   [DOI](https://doi.org/10.1364/OSAC.422924).
2. D. L. Windt, "IMD: Software for modeling the optical
   properties of multilayer films," Comput. Phys. **12**, 360
   (1998).
   [DOI](https://doi.org/10.1063/1.168689).
3. K. Shoemake, "Animating rotation with quaternion curves,"
   ACM SIGGRAPH Computer Graphics **19**, 245 (1985).
   [DOI](https://doi.org/10.1145/325165.325242).
