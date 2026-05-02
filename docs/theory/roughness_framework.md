# Roughness framework

## Scope of this page

Roughness physics enters `refloxide` at two distinct points in
the computation graph and through three complementary models.
This page fixes the architectural choice of where each model
attaches to the Passler-Paarmann 4x4 pipeline, what the kernel
trait that the three models implement looks like, and how the
selection between models is delegated to either the user or to
the heuristic in [Selection guide](roughness_selection_guide.md).
The companion pages
[Nevot-Croce](roughness_nevot_croce.md),
[Debye-Waller](roughness_debye_waller.md), and
[Graded interface](roughness_graded_interface.md)
treat the three models in physical detail.

The framing follows Esashi and coworkers [[1](#references)],
who compare the Nevot-Croce factor and the graded-interface
approach in the Parratt formalism for X-ray and EUV
reflectivity. Our adaptation generalizes their treatment from
scalar isotropic Parratt to the tensor anisotropic 4x4
formalism, and adds an explicit interface to the Berreman
state vector.

## Where roughness enters the pipeline

The 4x4 pipeline factorizes naturally into stage 3, the
per-interface matrix $A_i$ assembly of
[Interface matrices](interface_matrices.md), and stage 4,
the propagation product $T_{\text{tot}}$ of
[Propagation and assembly](propagation_and_assembly.md).
Roughness can be inserted at either stage with very different
physical and computational consequences.

The first insertion point is multiplicative correction at the
interface matrices. The Nevot-Croce and Debye-Waller models
both act here, attenuating individual Fresnel-like amplitudes
inside $A_i^{-1}A_i$ by closed-form Gaussian factors that
depend on the rms roughness $\sigma_{i,i+1}$ and on the
out-of-plane wave vector projection in each medium. The
correction does not change the dimension of the $A_i$ matrix,
does not introduce new layers, and adds essentially zero cost
to the stack-level product. The mathematical price is restricted
validity, $\sigma k_z < 1$ in either medium, and the physical
price is loss of the polarization-mixing channels that arise
when roughness has off-diagonal anisotropy.

The second insertion point is structural pre-processing, where
each rough interface is replaced before stage 3 by a stack of
thin sublayers whose dielectric tensors interpolate between the
two adjacent media along the longitudinal direction. The graded
interface approach acts here. The pre-processing inflates the
layer count by a factor of order $30$ to $100$ per rough
interface and incurs a proportional runtime cost in stages 3
through 5, but it makes no small-roughness approximation and
gracefully accommodates non-Gaussian profiles, anisotropic
roughness, and chemical interdiffusion that is not strictly a
morphological effect.

The two insertion points are mutually exclusive at a given
interface but mixable across interfaces. A stack with a
chemically sharp top interface and a graded buried interface
should use Nevot-Croce on the top and graded-interface
discretization on the buried boundary, and the kernel must
support this per-interface choice rather than imposing one
model on the entire stack.

## The shared trait

The three models implement a single trait that the kernel
dispatches on at the interface level. The trait sketch is

```rust
pub trait RoughnessModel {
    /// Validate the model parameters against the local stack
    /// geometry and the probe wavelength. Returns Ok(()) when
    /// the model is in its documented validity region.
    fn validate(
        &self,
        wavelength_m: f64,
        eps_above: &Matrix3<C64>,
        eps_below: &Matrix3<C64>,
        theta_rad: f64,
    ) -> Result<(), KernelError>;

    /// Modify the per-interface matrix product before it enters
    /// the propagation chain. Multiplicative models override
    /// this, structural models do not.
    fn correct_interface(
        &self,
        a_above: &mut Matrix4<C64>,
        a_below: &mut Matrix4<C64>,
        kz_above: C64,
        kz_below: C64,
    ) -> Result<(), KernelError> {
        Ok(())
    }

    /// Replace a rough interface by a sequence of thin sublayers
    /// before the per-layer eigensolve. Structural models
    /// override this, multiplicative models do not.
    fn discretize_interface(
        &self,
        layer_above: &Layer,
        layer_below: &Layer,
    ) -> Result<Vec<Layer>, KernelError> {
        Ok(Vec::new())
    }
}
```

A model implements either `correct_interface` or
`discretize_interface` but never both, and the default
implementations make the unused method a no-op. The dispatcher
walks the stack, calls `discretize_interface` first to expand
the stack, then runs stages 3 and 4 with the expanded stack and
calls `correct_interface` at each remaining interface that
carries a multiplicative model.

The `validate` method runs once per evaluation and emits a
`KernelError::InvalidGeometry` with a precise diagnostic when
the user-selected model violates its documented validity bound.
The kernel does not silently fall back to a different model
because silent fallback masks model-mismatch errors that should
be visible to the user. The
[Selection guide](roughness_selection_guide.md) recommends the
right model up front, and the kernel verifies the choice rather
than overriding it.

## Per-interface roughness specification

Each interface in a stack carries a `RoughnessSpec` payload that
the user supplies alongside the layer thicknesses and dielectric
tensors. The spec is

```rust
pub struct RoughnessSpec {
    pub sigma_nm: f64,
    pub correlation_length_nm: Option<f64>,
    pub model: RoughnessChoice,
    pub profile: ProfileShape,
}

pub enum RoughnessChoice {
    Sharp,
    NevotCroce,
    DebyeWaller,
    Graded { sublayer_count: usize },
    Auto,
}

pub enum ProfileShape {
    Gaussian,
    Linear,
    Sine,
    TanhSech2,
    Custom(Box<dyn Fn(f64) -> f64 + Send + Sync>),
}
```

The `correlation_length_nm` is mandatory for `Auto` selection
and for `DebyeWaller` validation, and is optional for the other
two models because they do not depend on the correlation length
explicitly. The `ProfileShape` is consulted only by the graded
model, since the Nevot-Croce and Debye-Waller closed-form
factors derive from a Gaussian distribution and silently produce
incorrect results for non-Gaussian roughness if the user
overrides the profile. The kernel rejects this combination at
`validate`.

The `Auto` choice dispatches per the criteria in the
[Selection guide](roughness_selection_guide.md), substituting
`Sharp`, `NevotCroce`, or `Graded` based on the local
$\sigma k_z$ and the correlation length to wavelength ratio.
The dispatcher records its choice in a stack-level audit log so
the user can verify after the fact what the kernel actually
applied.

## Composition with the Passler 4x4 pipeline

The integration is the strongest test of the roughness layer
because the three models must respect the conventions of stages
1 through 6, particularly the Berreman state ordering, the Xu
piecewise dispatch, and the erratum-corrected coefficients.

For Nevot-Croce and Debye-Waller, the multiplicative correction
applies to the eight Passler-Paarmann amplitudes
$r_{kl}, t_{kl}$ rather than to the tangential field components
inside $A_i$. The kernel therefore evaluates the unrough
amplitudes through stages 1 through 5, and applies the
correction at stage 5 output. This ordering preserves the Xu
degenerate-branch dispatch and the erratum-corrected
$t_{ss}, t_{ps}, t_{sp}$ formulas, both of which would be
invalidated by attempting to insert the correction inside
$A_i$. The companion pages give the explicit formulas for the
four channels, including the cross-polarization
$r_{ps}, r_{sp}$ that the Esashi treatment of scalar Parratt
does not need.

For the graded interface, the discretization runs before stage
1. Each rough interface inflates into a stack of thin sublayers
whose principal dielectric tensors are linearly interpolated
through the profile function, and whose Euler angles are
interpolated by spherical linear interpolation when the two
adjacent layers carry different optic-axis orientations. The
inflated stack then enters stage 1 unchanged. The eigensolve
and interface assembly absorb the inflated layer count without
any algorithmic modification, and the regression scaffold's
self-consistency tests at $\sigma \to 0$ recover the
unmodified pipeline by construction.

## Where the code lives

In `refloxide`, the roughness layer is the `core::roughness`
module of the Rust kernel. The three models live in submodules
`core::roughness::nevot_croce`, `core::roughness::debye_waller`,
and `core::roughness::graded`. The `RoughnessModel` trait sits
in `core::roughness::traits`. The dispatcher and the
`RoughnessSpec` enum live in `core::roughness::dispatch`.

The Python wrapper exposes a single `Roughness` dataclass that
maps onto `RoughnessSpec`, plus three convenience constructors
`Roughness.nevot_croce(sigma_nm, ...)`,
`Roughness.debye_waller(sigma_nm, correlation_nm, ...)`, and
`Roughness.graded(sigma_nm, profile, sublayer_count, ...)`.
The `Roughness.auto(sigma_nm, correlation_nm)` constructor
delegates to the Rust dispatcher and is the recommended default
for users who do not want to choose explicitly.

## Validation strategy

Three classes of test land in the regression scaffold once the
roughness layer is implemented. The first is sharp-interface
recovery, where each model with $\sigma \to 0$ must reproduce
the unrough amplitudes to within the analytical-benchmark
tolerance of $10^{-10}$. The second is graded-interface
convergence under sublayer refinement, where the amplitudes
must converge to a fixed point as the sublayer count is
increased. The third is multiplicative-model agreement with the
graded model in the small-$\sigma$ regime, where the three
models must agree to within $10^{-6}$ when $\sigma k_z < 0.3$,
matching the published convergence behavior in Esashi
[[1](#references), Figs. 7-10].

The Esashi paper itself provides four reference scans that the
regression scaffold should reproduce as cross-checks once the
roughness layer is implemented. These are the Si/MoSi$_2$/Mo at
13.5 nm scan, the Si/Mo bilayer at 13.5 nm scan, the W/B$_4$C
multilayer at 0.154 nm scan, and the surface-roughness sweep on
a Si substrate at 0.154 nm. We will add an Esashi-comparison
test module under `tests/regression/test_roughness_against_esashi.py`
once the roughness models land in the kernel.

## References

1. Y. Esashi, M. Tanksalvala, Z. Zhang, N. W. Jenkins, H. C.
   Kapteyn, and M. M. Murnane, "Influence of surface and
   interface roughness on X-ray and extreme ultraviolet
   reflectance: A comparative numerical study," OSA Continuum
   **4**, 1497 (2021).
   [DOI](https://doi.org/10.1364/OSAC.422924).
2. L. Nevot and P. Croce, "Caracterisation des surfaces par
   reflexion rasante de rayons X. Application a l'etude du
   polissage de quelques verres silicates," Rev. Phys. Appl.
   **15**, 761 (1980).
   [DOI](https://doi.org/10.1051/rphysap:01980001503076100).
3. P. Croce and L. Nevot, "Etude des couches minces et des
   surfaces par reflexion rasante, speculaire ou diffuse, de
   rayons X," Rev. Phys. Appl. **11**, 113 (1976).
   [DOI](https://doi.org/10.1051/rphysap:01976001101011300).
4. S. K. Sinha, E. B. Sirota, S. Garoff, and H. B. Stanley,
   "X-ray and neutron scattering from rough surfaces," Phys.
   Rev. B **38**, 2297 (1988).
   [DOI](https://doi.org/10.1103/PhysRevB.38.2297).
