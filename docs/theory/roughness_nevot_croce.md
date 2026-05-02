# Roughness model, Nevot-Croce

## Scope of this page

This page documents the Nevot-Croce factor [[1](#references),
[2](#references)] as the multiplicative roughness correction
applied to the eight Passler-Paarmann amplitudes
$r_{kl}, t_{kl}$ at each interface in `refloxide`. The treatment
follows the derivation that Esashi and coworkers
[[3](#references), Sec. 3] reproduce from the distorted-wave
Born approximation [[4](#references)], adapted to the tensor
4x4 setting. The correction has the lowest computational cost
of the three roughness models in the kernel and is the default
choice in the small-roughness regime defined by
$\sigma k_{i,z} < 1$ in either of the two adjacent media.

The companion pages
[Debye-Waller](roughness_debye_waller.md) and
[Graded interface](roughness_graded_interface.md)
treat the long-correlation limit and the structural
discretization respectively. The
[Selection guide](roughness_selection_guide.md) gives the
decision criterion that maps a given physical regime onto one
of the three.

## Closed-form correction factors

The Nevot-Croce factor models a Gaussian distribution of
interface heights with rms amplitude $\sigma_{i,i+1}$ across the
boundary between layers $i$ and $i+1$. For the reflection and
transmission amplitudes at a single interface, the corrections
read [[3](#references), Eq. (1)]

$$
r_{i,i+1}^{\text{rough}} = r_{i,i+1}^{\text{sharp}}\,
\exp\!\big(-2\,k_{i,z}\,k_{i+1,z}\,\sigma_{i,i+1}^{2}\big),
$$

$$
t_{i,i+1}^{\text{rough}} = t_{i,i+1}^{\text{sharp}}\,
\exp\!\big((k_{i,z} - k_{i+1,z})^{2}\,\sigma_{i,i+1}^{2}/2\big),
$$

where $k_{i,z} = (\omega/c)\,q_{ij}$ for the relevant mode $j$
of layer $i$, and the factor on the transmission coefficient is
the symmetric counterpart of the reflection factor in the limit
of vanishing index contrast. Two structural features distinguish
the Nevot-Croce form from the Debye-Waller form of the next
page. First, the reflection exponent contains the product
$k_{i,z}k_{i+1,z}$ rather than $k_{i,z}^{2}$, which embeds the
index contrast across the boundary. Second, the transmission
exponent carries the squared difference $(k_{i,z} - k_{i+1,z})^{2}$
with a positive sign in the exponent, which encodes the
amplitude enhancement of forward-propagating modes when the
roughness scatters energy out of the specular reflection
channel.

For absorbing media the wave-vector projections $k_{i,z}$ are
complex, and the exponential factors carry both modulus and
phase. The phase contribution modifies the field reconstruction
inside the layer through stage 6 of the pipeline and is the
mechanism by which the Nevot-Croce factor influences observables
beyond the specular reflectance, including ellipsometric phase
and standing-wave field profiles. We surmise that any
implementation that drops the phase by extracting only the
magnitude of the exponential will silently introduce errors in
the X-ray standing wave intensity that scale linearly with
$\sigma$.

## Application to the eight Passler-Paarmann amplitudes

The Esashi treatment is scalar and addresses only the two
co-polarized channels $r_{pp}, r_{ss}$ and $t_{pp}, t_{ss}$.
The 4x4 formalism carries four additional cross-polarization
channels $r_{ps}, r_{sp}, t_{ps}, t_{sp}$ that arise in
birefringent layers. The kernel applies the Nevot-Croce factor
to each of the eight channels individually using the
mode-resolved $k_{i,z}$ on each side of the interface.

For the co-polarized reflection amplitudes the assignment is
unambiguous,

$$
r_{pp}^{\text{rough}} = r_{pp}^{\text{sharp}}\,
\exp\!\big(-2\,k_{i,z}^{(p)}\,k_{i+1,z}^{(p)}\,\sigma^{2}\big),
\quad
r_{ss}^{\text{rough}} = r_{ss}^{\text{sharp}}\,
\exp\!\big(-2\,k_{i,z}^{(s)}\,k_{i+1,z}^{(s)}\,\sigma^{2}\big),
$$

with $k_{i,z}^{(p)} = (\omega/c)\,q_{i1}$ and
$k_{i,z}^{(s)} = (\omega/c)\,q_{i2}$ in the Passler sorted-mode
labels of [Eigenmode analysis](eigenmode_analysis.md).

For the cross-polarization channels the natural choice is the
geometric mean of the two polarization-specific projections,

$$
r_{ps}^{\text{rough}} = r_{ps}^{\text{sharp}}\,
\exp\!\Big(-2\sqrt{k_{i,z}^{(p)}k_{i,z}^{(s)}}\,
\sqrt{k_{i+1,z}^{(p)}k_{i+1,z}^{(s)}}\,\sigma^{2}\Big),
$$

and analogously for $r_{sp}^{\text{rough}}$. The geometric-mean
choice reduces to the co-polarized form in the limit of weak
birefringence, where $q_{i1} \approx q_{i2}$, and makes physical
sense because $r_{ps}$ couples a $p$ incident mode to an $s$
reflected mode, so its decoherence integral involves both
projections in symmetric combination. We hypothesize that this
prescription is correct to leading order in the roughness but
note that no published derivation extends Nevot-Croce to the
birefringent cross-polarization regime, and the regression
scaffold should compare it against a graded-interface
calculation in a uniaxial substrate before adopting it as the
production default. If the comparison fails, the kernel should
fall back to the graded approach in the cross-polarized
channels and emit a `KernelError::UnsupportedConversion`
diagnostic flagging the discrepancy.

The transmission corrections follow the same logic with the
substitution

$$
t_{kl}^{\text{rough}} = t_{kl}^{\text{sharp}}\,
\exp\!\Big((k_{i,z}^{(\text{in})} - k_{i+1,z}^{(\text{out})})^{2}\,
\sigma^{2}/2\Big),
$$

where the in and out projections are determined by the input
and output polarization labels of the channel.

## Validity regime

The Nevot-Croce factor descends from a perturbative expansion
in $\sigma k_z$ and is valid only when the expansion parameter
remains small. Esashi [[3](#references), text following Eq.
(1)] cites $\sigma k_{i,z} < 1$ as the canonical bound. Two
secondary conditions also apply. First, the index contrast
across the interface must be small enough that the unperturbed
Fresnel amplitude is not itself sensitive to the roughness
profile shape, which is the regime in which X-ray and EUV
optics naturally sit but is violated for visible-light
multilayers with metal-dielectric contrast. Second, the
correlation length of the roughness must be short compared to
the lateral coherence length of the probe, which selects the
short-correlation branch of the Sinha-Sirota-Garoff-Stanley
classification [[4](#references)] and excludes the
long-correlation Debye-Waller regime.

For X-ray reflectivity at $\lambda = 0.154$ nm and $\theta = 5
\,\text{deg}$ from grazing in vacuum, the bound translates to
$\sigma < 0.28$ nm. For EUV reflectivity at $\lambda = 13.5$
nm and $\theta = 30\,\text{deg}$ from grazing, it relaxes to
$\sigma < 4.3$ nm [[3](#references), text after Eq. (1)]. For
soft X-ray and visible-wavelength reflectometry the bound is
correspondingly relaxed, but the index-contrast condition
becomes the binding constraint instead. The kernel evaluates
both conditions in `RoughnessModel::validate` and returns
`KernelError::InvalidGeometry` with both numerical bounds in
the diagnostic message when either fails.

## Where the code lives

In `refloxide`, the Nevot-Croce model is implemented in
`core::roughness::nevot_croce`. The module exposes a single
`NevotCroce { sigma_nm: f64 }` struct that implements
`RoughnessModel`. The `correct_interface` method reads the
mode-resolved $q_{ij}$ from the upstream stack output, computes
the eight exponential corrections, and applies them in place to
a four-by-four channel-resolved amplitude matrix that the
amplitude solver carries forward. The `validate` method checks
the two regime conditions above and returns the diagnostic
error.

The Python wrapper exposes the model through the convenience
constructor `Roughness.nevot_croce(sigma_nm=...)`. No additional
parameters are required because the correlation length does not
enter the closed-form factor. Users who need to specify a
correlation length for the validity-region test pass it as
`Roughness.nevot_croce(sigma_nm=..., correlation_length_nm=...)`,
which the validator uses but the algorithm ignores.

## Validation

Three regression tests gate the Nevot-Croce implementation.

The first is sharp-interface recovery,
$\sigma \to 0 \Rightarrow r_{kl}^{\text{rough}} \to r_{kl}^{\text{sharp}}$,
which holds analytically and is asserted at $10^{-12}$
absolute tolerance in
`tests/regression/test_roughness_sharp_limit.py`.

The second is co-polarized agreement with the graded model in
the small-roughness regime, where the Nevot-Croce co-polarized
amplitudes must agree with a fully discretized graded-interface
calculation to within $10^{-6}$ when $\sigma k_z < 0.3$. This
test exercises the regime overlap and catches sign or
factor-of-two errors in the exponent. It lives in
`tests/regression/test_roughness_nevot_vs_graded.py`.

The third is cross-polarization sanity in a uniaxial substrate,
where the geometric-mean prescription for $r_{ps}, r_{sp}$ is
compared against a graded-interface calculation with the same
roughness. The test passes when the two agree to $10^{-4}$ in
the small-roughness regime, which is looser than the
co-polarized tolerance because the geometric-mean ansatz is
itself approximate. The test lives in
`tests/regression/test_roughness_cross_pol_uniaxial.py` and is
the only regression test in the roughness battery that may xfail
permanently if the geometric-mean ansatz is found to be
inadequate.

## Limitations and known failure modes

The Nevot-Croce factor fails in three regimes that are
catalogued here so that downstream users do not rediscover them
by debugging mismatched fits.

The first failure is the large-roughness regime
$\sigma k_z \gtrsim 1$, where the perturbative expansion
breaks down and the correction overcorrects the amplitudes. The
graded-interface model is the correct fallback.

The second failure is the high-index-contrast regime
characteristic of metal-dielectric multilayers in the visible.
The unperturbed Fresnel amplitude depends sensitively on the
roughness profile shape, and the closed-form Gaussian-derived
factor produces results that disagree with both the graded
model and with experiment. There is no clean fallback in this
regime, and the kernel emits a warning rather than an error
because the user may still want a fast estimate.

The third failure is non-Gaussian roughness profiles. The
closed-form factor descends specifically from a Gaussian height
distribution. For exponential or Lorentzian distributions, a
distinct factor is required and is documented in the
roughness-distribution literature [[5](#references)]. The
kernel does not currently implement the non-Gaussian variants,
and the `Roughness.nevot_croce` constructor rejects any
`ProfileShape` other than `Gaussian` at validation time.

## References

1. L. Nevot and P. Croce, "Caracterisation des surfaces par
   reflexion rasante de rayons X. Application a l'etude du
   polissage de quelques verres silicates," Rev. Phys. Appl.
   **15**, 761 (1980).
   [DOI](https://doi.org/10.1051/rphysap:01980001503076100).
2. P. Croce and L. Nevot, "Etude des couches minces et des
   surfaces par reflexion rasante, speculaire ou diffuse, de
   rayons X," Rev. Phys. Appl. **11**, 113 (1976).
   [DOI](https://doi.org/10.1051/rphysap:01976001101011300).
3. Y. Esashi, M. Tanksalvala, Z. Zhang, N. W. Jenkins, H. C.
   Kapteyn, and M. M. Murnane, "Influence of surface and
   interface roughness on X-ray and extreme ultraviolet
   reflectance: A comparative numerical study," OSA Continuum
   **4**, 1497 (2021).
   [DOI](https://doi.org/10.1364/OSAC.422924).
4. S. K. Sinha, E. B. Sirota, S. Garoff, and H. B. Stanley,
   "X-ray and neutron scattering from rough surfaces," Phys.
   Rev. B **38**, 2297 (1988).
   [DOI](https://doi.org/10.1103/PhysRevB.38.2297).
5. D. K. G. de Boer, "X-ray reflection and transmission by
   rough surfaces," Phys. Rev. B **51**, 5297 (1995).
   [DOI](https://doi.org/10.1103/PhysRevB.51.5297).
