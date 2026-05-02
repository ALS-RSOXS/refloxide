# Roughness model, Debye-Waller

## Scope of this page

This page documents the Debye-Waller factor as the
multiplicative roughness correction appropriate to the
long-correlation-length regime of interface roughness, where
the diffuse-scattering contribution sits angularly close to the
specular reflection and cannot be cleanly separated by a
finite-aperture detector. The treatment follows Esashi
[[1](#references), Sec. 3, Eq. (2)] with explicit reference to
the original Debye-Waller and Sinha-Sirota-Garoff-Stanley
derivations [[2](#references), [3](#references)]. The
Debye-Waller factor sits as the long-correlation companion to
the Nevot-Croce factor of
[Nevot-Croce](roughness_nevot_croce.md), and the
choice between them is governed by the criterion in the
[Selection guide](roughness_selection_guide.md).

## Closed-form correction factors

For a Gaussian distribution of interface heights with rms
amplitude $\sigma_{i,i+1}$, the Debye-Waller corrections to the
reflection and transmission Fresnel amplitudes at the interface
between layers $i$ and $i+1$ are [[1](#references), Eq. (2)]

$$
r_{i,i+1}^{\text{rough}} = r_{i,i+1}^{\text{sharp}}\,
\exp\!\big(-2\,k_{i,z}^{2}\,\sigma_{i,i+1}^{2}\big),
$$

$$
t_{i,i+1}^{\text{rough}} = t_{i,i+1}^{\text{sharp}}\,
\exp\!\big(-(k_{i,z} - k_{i+1,z})^{2}\,\sigma_{i,i+1}^{2}/2\big).
$$

The reflection exponent carries $k_{i,z}^{2}$ rather than the
$k_{i,z}k_{i+1,z}$ product that appears in the Nevot-Croce
factor, and the transmission exponent has the opposite overall
sign relative to the Nevot-Croce form. The two structural
differences trace to the underlying scattering geometry
[[3](#references)]. The Nevot-Croce factor follows from the
distorted-wave Born approximation in the regime where diffuse
scattering escapes the specular cone, so the reflection
amplitude is only the coherent contribution and the difference
exponent in the transmission factor reflects the energy
conservation between specular and diffuse channels. The
Debye-Waller factor follows from the same DWBA but in the
opposite limit, where diffuse scattering remains inside the
specular cone and is folded into the measured signal. The
correction therefore symmetrizes around $k_{i,z}^{2}$ rather
than $k_{i,z}k_{i+1,z}$, and the transmission exponent flips
sign because the diffuse channel removes rather than adds to
the forward amplitude.

The two factors agree in the limit $k_{i,z} \to k_{i+1,z}$,
where the index contrast vanishes, and they agree numerically
to sub-percent precision for $\sigma < 1$ nm in typical X-ray
geometries [[1](#references), Figs. 15-18]. They diverge most
strongly in the total external reflection regime near the
critical angle, where $k_{i+1,z}$ becomes nearly imaginary and
the product $k_{i,z}k_{i+1,z}$ acquires a strong phase that the
Debye-Waller form does not carry.

## Application to the eight Passler-Paarmann amplitudes

The application logic mirrors the Nevot-Croce treatment. For
co-polarized channels the projection $k_{i,z}^{(p)}$ uses the
$p$-mode eigenvalue $q_{i1}$ and similarly for the $s$ channel.
For the cross-polarization channels we adopt the same
geometric-mean prescription as in
[Nevot-Croce](roughness_nevot_croce.md) under the
substitution $k_{i,z}k_{i+1,z} \to k_{i,z}^{2}$,

$$
r_{ps}^{\text{rough}} = r_{ps}^{\text{sharp}}\,
\exp\!\Big(-2\,k_{i,z}^{(p)}\,k_{i,z}^{(s)}\,\sigma^{2}\Big),
$$

with the analogous expression for $r_{sp}^{\text{rough}}$. As
with Nevot-Croce, the cross-polarization extension lacks a
published derivation in the birefringent setting, and the
regression scaffold is the audit instrument for the choice. The
kernel emits a `KernelError::UnsupportedConversion` diagnostic
if the comparison against the graded model fails by more than
$10^{-4}$ in the small-roughness regime.

The transmission corrections take the same form,

$$
t_{kl}^{\text{rough}} = t_{kl}^{\text{sharp}}\,
\exp\!\Big(-(k_{i,z}^{(\text{in})} - k_{i+1,z}^{(\text{out})})^{2}\,
\sigma^{2}/2\Big),
$$

with the explicit minus sign that distinguishes the
Debye-Waller transmission factor from the Nevot-Croce form.

## When to choose Debye-Waller over Nevot-Croce

The selection between the two multiplicative factors is
controlled by the correlation-length regime of the roughness.
Define the angularly-resolved diffuse-scattering bandwidth
$\xi_{\text{diff}} = \lambda/(1 - \cos\theta)$, where $\lambda$
is the vacuum wavelength and $\theta$ is the grazing angle of
incidence [[1](#references), text following Eq. (2)]. If the
in-plane correlation length $\xi$ of the roughness satisfies
$\xi \ll \xi_{\text{diff}}$, the diffuse scattering is
angularly broad and can be separated from the specular
reflection by a finite-aperture detector. Nevot-Croce applies
in this regime. If instead $\xi \gg \xi_{\text{diff}}$, the
diffuse scattering crowds against the specular peak and the
detector integrates over both. Debye-Waller applies in this
regime.

For X-ray and EUV reflectivity the binding case is short
correlation length. Polished silicon wafers have $\xi \sim 1$
$\mu$m, and metal-coated fused quartz has $\xi$ in the
$100\text{-}200$ nm range [[1](#references), references 42 and
43]. At $\lambda = 0.154$ nm and $\theta = 5\,\text{deg}$,
$\xi_{\text{diff}} \approx 40$ nm, which sits below the typical
roughness correlation length and thereby places most XRR
geometries in the Nevot-Croce regime. At $\lambda = 13.5$ nm
and $\theta = 30\,\text{deg}$, $\xi_{\text{diff}} \approx 100$
nm, which puts EUV reflectometry on metal-coated optics close
to the regime boundary, where neither factor is strictly
correct and the graded-interface approach is the safer choice.

For visible-wavelength magneto-optic applications the picture
inverts. At $\lambda = 500$ nm and $\theta = 45\,\text{deg}$,
$\xi_{\text{diff}} \approx 1.7$ $\mu$m, which exceeds typical
roughness correlation lengths on polished substrates. The
Debye-Waller factor becomes the relevant multiplicative model,
not Nevot-Croce. This inversion is the reason the kernel does
not silently default to Nevot-Croce across all wavelengths but
requires the user to either pass an explicit choice or rely on
the auto-dispatcher in the
[Selection guide](roughness_selection_guide.md).

## Validity regime

The same perturbative bound $\sigma k_{i,z} < 1$ governs
Debye-Waller as it does Nevot-Croce, and the same
small-index-contrast caveat applies. The third condition
specific to Debye-Waller is the long-correlation requirement
$\xi \gg \xi_{\text{diff}}$. The kernel rejects a Debye-Waller
choice that violates either bound at validation time. When the
correlation length sits inside the crossover region
$\xi \sim \xi_{\text{diff}}$, the kernel issues a warning and
the user is advised to compare against a graded-interface
calculation rather than rely on either multiplicative model.

The numerical disagreement between Nevot-Croce and Debye-Waller
in their respective domains of strict validity remains modest
in the small-$\sigma$ regime [[1](#references), Figs. 15-18],
but the disagreement grows rapidly through the total external
reflection region near the critical angle. Users fitting XRR
data near the critical angle should be especially cautious
about model selection, and the regression scaffold tracks this
sensitivity through a dedicated test that sweeps $\theta$
across the critical angle for both factors and reports the
pointwise discrepancy.

## Where the code lives

In `refloxide`, the Debye-Waller model is implemented in
`core::roughness::debye_waller`. The module exposes
`DebyeWaller { sigma_nm: f64, correlation_length_nm: f64 }`
implementing `RoughnessModel`. The `correct_interface` method
applies the eight exponential corrections analogously to
Nevot-Croce, and the `validate` method enforces the three
conditions $\sigma k_z < 1$, small index contrast, and
$\xi \gg \xi_{\text{diff}}$.

The Python wrapper exposes the model through
`Roughness.debye_waller(sigma_nm=..., correlation_length_nm=...)`.
The correlation length is mandatory because it gates the
validity check, and the kernel rejects construction without it.

## Validation

Two regression tests gate the Debye-Waller implementation in
addition to the sharp-interface recovery shared with the other
two models.

The first is critical-angle agreement with the graded model,
where the Debye-Waller amplitudes must agree with a fully
discretized graded calculation across the total external
reflection region to within $10^{-5}$. This test catches the
most common Debye-Waller failure mode, sign errors in the
transmission exponent that produce subtle phase shifts in the
critical-angle dip. It lives in
`tests/regression/test_roughness_debye_critical_angle.py`.

The second is co-polarized agreement with Nevot-Croce in the
small-roughness, low-contrast regime, where the two
multiplicative factors must agree to $10^{-4}$ when
$\sigma k_z < 0.1$ and the index contrast is below $10^{-3}$.
This is the regime where both factors should reduce to the
same DWBA leading-order correction, and disagreement between
them indicates a transcription error in one of the two
implementations. It lives in
`tests/regression/test_roughness_debye_vs_nevot.py`.

## Limitations and known failure modes

The Debye-Waller failure modes parallel those of Nevot-Croce
with one addition. The shared failures are large $\sigma k_z$,
high index contrast, and non-Gaussian profiles. The
Debye-Waller-specific failure is the short-correlation regime,
where the diffuse scattering escapes the specular cone and the
Debye-Waller assumption that all scattered power remains
detected becomes incorrect. The kernel rejects this regime at
validation time, so the failure mode is loud rather than
silent.

A second Debye-Waller-specific concern is the absence of a
published derivation for cross-polarization channels in
birefringent media, identical to the Nevot-Croce situation. The
geometric-mean prescription used in the kernel is a working
ansatz, and the regression scaffold catches its breakdown
through the cross-polarization comparison test against the
graded model.

## References

1. Y. Esashi, M. Tanksalvala, Z. Zhang, N. W. Jenkins, H. C.
   Kapteyn, and M. M. Murnane, "Influence of surface and
   interface roughness on X-ray and extreme ultraviolet
   reflectance: A comparative numerical study," OSA Continuum
   **4**, 1497 (2021).
   [DOI](https://doi.org/10.1364/OSAC.422924).
2. P. Debye, "Interferenz von Roentgenstrahlen und
   Waermebewegung," Ann. Phys. **348**, 49 (1913).
   [DOI](https://doi.org/10.1002/andp.19133480105).
3. S. K. Sinha, E. B. Sirota, S. Garoff, and H. B. Stanley,
   "X-ray and neutron scattering from rough surfaces," Phys.
   Rev. B **38**, 2297 (1988).
   [DOI](https://doi.org/10.1103/PhysRevB.38.2297).
4. V. Holy, J. Kubena, I. Ohlidal, K. Lischka, and W. Plotz,
   "X-ray reflection from rough layered systems," Phys. Rev. B
   **47**, 15896 (1993).
   [DOI](https://doi.org/10.1103/PhysRevB.47.15896).
