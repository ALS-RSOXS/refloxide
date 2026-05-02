# Roughness model selection guide

## Scope of this page

This page is the practical decision framework for choosing
among the three roughness treatments in `refloxide`. The
underlying physics is treated in
[Nevot-Croce](roughness_nevot_croce.md),
[Debye-Waller](roughness_debye_waller.md), and
[Graded interface](roughness_graded_interface.md).
The architectural composition with the 4x4 pipeline is treated
in [Framework](roughness_framework.md). This page
sits on top of all four and answers the user-facing question of
which model to pick for a given physical regime.

The framing follows Esashi [[1](#references), Sec. 3 and Sec.
4] for the small-roughness, large-roughness, and high-contrast
regimes that distinguish the three models, with extensions for
the birefringent and visible-wavelength regimes that the
Esashi treatment does not cover.

## Two governing dimensionless quantities

Two dimensionless quantities determine which model is correct
for a given physical situation. Both are evaluated at the
relevant interface, in the relevant medium, at the relevant
incidence angle.

The first is the Esashi small-roughness parameter
$\alpha = \sigma\,k_z$, where $\sigma$ is the rms roughness
amplitude and $k_z = (2\pi n/\lambda_0)\sin\theta$ is the
out-of-plane wave-vector projection in the medium just above
the interface. The DWBA expansion that produces both
multiplicative factors is valid only when $\alpha \lesssim 1$
[[1](#references), text following Eq. (1)]. We adopt the
operational rule

$$
\alpha < 0.3 \Rightarrow \text{multiplicative regime},
\quad
\alpha > 1 \Rightarrow \text{graded regime},
$$

with the intermediate region $0.3 < \alpha < 1$ flagged as
crossover where the kernel emits a warning and recommends a
graded calculation as the verification reference.

The second is the Esashi correlation-length ratio
$\beta = \xi / \xi_{\text{diff}}$, where $\xi$ is the in-plane
correlation length of the roughness and
$\xi_{\text{diff}} = \lambda/(1 - \cos\theta)$ is the
diffuse-scattering bandwidth set by the geometry. The
Nevot-Croce factor applies when $\beta \ll 1$ and the
Debye-Waller factor applies when $\beta \gg 1$
[[1](#references), Eq. (2) and surrounding text]. The
operational rule is

$$
\beta < 0.3 \Rightarrow \text{Nevot-Croce},
\quad
\beta > 3 \Rightarrow \text{Debye-Waller},
$$

with the intermediate region as a second crossover that
recommends the graded calculation.

The two dimensionless quantities are independent in principle.
A stack with small roughness and short correlation length sits
in the Nevot-Croce regime. A stack with large roughness and
short correlation length sits in the graded regime. A stack
with small roughness and long correlation length sits in the
Debye-Waller regime. A stack with large roughness and long
correlation length sits in the graded regime regardless. The
graded model is the universal fallback when either quantity
falls outside its operating range.

## Decision matrix

The four-way matrix below summarizes the dispatch as
implemented in `core::roughness::dispatch::auto_select`.

| $\alpha$ regime | $\beta$ regime | Primary model | Reason |
|-----------------|---------------|---------------|--------|
| $\alpha < 0.3$  | $\beta < 0.3$  | Nevot-Croce  | Both DWBA conditions satisfied, short correlation length |
| $\alpha < 0.3$  | $\beta > 3$    | Debye-Waller | Both DWBA conditions satisfied, long correlation length |
| $\alpha < 0.3$  | $0.3 \le \beta \le 3$ | Graded | Correlation length crossover, multiplicative models ambiguous |
| $0.3 \le \alpha \le 1$ | any | Graded with warning | Approaching DWBA breakdown, graded recommended for verification |
| $\alpha > 1$    | any | Graded | DWBA invalid, graded mandatory |

A separate axis applies independently to all four cells: the
index-contrast regime. When the relative dielectric contrast
$|\Delta\varepsilon|/\varepsilon$ across the interface exceeds
$0.3$ in either direction, the multiplicative models lose
validity even when $\alpha$ and $\beta$ sit in their nominal
operating regions [[1](#references), text following Eq. (1)].
The kernel applies the override rule

$$
|\Delta\varepsilon|/\varepsilon > 0.3 \Rightarrow \text{Graded},
$$

regardless of the $\alpha$ and $\beta$ values. This regime is
characteristic of metal-dielectric multilayers in the visible
and near-infrared, and of the Au and Pt overlayers used in
soft-X-ray reflectivity standards.

A third axis applies to the polarization regime. For
birefringent substrates with cross-polarization channels
$r_{ps}, r_{sp} \neq 0$, the kernel uses the geometric-mean
prescription described in
[Nevot-Croce](roughness_nevot_croce.md) for the
multiplicative models, but the prescription is unverified
against published derivations. When the cross-polarization
channels carry more than $1\%$ of the co-polarized amplitude,
the kernel emits a warning recommending graded verification.

## Recommended workflow

The recommended workflow has four steps.

The first step is initial dispatch. Pass the roughness
parameters and let the kernel auto-dispatch through the matrix
above. The dispatcher records its choice in the audit log,
which is exposed through `Pipeline::audit_log()` in Rust and
through the `log` attribute of the result struct in Python. For
production fitting workflows the auto-dispatch is the
appropriate default because it captures the broadest range of
physical regimes and prevents the user from accidentally
selecting an out-of-validity model.

The second step is regime sanity check. After the initial
dispatch, run the same calculation with `Roughness.graded(...)`
and compare. If the two agree to $10^{-6}$, the dispatcher
choice is verified and the user can proceed with the cheaper
multiplicative model in subsequent sweeps. If they disagree,
the dispatcher choice was inappropriate for the physical
regime, and the user must fall back to the graded model or
revise the roughness parameters.

The third step is convergence verification when the graded
model is in use. Set `Roughness.graded(..., convergence_check=True)`
to enable automatic refinement and convergence assertion at the
default tolerance of $10^{-6}$ on absolute amplitude
differences. The convergence check doubles the runtime but is
the only way to certify the graded result.

The fourth step is fit re-evaluation. When fitting experimental
data, the optimizer will move the roughness parameters across
the regime boundaries during the fit, and a model that was
appropriate at the initial guess may become inappropriate at
the converged point. Re-running the dispatcher at the converged
parameters is the safe practice. The kernel supports this
through a `Pipeline::reaudit()` method that re-evaluates the
dispatch on the current parameters and emits a warning if the
recommended model has changed.

## Recommended defaults by application

The following defaults are appropriate starting points for the
common application classes. Each carries an explicit caveat
about the regime in which it should be re-evaluated.

For X-ray reflectivity at synchrotron wavelengths
($\lambda \sim 0.1\text{-}0.2$ nm), the default is Nevot-Croce
with $\sigma$ from the experimental fit. The DWBA conditions
are typically satisfied for polished substrates and metallic
overlayers. Re-evaluate when the fit pushes $\sigma > 1$ nm or
when the data span the critical-angle region for a heavy
substrate.

For EUV reflectometry ($\lambda \sim 13.5$ nm), the default
flips depending on the substrate. For polished silicon the
short-correlation length keeps Nevot-Croce valid up to
$\sigma \sim 4$ nm. For metal-coated optics the
correlation-length ratio approaches the crossover, and the
graded model is the safer default. Re-evaluate after the fit
converges.

For soft X-ray magneto-optic reflectivity at the resonance
edges of 3d transition metals, the default is graded. The
combination of substantial absorption and tensor anisotropy
puts the multiplicative models close to their validity bounds,
and the cross-polarization channels are sensitive to
roughness-induced phase shifts that the graded model captures
faithfully.

For visible-wavelength magneto-optical Kerr effect (MOKE)
spectroscopy, the default is graded. The Debye-Waller regime
applies in principle, but the high-contrast condition is
typically violated by metal-substrate interfaces, and the
graded model is the only physically defensible choice.

For neutron reflectometry, the default is Nevot-Croce. The
neutron wavelengths sit in the same range as X-rays for
typical experiments, the index contrasts are modest, and the
correlation lengths are short. The kernel reports the same
dispatch as for the analogous X-ray geometry.

## Output of the auto-dispatcher

The auto-dispatcher returns a `RoughnessDispatchResult` struct
that records the model chosen, the values of $\alpha$, $\beta$,
and the index-contrast magnitude at every interface, and any
warnings issued during the dispatch. A typical output for a
three-interface stack reads

```
RoughnessDispatchResult {
    interface_0: { model: NevotCroce, alpha: 0.18, beta: 0.05, contrast: 0.02 },
    interface_1: { model: Graded,     alpha: 0.42, beta: 0.31, contrast: 0.15,
                   warning: "alpha in crossover region 0.3-1.0" },
    interface_2: { model: NevotCroce, alpha: 0.09, beta: 0.04, contrast: 0.01 },
}
```

The struct is the canonical record of the dispatch decision,
and is included in any saved result so that downstream
re-analysis sees both the result and the model that produced
it. We surmise that this audit-trail behavior is essential for
reproducibility in fitting workflows where the same data are
re-analyzed months later by a different team member.

## When to override the auto-dispatcher

Three classes of situation justify overriding the auto-dispatch.

The first is when the user has a physical reason to expect a
specific model regardless of the dispatcher choice. A
chemically interdiffused interface is graded by physics, not
by morphology, and should be modeled with the graded approach
even when the multiplicative dispatcher would accept it.

The second is when computational budget forbids the graded
model and the user accepts the resulting accuracy degradation.
For large parameter sweeps over many wavelengths or many
incidence angles, the $30\times$ to $100\times$ overhead of the
graded model can be prohibitive, and a multiplicative model
with documented inaccuracy is preferable to no result. The
kernel emits a warning in this case but does not refuse the
override.

The third is when comparing against published reference data
that was computed with a specific model. Reproducing the
Esashi figures, for instance, requires forcing the model that
Esashi used at each panel rather than letting the dispatcher
choose. The kernel exposes the override through the
`RoughnessChoice` enum on the `RoughnessSpec`, and explicit
choices are honored without dispatch.

## References

1. Y. Esashi, M. Tanksalvala, Z. Zhang, N. W. Jenkins, H. C.
   Kapteyn, and M. M. Murnane, "Influence of surface and
   interface roughness on X-ray and extreme ultraviolet
   reflectance: A comparative numerical study," OSA Continuum
   **4**, 1497 (2021).
   [DOI](https://doi.org/10.1364/OSAC.422924).
2. S. K. Sinha, E. B. Sirota, S. Garoff, and H. B. Stanley,
   "X-ray and neutron scattering from rough surfaces," Phys.
   Rev. B **38**, 2297 (1988).
   [DOI](https://doi.org/10.1103/PhysRevB.38.2297).
3. D. L. Windt, "IMD: Software for modeling the optical
   properties of multilayer films," Comput. Phys. **12**, 360
   (1998).
   [DOI](https://doi.org/10.1063/1.168689).
