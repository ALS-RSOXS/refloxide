# Reflection and transmission coefficients

## Scope of this page

This page treats stage 5 of the pipeline. Given the Yeh-layout
transfer matrix $\tilde{\Gamma}*N$ from
`[propagation_and_assembly.md](propagation_and_assembly.md)`, we
extract the eight complex amplitude coefficients
$r*{pp}, r_{ss}, r_{ps}, r_{sp}, t_{pp}, t_{ss}, t_{ps}, t_{sp}$
from rational combinations of its matrix elements, and we document
the sign and normalization corrections imposed by the
Passler-Paarmann erratum [[1](#references)]. We also record the
relabeling $p/s \to o/e$ required for birefringent substrates, and
the open status of the transmittance identity in anisotropic
substrates.

The closed-form expressions follow Yeh [[2](#references)] and are
tabulated by Passler and Paarmann [[3](#references), Eqs.
(33)-(36)], with the sign corrections of [[1](#references), Eqs.
(33*)-(36*)].

## From $\tilde{\Gamma}_N$ to the amplitude coefficients

Let $\tilde{\Gamma}*N$ have matrix elements $\Gamma*{\alpha\beta}$,
where the $(\alpha, \beta)$ indexing uses the Yeh row order
$(E^p_{\text{trans}}, E^p_{\text{refl}},
  E^s_{\text{trans}}, E^s_{\text{refl}})$ [[3](#references),
Eq. (30)]. For an incident medium that is isotropic and
non-absorbing, the reflection and transmission coefficients read,
using the erratum-corrected sign convention [[1](#references), Eqs.
(33*)-(36*)],

$$
r_{pp} = \frac{\Gamma_{21}\Gamma_{33} - \Gamma_{23}\Gamma_{31}}
              {\Gamma_{11}\Gamma_{33} - \Gamma_{13}\Gamma_{31}},
\qquad
t_{pp} = \frac{\Gamma_{33}}
              {\Gamma_{11}\Gamma_{33} - \Gamma_{13}\Gamma_{31}},
$$

$$
r_{ss} = \frac{\Gamma_{11}\Gamma_{43} - \Gamma_{41}\Gamma_{13}}
              {\Gamma_{11}\Gamma_{33} - \Gamma_{13}\Gamma_{31}},
\qquad
t_{ss} = \frac{\Gamma_{11}}
              {\Gamma_{11}\Gamma_{33} - \Gamma_{13}\Gamma_{31}},
$$

$$
r_{ps} = \frac{\Gamma_{41}\Gamma_{33} - \Gamma_{43}\Gamma_{31}}
              {\Gamma_{11}\Gamma_{33} - \Gamma_{13}\Gamma_{31}},
\qquad
t_{ps} = -\frac{\Gamma_{31}}
                {\Gamma_{11}\Gamma_{33} - \Gamma_{13}\Gamma_{31}},
$$

$$
r_{sp} = \frac{\Gamma_{11}\Gamma_{23} - \Gamma_{21}\Gamma_{13}}
              {\Gamma_{11}\Gamma_{33} - \Gamma_{13}\Gamma_{31}},
\qquad
t_{sp} = -\frac{\Gamma_{13}}
                {\Gamma_{11}\Gamma_{33} - \Gamma_{13}\Gamma_{31}}.
$$

The subscript convention is the usual one in ellipsometry. The
first subscript $k$ labels the outgoing polarization and the
second $l$ labels the incident polarization, so $r_{ps}$ is the
amplitude of the outgoing $p$ wave per unit incident $s$ wave
[[3](#references), text preceding Eq. (33)]. The common
denominator $\Gamma_{11}\Gamma_{33} - \Gamma_{13}\Gamma_{31}$ is
the same in all eight expressions and is the determinant of the
forward-transmitted sub-block of $\tilde{\Gamma}_N$.

## The sign correction of the 2019 erratum

The 2017 paper [[3](#references), Eqs. (33)-(36)] gave $t_{ss}$,
$t_{ps}$, and $t_{sp}$ with an opposite sign convention from
Yeh's original derivation [[2](#references)], carried through
deliberately for consistency with the field reconstruction of the
2017 paper. The 2019 erratum restores the Yeh convention, because
the opposite convention produced incorrect electric-field
components in birefringent media [[1](#references), Sec. 2.B]. The
transmittance $T_{kl} = |t_{kl}|^2$ is unchanged, but the phase of
the transmitted field is not, and the phase matters for the field
reconstruction of stage 6 (see
`[electric_field_distribution.md](electric_field_distribution.md)`).
`refloxide` implements the erratum-corrected signs. A library that
implements the 2017 signs verbatim and then uses the 2017 field
reconstruction recipe produces self-consistent but incorrect
fields in birefringent stacks.

## Reflectance

Because the incident medium is isotropic and non-absorbing (see the
assumptions in `[overview.md](overview.md#assumptions)`), the
reflectance is the modulus-squared of the amplitude coefficient,

$$
R_{kl} = |r_{kl}|^2 \quad \text{for } k, l \in s, p.
$$

This identity follows from the Poynting flux projection in the
incident half-space and does not require any anisotropy adjustment
on the incident side [[3](#references), Sec. 2.B]. The full angular
distribution of reflected intensity is the $2 \times 2$ Jones
reflectance matrix with components $R_{pp}$, $R_{ss}$, $R_{ps}$,
$R_{sp}$.

## Transmittance, and the open question

The symmetric identity $T_{kl} = |t_{kl}|^2$ is true only in the
degenerate case where the substrate is vacuum. For a general
anisotropic substrate the transmittance depends on the Poynting
projection on the substrate side, which in turn depends on the
eigenvalue $q_{i,N+1}$ of the transmitted mode, on the dielectric
tensor of the substrate, and on the refractive index of the
incident medium [[1](#references), Sec. 2.B, discussion preceding
Eqs. (33*)-(36*)]. The correct expression was deferred by Passler
and Paarmann to a later publication [[1](#references), Ref. 6 of
the erratum]. As of the sources we cite here, no general
closed-form transmittance identity has been published.

We flag this as a known gap. The kernel computes and returns the
eight $t_{kl}$ amplitudes, which are physically meaningful and
complete. We do not currently expose $T_{kl}$ as a derived
observable, and we prefer to leave the conversion to the caller
(with the caveat that the isotropic-substrate identity
$T_{kl} = |t_{kl}|^2 \operatorname{Re}(n_{N+1}\cos\theta_{N+1})
/ \operatorname{Re}(n_0\cos\theta_0)$ is the one classical result
that we cover in a docstring example). A generalized transmittance
module will be added once the Passler followup publication is
available, or once we are willing to derive it independently.

## Birefringent substrates and the $o/e$ relabeling

When the substrate is birefringent, the two transmitted modes are
no longer purely $p$-polarized and $s$-polarized but are the
ordinary (o) and extraordinary (e) eigenmodes of the substrate's
dielectric tensor [[1](#references), Sec. 2.B, text following Eqs.
(33*)-(36*)]. Both eigenmodes carry non-zero $E_x$ and non-zero
$E_y$ in general, so the Jones layout $(p, s)$ on the outgoing side
is no longer natural.

The relabeling takes the 2017 coefficients and renames them
[[1](#references), Sec. 2.B],

$$
t_{pp} \to t_{po},\quad
t_{ss} \to t_{se},\quad
t_{ps} \to t_{pe},\quad
t_{sp} \to t_{so},
$$

with the analogous renaming applied to $r_{kl}$. The numerical
values of the matrix elements $\Gamma_{\alpha\beta}$ are unchanged,
only the semantic labeling of which transmitted mode each
coefficient addresses. The calling code is responsible for
interpreting these in the ordinary and extraordinary eigenbasis of
the substrate.

For lab-frame-diagonal substrates, which is the common case, the
ordinary and extraordinary modes coincide with the canonical $s$
and $p$ labels, and the relabeling is the identity. A library that
wants to support both regimes uniformly, as `refloxide` does,
returns the eight coefficients under the neutral names $t_{11}$,
$t_{22}$, $t_{12}$, $t_{21}$ (and analogously for $r$) and exposes
label aliases in a separate semantic layer.

## Ellipsometric observables

The amplitude coefficients feed directly into the standard
ellipsometric observables. For a single reflected $p$ and $s$ ratio,

$$
\rho = \tan\Psi  e^{i\Delta} = \frac{r_{pp}}{r_{ss}},
$$

which defines the ellipsometric angles $(\Psi, \Delta)$ consumed by
fitting frameworks. For anisotropic samples with non-negligible
cross-polarization, the full $2 \times 2$ Jones or $4 \times 4$
Mueller matrix constructed from the $r_{kl}$ set is the appropriate
observable. We do not construct Mueller matrices in the kernel,
because the incoherent averaging they encode is a depolarization
operation that belongs in the fitting layer.

## Where the code lives

Stage 5 is the `core::coefficients` module. It consumes
$\tilde{\Gamma}_N$ and returns the eight amplitudes as a named
struct. The rational expressions are written out verbatim from
[[1](#references), Eqs. (33*)-(36*)] rather than being rederived in
code, so that a reader comparing the implementation to the erratum
can trace the mapping line by line. The common denominator is
computed once per call and shared across the eight returns.

## Numerical notes

The denominator $\Gamma_{11}\Gamma_{33} - \Gamma_{13}\Gamma_{31}$
vanishes in two regimes. The first is total reflection with a
non-absorbing substrate, where the incident mode couples entirely
to the reflected basis and the forward-transmitted sub-block is
rank-deficient. The second is at a guided-mode resonance of the
stack, where a lossless pole of $r_{kl}$ coincides with a zero of
the denominator. Both regimes produce physically finite
observables when evaluated carefully, because the numerators
vanish at the same rate as the denominator. The library does not
branch on proximity to these zeros, because the complex arithmetic
survives the $0/0$ cancellation at standard double precision when
the stack is built correctly.

## References

1. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
  for light propagation in anisotropic stratified media, erratum,"
   J. Opt. Soc. Am. B **36**, 3246 (2019).
   [DOI](https://doi.org/10.1364/JOSAB.36.003246).
2. P. Yeh, "Electromagnetic propagation in birefringent layered
  media," J. Opt. Soc. Am. **69**, 742 (1979).
   [DOI](https://doi.org/10.1364/JOSA.69.000742).
3. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
  for light propagation in anisotropic stratified media," J. Opt.
   Soc. Am. B **34**, 2128 (2017).
   [DOI](https://doi.org/10.1364/JOSAB.34.002128).

