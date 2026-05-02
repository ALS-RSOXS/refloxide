# Eigenmode analysis per layer

## Scope of this page

This page treats stage 2 of the pipeline. Given the 4x4 Berreman
matrix $\Delta_i$ produced in [Foundations](foundations.md),
we solve the algebraic eigenvalue problem in each homogeneous layer
$i$, partition the four resulting modes into forward and backward
propagation, and impose the continuity ordering of Li, Sullivan,
and Parsons [[1](#references)] so that mode labels remain consistent
when the dielectric tensor is swept smoothly across parameter space.
The treatment follows Passler and Paarmann [[2](#references), Sec.
2.A.1 and 2.A.2, Eqs. (11)-(18)], who adapted the Li sorting rule
into the 4x4 machinery.

## The eigenvalue problem

Because $\Delta_i$ is $z$-independent inside layer $i$, the
spatial wave equation
$\partial_z \Psi = i(\omega/c)\,\Delta_i\,\Psi$
decouples into four plane-wave solutions that share the same
in-plane wave vector $\xi$ but carry distinct out-of-plane
components $q_{ij}$ [[2](#references), Eq. (11)],

$$
q_{ij}\,\Psi_{ij}(\Delta_i) = \Delta_i\,\Psi_{ij}(\Delta_i),
\quad j = 1, 2, 3, 4.
$$

The four $q_{ij}$ are the eigenvalues of $\Delta_i$, which in
numerical practice are obtained via a standard dense eigensolver.
The four $\Psi_{ij}$ are the right eigenvectors and carry the
polarization content of each mode in the ordered component basis
$(E_x, H_y, E_y, -H_x)^\top$ fixed by [[2](#references), Eq. (7)].

## Forward and backward partition

The first sorting step separates the four modes into the two pairs
that propagate or decay toward $+\hat{z}$ (transmitted) and toward
$-\hat{z}$ (reflected). The rule given by [[2](#references),
Eq. (12)] is

$$
\begin{aligned}
q_{ij} \text{ real,} &\quad q_{ij} \ge 0 \Rightarrow \text{transmitted},
\quad q_{ij} < 0 \Rightarrow \text{reflected}, \\
q_{ij} \text{ complex,} &\quad \operatorname{Im}(q_{ij}) \ge 0
\Rightarrow \text{transmitted},\\
&\quad \operatorname{Im}(q_{ij}) < 0 \Rightarrow \text{reflected}.
\end{aligned}
$$

The two rules are consistent, because a real $q_{ij}$ points a
Poynting vector in its own direction whereas a complex $q_{ij}$
with $\operatorname{Im}(q_{ij}) > 0$ describes an exponentially
damped wave, which in the assumed sign convention
$\exp(-i(\omega/c)q_{ij}z)$ corresponds to decay in the $+\hat{z}$
direction.

The two transmitted modes are labeled $q_{i1}$ and $q_{i2}$, and
the two reflected modes are labeled $q_{i3}$ and $q_{i4}$
[[2](#references), text following Eq. (12)]. Which of the two
transmitted modes is assigned index $1$ rather than $2$ is the
question answered by the next step.

## Continuity sorting within each pair

A naive eigensolver returns eigenvectors in arbitrary permutation,
so a dielectric tensor that is swept continuously through parameter
space produces output spectra that flip labels discontinuously. The
physical observables $r_{kl}$ and $t_{kl}$ then appear to jump
between $kl = pp$ and $kl = ss$ branches across the sweep. This
failure mode was diagnosed and corrected by Li, Sullivan, and
Parsons in the context of magneto-optic recording layers
[[1](#references)], and we adopt their projection rule as the
within-pair sorting criterion.

The rule uses a scalar projection functional of the eigenvector
itself. For moderately anisotropic tensors,

$$
C(q_{ij}) = \frac{|\Psi_{ij,1}|^2}
                 {|\Psi_{ij,1}|^2 + |\Psi_{ij,3}|^2},
$$

is the ratio of the squared $E_x$ amplitude to the squared in-plane
electric amplitude of the mode [[2](#references), Eq. (13)]. For a
canonically $p$-polarized mode the electric field lies in the $x$-
$z$ plane, so $\Psi_{ij,3} = E_y = 0$ and $C(q_{ij}) = 1$. For a
canonically $s$-polarized mode the electric field is along
$\hat{y}$, so $\Psi_{ij,1} = E_x = 0$ and $C(q_{ij}) = 0$. The
sorting rule is then [[2](#references), Eq. (14)]

$$
C(q_{i1}) > C(q_{i2}) \quad\text{and}\quad
C(q_{i3}) > C(q_{i4}),
$$

so $q_{i1}$ and $q_{i3}$ are the $p$-like transmitted and reflected
modes, and $q_{i2}$ and $q_{i4}$ are the $s$-like ones.

## Birefringent fallback to the Poynting criterion

When a principal axis of the dielectric tensor lies outside the
$x$-$z$ plane and outside $\hat{y}$, the eigenmodes are no longer
purely $p$ or $s$ and the $E$-field projection becomes ambiguous.
Both $\Psi_{ij,1}$ and $\Psi_{ij,3}$ are non-zero for all four
modes, and the $C$-functional no longer cleanly separates the
pair. In this regime Passler and Paarmann replace the electric
projection with the analogous Poynting-vector projection
[[2](#references), Eq. (15)],

$$
C(q_{ij}) = \frac{|S_{ij,x}|^2}
                 {|S_{ij,x}|^2 + |S_{ij,y}|^2},
$$

where $\vec{S}_{ij} = \vec{E}_{ij} \times \vec{H}_{ij}$ is
evaluated component-wise. The tangential Poynting components are
recovered from the eigenvector using the identifications
$E_{ij,x} = \Psi_{ij,1}$, $E_{ij,y} = \Psi_{ij,3}$,
$H_{ij,x} = -\Psi_{ij,4}$, $H_{ij,y} = \Psi_{ij,2}$, with the
longitudinal components $E_{ij,z}$ and $H_{ij,z}$ recovered from
the auxiliary linear combinations of the $a_{3n}$ and $a_{6n}$
coefficients,

$$
\begin{aligned}
E_{ij,z} &= a_{31}(i)\,E_{ij,x} + a_{32}(i)\,E_{ij,y}
          + a_{34}(i)\,H_{ij,x} + a_{35}(i)\,H_{ij,y}, \\
H_{ij,z} &= a_{61}(i)\,E_{ij,x} + a_{62}(i)\,E_{ij,y}
          + a_{64}(i)\,H_{ij,x} + a_{65}(i)\,H_{ij,y},
\end{aligned}
$$

as enumerated in [[2](#references), Eqs. (17)-(18)]. This is the
same $a_{3n}$ and $a_{6n}$ tabulation [[2](#references), Eq. (9)]
that populates $\Delta$ itself, now reused for the Poynting
reconstruction.

The Poynting criterion is more expensive than the electric
criterion because it requires assembling $\vec{E}$ and $\vec{H}$
fully before the projection, whereas the electric criterion reads
directly off the eigenvector. We surmise that a practical
implementation should dispatch on the tensor geometry, defaulting
to the cheaper electric projection and switching to the Poynting
projection whenever the principal-axis alignment fails.

## What the sorting buys the rest of the pipeline

The ordered mode quadruplet $(q_{i1}, q_{i2}, q_{i3}, q_{i4})$
becomes the convention relied on in stages 3 through 6. Stage 3
(see [Interface matrices](interface_matrices.md)) assembles
the interface matrix $A_i$ with column layout
$(\text{p-trans}, \text{s-trans}, \text{p-refl}, \text{s-refl})$,
the Xu $\gamma_{ij}$ eigenvectors [[2](#references), Eqs. (19)-(20);
[3](#references)] are written in the same order, and the
Yeh-layout permutation $\Lambda_{1324}$ introduced in
[Propagation and assembly](propagation_and_assembly.md) relies
on exactly this ordering. A violation of the Li sorting would
therefore propagate as a mis-permutation of $\Gamma_N$ and would
surface as discontinuous $r_{kl}(\omega)$ or $r_{kl}(\theta)$
spectra in whatever observable is requested.

## Where the code lives

In `refloxide`, the eigensolve and the sorting live in the
`core::modes` module. The module consumes $\Delta_i$ from
`core::delta`, returns the quadruplet $(q_{ij}, \Psi_{ij})$ in the
ordered layout above, and exposes the sorting criterion as a
user-visible switch so that the Poynting fallback can be forced on
when the caller knows a principal axis is tilted. Targeted unit
tests against the Fresnel limit, against Yeh's birefringent
multilayer benchmark, and against a smoothly swept dielectric
tensor near a degenerate point are the appropriate validation
battery.

## Edge cases and limitations

The Li criterion is well defined whenever exactly two of the four
modes are forward-propagating, which is the generic case.
Four-fold degeneracies, which occur in exactly isotropic media at
exactly normal incidence, leave the within-pair order formally
ambiguous. In that limit the $p$ and $s$ labels carry no physical
content and the pipeline returns the correct answer regardless of
the label assignment. We note that this degeneracy is the one Xu,
Wood, and Golding treat separately at the eigenvector level
[[3](#references); [2](#references), Eq. (19) branch
$q_{i1} = q_{i2}$], and the Xu piecewise construction (see
[Interface matrices](interface_matrices.md)) is the mechanism
that keeps the rest of the pipeline finite through the degenerate
limit.

The forward and backward classification on $\operatorname{Im}(q)$
alone presumes the loss sign convention built into the dielectric
tensor. A caller who feeds a gain medium with
$\operatorname{Im}(\varepsilon) < 0$ inverts the classification and
produces physically backward-decaying modes labeled forward. The
library does not guard against this, because a consistent gain
convention is the caller's responsibility.

## References

1. Z.-M. Li, B. T. Sullivan, and R. R. Parsons, "Use of the 4x4
   matrix method in the optics of multilayer magneto-optic
   recording media," Appl. Opt. **27**, 1334 (1988).
   [DOI](https://doi.org/10.1364/AO.27.001334).
2. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
   for light propagation in anisotropic stratified media," J. Opt.
   Soc. Am. B **34**, 2128 (2017).
   [DOI](https://doi.org/10.1364/JOSAB.34.002128).
3. W. Xu, L. T. Wood, and T. D. Golding, "Optical degeneracies in
   anisotropic layered media," Phys. Rev. B **61**, 1740 (2000).
   [DOI](https://doi.org/10.1103/PhysRevB.61.1740).
