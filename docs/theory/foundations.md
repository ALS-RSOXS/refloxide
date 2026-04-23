# Foundations

## Scope of this page

This page derives the 4x4 differential system that underlies every
subsequent stage of the pipeline. It follows the reduction given by
Berreman [[1](#references)] and recast in modern notation by
Passler and Paarmann [[2](#references), Sec. 2.A.1 and Eqs.
(1)-(10)]. The reader arriving from the [overview](overview.md) or
the [pipeline summary](pipeline.md) will find the algebraic steps
here and will then continue to the [eigenmode
analysis](eigenmode_analysis.md).

## Geometry and conventions

Each layer $i$ is homogeneous in its dielectric response, is bounded
by planar interfaces normal to $\hat{z}$, and has thickness $d_i$.
The incident beam propagates in the $x$-$z$ plane, so the in-plane
wave vector component
$\xi = \sqrt{\varepsilon_{\text{inc}}}\sin\theta$
is real and common to every layer [[2](#references), Eq. (1)]. The
out-of-plane component is $q_i$, dimensionless, in units of
$\omega/c$. The incident medium is $i = 0$, the substrate is
$i = N+1$, and the origin of $z$ is placed at the first interface.

The dielectric tensor of each layer is specified in the principal
frame as $\operatorname{diag}(\varepsilon_x, \varepsilon_y,
\varepsilon_z)$ and rotated into the lab frame by three Euler
angles,

$$
\bar{\varepsilon}' = \Omega \bar{\varepsilon} \Omega^{-1},
$$

with $\Omega$ as in [[2](#references), Eq. (2)] and the convention
given there. Optical activity and static-field responses can be
folded into $\bar{\varepsilon}$ at the cost of making it
non-diagonal in the principal frame [[2](#references), Sec. 2.A.1].
We do not carry those terms explicitly in the foundation, although
the formalism admits them.

## Six-component Maxwell system

Time-harmonic fields of frequency $\omega$ satisfy Maxwell's curl
equations in the form that Berreman wrote as a 6x6 system relating
the six-component generalized field
$\vec{G} = (E_x, E_y, E_z, H_x, H_y, H_z)^\top$
to its spatial derivatives [[1](#references); [2](#references),
Eq. (3)]. Passler and Paarmann retain Berreman's geometric operator
but invert the direction of the constitutive closure, writing

$$
\vec{C} = M\vec{G} \equiv
\begin{pmatrix} \bar{\varepsilon} & \bar{\rho}_1 
                \bar{\rho}_2 & \bar{\mu} \end{pmatrix} \vec{G},
$$

where $\vec{C}$ collects $(\vec{D}, \vec{B})$ and $\bar{\rho}_{1,2}$
are optical rotation tensors [[2](#references), Eq. (4)]. This
matters for cross-checking against the Berreman original, where the
inverse relation $\vec{G} = M\vec{C}$ is tabulated instead
[[2](#references), remark after Eq. (4)].

Elimination of the in-plane derivatives in the curl equations then
produces a spatial wave equation

$$
R\vec{g} = -i\omega M\vec{g},
$$

with $\vec{g}$ the spatial part of $\vec{G} = \vec{g}e^{-i\omega
t}$ and $R$ the 6x6 matrix of $z$-derivatives with the in-plane
$\xi$ already inserted [[2](#references), Eq. (5)]. The explicit
form of $R$ is the Berreman block structure in which only $E_z$ and
$H_z$ appear as static rows [[1](#references)].

## Elimination of the longitudinal components

The two rows of $R\vec{g} = -i\omega M\vec{g}$ that involve $E_z$
and $H_z$ without $\partial_z$ derivatives are algebraic. Solving
them for $E_z$ and $H_z$ in terms of the remaining four components
and back-substituting yields a closed 4x4 system in
$\Psi = (E_x, H_y, E_y, -H_x)^\top$ [[2](#references), Eq. (7)].
The resulting equation is

$$
\frac{\partial \Psi}{\partial z} = i\frac{\omega}{c}\Delta\Psi,
$$

the form that Berreman called the "matrix method" and that every
subsequent 4x4 formalism is a specialization of [[1](#references);
[2](#references), Eq. (6)]. The component ordering of $\Psi$ was
chosen by Berreman to make the boundary conditions at an interface
manifest, since $E_x$, $H_y$, $E_y$, and $H_x$ are the tangential
components that match across a planar interface.

## The Berreman 4x4 matrix

The elements of $\Delta$ are bilinear combinations of elements of
$M$, the in-plane wave vector $\xi$, and the auxiliary coefficients
$a_{3n}$ and $a_{6n}$ that arose from eliminating $E_z$ and $H_z$.
The complete tabulation is given in [[2](#references), Eq. (8)], of
which we reproduce representative entries,

$$
\begin{aligned}
\Delta_{11} &= M_{51} + (M_{53} + \xi)a_{31} + M_{56}a_{61}, \\
\Delta_{21} &= M_{11} + M_{13}a_{31} + M_{16}a_{61}, \\
\Delta_{31} &= -M_{41} - M_{43}a_{31} - M_{46}a_{61}, \\
\Delta_{41} &= M_{21} + M_{23}a_{31} + (M_{26} - \xi)a_{61}.
\end{aligned}
$$

The full 16 entries, together with the $a_{3n}$ and $a_{6n}$
coefficients that populate them, are enumerated in
[[2](#references), Eqs. (8)-(10)]. The scalar $b = M_{33}M_{66} -
M_{36}M_{63}$ of [[2](#references), Eq. (10)] appears in every
$a_{3n}$ and $a_{6n}$ denominator and sets the condition under
which the longitudinal elimination is non-singular.

Two conventions merit explicit mention so that a reader comparing
to Berreman does not confuse herself. Passler and Paarmann absorb
the factor $\omega/c$ into the overall prefactor of $\Delta$, so
$\Delta$ itself is dimensionless and $\xi$ appears without a
prefactor [[2](#references), remark after Eq. (10)]. Berreman's
original tabulation in [[1](#references), Eq. (7)] carries the
$\omega/c$ inside $\xi$ instead. The two forms are algebraically
equivalent.

## Where the code lives

In `refloxide`, the $\Delta$ matrix is the output of the
`core::delta` module. That module consumes a rotated dielectric
tensor and returns the 4x4 $\Delta_i$ per layer, evaluated at a
specified $(\xi, \omega)$. The coefficients $a_{3n}$, $a_{6n}$, and
$b$ live inside the same module because they are strictly
intermediate and serve no downstream purpose outside populating
$\Delta$. The layer-dependent Euler rotation of the principal
tensor [[2](#references), Eq. (2)] is a pre-processing step applied
before $\Delta$ is built.

## Assumptions implicit in the reduction

The reduction used here assumes $z$-independence of $M$ within a
layer, i.e. that each layer is genuinely homogeneous
[[2](#references), remark preceding Eq. (11)]. A $z$-dependent $M$
would require the numerical ODE integration sketched by Berreman
[[1](#references)], which is what `refloxide` avoids by discretizing
graded interfaces into thin homogeneous sublayers inside the
roughness pipeline (see
`[roughness_graded_interface.md](roughness_graded_interface.md)`).

The elimination of $E_z$ and $H_z$ assumes $b \neq 0$. The scalar
$b$ vanishes only in pathological constitutive relations that we do
not encounter in ordinary magneto-dielectric media, so no numerical
safeguard is required at this stage. The guard against singularities
enters later, at the eigenvector level, through the Xu piecewise
definitions [[3](#references); [4](#references)] treated in
`[interface_matrices.md](interface_matrices.md)`.

## References

1. D. W. Berreman, "Optics in stratified and anisotropic media, 4x4
  matrix formulation," J. Opt. Soc. Am. **62**, 502 (1972).
   [DOI](https://doi.org/10.1364/JOSA.62.000502).
2. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
  for light propagation in anisotropic stratified media," J. Opt.
   Soc. Am. B **34**, 2128 (2017).
   [DOI](https://doi.org/10.1364/JOSAB.34.002128).
3. W. Xu, L. T. Wood, and T. D. Golding, "Optical degeneracies in
  anisotropic layered media," Phys. Rev. B **61**, 1740 (2000).
   [DOI](https://doi.org/10.1103/PhysRevB.61.1740).
4. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
  for light propagation in anisotropic stratified media, erratum,"
   J. Opt. Soc. Am. B **36**, 3246 (2019).
   [DOI](https://doi.org/10.1364/JOSAB.36.003246).