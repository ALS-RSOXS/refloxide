# Theory overview

## What this page is

This is the landing page for the theoretical documentation of
`refloxide`. It states the problem solved by the package, fixes the
assumptions that underlie the solution, contrasts the 4x4 transfer
matrix method with the full-wave Maxwell solvers a reader might
otherwise reach for, lists the experimental regimes where the
formalism is the right tool, and routes the reader to the derivation
companion files. Nothing is derived on this page. Derivations live
in the companion files and are cross-linked from the table of
contents below.

A reader who wants historical lineage should proceed to
[History](history.md). A reader who wants a stage-by-stage
summary of the algorithm, without full derivations, should proceed
to [Pipeline at a glance](pipeline.md). Everything else on this page is
scope, applications, and pointers.

## Table of contents

The theory section is organized along two axes, the per-stage
breakdown of the core 4x4 pipeline and the orthogonal treatment of
interfacial roughness. Narrative context pages come first.

Narrative context.

- [History](history.md), the lineage from Fresnel through
Abelès, Parratt, Berreman, Yeh, Lin-Chung and Teitler, Xu, Li, and
Passler, plus the separate lineage of the three roughness models.
- [Pipeline at a glance](pipeline.md), a compact walkthrough of the six
stages executed per measurement coordinate.

Core pipeline.

- [Foundations](foundations.md), the 6x6 Maxwell system, the
constitutive relation $\vec{C} = M\vec{G}$, elimination of $E_z$
and $H_z$, and the resulting Berreman 4x4 matrix $\Delta$.
- [Eigenmode analysis](eigenmode_analysis.md), the per-layer
eigenproblem $q_{ij}\Psi_{ij} = \Delta_i \Psi_{ij}$, the
forward-backward partition, and the Li-Sullivan-Parsons ordering
rule.
- [Interface matrices](interface_matrices.md), the
Xu-Wood-Golding piecewise eigenvectors $\vec{\gamma}_{ij}$, the
erratum-corrected components and normalization convention, and
the boundary-matching matrix $A_i$.
- [Propagation and assembly](propagation_and_assembly.md),
the diagonal propagation matrix $P_i$, the layer transfer matrix
$T_i = A_i P_i A_i^{-1}$, the stack product
$\Gamma_N = L_1 P_1 L_2 P_2 \cdots L_N P_N L_{N+1}$, and the
$\Lambda_{1324}$ permutation.
- [Reflection and transmission](reflection_transmission.md), the
rational extraction of the eight amplitude coefficients
$r_{kl}$ and $t_{kl}$ from $\Gamma_N$, with ordinary and
extraordinary relabeling for birefringent substrates.
- [Electric field distribution](electric_field_distribution.md),
the erratum-corrected reconstruction of $\vec{E}(x, y, z)$ inside
the stack.

Roughness models.

- [Framework](roughness_framework.md), how the three
roughness models plug into (or around) the core pipeline.
- [Nevot-Croce](roughness_nevot_croce.md), the
distorted-wave Born approximation factor applicable to short
correlation lengths.
- [Debye-Waller](roughness_debye_waller.md), the
long-correlation-length counterpart and its relation to the
crystallographic factor.
- [Graded interface](roughness_graded_interface.md),
the Stearns slide method that replaces a sharp step by a
discretized continuous index profile.
- [Selection guide](roughness_selection_guide.md),
the decision framework that mirrors the quantitative comparison
of [[Esashi et al. 2021](#references)].

## Problem statement

A plane wave of angular frequency $\omega$ and in-plane wave vector
component
$\xi = \sqrt{\varepsilon_{\text{inc}}}\sin\theta$
is incident from an isotropic, non-absorbing half space $i = 0$ onto
a stack of $N$ homogeneous layers, each of thickness $d_i$ and
characterized by a (generally complex, generally anisotropic)
dielectric tensor $\bar{\varepsilon}_i$. The substrate $i = N+1$ is
a second half space, which may itself be anisotropic and absorbing.
The observables are the four reflection amplitudes $r_{pp}$,
$r_{ss}$, $r_{ps}$, $r_{sp}$, the four transmission amplitudes
$t_{pp}$, $t_{ss}$, $t_{ps}$, $t_{sp}$, and optionally the full
$\vec{E}(x, y, z)$ distribution inside every layer.

Throughout the theory section we use dimensionless units in which
the in-plane wave vector component is $\xi$ and the out-of-plane
components $q_{ij}$ (four modes $j$ per layer $i$) are the
$z$-projections of $\vec{k}$ in units of $\omega/c$.

## Applications

The formalism is well matched to the experimental techniques and
device classes enumerated below. In every case the sample is either
genuinely planar or well approximated as such on the relevant length
scale.

Optical reflectometry and
[ellipsometry](https://en.wikipedia.org/wiki/Ellipsometry) on thin
films, multilayer coatings, and heterostructures. These measurements
extract dielectric functions and layer thicknesses by fitting the
model to data, and they exercise every element of the pipeline.

[X-ray reflectometry
(XRR)](https://en.wikipedia.org/wiki/X-ray_reflectivity) and extreme
ultraviolet reflectometry (EUVR) on polished substrates, multilayer
mirrors, and semiconductor stacks. These techniques are the primary
driver of interest in the three roughness models and motivated the
Esashi comparative study [[Esashi et al. 2021](#references)].

[Neutron reflectometry](https://en.wikipedia.org/wiki/Neutron_reflectometry)
on thin films and soft-matter multilayers. The same 4x4 machinery
applies when the wave equation is replaced by the Schrödinger
equation with a suitable optical potential.

Infrared and terahertz studies of [surface phonon
polaritons](https://en.wikipedia.org/wiki/Surface_phonon_polariton)
in polar dielectric heterostructures, including the Otto geometry
simulations worked out in [[Passler and Paarmann 2017](#references)].
Accurate treatment of anisotropic dielectric tensors is essential
here, which rules out scalar formalisms.

Resonant soft x-ray reflectivity with tensorial magnetic or orbital
contrast, where the dielectric tensor itself is the observable and
scalar treatments are insufficient.

Optical thin-film design for anti-reflection coatings, dielectric
mirrors, and beam splitters, where the forward calculation must be
fast enough to drive optimization loops.

The formalism is not well suited to sub-wavelength patterned
surfaces, metasurfaces with transverse structure, localized
resonances in nanoparticles, or problems that depend on diffuse
scattering rather than specular reflection.

## Assumptions

The derivation is only as good as its premises. We list them
explicitly so that the user knows when to reach for a different
tool.

The medium is assumed to be a stack of homogeneous layers separated
by planar, transversely infinite interfaces normal to $\hat{z}$.
Lateral structure (gratings, patterned surfaces, diffusion profiles
that vary in the $x$ or $y$ direction) is outside the model.
Problems with in-plane periodicity should be handled by [rigorous
coupled-wave analysis](https://en.wikipedia.org/wiki/Rigorous_coupled-wave_analysis)
or its generalizations.

The incident field is a monochromatic [plane
wave](https://en.wikipedia.org/wiki/Plane_wave). Time-domain or
broadband problems are recovered by Fourier synthesis over frequency
after the fact. Focused beams, Gaussian beams, and near-field tips
are outside the model, though a plane-wave expansion can extend the
formalism to those cases at additional cost.

The media are linear. Non-linear effects (second-harmonic
generation, Kerr, and so on) are not in the core kernel, although
the formalism is known to extend straightforwardly into the
non-linear regime with distributed sources, as sketched by Passler
and Paarmann [[Passler and Paarmann 2017](#references)].

The magnetic permeability is taken as a scalar, $\bar{\mu}_i =
\mu_i \mathbb{1}$, and no optical activity or magnetic anisotropy
is included. The algorithm admits those extensions, but they
require different $\gamma$ eigenvectors and are not part of the
initial implementation.

Roughness is treated in a transversely averaged sense. Either it is
small enough and short-correlated enough that the
[Névot-Croce](https://en.wikipedia.org/wiki/Nevot%E2%80%93Croce_factor)
(or [Debye-Waller](https://en.wikipedia.org/wiki/Debye%E2%80%93Waller_factor))
multiplicative correction to Fresnel coefficients applies, or else
the index profile is replaced by a graded stack. Diffuse scattering
and coherent off-specular features are not modeled. Problems where
diffuse scattering carries the information of interest (grazing
incidence small-angle scattering, for example) require a different
framework.

The incident medium is assumed non-absorbing and isotropic, so that
$k_z$ in the fronting medium is real (above the critical angle the
substrate can be evanescent) and the incident polarization basis is
the standard $s$ and $p$.

## How this differs from full-wave solvers

The 4x4 transfer matrix method is a semi-analytic technique, not a
numerical field solver. It exploits the fact that in a piecewise
homogeneous medium with only $z$-dependence, Maxwell's equations
reduce to an algebraic eigenproblem per layer plus an exponential
propagation through each layer. The cost scales linearly in the
number of layers and is independent of wavelength or spot size. For
a stack of $N$ layers the work is $\mathcal{O}(N)$ 4x4 matrix
products plus one 4x4 eigensolve per layer.

Full-wave solvers discretize the electromagnetic fields (or their
sources) on a grid or mesh and solve the resulting large sparse
system directly.

The [finite-difference time-domain method
(FDTD)](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method)
steps Maxwell's curl equations forward in time on a Yee grid. It
handles arbitrary geometry, dispersion, and non-linearity, but each
simulation yields a time-domain response at a single source
configuration, the spatial grid must resolve the shortest wavelength
everywhere, and the cost scales roughly as $\lambda^{-4}$ in three
dimensions.

The [finite element method
(FEM)](https://en.wikipedia.org/wiki/Finite_element_method)
discretizes the frequency-domain Helmholtz problem on an
unstructured mesh and solves the resulting system by sparse linear
algebra. It handles complex geometry well and supports higher-order
basis functions, but each solve is expensive and the matrices scale
poorly for large three-dimensional problems.

The [method of moments
(MoM)](https://en.wikipedia.org/wiki/Method_of_moments_(electromagnetics))
reformulates the problem as a surface integral equation using
Green's functions, trading a volumetric mesh for a surface mesh. It
is efficient for electrically large, open-domain problems but
produces dense matrices and requires careful Green's-function
tabulation.

The 4x4 transfer matrix method sits in a very different regime from
all three. Because it assumes the geometry is a planar stack, it
pays no cost for a large wavelength-to-thickness ratio, it produces
an exact result up to eigensolver precision rather than a
discretization error, and it trivially parallelizes over frequency
and angle samples. Its limitation is the geometry constraint. Any
problem where the 1D-planar assumption holds is best attacked with
this family of methods. Any problem that violates it requires a
full-wave solver.

## Roughness models at a glance

No real interface is atomically sharp. `refloxide` supports three
models for incorporating finite interfacial roughness $\sigma$ into
the 4x4 pipeline, following the comparative study of [[Esashi et
al. 2021](#references)]. The three models differ in where they enter
the pipeline and in which physical regime they are valid. The
**Névot-Croce factor** is a distorted-wave Born approximation
correction appropriate for short correlation lengths and small
index contrast. The **Debye-Waller factor** is its long-correlation
counterpart and is essentially indistinguishable from Névot-Croce
for $\sigma < 1$ nm away from the critical angle. The
**graded-interface (slide) method** replaces the sharp index step
by a discretized continuous profile and is the only one of the
three that remains correct for large $\sigma$, large $\Delta n$,
non-Gaussian statistics, or phase-sensitive measurements. The
decision framework is enumerated in
[Selection guide](roughness_selection_guide.md).

## Intended implementation path

The numerical kernel is a pure function mapping a
`(Structure, Measurement)` pair to an output array. The `Structure`
carries the ordered list of layers (thickness, dielectric tensor
parameterization, roughness scalar, roughness model tag) together
with the fronting and backing media. The `Measurement` carries the
sample grid (angle, energy, or wavelength) and the requested
observables (reflectance, transmittance, field distribution, or a
subset). The 4x4 kernel itself is a pure function of its arguments
with no mutable state.

Stages 1 through 6 of [Pipeline at a glance](pipeline.md) map onto the six
modules of the kernel. Stage 1 (the Berreman $\Delta$ matrix and
the dielectric tensor rotation) lives in the `core::delta` module.
Stage 2 (the eigensolve and mode sorting) lives in `core::modes`.
Stage 3 (the $\gamma$ construction, normalization, and $A_i$
assembly) lives in `core::interface`, with the Xu piecewise
formulas factored into a helper so that the $q_{i1} = q_{i2}$
branch can be exercised by targeted unit tests. Stage 4 (the
product of $T_i$ and the $\Lambda_{1324}$ permutation) lives in
`core::transfer`. Stage 5 (the $r$ and $t$ extraction) lives in
`core::coefficients`. Stage 6 (the amplitude propagation and field
reconstruction) lives in `core::field`.

The three roughness models are orthogonal to stages 1 through 6.
The Névot-Croce and Debye-Waller models are post-processing
multipliers applied to $A_i$ or $L_i$ at stage 3. The
graded-interface model is a pre-processing expansion of the
`Structure` before stage 2, producing a longer layer list that is
then fed to the unmodified pipeline. This orthogonality is what
makes the three models simultaneously supportable.

Validation targets include the published MATLAB reference
implementation [[Passler and Paarmann 2019 code](#references)], the
Python port by Jeannin [[Jeannin 2019](#references)], the Fresnel
limit for a scalar index, the Yeh birefringent-multilayer
benchmark, and the Parratt-limit agreement with
`[refnx](https://github.com/refnx/refnx)` and
`[refl1d](https://github.com/refl1d/refl1d)` for zero-anisotropy
structures.

## Related projects

Several open-source projects overlap with `refloxide` in scope.
Listing them here serves two purposes, to give the reader a route
out of `refloxide` when its scope is wrong for their problem and to
credit the codebases that form our validation benchmarks.

Scalar isotropic reflectometry engines.

- `[refnx](https://github.com/refnx/refnx)`, a Python package for
neutron and x-ray reflectometry fitting built on Abelès 2x2
matrices.
- `[refl1d](https://github.com/refl1d/refl1d)`, a companion package
from the NIST Center for Neutron Research, with emphasis on model
composition and magnetism.
- `[IMD](https://www.rxollc.com/idl/)`, David Windt's IDL-based
multilayer reflectivity code, widely used in the EUV community.
- `[GenX](https://aglavic.github.io/genx/)`, a Python multilayer
fitting suite aimed at x-ray and neutron reflectometry with
magnetic extensions.
- `[tmm](https://github.com/sbyrnes321/tmm)`, Steven Byrnes's
compact Python implementation of the scalar transfer matrix
method.

4x4 anisotropic engines.

- [Passler and Paarmann MATLAB
implementation](https://doi.org/10.5281/zenodo.601496), the
reference code accompanying the 2017 paper and 2019 erratum.
- [Jeannin Python
port](https://doi.org/10.5281/zenodo.3417751), a Python reimplementation
of the same algorithm used as a validation target here.
- `[PyMoosh](https://github.com/AnMoreau/PyMoosh)`, a Python library
for multilayer optics that supports anisotropic media and
non-standard layer models.

Adjacent full-wave and grating solvers.

- `[meep](https://github.com/NanoComp/meep)`, an open-source FDTD
package for arbitrary geometries.
- `[S4](https://web.stanford.edu/group/fan/S4/)`, a rigorous
coupled-wave analysis (RCWA) engine for periodic multilayer
gratings.

`refloxide` occupies the niche of a fast, anisotropy-capable, 4x4
engine with first-class support for three interfacial roughness
models in a single package. Where scalar isotropy suffices, the
established Python tools above are more mature and we recommend
them.

## Where this page falls short

This overview is deliberately narrative and does not derive
anything. It is also silent on several topics that will need their
own companion files before the documentation is complete.

Energy conservation and the correct expression for the transmittance
in anisotropic substrates are left to a future companion file,
since they were themselves left to a future publication by the
original authors [[Passler and Paarmann 2019](#references)].

Non-linear source terms, optical activity, magnetic anisotropy, and
gyrotropy are mentioned only in passing. Each admits an extension
of the $\gamma$ eigenvectors and each merits its own file if and
when the kernel grows to support it.

The tensor generalization of the Névot-Croce and Debye-Waller
factors beyond the scalar Parratt case treated by [[Esashi et al.
2021](#references)] is not published in the sources we cite. The
implementation choice made in
[Nevot-Croce](roughness_nevot_croce.md) and
[Debye-Waller](roughness_debye_waller.md), namely
that the scalar multiplier is applied block-diagonally to $A_i$ in
the $s$ and $p$ eigen-channels, is an implementation decision that
we flag explicitly rather than attribute to a published derivation.

Fitting, uncertainty propagation, and resolution convolution are
outside the scope of this package. The 4x4 kernel is designed to
plug cleanly into fitting frameworks that handle those concerns.

## References

1. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
  for light propagation in anisotropic stratified media, study of
   surface phonon polaritons in polar dielectric heterostructures,"
   J. Opt. Soc. Am. B **34**, 2128 (2017).
   [DOI](https://doi.org/10.1364/JOSAB.34.002128).
2. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
  for light propagation in anisotropic stratified media, erratum,"
   J. Opt. Soc. Am. B **36**, 3246 (2019).
   [DOI](https://doi.org/10.1364/JOSAB.36.003246).
3. D. W. Berreman, "Optics in stratified and anisotropic media, 4x4
  matrix formulation," J. Opt. Soc. Am. **62**, 502 (1972).
   [DOI](https://doi.org/10.1364/JOSA.62.000502).
4. F. Abelès, "Recherches sur la propagation des ondes
  électromagnétiques sinusoïdales dans les milieux stratifiés.
   Application aux couches minces," Ann. Phys. **12**, 596 (1950).
5. L. G. Parratt, "Surface studies of solids by total reflection of
  x-rays," Phys. Rev. **95**, 359 (1954).
   [DOI](https://doi.org/10.1103/PhysRev.95.359).
6. P. J. Lin-Chung and S. Teitler, "4x4 matrix formalisms for optics
  in stratified anisotropic media," J. Opt. Soc. Am. A **1**, 703
   (1984).
   [DOI](https://doi.org/10.1364/JOSAA.1.000703).
7. P. Yeh, "Electromagnetic propagation in birefringent layered
  media," J. Opt. Soc. Am. **69**, 742 (1979).
   [DOI](https://doi.org/10.1364/JOSA.69.000742).
8. W. Xu, L. T. Wood, and T. D. Golding, "Optical degeneracies in
  anisotropic layered media, treatment of singularities in a 4x4
   matrix formalism," Phys. Rev. B **61**, 1740 (2000).
   [DOI](https://doi.org/10.1103/PhysRevB.61.1740).
9. Z.-M. Li, B. T. Sullivan, and R. R. Parsons, "Use of the 4x4
  matrix method in the optics of multilayer magneto-optic recording
   media," Appl. Opt. **27**, 1334 (1988).
   [DOI](https://doi.org/10.1364/AO.27.001334).
10. L. Névot and P. Croce, "Caractérisation des surfaces par
  réflexion rasante de rayons X. Application à l'étude du
    polissage de quelques verres silicates," Rev. Phys. Appl. **15**,
    761 (1980).
    [DOI](https://doi.org/10.1051/rphysap:01980001503076100).
11. D. G. Stearns, "The scattering of x rays from nonideal multilayer
  structures," J. Appl. Phys. **65**, 491 (1989).
    [DOI](https://doi.org/10.1063/1.343131).
12. Y. Esashi, M. Tanksalvala, Z. Zhang, N. W. Jenkins, H. C.
  Kapteyn, and M. M. Murnane, "Influence of surface and interface
    roughness on X-ray and extreme ultraviolet reflectance, a
    comparative numerical study," OSA Continuum **4**, 1497 (2021).
    [DOI](https://doi.org/10.1364/OSAC.422924).
13. N. C. Passler and A. Paarmann, MATLAB implementation,
  [Zenodo](https://doi.org/10.5281/zenodo.601496) (2019).
14. M. Jeannin, Python implementation,
  [Zenodo](https://doi.org/10.5281/zenodo.3417751) (2019).
