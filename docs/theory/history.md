# A brief history of the 4x4 transfer matrix method

The problem of calculating the reflection of a plane wave from a
layered medium has been studied continuously since the nineteenth
century. What `refloxide` implements sits at the current end of a
long lineage of successive generalizations.

The single-interface case is the [Fresnel
equations](https://en.wikipedia.org/wiki/Fresnel_equations) of 1823,
which give the amplitude ratios $r$ and $t$ for a plane wave
crossing a sharp dielectric boundary, decomposed into $s$ and $p$
polarizations. Every subsequent method for stratified media is, in
effect, a bookkeeping scheme for chaining Fresnel coefficients
together.

The first systematic chaining scheme is the Abelès matrix
formalism [[Abelès 1950](#references)], later popularized by Born and
Wolf and reviewed on the Wikipedia entry for the [transfer-matrix
method in optics](https://en.wikipedia.org/wiki/Transfer-matrix_method_(optics)).
Abelès represented each homogeneous layer by a 2x2 matrix acting on
a two-component field and accumulated the stack by matrix product.
This is the isotropic, scalar-index formalism still used today by
tools such as [`refnx`](https://github.com/refnx/refnx) and
[`refl1d`](https://github.com/refl1d/refl1d).

For grazing-incidence x-ray problems, Parratt [[Parratt
1954](#references)] recast the same physics as a recursion on complex
field ratios rather than a matrix product. The
[Parratt recursion](https://en.wikipedia.org/wiki/X-ray_reflectivity)
is numerically equivalent to the Abelès product for isotropic media
but is easier to implement on small stacks, and it remains the
standard for [x-ray reflectometry](https://en.wikipedia.org/wiki/X-ray_reflectivity).

Neither the Abelès nor the Parratt formalism handles
[birefringent](https://en.wikipedia.org/wiki/Birefringence) or
otherwise tensor-valued [permittivity](https://en.wikipedia.org/wiki/Permittivity).
Berreman [[Berreman 1972](#references)] generalized the problem by
casting [Maxwell's equations](https://en.wikipedia.org/wiki/Maxwell%27s_equations)
as a 6x6 first-order system in the six components of $\vec{E}$ and
$\vec{H}$, eliminating the two longitudinal components algebraically,
and solving the remaining 4x4 eigenproblem numerically. Lin-Chung
and Teitler [[Lin-Chung and Teitler 1984](#references)] recast
Berreman's algorithm in a form closer to the Abelès product. Yeh
[[Yeh 1979](#references)] independently produced an equivalent 4x4
formalism from a plane-wave ansatz directly.

All three of these early 4x4 formalisms are numerically unstable in
important special cases. When the principal axes of the dielectric
tensor align with the lab frame, several of the matrix elements
diverge even though the physical answer is finite. Xu, Wood, and
Golding [[Xu et al. 2000](#references)] wrote out the four
eigenvectors in a piecewise form that remains finite through the
degenerate limit. Li, Sullivan, and Parsons [[Li et al.
1988](#references)] solved a separate continuity problem, namely that
a naive eigensolver can permute the labels of the four modes
arbitrarily when the dielectric tensor is swept smoothly through
parameter space, producing discontinuous spectra.

Passler and Paarmann [[Passler and Paarmann
2017](#references), [2019](#references)] assembled these pieces into
a single algorithm, with the Berreman reduction at stage 1, the Xu
piecewise eigenvectors at stage 3, the Li mode-sorting rule at
stage 2, and the Yeh $r/t$ extraction at stage 5. Their 2019
erratum corrects two typographical errors in the eigenvector
components, imposes a normalization that was implicit rather than
explicit in the 2017 paper, and replaces the field reconstruction
recipe so that it works for birefringent substrates. This is the
algorithm `refloxide` implements.

## Roughness is a separate lineage

The [Debye-Waller factor](https://en.wikipedia.org/wiki/Debye%E2%80%93Waller_factor)
originates in crystallographic diffraction, where thermal atomic
displacement reduces the coherent Bragg intensity by an exponential
of the mean-square displacement. It was adapted to reflection from
rough surfaces by analogy. The Névot-Croce factor [[Névot and Croce
1980](#references)] is a more careful derivation based on the
[distorted-wave Born approximation](https://en.wikipedia.org/wiki/Distorted_wave_Born_approximation),
appropriate when the correlation length of the roughness is short
enough that diffuse scattering is angularly separated from specular.
The graded-interface (slide) method was introduced by Stearns
[[Stearns 1989](#references)] and has been compared quantitatively
against Névot-Croce by Esashi et al. [[Esashi et al.
2021](#references)].

## References

The references enumerated here are the same set collected on the
[theory overview](overview.md#references) page. Numbering is
preserved so that in-text citations resolve identically across
companion files.

1. F. Abelès, "Recherches sur la propagation des ondes
   électromagnétiques sinusoïdales dans les milieux stratifiés.
   Application aux couches minces," Ann. Phys. **12**, 596 (1950).
2. L. G. Parratt, "Surface studies of solids by total reflection of
   x-rays," Phys. Rev. **95**, 359 (1954).
   [DOI](https://doi.org/10.1103/PhysRev.95.359).
3. D. W. Berreman, "Optics in stratified and anisotropic media, 4x4
   matrix formulation," J. Opt. Soc. Am. **62**, 502 (1972).
   [DOI](https://doi.org/10.1364/JOSA.62.000502).
4. P. Yeh, "Electromagnetic propagation in birefringent layered
   media," J. Opt. Soc. Am. **69**, 742 (1979).
   [DOI](https://doi.org/10.1364/JOSA.69.000742).
5. L. Névot and P. Croce, "Caractérisation des surfaces par
   réflexion rasante de rayons X," Rev. Phys. Appl. **15**, 761
   (1980).
   [DOI](https://doi.org/10.1051/rphysap:01980001503076100).
6. P. J. Lin-Chung and S. Teitler, "4x4 matrix formalisms for optics
   in stratified anisotropic media," J. Opt. Soc. Am. A **1**, 703
   (1984).
   [DOI](https://doi.org/10.1364/JOSAA.1.000703).
7. Z.-M. Li, B. T. Sullivan, and R. R. Parsons, "Use of the 4x4
   matrix method in the optics of multilayer magneto-optic recording
   media," Appl. Opt. **27**, 1334 (1988).
   [DOI](https://doi.org/10.1364/AO.27.001334).
8. D. G. Stearns, "The scattering of x rays from nonideal multilayer
   structures," J. Appl. Phys. **65**, 491 (1989).
   [DOI](https://doi.org/10.1063/1.343131).
9. W. Xu, L. T. Wood, and T. D. Golding, "Optical degeneracies in
   anisotropic layered media," Phys. Rev. B **61**, 1740 (2000).
   [DOI](https://doi.org/10.1103/PhysRevB.61.1740).
10. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
    for light propagation in anisotropic stratified media,"
    J. Opt. Soc. Am. B **34**, 2128 (2017).
    [DOI](https://doi.org/10.1364/JOSAB.34.002128).
11. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
    for light propagation in anisotropic stratified media, erratum,"
    J. Opt. Soc. Am. B **36**, 3246 (2019).
    [DOI](https://doi.org/10.1364/JOSAB.36.003246).
12. Y. Esashi et al., "Influence of surface and interface roughness
    on X-ray and extreme ultraviolet reflectance, a comparative
    numerical study," OSA Continuum **4**, 1497 (2021).
    [DOI](https://doi.org/10.1364/OSAC.422924).
