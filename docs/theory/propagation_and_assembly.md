# Propagation and stack assembly

## Scope of this page

This page treats stage 4 of the pipeline. Given the per-layer
interface matrices $A_i$ from
`[interface_matrices.md](interface_matrices.md)`, we assemble the
single-layer transfer matrix $T_i$, chain the $N$ per-layer
transfers into a total transfer $T_{\text{tot}}$, sandwich the
result between the two cladding matrices, and apply the
$\Lambda_{1324}$ basis permutation that reconciles the sorted
Passler layout with the $p/s$ layout required by the Yeh extraction
of stage 5. The construction follows Passler and Paarmann
[[1](#references), Eqs. (25)-(32)], building on Yeh
[[2](#references)].

## Propagation inside a homogeneous layer

Inside layer $i$ each mode propagates with its own $q_{ij}$, so the
four-component amplitude vector accumulates a diagonal phase across
thickness $d_i$. The propagation matrix is
[[1](#references), Eq. (25)]

$$
P_i =
\begin{pmatrix}
e^{-i(\omega/c)q_{i1} d_i} & 0 & 0 & 0 
0 & e^{-i(\omega/c)q_{i2} d_i} & 0 & 0 
0 & 0 & e^{-i(\omega/c)q_{i3} d_i} & 0 
0 & 0 & 0 & e^{-i(\omega/c)q_{i4} d_i}
\end{pmatrix}.
$$

The sign convention in the exponent follows Passler and Paarmann
[[1](#references), text preceding Eq. (25)] and is consistent with
the assignment of forward modes as those with
$\operatorname{Im}(q_{ij}) \ge 0$ in
`[eigenmode_analysis.md](eigenmode_analysis.md)`.

Two practical remarks are worth flagging. First, $P_i$ is evaluated
at the full layer thickness $d_i$ for the stack assembly, but it
can also be evaluated at a partial thickness $0 < z < d_i$ for the
field reconstruction of stage 6 (see
`[electric_field_distribution.md](electric_field_distribution.md)`),
where the diagonal entries become $e^{-i(\omega/c)q_{ij} z}$
[[1](#references), Eq. (38)]. Second, exponentials of large
positive imaginary argument grow without bound, which is the
characteristic overflow mode of the Abelès product in thick
absorbing stacks. The Passler recommendation of the
$A_0^{-1} T_{\text{tot}} A_{N+1}$ form mitigates this because each
$T_i$ retains both growing and decaying modes in balance, so the
cancellation happens inside the matrix product rather than after.

## Single-layer transfer matrix

The single-layer transfer matrix rotates the amplitude vector into
the layer's mode basis, accumulates the per-mode phase, and rotates
back out [[1](#references), Eq. (26)],

$$
T_i = A_iP_iA_i^{-1}.
$$

$T_i$ acts on the same tangential-field representation that $A_i$
emits, so consecutive $T_i$ chain directly by matrix product without
intermediate basis changes.

## Total transfer and the stack product

The stack-level transfer is the product of the $T_i$ over all $N$
intermediate layers [[1](#references), Eq. (27)],

$$
T_{\text{tot}} = \prod_{i=1}^{N} T_i.
$$

The full multilayer transfer matrix, which connects the amplitude
vector in the incident medium to the amplitude vector in the
substrate, sandwiches $T_{\text{tot}}$ between the inverse of the
incident interface matrix and the substrate interface matrix
[[1](#references), Eq. (28)],

$$
\Gamma_N = A_0^{-1}T_{\text{tot}}A_{N+1}
        = A_0^{-1}T_1T_2\cdots T_NA_{N+1}
        = L_1P_1L_2P_2 \cdots L_NP_NL_{N+1}.
$$

The three forms are algebraically equivalent. The first is the
preferred numerical form because it confines the ill-conditioned
inversion to the two cladding matrices $A_0$ and $A_{N+1}$. The
third is the one that makes the pipeline's structural decomposition
manifest, a concatenation of per-interface operations $L_i$ and
per-layer operations $P_i$ [[1](#references), text following Eq.
(28)], which is the form used by Lin-Chung and Teitler
[[3](#references)] and by Yeh [[2](#references)].

## Basis permutation $\Lambda_{1324}$

The Li-Sullivan-Parsons sorting rule places the amplitude vector
into the order
$(E^p_{\text{trans}}, E^s_{\text{trans}},
  E^p_{\text{refl}}, E^s_{\text{refl}})^\top$
[[1](#references), Eq. (23); see also
`[eigenmode_analysis.md](eigenmode_analysis.md)`]. The Yeh
extraction of stage 5, by contrast, requires the layout
$(E^p_{\text{trans}}, E^p_{\text{refl}},
  E^s_{\text{trans}}, E^s_{\text{refl}})^\top$
[[1](#references), Eq. (30); [2](#references)], in which the
forward and backward $p$ modes are contiguous and likewise for
$s$. A similarity transformation by the permutation matrix

$$
\Lambda_{1324} =
\begin{pmatrix}
1 & 0 & 0 & 0 
0 & 0 & 1 & 0 
0 & 1 & 0 & 0 
0 & 0 & 0 & 1
\end{pmatrix}
$$

of [[1](#references), Eq. (32)] brings the two layouts into
agreement. The Yeh-compatible transfer matrix is

$$
\tilde{\Gamma}*N = \Lambda*{1324}^{-1}\Gamma_N\Lambda_{1324},
$$

as in [[1](#references), Eq. (31)]. $\Lambda_{1324}$ is its own
inverse (it is an involution on the four-element basis), so the
similarity transformation incurs no numerical inversion penalty.

Stages 5 and 6 consume $\tilde{\Gamma}*N$. Every closed-form
expression for the amplitude coefficients displayed in
`[reflection_transmission.md](reflection_transmission.md)` reads
matrix elements of $\tilde{\Gamma}N$, not of the pre-permutation
$\Gamma_N$. A library that forgets this permutation returns
$r{pp}$ where $r*{ss}$ is expected, and vice versa, which is a
classic failure mode of a naive implementation.

## The $N = 0$ limit and the Fresnel benchmark

The stack formula reduces cleanly when no intermediate layers are
present. Setting $N = 0$ in [[1](#references), Eq. (28)] leaves
$\Gamma_0 = A_0^{-1} A_1$, the single-interface transfer between
two half-spaces. When $A_0$ and $A_1$ describe isotropic media,
$\Gamma_0$ reproduces the scalar Fresnel coefficients of classical
optics. This is the natural unit test for the entire stack
assembly, because it exercises $A$ construction, $A^{-1}$ inversion,
and the $\Lambda_{1324}$ permutation without invoking any $P_i$ at
all.

## Numerical stability considerations

The dominant failure mode of an Abelès-style product over thick
absorbing stacks is loss of precision in the cancellation between
exponentially growing and exponentially decaying mode amplitudes,
which appear together as diagonal entries of $P_i$. The 4x4
formalism carries this cancellation throughout the product, and
the product form
$\Gamma_N = A_0^{-1}T_{\text{tot}}A_{N+1}$
is numerically better conditioned than the alternating
$L_iP_i$ chain because $T_i = A_iP_iA_i^{-1}$ preserves the
mode-balanced structure at each step.

We surmise that a production implementation should additionally
guard against the scenario where a single layer has
$\operatorname{Im}(q_{ij})d_i \gg 1$, which produces an entry of
$P_i$ that overflows double precision. The standard remedy in the
scalar transfer-matrix literature is scattering-matrix
reformulation, which operates on ratios rather than amplitudes and
is immune to overflow. We do not implement scattering matrices in
the initial kernel, and we flag the overflow regime as a known
limitation to be revisited when profiled.

## Where the code lives

Stage 4 is the `core::transfer` module. The module consumes
$(A_i, q_{ij}, d_i)*{i=1}^N$ and the two claddings $A_0$ and
$A*{N+1}$, and returns $\tilde{\Gamma}*N$. The $\Lambda*{1324}$
permutation is implemented as a constant 4x4 matrix rather than a
row-shuffle, because the library's matrix primitives are
consistently 4x4 and a constant multiplication is easier to reason
about than an index remap. The partial propagation form
$P_i(z) = \operatorname{diag}(e^{-i(\omega/c)q_{ij}z})$ is exposed
as a separate callable for use by `core::field` at stage 6.

## References

1. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
  for light propagation in anisotropic stratified media," J. Opt.
   Soc. Am. B **34**, 2128 (2017).
   [DOI](https://doi.org/10.1364/JOSAB.34.002128).
2. P. Yeh, "Electromagnetic propagation in birefringent layered
  media," J. Opt. Soc. Am. **69**, 742 (1979).
   [DOI](https://doi.org/10.1364/JOSA.69.000742).
3. P. J. Lin-Chung and S. Teitler, "4x4 matrix formalisms for
  optics in stratified anisotropic media," J. Opt. Soc. Am. A
   **1**, 703 (1984).
   [DOI](https://doi.org/10.1364/JOSAA.1.000703).