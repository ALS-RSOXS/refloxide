# Electric field distribution

## Scope of this page

This page treats stage 6 of the pipeline. Given the amplitude
coefficients from
[Reflection and transmission](reflection_transmission.md), together
with the per-interface matrices $L_i$ and the partial propagation
matrices $P_i(z)$ from
[Propagation and assembly](propagation_and_assembly.md) and
the normalized eigenvectors $\hat{\vec{\gamma}}_{ij}$ from
[Interface matrices](interface_matrices.md), we reconstruct
the complex three-component electric field $\vec{E}(x, y, z)$
inside every layer of the stack. The procedure follows the
erratum-corrected recipe of Passler and Paarmann
[[1](#references), Sec. 2.C and Eq. (E2)]. The 2017 recipe
[[2](#references), Eqs. (37)-(41)] is superseded, because it
assumes a lab-frame-diagonal dielectric tensor and gives the wrong
answer for birefringent substrates.

## Inputs

The reconstruction consumes four quantities from prior stages, each
of which must be available per layer $i$. The per-layer eigenvalues
$q_{ij}$ and normalized eigenvectors $\hat{\vec{\gamma}}_{ij}$ are
produced by `core::interface` at stage 3. The per-layer partial
propagation matrix $P_i(z)$, with diagonal entries
$e^{-i(\omega/c)q_{ij} z}$ evaluated at relative position
$0 \le z \le d_i$ inside layer $i$ [[2](#references), Eq. (38)], is
produced by `core::transfer` at stage 4. The per-interface matrix
$L_i = A_{i-1}^{-1} A_i$, which propagates the four-component
amplitude vector from the mode basis of layer $i$ to that of layer
$i-1$, is likewise a stage 4 artifact. The amplitude coefficients
$t_{kl}$ are the erratum-corrected rational expressions of
[[1](#references), Eqs. (33*)-(36*)] from stage 5.

## Amplitude vector in the substrate

The amplitude vector $\vec{E}_{N+1}^{+}$ on the substrate side of
the final interface is constructed separately for $p$-polarized
and $s$-polarized incident light [[1](#references), Eq. (37*)]. The
separation is the key structural change from the 2017 recipe. For
$p$-polarized incidence, the only non-zero substrate amplitudes are
the two transmitted modes of the substrate,

$$
\vec{E}_{N+1\,|\,p\text{-in}}^{+}
=
\begin{pmatrix}
E^{p/o}_{\text{trans}} \\ E^{s/e}_{\text{trans}} \\
E^{p/o}_{\text{refl}} \\ E^{s/e}_{\text{refl}}
\end{pmatrix}_{p\text{-in}}
=
\begin{pmatrix}
t_{p(p/o)} \\ t_{p(s/e)} \\ 0 \\ 0
\end{pmatrix},
$$

and for $s$-polarized incidence the analogous expression carries
$t_{s(p/o)}$ and $t_{s(s/e)}$ in the transmitted slots. The
reflected slots vanish identically because no source sits below the
substrate [[1](#references), text following Eq. (37*)]. The
$p/o$ and $s/e$ pairing reflects the relabeling discussion of
[Reflection and transmission](reflection_transmission.md).
Lab-frame-diagonal substrates take the $p/s$ reading, birefringent
substrates take the $o/e$ reading, and the formalism does not
otherwise distinguish the two cases.

A subtlety worth flagging is that the stage 5 extraction was
carried out in the Yeh-layout basis $\tilde{\Gamma}_N$, but the
amplitude-vector reconstruction uses the sorted Passler basis of
[[2](#references), Eq. (23)]. The two layouts are connected by
$\Lambda_{1324}$, and the library stores the sorted layout for the
downstream field computation so that no further permutation is
needed. This is the reason [[1](#references), text following Eq.
(37*)] remarks that the field reconstruction uses the same
interface and propagation matrices that were computed on the way
to $\tilde{\Gamma}_N$.

## Backward propagation through the stack

Starting from the substrate-side amplitude vector, the
four-component amplitude in layer $i$ at relative depth $z$ is
obtained by repeated application of the appropriate $L$ and $P(z)$
operators. Passler and Paarmann give the recursion in the form
[[1](#references), Eq. (38)]

$$
\vec{E}_i(z) = P_i(z)\,\vec{E}_i^{-},
$$

where $\vec{E}_i^{-}$ is the amplitude vector at the top of layer
$i$, i.e. on the layer $i-1$ side of the $(i-1, i)$ interface. The
transition from $\vec{E}_i^{-}$ to the amplitude on the opposite
side of the next interface toward the substrate reads
$\vec{E}_{i-1}^{-} = L_i\,\vec{E}_i^{+}$, in the notation of
[[1](#references), Eq. (28) and Eq. (38)]. The backward walk from
layer $N+1$ to the layer of interest therefore alternates
per-interface $L$ with per-layer $P$ at full thickness, followed
by a single $P(z)$ at partial thickness once the layer of interest
is reached.

In operational form, the amplitude vector in layer $k$ at relative
position $z$ is

$$
\vec{E}_k(z)
= P_k(z)\,L_{k+1}\,P_{k+1}(d_{k+1})\,L_{k+2}\,P_{k+2}(d_{k+2})\cdots
  L_N\,P_N(d_N)\,L_{N+1}\,\vec{E}_{N+1}^{+}.
$$

The library evaluates this product left-to-right starting from
$\vec{E}_{N+1}^{+}$, which preserves the conditioning advantage of
sandwiching the exponential-carrying $P_i$ matrices between the
well-conditioned $L_i$.

## Reconstruction of the three-component field

Having $\vec{E}_k(z)$, the reconstruction of $\vec{E}(x, y, z)$
inside layer $k$ proceeds mode by mode. The four components of
$\vec{E}_k(z)$ are the four mode amplitudes, and each amplitude
multiplies its associated normalized eigenvector to produce a
three-component contribution. The erratum gives the explicit
mapping for each of the four modes separately for $p$- and
$s$-incidence [[1](#references), Eq. (E2)],

$$
\vec{E}^{\,p/o}_{\text{trans}\,|\,p/s\text{-in}} =
E^{p/o}_{\text{trans}\,|\,p/s\text{-in}}\,
\begin{pmatrix}
\hat{\gamma}_{i11} \\ \hat{\gamma}_{i12} \\ \hat{\gamma}_{i13}
\end{pmatrix},
\quad
\vec{E}^{\,s/e}_{\text{trans}\,|\,p/s\text{-in}} =
E^{s/e}_{\text{trans}\,|\,p/s\text{-in}}\,
\begin{pmatrix}
\hat{\gamma}_{i21} \\ \hat{\gamma}_{i22} \\ \hat{\gamma}_{i23}
\end{pmatrix},
$$

with the analogous expressions for the reflected pair using
$\hat{\vec{\gamma}}_{i3}$ and $\hat{\vec{\gamma}}_{i4}$ of the
same layer. The full three-component field at $z$ is the sum of
these four contributions,

$$
\vec{E}(x, y, z) =
e^{i\xi(\omega/c) x}
\sum_{j=1}^{4}
E^{(j)}_{i\,|\,p/s\text{-in}}(z)\,
\begin{pmatrix}
\hat{\gamma}_{ij1} \\ \hat{\gamma}_{ij2} \\ \hat{\gamma}_{ij3}
\end{pmatrix},
$$

where $\xi(\omega/c)$ is the in-plane wavenumber of stage 1 and the
$x$-phase factor is common to every mode because $\xi$ is conserved
across the stack. The $y$ dependence is absent because the plane of
incidence was chosen as the $x$-$z$ plane in
[Foundations](foundations.md).

## Why the eigenvector phases already account for reflection

The 2017 recipe carried an explicit minus sign on $E_x$ and $E_z$
for reflected modes, which implemented the usual phase flip on
reflection at the interface [[2](#references), Eqs. (39)-(41) and
text preceding Eq. (41)]. The 2019 erratum shows that this was a
consequence of using the unnormalized $\vec{\gamma}_{ij}$ and of
restricting attention to lab-frame-diagonal tensors
[[1](#references), text following Eq. (E2)]. Once the normalized
$\hat{\vec{\gamma}}_{ij}$ are used throughout, the phase flip on
reflection is encoded inside $\hat{\gamma}_{i31}$, $\hat{\gamma}_{i32}$,
$\hat{\gamma}_{i33}$, $\hat{\gamma}_{i41}$, $\hat{\gamma}_{i42}$,
$\hat{\gamma}_{i43}$ automatically, and no separate sign bookkeeping
is needed. The library therefore does not carry any reflection
sign convention at the field layer, and we explicitly warn in the
module docstring against reintroducing one.

## Evanescent modes and the Otto geometry

When a mode has $\operatorname{Im}(q_{ij}) > 0$, the diagonal entry
$e^{-i(\omega/c)q_{ij}z}$ of $P_i(z)$ is an exponential decay in
$+\hat{z}$. For the reflected modes, $\operatorname{Im}(q_{ij}) < 0$
produces exponential growth toward the substrate, which is
physical. The characteristic field profile in this regime shows
peaks at interfaces and valleys inside the layers
[[2](#references), text preceding Sec. 2.D and Fig. 1(a)]. This is
the regime of the Otto coupling geometry [[2](#references),
Sec. 3.C], where an incident angle past the total internal
reflection critical angle produces an evanescent tail inside the
intermediate dielectric and drives resonant excitation of the
underlying surface mode.

The library does not special-case the evanescent regime. The same
rational expressions for $t_{kl}$ and the same recursion
$\vec{E}_k(z) = P_k(z)\,L_{k+1}\,\cdots$ apply, and the exponential
decay is carried inside $P_i(z)$. Callers who want a field-energy
plot should be careful that the $|\vec{E}|^2$ at an evanescent peak
can exceed the incident intensity by a resonance factor that has
nothing to do with energy conservation, because evanescent fields
carry no time-averaged Poynting flux through the interface they
decorate.

## Where the code lives

Stage 6 is the `core::field` module. It consumes the per-layer
$(q_{ij}, \hat{\vec{\gamma}}_{ij}, d_i)$ tuples from `core::interface`,
the $L_i$ and $P_i(z)$ callables from `core::transfer`, and the
eight $t_{kl}$ amplitudes from `core::coefficients`. It emits a
callable $\vec{E}(x, y, z)$ that evaluates at requested $z$ inside
any layer, or at a dense $z$-grid for depth-profile plots. The
$p$-incidence and $s$-incidence reconstructions are performed
independently, as the erratum requires
[[1](#references), text following Eq. (37*)], and the results are
returned as two separate callables so that arbitrary incident
polarization can be composed linearly.

## Validation targets

Four benchmarks are appropriate for the field reconstruction.
Tangential continuity of $(E_x, H_y, E_y, H_x)$ across every
interface is the basic consistency check and follows from the
$L_i = A_{i-1}^{-1} A_i$ construction; violations indicate an
ordering or normalization bug. Agreement with the Passler MATLAB
reference [[3](#references)] for the SiC/GaN/SiC Otto-geometry
example of [[2](#references), Sec. 3.C] exercises the evanescent
regime. Agreement with the Jeannin Python port
[[4](#references)] for a birefringent test stack validates the
erratum-corrected path. Convergence of the reconstructed field
against finer $z$-sampling in graded-interface expansions is the
test that couples stage 6 to the roughness machinery of
[Graded interface](roughness_graded_interface.md).

## References

1. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
   for light propagation in anisotropic stratified media, erratum,"
   J. Opt. Soc. Am. B **36**, 3246 (2019).
   [DOI](https://doi.org/10.1364/JOSAB.36.003246).
2. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
   for light propagation in anisotropic stratified media," J. Opt.
   Soc. Am. B **34**, 2128 (2017).
   [DOI](https://doi.org/10.1364/JOSAB.34.002128).
3. N. C. Passler and A. Paarmann, MATLAB implementation,
   [Zenodo](https://doi.org/10.5281/zenodo.601496) (2019).
4. M. Jeannin, Python implementation,
   [Zenodo](https://doi.org/10.5281/zenodo.3417751) (2019).
