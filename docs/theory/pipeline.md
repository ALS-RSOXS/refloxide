# The pipeline at a glance

The 4x4 transfer matrix algorithm as implemented in `refloxide`
decomposes into six stages, executed sequentially per measurement
coordinate (angle, energy, or wavelength). Each stage is treated in
depth in a dedicated companion file and is summarized here so that
the reader can locate the relevant discussion without reading the
full derivation. The stages are the same ones assembled by Passler
and Paarmann [[Passler and Paarmann
2017](overview.md#references), [2019](overview.md#references)], with
corrections from the 2019 erratum folded into stages 3, 5, and 6.

## 1. Foundations

Maxwell's equations are cast into a 6x6 matrix system that relates
the field vector $(E_x, E_y, E_z, H_x, H_y, H_z)$ to itself through
spatial and temporal derivatives. The constitutive relation
$\vec{C} = M\vec{G}$ encodes $\bar{\varepsilon}$, the permeability
tensor $\bar{\mu}$, and any optical rotation tensors. The
longitudinal components $E_z$ and $H_z$ satisfy algebraic relations
and are eliminated. The remaining four-component field
$\Psi = (E_x, H_y, E_y, -H_x)^\top$ obeys
$\partial_z \Psi = i(\omega/c)\Delta\Psi$,
where $\Delta$ is the 4x4 Berreman matrix tabulated in [[Passler and
Paarmann 2017](overview.md#references), Eq. (8)]. This reduction is
the entry point to the whole formalism. See
[Foundations](foundations.md).

## 2. Eigenmode analysis per layer

Because each layer is $z$-homogeneous, $\Delta_i$ is constant and
the solution reduces to the algebraic eigenvalue problem
$q_{ij}\Psi_{ij} = \Delta_i \Psi_{ij}$, which has four roots
$q_{ij}$, $j = 1, 2, 3, 4$. These are the $z$-components of the
four plane waves supported in layer $i$. The modes are partitioned
into forward (transmitted) and backward (reflected) by the sign of
$\operatorname{Re}(q_{ij})$ for propagating modes and the sign of
$\operatorname{Im}(q_{ij})$ for evanescent modes. Within each pair
the two solutions are ordered by a projection functional. For
moderately anisotropic tensors we use
$C(q_{ij}) = |\Psi_{ij,1}|^2 / (|\Psi_{ij,1}|^2 + |\Psi_{ij,3}|^2)$,
which separates the $p$-like from the $s$-like mode. For strongly
birefringent tensors whose principal axes are not aligned with the
plane of incidence this functional is ambiguous, and we fall back
to the analogous Poynting-vector projection
$C(q_{ij}) = |S_{ij,x}|^2 / (|S_{ij,x}|^2 + |S_{ij,y}|^2)$,
following [[Li et al. 1988](overview.md#references)]. This ordering
is the step that guarantees continuity of the solution branches
when the dielectric tensor varies smoothly across parameter space.
See [Eigenmode analysis](eigenmode_analysis.md).

## 3. Interface matrices free of singularities

Naive eigenvector extraction from $\Delta_i$ fails when the modes
degenerate, which happens generically for isotropic media or
whenever the principal dielectric axes align with the lab frame. We
therefore use the closed-form electric-field eigenvectors
$\vec{\gamma}_{ij}$ of [[Xu et al. 2000](overview.md#references)],
which are piecewise defined for $q_{i1} = q_{i2}$ and $q_{i1} \neq
q_{i2}$ and remain finite and continuous through these limits. The
erratum [[Passler and Paarmann 2019](overview.md#references)]
corrects two components, $\gamma_{i13}$ and $\gamma_{i33}$, that
were mistyped in the original paper, and it requires that each
$\vec{\gamma}_{ij}$ be normalized,
$\hat{\vec{\gamma}}_{ij} = \vec{\gamma}_{ij}/|\vec{\gamma}_{ij}|$,
so that cross-polarization coefficients in birefringent substrates
come out correctly. The interface matrix $A_i$ is the 4x4 matrix
whose columns are the four $\hat{\vec{\gamma}}_{ij}$ augmented with
their associated $(H_x, H_y)$ components. The boundary-matching
step across interface $i$ then reads
$A_{i-1}\vec{E}_{i-1} = A_i\vec{E}_i$, and
$L_i = A_{i-1}^{-1} A_i$ is the interface matrix that projects the
mode basis of layer $i$ onto that of layer $i-1$. See
[Interface matrices](interface_matrices.md).

## 4. Propagation and assembly

Inside layer $i$, each mode accumulates a phase
$\exp(-i(\omega/c) q_{ij} d_i)$. These phases populate a diagonal
4x4 propagation matrix $P_i$. The single-layer transfer matrix is
$T_i = A_i P_i A_i^{-1}$, and the full multilayer transfer matrix
is

$$
\Gamma_N = A_0^{-1} \left(\prod_{i=1}^{N} T_i \right) A_{N+1}
        = L_1 P_1 L_2 P_2 \cdots L_N P_N L_{N+1},
$$

where the second equality makes it manifest that $\Gamma_N$ is a
concatenation of per-interface and per-layer operations. The
product form is the preferred implementation target because it
separates the two numerically distinct operations, namely mode
basis change (real, often ill-conditioned) and phase accumulation
(trivially diagonal). Because the eigenvector ordering produced by
the sorting rules is
$(E^p_{\text{trans}}, E^s_{\text{trans}}, E^p_{\text{refl}},
E^s_{\text{refl}})$, but Yeh's $r/t$ expressions expect the layout
$(E^p_{\text{trans}}, E^p_{\text{refl}}, E^s_{\text{trans}},
E^s_{\text{refl}})$, we apply a permutation $\Lambda_{1324}$ after
$\Gamma_N$ is built. See
[Propagation and assembly](propagation_and_assembly.md).

## 5. Reflection and transmission coefficients

The eight amplitude coefficients are rational functions of four
components of $\Gamma_N$. For lab-frame-diagonal substrates the
labels are the usual $p$ and $s$, and the expressions follow Yeh
[[Yeh 1979](overview.md#references)] up to the sign convention
established in the erratum [[Passler and Paarmann
2019](overview.md#references)]. For birefringent substrates the
natural eigen-labels are ordinary (o) and extraordinary (e), and
the coefficients are relabeled $t_{po}$, $t_{se}$, $t_{pe}$,
$t_{so}$ (and analogously for $r$). The reflectance is
$R_{kl} = |r_{kl}|^2$ because the incident medium is isotropic.
The transmittance, on the other hand, is not in general
$|t_{kl}|^2$. A proper energy-conservation treatment for
anisotropic substrates was deferred to a later publication
[[Passler and Paarmann 2019](overview.md#references)] and is
outside the scope of this initial implementation. See
[Reflection and transmission](reflection_transmission.md).

## 6. Electric field distribution

Reconstructing $\vec{E}(x, y, z)$ inside the stack requires one
additional step per depth. The erratum-corrected procedure
[[Passler and Paarmann 2019](overview.md#references)] is to
propagate the amplitude vector through the appropriate sequence of
$L_i$ and $P_i(z)$ from the substrate backward, and at each $z$
compose the three-component field by summing each mode's amplitude
times its normalized eigenvector $\hat{\vec{\gamma}}_{ij}$. The
calculation is performed separately for $p$-polarized and
$s$-polarized incident light so that birefringent cross-coupling
is represented correctly. Because $\hat{\vec{\gamma}}_{ij}$
already carries the phase information, no separate
reflection-phase bookkeeping is needed. See
[Electric field distribution](electric_field_distribution.md).