# Interface matrices with treatment of singularities

## Scope of this page

This page treats stage 3 of the pipeline. Given the per-layer
eigenvalues $q_{ij}$ and sorted eigenvectors $\Psi_{ij}$ produced
in [`eigenmode_analysis.md`](eigenmode_analysis.md), we construct
the singularity-free electric-field eigenvectors
$\vec{\gamma}_{ij}$ of Xu, Wood, and Golding [[1](#references)],
apply the normalization and component corrections imposed by the
Passler-Paarmann erratum [[2](#references)], and assemble the
layer interface matrix $A_i$ and the pairwise interface matrix
$L_i = A_{i-1}^{-1} A_i$ of [[3](#references), Eqs. (19)-(24)].

## Why not use $\Psi_{ij}$ directly

The eigenvectors returned by a dense eigensolver for $\Delta_i$ are
formally correct solutions of $q_{ij}\Psi_{ij} = \Delta_i \Psi_{ij}$
but are not numerically usable when two eigenvalues coincide,
because the eigenspace is then two-dimensional and the solver
returns an arbitrary basis within it. The coincidence occurs
generically for isotropic media and whenever the principal
dielectric axes align with the lab frame, i.e. exactly the cases
most users care about [[3](#references), text introducing Sec.
2.A.3]. The Berreman [[4](#references)], Yeh [[5](#references)],
and Lin-Chung-Teitler [[6](#references)] formalisms all exhibit
this failure.

Xu, Wood, and Golding resolved the degeneracy by writing out the
four $\vec{\gamma}_{ij}$ in closed form as piecewise functions of
the dielectric tensor and of the eigenvalue pair
$(q_{i1}, q_{i2})$ or $(q_{i3}, q_{i4})$, and by choosing branches
that remain finite and continuous through the coincidence limit
[[1](#references)]. Passler and Paarmann adopted these formulas
and tabulated them in a single place as [[3](#references), Eq.
(20)], which is the form we implement.

## The Xu piecewise eigenvectors

Each $\vec{\gamma}_{ij} = (\gamma_{ij1}, \gamma_{ij2},
\gamma_{ij3})^\top$ is the electric-field eigenvector of mode $j$
in layer $i$. By the Xu convention [[1](#references);
[3](#references), Eq. (19)], four components are fixed by
normalization,

$$
\gamma_{i11} = \gamma_{i22} = \gamma_{i42} = 1,
\quad \gamma_{i31} = -1,
$$

and the remaining eight components are fixed by the piecewise
rule of [[3](#references), Eq. (20)] with the two entries
$\gamma_{i13}$ and $\gamma_{i33}$ replaced by the erratum forms
of [[2](#references), Eq. (20*)]. We tabulate all eight so that
the implementation in `core::interface` can be diffed
component-by-component against this page.

The transmitted pair $\vec{\gamma}_{i1}, \vec{\gamma}_{i2}$ reads

$$
\gamma_{i12} =
\begin{cases}
0, & q_{i1} = q_{i2}, \\[4pt]
\dfrac{\mu_i\varepsilon_{i23}(\mu_i\varepsilon_{i31} + \xi q_{i1})
     - \mu_i\varepsilon_{i21}(\mu_i\varepsilon_{i33} - \xi^2)}
      {(\mu_i\varepsilon_{i33} - \xi^2)(\mu_i\varepsilon_{i22} -
        \xi^2 - q_{i1}^2) - \mu_i^2\,\varepsilon_{i23}\varepsilon_{i32}},
& q_{i1} \neq q_{i2},
\end{cases}
$$

$$
\gamma_{i13} =
\begin{cases}
-\dfrac{\mu_i\varepsilon_{i31} + \xi q_{i1}}
       {\mu_i\varepsilon_{i33} - \xi^2},
& q_{i1} = q_{i2}, \\[8pt]
-\dfrac{\mu_i\varepsilon_{i31} + \xi q_{i1}}
       {\mu_i\varepsilon_{i33} - \xi^2}
-\dfrac{\mu_i\varepsilon_{i32}}
       {\mu_i\varepsilon_{i33} - \xi^2}\,\gamma_{i12},
& q_{i1} \neq q_{i2},
\end{cases}
$$

$$
\gamma_{i21} =
\begin{cases}
0, & q_{i1} = q_{i2}, \\[4pt]
\dfrac{\mu_i\varepsilon_{i32}(\mu_i\varepsilon_{i13} + \xi q_{i2})
     - \mu_i\varepsilon_{i12}(\mu_i\varepsilon_{i33} - \xi^2)}
      {(\mu_i\varepsilon_{i33} - \xi^2)(\mu_i\varepsilon_{i11}
        - q_{i2}^2) - (\mu_i\varepsilon_{i13} + \xi q_{i2})
        (\mu_i\varepsilon_{i31} + \xi q_{i2})},
& q_{i1} \neq q_{i2},
\end{cases}
$$

$$
\gamma_{i23} =
\begin{cases}
-\dfrac{\mu_i\varepsilon_{i32}}
       {\mu_i\varepsilon_{i33} - \xi^2},
& q_{i1} = q_{i2}, \\[8pt]
-\dfrac{\mu_i\varepsilon_{i31} + \xi q_{i2}}
       {\mu_i\varepsilon_{i33} - \xi^2}\,\gamma_{i21}
-\dfrac{\mu_i\varepsilon_{i32}}
       {\mu_i\varepsilon_{i33} - \xi^2},
& q_{i1} \neq q_{i2}.
\end{cases}
$$

The reflected pair $\vec{\gamma}_{i3}, \vec{\gamma}_{i4}$ is
structurally identical with $q_{i1} \leftrightarrow q_{i3}$ and
$q_{i2} \leftrightarrow q_{i4}$ substitutions and the sign
reversals of [[3](#references), Eq. (20)],

$$
\gamma_{i32} =
\begin{cases}
0, & q_{i3} = q_{i4}, \\[4pt]
\dfrac{\mu_i\varepsilon_{i21}(\mu_i\varepsilon_{i33} - \xi^2)
     - \mu_i\varepsilon_{i23}(\mu_i\varepsilon_{i31} + \xi q_{i3})}
      {(\mu_i\varepsilon_{i33} - \xi^2)(\mu_i\varepsilon_{i22}
        - \xi^2 - q_{i3}^2) - \mu_i^2\,\varepsilon_{i23}\varepsilon_{i32}},
& q_{i3} \neq q_{i4},
\end{cases}
$$

$$
\gamma_{i33} =
\begin{cases}
\dfrac{\mu_i\varepsilon_{i31} + \xi q_{i3}}
       {\mu_i\varepsilon_{i33} - \xi^2},
& q_{i3} = q_{i4}, \\[8pt]
\dfrac{\mu_i\varepsilon_{i31} + \xi q_{i3}}
       {\mu_i\varepsilon_{i33} - \xi^2}
+\dfrac{\mu_i\varepsilon_{i32}}
       {\mu_i\varepsilon_{i33} - \xi^2}\,\gamma_{i32},
& q_{i3} \neq q_{i4},
\end{cases}
$$

$$
\gamma_{i41} =
\begin{cases}
0, & q_{i3} = q_{i4}, \\[4pt]
\dfrac{\mu_i\varepsilon_{i32}(\mu_i\varepsilon_{i13} + \xi q_{i4})
     - \mu_i\varepsilon_{i12}(\mu_i\varepsilon_{i33} - \xi^2)}
      {(\mu_i\varepsilon_{i33} - \xi^2)(\mu_i\varepsilon_{i11}
        - q_{i4}^2) - (\mu_i\varepsilon_{i13} + \xi q_{i4})
        (\mu_i\varepsilon_{i31} + \xi q_{i4})},
& q_{i3} \neq q_{i4},
\end{cases}
$$

$$
\gamma_{i43} =
\begin{cases}
-\dfrac{\mu_i\varepsilon_{i32}}
       {\mu_i\varepsilon_{i33} - \xi^2},
& q_{i3} = q_{i4}, \\[8pt]
-\dfrac{\mu_i\varepsilon_{i31} + \xi q_{i4}}
       {\mu_i\varepsilon_{i33} - \xi^2}\,\gamma_{i41}
-\dfrac{\mu_i\varepsilon_{i32}}
       {\mu_i\varepsilon_{i33} - \xi^2},
& q_{i3} \neq q_{i4}.
\end{cases}
$$

The twelve entries enumerated above, together with the four
fixed unit-normalizations, saturate the eigenvector layout and
give `core::interface` a one-to-one correspondence between Rust
expressions and PP2017/PP2019 equations. Two entries merit
flagging. First, $\gamma_{i13}$ and $\gamma_{i33}$ are the
erratum-corrected components of [[2](#references), Eq. (20*)].
Second, the $\gamma_{i12}$ and $\gamma_{i32}$ denominators carry
$\mu_i^2\varepsilon_{i23}\varepsilon_{i32}$, not
$\mu_i\varepsilon_{i23}\varepsilon_{i32}$. The distinction is
invisible for non-magnetic media where $\mu_i = 1$ but matters
for magneto-optic layers, and silent loss of the quadratic
$\mu_i$ factor is a recurring transcription failure mode in
downstream ports.

The piecewise definition is the key to the algorithm's
well-posedness. In the degenerate branch $q_{i1} = q_{i2}$, the
rational expression in the $q_{i1} \neq q_{i2}$ branch develops a
$0/0$ that is cancelled by the analytic continuation Xu chose.
Taking the limit directly on the non-degenerate formula is
numerically fragile, whereas dispatching on the branch is exact.
The library therefore switches on the eigenvalue separation
$|q_{i1} - q_{i2}|$ relative to a machine-epsilon threshold,
rather than attempting to evaluate the non-degenerate form near
degeneracy.

## Erratum-corrected components

The original 2017 tabulation contained typographical errors in
$\gamma_{i13}$ and $\gamma_{i33}$ [[2](#references), discussion
preceding Eq. (20*)]. The corrected forms are the ones displayed
above. The library must implement the erratum version, because the
2017 version gives incorrect cross-polarization coefficients for
birefringent substrates. We flag this loudly in the module-level
docstring of `core::interface`.

## Normalization

The erratum additionally requires that each $\vec{\gamma}_{ij}$ be
normalized before it enters the interface matrix
[[2](#references), Eq. (E1)],

$$
\hat{\vec{\gamma}}_{ij} = \frac{\vec{\gamma}_{ij}}{|\vec{\gamma}_{ij}|}.
$$

The normalization has no effect on the amplitude coefficients
$t_{kl}$ and $r_{kl}$ for media with diagonal dielectric tensors,
because the magnitudes cancel between columns of $A_i$ and rows of
$A_i^{-1}$. It does affect the cross-polarization coefficients for
birefringent substrates [[2](#references), text following Eq.
(E1)], and it is essential for the field reconstruction of stage 6
(see [`electric_field_distribution.md`](electric_field_distribution.md)).
A library that omits the normalization silently produces wrong
$r_{ps}$, $r_{sp}$, $t_{pe}$, and $t_{so}$ in birefringent stacks.
The module therefore applies $\hat{\vec{\gamma}}$ in place of
$\vec{\gamma}$ at the point of $A_i$ assembly, and never allows
the unnormalized form to escape.

## Assembly of $A_i$

The interface matrix $A_i$ is the 4x4 matrix whose columns are
the four $\hat{\vec{\gamma}}_{ij}$ augmented with their associated
tangential $\vec{H}$ components. Following [[3](#references), Eq.
(22)],

$$
A_i =
\begin{pmatrix}
\hat{\gamma}_{i11} & \hat{\gamma}_{i21} & \hat{\gamma}_{i31} & \hat{\gamma}_{i41} \\
\hat{\gamma}_{i12} & \hat{\gamma}_{i22} & \hat{\gamma}_{i32} & \hat{\gamma}_{i42} \\
\dfrac{q_{i1}\hat{\gamma}_{i11} - \xi\hat{\gamma}_{i13}}{\mu_i}
 & \dfrac{q_{i2}\hat{\gamma}_{i21} - \xi\hat{\gamma}_{i23}}{\mu_i}
 & \dfrac{q_{i3}\hat{\gamma}_{i31} - \xi\hat{\gamma}_{i33}}{\mu_i}
 & \dfrac{q_{i4}\hat{\gamma}_{i41} - \xi\hat{\gamma}_{i43}}{\mu_i} \\
\dfrac{q_{i1}\hat{\gamma}_{i12}}{\mu_i}
 & \dfrac{q_{i2}\hat{\gamma}_{i22}}{\mu_i}
 & \dfrac{q_{i3}\hat{\gamma}_{i32}}{\mu_i}
 & \dfrac{q_{i4}\hat{\gamma}_{i42}}{\mu_i}
\end{pmatrix},
$$

where rows 3 and 4 carry the $\vec{H}$ components recovered from
Ampère's law within the layer. The column order reflects the
Passler sorting of [`eigenmode_analysis.md`](eigenmode_analysis.md),
namely
$(\text{p-trans}, \text{s-trans}, \text{p-refl}, \text{s-refl})$.

The action of $A_i$ on the four-component amplitude vector
$\vec{E}_i = (E^p_{\text{trans}}, E^s_{\text{trans}},
E^p_{\text{refl}}, E^s_{\text{refl}})^\top$ of
[[3](#references), Eq. (23)] projects the amplitude description
onto the tangential-field description that matches at an interface.

## Interface matrix $L_i$ between layers

The tangential fields on the two sides of the interface between
layers $i-1$ and $i$ match, so
$A_{i-1}\vec{E}_{i-1} = A_i\vec{E}_i$ [[3](#references), Eq. (21)].
Rearranging,

$$
\vec{E}_{i-1} = A_{i-1}^{-1} A_i \vec{E}_i \equiv L_i\,\vec{E}_i,
$$

which defines the interface matrix [[3](#references), Eq. (24)].
$L_i$ projects the mode basis of layer $i$ onto the mode basis of
layer $i-1$. The stack-level product constructed in
[`propagation_and_assembly.md`](propagation_and_assembly.md)
alternates $L_i$ and $P_i$ and is algebraically equivalent to the
$A^{-1} T A$ form of [[3](#references), Eq. (28)].

## Numerical notes

The inverse $A_{i-1}^{-1}$ is in general ill-conditioned. Two
conditions drive the ill-conditioning. The first is high optical
contrast at the interface, which makes $A_{i-1}$ close to singular
when one pair of modes dominates. The second is large exponential
phase in the adjacent propagation matrix, which the numerical
linear algebra sees as a coupled conditioning problem when $L_i$
is multiplied by $P_{i-1}$ downstream. Passler and Paarmann
explicitly recommend the form
$\Gamma_N = A_0^{-1}\,T_{\text{tot}}\,A_{N+1}$ [[3](#references),
Eq. (28), first line] as the most stable implementation route,
because it restricts inversion to the two cladding matrices.
`refloxide` follows this recommendation but retains the $L_i$ form
internally for field reconstruction (see
[`electric_field_distribution.md`](electric_field_distribution.md)),
where the per-interface operations are needed explicitly.

## Where the code lives

Stage 3 is the `core::interface` module. The Xu piecewise formulas
[[3](#references), Eq. (20); [2](#references), Eq. (20*)] are
implemented as a helper that takes $(\varepsilon_i, \mu_i, \xi,
q_{i1}, q_{i2})$ (or the analogous quadruple for the reflected
pair) and returns $\vec{\gamma}_{ij}$ on the appropriate branch.
The normalization $\hat{\vec{\gamma}}_{ij}$ is applied inside the
helper rather than at the call site, so the unnormalized form
never escapes. The interface matrix $A_i$ is assembled from the
four $\hat{\vec{\gamma}}_{ij}$ and the corresponding $\vec{H}$
rows. A dedicated unit-test module exercises each branch of the
piecewise dispatch.

## References

1. W. Xu, L. T. Wood, and T. D. Golding, "Optical degeneracies in
   anisotropic layered media," Phys. Rev. B **61**, 1740 (2000).
   [DOI](https://doi.org/10.1103/PhysRevB.61.1740).
2. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
   for light propagation in anisotropic stratified media, erratum,"
   J. Opt. Soc. Am. B **36**, 3246 (2019).
   [DOI](https://doi.org/10.1364/JOSAB.36.003246).
3. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix formalism
   for light propagation in anisotropic stratified media," J. Opt.
   Soc. Am. B **34**, 2128 (2017).
   [DOI](https://doi.org/10.1364/JOSAB.34.002128).
4. D. W. Berreman, "Optics in stratified and anisotropic media, 4x4
   matrix formulation," J. Opt. Soc. Am. **62**, 502 (1972).
   [DOI](https://doi.org/10.1364/JOSA.62.000502).
5. P. Yeh, "Electromagnetic propagation in birefringent layered
   media," J. Opt. Soc. Am. **69**, 742 (1979).
   [DOI](https://doi.org/10.1364/JOSA.69.000742).
6. P. J. Lin-Chung and S. Teitler, "4x4 matrix formalisms for
   optics in stratified anisotropic media," J. Opt. Soc. Am. A
   **1**, 703 (1984).
   [DOI](https://doi.org/10.1364/JOSAA.1.000703).
