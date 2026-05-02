# Algorithm audit and traceability matrix

## Scope of this page

This page is the single-source audit log for the
Passler-Paarmann 4x4 formalism as implemented in `refloxide`. It
maps each load-bearing equation in PP2017 [[1](#references)] and
its 2019 erratum [[2](#references)] onto the Rust module that
will implement it and the regression test that exercises it. A
reviewer can sit with the two PDFs open and verify that every
equation that needs to be carried into the kernel has a home in
the code, that the home is named, and that the home has a test.

The page is intentionally redundant with the per-stage
discussions in [Foundations](foundations.md),
[Eigenmode analysis](eigenmode_analysis.md), and
[Interface matrices](interface_matrices.md). The redundancy
is structural. The per-stage pages explain why each step is
correct. This page enumerates the steps so that none is forgotten.

## How to use this page

The kernel implementation lands stage-by-stage. Each stage adds
new equations to the kernel, and each equation that lands gets
a function reference and a passing test. The audit is satisfied
when every row in the table below points to a Rust function and
to a regression test that no longer xfails.

Three column conventions are used. The "kernel function" column
names the Rust path that will implement the equation, using the
convention `crate::module::function`. The "regression test"
column names the pytest test by file and function. A column
entry of $\varnothing$ means the equation is not implemented or
tested directly because it serves only as a definition or as a
sub-step of an enclosing equation, with the enclosing equation
listed in a separate row.

## Traceability matrix

### Stage 1, the constitutive system and $\Delta$ reduction

| Equation                       | Description                                                | Kernel function                          | Regression test                                            |
|--------------------------------|------------------------------------------------------------|------------------------------------------|------------------------------------------------------------|
| PP2017 Eq. (1)                 | Snell invariant $\xi$                                      | `core::geometry::tangential_xi`          | covered by all amplitude tests                             |
| PP2017 Eq. (2)                 | Euler rotation of principal $\bar{\varepsilon}$            | `core::tensor::rotate_principal`         | `test_uniaxial_o_and_e_channels_match_closed_form`         |
| PP2017 Eq. (4)                 | Constitutive map $\vec{C} = M\vec{G}$                      | `core::constitutive::build_M`            | $\varnothing$ (audited via $\Delta$)                       |
| PP2017 Eqs. (8)-(10)           | Berreman $\Delta$ matrix and $a_{3n}, a_{6n}, b$           | `core::delta::build_delta`               | `test_fresnel_reflection_matches_closed_form`              |
| PP2017 Eq. (5), (6), (7)       | Berreman 4x4 spatial wave equation, $\Psi$ basis           | $\varnothing$ (definition)               | $\varnothing$                                              |

### Stage 2, the per-layer eigenmode analysis

| Equation                       | Description                                                | Kernel function                          | Regression test                                            |
|--------------------------------|------------------------------------------------------------|------------------------------------------|------------------------------------------------------------|
| PP2017 Eq. (11)                | Eigenvalue problem $q\Psi = \Delta\Psi$                    | `core::modes::solve_eigenmodes`          | `test_fresnel_reflection_matches_closed_form`              |
| PP2017 Eq. (12)                | Forward and backward partition on $\operatorname{Im}(q)$   | `core::modes::partition_modes`           | `test_fresnel_reflection_matches_closed_form`              |
| PP2017 Eq. (13), (14)          | Li-Sullivan-Parsons electric-projection sort                | `core::modes::sort_li`                   | `test_uniaxial_o_and_e_channels_match_closed_form`         |
| PP2017 Eq. (15), (16)          | Poynting fallback in birefringent regime                   | `core::modes::sort_poynting`             | `test_sic_gan_sic_otto_amplitudes_match_jeannin`           |
| PP2017 Eqs. (17), (18)         | $E_z, H_z$ recovery from $a_{3n}, a_{6n}$                  | `core::modes::longitudinal_components`   | `test_e_tangential_continuity_across_gan_sic_interfaces`   |

### Stage 3, the interface matrices and erratum corrections

| Equation                       | Description                                                | Kernel function                          | Regression test                                            |
|--------------------------------|------------------------------------------------------------|------------------------------------------|------------------------------------------------------------|
| PP2017 Eq. (19)                | Definition of $\vec{\gamma}_{ij}$                          | $\varnothing$ (notation)                 | $\varnothing$                                              |
| PP2017 Eq. (20), four entries  | Unit-normalized rows $\gamma_{i11} = \gamma_{i22} = \gamma_{i42} = 1, \gamma_{i31} = -1$ | `core::interface::gamma_unit_rows`       | $\varnothing$ (constants)                                  |
| PP2017 Eq. (20), $\gamma_{i12}, \gamma_{i32}$ | Both branches each, with $\mu_i^2$ in denominator     | `core::interface::gamma_p_branch_2`      | `test_sic_gan_sic_otto_amplitudes_match_jeannin`           |
| PP2017 Eq. (20), $\gamma_{i21}, \gamma_{i41}$ | Both branches each                                    | `core::interface::gamma_s_branch_1`      | `test_sic_gan_sic_otto_amplitudes_match_jeannin`           |
| PP2017 Eq. (20), $\gamma_{i23}, \gamma_{i43}$ | Both branches each                                    | `core::interface::gamma_s_branch_3`      | `test_sic_gan_sic_otto_amplitudes_match_jeannin`           |
| PP2019 Eq. (20\*), $\gamma_{i13}, \gamma_{i33}$ | Erratum-corrected components                       | `core::interface::gamma_p_branch_3`      | `test_uniaxial_substrate_matches_jeannin`                  |
| PP2019 Eq. (E1)                | Normalization $\hat{\vec{\gamma}}_{ij}$                    | `core::interface::normalize_gamma`       | `test_uniaxial_substrate_matches_jeannin`                  |
| PP2017 Eq. (21)                | Boundary-matching $A_{i-1}\vec{E}_{i-1} = A_i\vec{E}_i$    | $\varnothing$ (algebraic identity)        | `test_e_tangential_continuity_across_gan_sic_interfaces`   |
| PP2017 Eq. (22)                | Assembly of $A_i$ with $\vec{H}$ rows                      | `core::interface::build_A`               | `test_fresnel_reflection_matches_closed_form`              |
| PP2017 Eq. (24)                | Interface matrix $L_i = A_{i-1}^{-1} A_i$                  | `core::interface::build_L`               | `test_e_tangential_continuity_across_gan_sic_interfaces`   |

### Stage 4, propagation and stack assembly

| Equation                       | Description                                                | Kernel function                          | Regression test                                            |
|--------------------------------|------------------------------------------------------------|------------------------------------------|------------------------------------------------------------|
| PP2017 Eq. (25)                | Per-layer propagation matrix $P_i$                         | `core::propagate::build_P`               | `test_e_tangential_continuity_across_gan_sic_interfaces`   |
| PP2017 Eq. (28), first line    | $\Gamma_N = A_0^{-1}\,T_{\text{tot}}\,A_{N+1}$             | `core::propagate::assemble_Gamma`        | `test_fresnel_reflection_matches_closed_form`              |
| PP2017 Eq. (28), second line   | $T_{\text{tot}}$ as alternating $A_i P_i A_i^{-1}$ product | $\varnothing$ (subsumed by line 1)       | $\varnothing$                                              |
| PP2017 unnumbered              | $\Lambda_{1324}$ permutation to Yeh layout                 | `core::propagate::permute_to_yeh`        | `test_sic_gan_sic_otto_amplitudes_match_passler_matlab`    |

### Stage 5, the reflection and transmission coefficients

| Equation                       | Description                                                | Kernel function                          | Regression test                                            |
|--------------------------------|------------------------------------------------------------|------------------------------------------|------------------------------------------------------------|
| PP2017 Eq. (33)                | $t_{pp}$ (unchanged by erratum)                            | `core::coefficients::t_pp`               | `test_fresnel_reflection_matches_closed_form`              |
| PP2019 Eq. (34\*)              | $t_{ss}$ (sign flip per erratum)                           | `core::coefficients::t_ss`               | `test_fresnel_reflection_matches_closed_form`              |
| PP2019 Eq. (35\*)              | $t_{ps}$ (sign flip per erratum)                           | `core::coefficients::t_ps`               | `test_uniaxial_substrate_matches_jeannin`                  |
| PP2019 Eq. (36\*)              | $t_{sp}$ (sign flip per erratum)                           | `core::coefficients::t_sp`               | `test_uniaxial_substrate_matches_jeannin`                  |
| PP2017 Eqs. (29)-(32)          | Four $r_{kl}$, untouched by erratum                        | `core::coefficients::r_kl`               | `test_fresnel_reflection_matches_closed_form`              |

### Stage 6, the electric-field reconstruction

| Equation                       | Description                                                | Kernel function                          | Regression test                                            |
|--------------------------------|------------------------------------------------------------|------------------------------------------|------------------------------------------------------------|
| PP2019 Eq. (37\*)              | Layer amplitude vector $\vec{E}_i$ from $r_{kl}, t_{kl}$   | `core::field::layer_amplitudes`          | `test_e_tangential_continuity_across_gan_sic_interfaces`   |
| PP2019 Eq. (E2)                | Field reconstruction inside layer $i$                      | `core::field::reconstruct`               | `test_e_tangential_continuity_across_gan_sic_interfaces`   |

## Outstanding correctness gaps

Three classes of correctness check are not addressed by the
matrix above and must land before the kernel is considered
production-ready.

The first class is the four-fold-degenerate limit at exactly
normal incidence in exactly isotropic media. Both $C(q_{ij})$
projections are formally indeterminate there, and the Xu
piecewise dispatch on $|q_{i1} - q_{i2}|$ has to fall back on a
labeling-agnostic branch. The current scaffold has no test for
this limit, because the analytical Fresnel benchmark uses a
finite incidence angle to sidestep it. A future test should
sweep $\theta \to 0$ in an isotropic medium and assert that the
amplitude ratios are continuous through the limit, even though
the $p$ and $s$ labels carry no physical content.

The second class is the loss-sign convention. The forward and
backward classification on $\operatorname{Im}(q)$ presumes a
passive medium with $\operatorname{Im}(\varepsilon) \ge 0$. A
gain medium inverts the classification. The library does not
guard against this, and the regression scaffold does not test it,
because the kernel has not yet committed to a documented sign
convention for gain media. A future test should fix the
convention by exercising a stack with $\operatorname{Im}(\varepsilon) < 0$
and asserting the expected behavior.

The third class is exactly satisfied $b = 0$ pathologies in the
constitutive matrix, where the longitudinal elimination of $E_z$
and $H_z$ is singular. The library presumes generic
magneto-dielectric media and does not guard against this. We
note it here for completeness rather than as an action item,
because no condensed-matter system encountered in the project's
target use cases produces $b = 0$.

## Verification protocol

The audit is run by working down the matrix and asking three
questions of each row. Has the named kernel function been
implemented and committed. Does the function carry a doc-comment
that quotes the equation number from PP2017 or PP2019. Does the
named regression test pass without xfail.

A row passes when all three answers are yes. A row fails when
any answer is no, and the kernel is not considered ready until
all rows pass.

The audit is intended to be lightweight and re-runnable. A
suggested implementation is a script under
`tests/regression/_audit.py` that walks this page, parses the
table, and asserts that each named function exists and that each
named test does not xfail. The script lives outside the formal
test suite because it is a structural audit rather than a
functional test, but its output is the canonical statement of
audit pass or fail.

## References

1. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix
   formalism for light propagation in anisotropic stratified
   media," J. Opt. Soc. Am. B **34**, 2128 (2017).
   [DOI](https://doi.org/10.1364/JOSAB.34.002128).
2. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix
   formalism for light propagation in anisotropic stratified
   media, erratum," J. Opt. Soc. Am. B **36**, 3246 (2019).
   [DOI](https://doi.org/10.1364/JOSAB.36.003246).
3. M. Jeannin, "Generalized 4x4 matrix algorithm for light
   propagation in anisotropic stratified media (Python files),"
   Zenodo (2019),
   [https://doi.org/10.5281/zenodo.3417751](https://doi.org/10.5281/zenodo.3417751).
4. N. C. Passler and A. Paarmann, "Generalized 4x4 matrix
   algorithm for light propagation in anisotropic stratified
   media (Matlab files)," Zenodo (2019),
   [https://doi.org/10.5281/zenodo.601496](https://doi.org/10.5281/zenodo.601496).
