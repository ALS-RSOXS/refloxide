"""Regression tests for the refloxide 4x4 transfer matrix kernel.

The suite cross-checks the kernel against three independent correctness
anchors.

The first anchor is closed-form analytical benchmarks that do not depend on
any external reference code, specifically the Fresnel coefficients of a
single isotropic interface (``N = 0`` limit) and the ordinary/extraordinary
Fresnel coefficients for a uniaxial substrate with optic axis along
``z_hat``.

The second anchor is the erratum-corrected reference implementation of
Passler and Paarmann, published as MATLAB at Zenodo 601496 and
reimplemented in Python by Jeannin at Zenodo 3417751. Tests that depend on
these references are marked ``reference_matlab`` or ``reference_jeannin``
and are skipped automatically when the associated golden data are not
present on disk.

The third anchor is self-consistency of the kernel. Tangential continuity
of ``(E_x, H_y, E_y, H_x)`` across every interface is a direct algebraic
consequence of the interface matrix construction ``L_i = A_{i-1}^{-1}
A_i``, so violations are implementation bugs rather than model errors.

References:
    N. C. Passler and A. Paarmann, J. Opt. Soc. Am. B 34, 2128 (2017),
    doi:10.1364/JOSAB.34.002128.

    N. C. Passler and A. Paarmann, J. Opt. Soc. Am. B 36, 3246 (2019)
    erratum, doi:10.1364/JOSAB.36.003246.
"""
