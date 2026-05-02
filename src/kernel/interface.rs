//! Stage 3, the interface matrices and Xu piecewise eigenvectors.
//!
//! Implements PP2017 Eq. (20) for the Xu piecewise `gamma_ij`
//! components, with the PP2019 Eq. (20*) erratum corrections
//! folded into `gamma_i13` and `gamma_i33`. The normalization
//! `hat_gamma = gamma / |gamma|` follows PP2019 Eq. (E1). The
//! interface matrix `A_i` follows PP2017 Eq. (22) with `H` rows
//! derived from Faraday's law.
//!
//! Full tabulation in `docs/theory/interface_matrices.md`.

use crate::error::{KernelError, KernelResult};
use crate::kernel::modes::LayerModes;
use crate::kernel::XU_DEGENERATE_THRESHOLD;
use crate::types::scalar::C64;
use crate::types::tensor::LabTensor;
use nalgebra::{Matrix4, Vector3};

/// One Xu electric-field eigenvector, three complex components.
pub type Gamma = Vector3<C64>;

#[inline]
fn e(lab: &LabTensor, i: usize, j: usize) -> C64 {
    lab.as_matrix()[(i - 1, j - 1)]
}

/// The four normalization constants of PP2017 Eq. (20),
/// `gamma_i11 = gamma_i22 = gamma_i42 = 1`, `gamma_i31 = -1`.
pub fn gamma_unit_rows() -> [(C64, C64); 4] {
    [
        (C64::new(1.0, 0.0), C64::new(1.0, 0.0)),
        (C64::new(1.0, 0.0), C64::new(1.0, 0.0)),
        (C64::new(-1.0, 0.0), C64::new(1.0, 0.0)),
        (C64::new(1.0, 0.0), C64::new(1.0, 0.0)),
    ]
}

fn denom_p12(lab: &LabTensor, mu: C64, xi: C64, q1: C64) -> C64 {
    let e33 = e(lab, 3, 3);
    let e22 = e(lab, 2, 2);
    let e23 = e(lab, 2, 3);
    let e32 = e(lab, 3, 2);
    (mu * e33 - xi * xi) * (mu * e22 - xi * xi - q1 * q1) - mu * mu * e23 * e32
}

fn denom_p21(lab: &LabTensor, mu: C64, xi: C64, q2: C64) -> C64 {
    let e33 = e(lab, 3, 3);
    let e11 = e(lab, 1, 1);
    let e13 = e(lab, 1, 3);
    let e31 = e(lab, 3, 1);
    (mu * e33 - xi * xi) * (mu * e11 - q2 * q2)
        - (mu * e13 + xi * q2) * (mu * e31 + xi * q2)
}

/// Compute `gamma_i12` and `gamma_i32` per PP2017 Eq. (20). Both
/// branches handled. Denominator carries `mu_i^2 epsilon_i23
/// epsilon_i32`, not `mu_i epsilon_i23 epsilon_i32`.
pub fn gamma_p_branch_2(
    epsilon_lab: &LabTensor,
    mu_scalar: C64,
    xi: C64,
    q_pair: (C64, C64),
) -> KernelResult<(C64, C64)> {
    let (q1, q2) = q_pair;
    let mu = mu_scalar;
    let e23 = e(epsilon_lab, 2, 3);
    let e31 = e(epsilon_lab, 3, 1);
    let e21 = e(epsilon_lab, 2, 1);
    let e33 = e(epsilon_lab, 3, 3);
    let g12_t = if (q1 - q2).norm() < XU_DEGENERATE_THRESHOLD {
        C64::new(0.0, 0.0)
    } else {
        let num =
            mu * e23 * (mu * e31 + xi * q1) - mu * e21 * (mu * e33 - xi * xi);
        let den = denom_p12(epsilon_lab, mu, xi, q1);
        if den.norm() < 1.0e-18 {
            return Err(KernelError::SingularConstitutive(den));
        }
        num / den
    };
    let g32_t = if (q1 - q2).norm() < XU_DEGENERATE_THRESHOLD {
        let den = mu * e33 - xi * xi;
        if den.norm() < 1.0e-18 {
            return Err(KernelError::SingularConstitutive(den));
        }
        -(mu * e23) / den
    } else {
        let den = mu * e33 - xi * xi;
        if den.norm() < 1.0e-18 {
            return Err(KernelError::SingularConstitutive(den));
        }
        -(mu * e31 + xi * q2) / den * g12_t - (mu * e23) / den
    };
    Ok((g12_t, g32_t))
}

/// Compute `gamma_i13` and `gamma_i33` per the PP2019 Eq. (20*)
/// erratum corrections. The transmitted branch returns
/// `gamma_i13`, the reflected branch returns `gamma_i33`.
pub fn gamma_p_branch_3(
    epsilon_lab: &LabTensor,
    mu_scalar: C64,
    xi: C64,
    q_pair: (C64, C64),
    gamma_branch_2: C64,
    is_reflected: bool,
) -> KernelResult<C64> {
    let (q1, _) = q_pair;
    let mu = mu_scalar;
    let e31 = e(epsilon_lab, 3, 1);
    let e33 = e(epsilon_lab, 3, 3);
    let e32 = e(epsilon_lab, 3, 2);
    let den = mu * e33 - xi * xi;
    if den.norm() < 1.0e-18 {
        return Err(KernelError::SingularConstitutive(den));
    }
    if !is_reflected {
        if (q_pair.0 - q_pair.1).norm() < XU_DEGENERATE_THRESHOLD {
            Ok(-(mu * e31 + xi * q1) / den)
        } else {
            Ok(-(mu * e31 + xi * q1) / den - (mu * e32) / den * gamma_branch_2)
        }
    } else if (q_pair.0 - q_pair.1).norm() < XU_DEGENERATE_THRESHOLD {
        Ok((mu * e31 + xi * q1) / den)
    } else {
        Ok((mu * e31 + xi * q1) / den + (mu * e32) / den * gamma_branch_2)
    }
}

/// Compute `gamma_i21` and `gamma_i41` per PP2017 Eq. (20).
pub fn gamma_s_branch_1(
    epsilon_lab: &LabTensor,
    mu_scalar: C64,
    xi: C64,
    q_pair: (C64, C64),
) -> KernelResult<(C64, C64)> {
    let (_, q2) = q_pair;
    let mu = mu_scalar;
    let e32 = e(epsilon_lab, 3, 2);
    let e13 = e(epsilon_lab, 1, 3);
    let e12 = e(epsilon_lab, 1, 2);
    let e33 = e(epsilon_lab, 3, 3);
    let g21_t = if (q_pair.0 - q_pair.1).norm() < XU_DEGENERATE_THRESHOLD {
        C64::new(0.0, 0.0)
    } else {
        let num =
            mu * e32 * (mu * e13 + xi * q2) - mu * e12 * (mu * e33 - xi * xi);
        let den = denom_p21(epsilon_lab, mu, xi, q2);
        if den.norm() < 1.0e-18 {
            return Err(KernelError::SingularConstitutive(den));
        }
        num / den
    };
    let e31 = e(epsilon_lab, 3, 1);
    let g41_t = if (q_pair.0 - q_pair.1).norm() < XU_DEGENERATE_THRESHOLD {
        let den = mu * e33 - xi * xi;
        if den.norm() < 1.0e-18 {
            return Err(KernelError::SingularConstitutive(den));
        }
        -(mu * e32) / den
    } else {
        let den = mu * e33 - xi * xi;
        if den.norm() < 1.0e-18 {
            return Err(KernelError::SingularConstitutive(den));
        }
        -(mu * e31 + xi * q2) / den * g21_t - (mu * e32) / den
    };
    Ok((g21_t, g41_t))
}

/// Compute `gamma_i23` and `gamma_i43` per PP2017 Eq. (20).
pub fn gamma_s_branch_3(
    epsilon_lab: &LabTensor,
    mu_scalar: C64,
    xi: C64,
    q_pair: (C64, C64),
    gamma_s_branch_1: C64,
) -> KernelResult<C64> {
    let (_, q2) = q_pair;
    let mu = mu_scalar;
    let e31 = e(epsilon_lab, 3, 1);
    let e33 = e(epsilon_lab, 3, 3);
    let e32 = e(epsilon_lab, 3, 2);
    let den = mu * e33 - xi * xi;
    if den.norm() < 1.0e-18 {
        return Err(KernelError::SingularConstitutive(den));
    }
    if (q_pair.0 - q_pair.1).norm() < XU_DEGENERATE_THRESHOLD {
        Ok(-(mu * e32) / den)
    } else {
        Ok(
            -(mu * e31 + xi * q2) / den * gamma_s_branch_1 - (mu * e32) / den,
        )
    }
}

/// Apply the PP2019 Eq. (E1) normalization `hat_gamma = gamma /
/// |gamma|`. The normalization is required for correct
/// cross-polarization amplitudes in birefringent substrates.
pub fn normalize_gamma(gamma: Gamma) -> KernelResult<Gamma> {
    let n = (gamma[0].norm_sqr() + gamma[1].norm_sqr() + gamma[2].norm_sqr()).sqrt();
    if n <= 1.0e-18 {
        return Err(KernelError::InvalidGeometry(
            "cannot normalize zero-length gamma".into(),
        ));
    }
    Ok(gamma.map(|z| z / n))
}

/// Build the interface matrix `A_i` per PP2017 Eq. (22) from the
/// four normalized eigenvectors and the four eigenvalues. The row
/// ordering is `(E_x, E_y, H_y, -H_x)` per
/// [`crate::kernel::INTERFACE_BASIS_ORDER`].
pub fn build_a(
    hat_gammas: &[Gamma; 4],
    modes: &LayerModes,
    mu_scalar: C64,
    xi: C64,
) -> KernelResult<Matrix4<C64>> {
    let mut a = Matrix4::zeros();
    for j in 0..4 {
        let g = &hat_gammas[j];
        let qj = modes.q[j];
        a[(0, j)] = g[0];
        a[(1, j)] = g[1];
        let hy = (qj * g[0] - xi * g[2]) / mu_scalar;
        let neg_hx_row = qj * g[1] / mu_scalar;
        a[(2, j)] = hy;
        a[(3, j)] = neg_hx_row;
    }
    Ok(a)
}

/// Build the per-interface matrix `L_i = A_{i-1}^{-1} A_i` per
/// PP2017 Eq. (24).
pub fn build_l(a_above: &Matrix4<C64>, a_below: &Matrix4<C64>) -> KernelResult<Matrix4<C64>> {
    let inv = a_above
        .try_inverse()
        .ok_or_else(|| KernelError::SingularCoefficientDenominator(0.0))?;
    Ok(inv * a_below)
}

fn reflected_gamma_pair(
    epsilon_lab: &LabTensor,
    mu: C64,
    xi: C64,
    q3: C64,
    q4: C64,
) -> KernelResult<(C64, C64, C64, C64)> {
    let e21 = e(epsilon_lab, 2, 1);
    let e23 = e(epsilon_lab, 2, 3);
    let e31 = e(epsilon_lab, 3, 1);
    let e33 = e(epsilon_lab, 3, 3);
    let e22 = e(epsilon_lab, 2, 2);
    let e32 = e(epsilon_lab, 3, 2);
    let e13 = e(epsilon_lab, 1, 3);
    let e11 = e(epsilon_lab, 1, 1);
    let e12 = e(epsilon_lab, 1, 2);
    let den_p32 = |q: C64| {
        (mu * e33 - xi * xi) * (mu * e22 - xi * xi - q * q) - mu * mu * e23 * e32
    };
    let g32 = if (q3 - q4).norm() < XU_DEGENERATE_THRESHOLD {
        C64::new(0.0, 0.0)
    } else {
        let num =
            mu * e21 * (mu * e33 - xi * xi) - mu * e23 * (mu * e31 + xi * q3);
        let den = den_p32(q3);
        if den.norm() < 1.0e-18 {
            return Err(KernelError::SingularConstitutive(den));
        }
        num / den
    };
    let den33 = mu * e33 - xi * xi;
    if den33.norm() < 1.0e-18 {
        return Err(KernelError::SingularConstitutive(den33));
    }
    let g33 = if (q3 - q4).norm() < XU_DEGENERATE_THRESHOLD {
        (mu * e31 + xi * q3) / den33
    } else {
        (mu * e31 + xi * q3) / den33 + (mu * e32) / den33 * g32
    };
    let den_p41 = |q: C64| {
        (mu * e33 - xi * xi) * (mu * e11 - q * q)
            - (mu * e13 + xi * q) * (mu * e31 + xi * q)
    };
    let g41 = if (q3 - q4).norm() < XU_DEGENERATE_THRESHOLD {
        C64::new(0.0, 0.0)
    } else {
        let num =
            mu * e32 * (mu * e13 + xi * q4) - mu * e12 * (mu * e33 - xi * xi);
        let den = den_p41(q4);
        if den.norm() < 1.0e-18 {
            return Err(KernelError::SingularConstitutive(den));
        }
        num / den
    };
    let g43 = if (q3 - q4).norm() < XU_DEGENERATE_THRESHOLD {
        -(mu * e32) / den33
    } else {
        -(mu * e31 + xi * q4) / den33 * g41 - (mu * e32) / den33
    };
    Ok((g32, g33, g41, g43))
}

/// Assemble normalized Xu eigenvectors for one layer in Passler
/// sort order `(p-trans, s-trans, p-refl, s-refl)`.
pub fn hat_gamma_quadruplet(
    epsilon_lab: &LabTensor,
    mu_scalar: C64,
    xi: C64,
    modes: &LayerModes,
) -> KernelResult<[Gamma; 4]> {
    let q = modes.q;
    let qt = (q[0], q[1]);
    let (g12, _) = gamma_p_branch_2(epsilon_lab, mu_scalar, xi, qt)?;
    let g13 = gamma_p_branch_3(epsilon_lab, mu_scalar, xi, qt, g12, false)?;
    let (g21, _) = gamma_s_branch_1(epsilon_lab, mu_scalar, xi, qt)?;
    let g23 = gamma_s_branch_3(epsilon_lab, mu_scalar, xi, qt, g21)?;
    let one = C64::new(1.0, 0.0);
    let v1 = normalize_gamma(Gamma::new(one, g12, g13))?;
    let v2 = normalize_gamma(Gamma::new(g21, one, g23))?;
    let (rg32, rg33, rg41, rg43) =
        reflected_gamma_pair(epsilon_lab, mu_scalar, xi, q[2], q[3])?;
    let v3 = normalize_gamma(Gamma::new(-one, rg32, rg33))?;
    let v4 = normalize_gamma(Gamma::new(rg41, one, rg43))?;
    Ok([v1, v2, v3, v4])
}
