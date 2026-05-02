//! Stage 4, the propagation matrix and the stack-level assembly.
//!
//! Implements PP2017 Eq. (25) for the per-layer propagation matrix
//! `P_i`, Eq. (28) first line for the stack-level assembly
//! `Gamma_N = A_0^{-1} T_tot A_{N+1}`, and the unnumbered
//! `Lambda_1324` permutation that maps the Passler `(p-trans,
//! s-trans, p-refl, s-refl)` ordering to the Yeh `(trans p, refl
//! p, trans s, refl s)` layout. See
//! `docs/theory/propagation_and_assembly.md`.

use crate::error::{KernelError, KernelResult};
use crate::types::scalar::{C64, SPEED_OF_LIGHT_M_S};
use nalgebra::Matrix4;

/// Build the diagonal propagation matrix `P_i` per PP2017 Eq.
/// (25). The four diagonal entries are
/// `exp(-i (omega/c) q_ij d_i)`.
pub fn build_p(
    q: &[C64; 4],
    thickness_nm: f64,
    omega_rad_per_s: f64,
) -> KernelResult<Matrix4<C64>> {
    if !thickness_nm.is_finite() || thickness_nm < 0.0 {
        return Err(KernelError::InvalidGeometry(
            "layer thickness_nm must be finite and non-negative".into(),
        ));
    }
    let k0 = omega_rad_per_s / SPEED_OF_LIGHT_M_S;
    let d_m = thickness_nm * 1.0e-9;
    let iwt = C64::new(0.0, -1.0) * k0 * d_m;
    let mut p = Matrix4::zeros();
    for j in 0..4 {
        p[(j, j)] = (iwt * q[j]).exp();
    }
    Ok(p)
}

/// Build partial propagation `P(z)` with thickness `z_nm` inside layer.
pub fn build_p_partial_z(q: &[C64; 4], z_nm: f64, omega_rad_per_s: f64) -> KernelResult<Matrix4<C64>> {
    build_p(q, z_nm, omega_rad_per_s)
}

/// The Passler-to-Yeh permutation `Lambda_1324`. Defined as a
/// 4x4 permutation matrix with ones at positions
/// `(0,0), (1,2), (2,1), (3,3)`.
pub fn lambda_1324() -> Matrix4<C64> {
    let mut m = Matrix4::zeros();
    m[(0, 0)] = C64::new(1.0, 0.0);
    m[(1, 2)] = C64::new(1.0, 0.0);
    m[(2, 1)] = C64::new(1.0, 0.0);
    m[(3, 3)] = C64::new(1.0, 0.0);
    m
}

/// Assemble the stack-level transfer matrix
/// `Gamma_N = A_0^{-1} T_tot A_{N+1}` per PP2017 Eq. (28). The
/// caller is responsible for supplying the cladding interface
/// matrices and the ordered list of per-layer
/// `(A_i, P_i, A_i^{-1})` triples.
pub fn assemble_gamma(
    a_incident: &Matrix4<C64>,
    layer_triples: &[(Matrix4<C64>, Matrix4<C64>, Matrix4<C64>)],
    a_substrate: &Matrix4<C64>,
) -> KernelResult<Matrix4<C64>> {
    let mut t_tot = Matrix4::identity();
    for (a, p, a_inv) in layer_triples.iter() {
        let ti = *a * *p * *a_inv;
        t_tot *= ti;
    }
    let a0_inv = a_incident.try_inverse().ok_or_else(|| {
        KernelError::SingularCoefficientDenominator(a_incident[(0, 0)].re.abs())
    })?;
    Ok(a0_inv * t_tot * a_substrate)
}

/// Apply [`lambda_1324`] to a `Gamma_N` matrix to convert from
/// Passler layout to Yeh layout for the coefficient assembly in
/// [`crate::kernel::coefficients`].
pub fn permute_to_yeh(gamma_passler: &Matrix4<C64>) -> Matrix4<C64> {
    let lam = lambda_1324();
    lam * gamma_passler * lam
}
