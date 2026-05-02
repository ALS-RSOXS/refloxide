//! Stage 4, the propagation matrix and the stack-level assembly.
//!
//! Implements PP2017 Eq. (25) for the per-layer propagation matrix
//! `P_i`, Eq. (28) first line for the stack-level assembly
//! `Gamma_N = A_0^{-1} T_tot A_{N+1}`, and the unnumbered
//! `Lambda_1324` permutation that maps the Passler `(p-trans,
//! s-trans, p-refl, s-refl)` ordering to the Yeh `(trans p, refl
//! p, trans s, refl s)` layout. See
//! `docs/theory/propagation_and_assembly.md`.

use crate::error::KernelResult;
use crate::types::scalar::C64;
use nalgebra::Matrix4;

/// Build the diagonal propagation matrix `P_i` per PP2017 Eq.
/// (25). The four diagonal entries are
/// `exp(-i (omega/c) q_ij d_i)`.
pub fn build_p(
    q: &[C64; 4],
    thickness_nm: f64,
    omega_rad_per_s: f64,
) -> KernelResult<Matrix4<C64>> {
    let _ = (q, thickness_nm, omega_rad_per_s);
    todo!("build_p not yet implemented")
}

/// The Passler-to-Yeh permutation `Lambda_1324`. Defined as a
/// 4x4 permutation matrix with ones at positions
/// `(0,0), (1,2), (2,1), (3,3)`.
pub fn lambda_1324() -> Matrix4<C64> {
    todo!("lambda_1324 not yet implemented")
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
    let _ = (a_incident, layer_triples, a_substrate);
    todo!("assemble_gamma not yet implemented")
}

/// Apply [`lambda_1324`] to a `Gamma_N` matrix to convert from
/// Passler layout to Yeh layout for the coefficient assembly in
/// [`crate::kernel::coefficients`].
pub fn permute_to_yeh(gamma_passler: &Matrix4<C64>) -> Matrix4<C64> {
    let _ = gamma_passler;
    todo!("permute_to_yeh not yet implemented")
}
