//! Stage 5, the eight Passler-Paarmann amplitude coefficients.
//!
//! Implements PP2017 Eq. (33) for `t_pp` (unchanged by erratum)
//! and PP2019 Eqs. (34*), (35*), (36*) for `t_ss`, `t_ps`, `t_sp`
//! (sign-flipped per erratum). The four reflection coefficients
//! `r_pp, r_ss, r_ps, r_sp` follow PP2017 Eqs. (29) through (32),
//! untouched by the erratum. See
//! `docs/theory/reflection_transmission.md`.

use crate::error::KernelResult;
use crate::types::scalar::C64;
use nalgebra::Matrix4;

/// Eight Passler-Paarmann amplitude coefficients.
#[derive(Debug, Clone, Copy)]
pub struct Amplitudes {
    pub r_pp: C64,
    pub r_ss: C64,
    pub r_ps: C64,
    pub r_sp: C64,
    pub t_pp: C64,
    pub t_ss: C64,
    pub t_ps: C64,
    pub t_sp: C64,
}

/// Solve for the four reflection coefficients per PP2017
/// Eqs. (29)-(32). Reads the Yeh-layout `Gamma_N` matrix.
pub fn build_r_kl(gamma_yeh: &Matrix4<C64>) -> KernelResult<(C64, C64, C64, C64)> {
    let _ = gamma_yeh;
    todo!("build_r_kl not yet implemented")
}

/// Solve for `t_pp` per PP2017 Eq. (33). Untouched by the
/// erratum.
pub fn build_t_pp(gamma_yeh: &Matrix4<C64>) -> KernelResult<C64> {
    let _ = gamma_yeh;
    todo!("build_t_pp not yet implemented")
}

/// Solve for `t_ss` per PP2019 Eq. (34*). Sign flipped relative
/// to the PP2017 form.
pub fn build_t_ss(gamma_yeh: &Matrix4<C64>) -> KernelResult<C64> {
    let _ = gamma_yeh;
    todo!("build_t_ss not yet implemented")
}

/// Solve for `t_ps` per PP2019 Eq. (35*). Sign flipped relative
/// to the PP2017 form.
pub fn build_t_ps(gamma_yeh: &Matrix4<C64>) -> KernelResult<C64> {
    let _ = gamma_yeh;
    todo!("build_t_ps not yet implemented")
}

/// Solve for `t_sp` per PP2019 Eq. (36*). Sign flipped relative
/// to the PP2017 form.
pub fn build_t_sp(gamma_yeh: &Matrix4<C64>) -> KernelResult<C64> {
    let _ = gamma_yeh;
    todo!("build_t_sp not yet implemented")
}

/// Compose all eight amplitudes from the Yeh-layout matrix.
pub fn build_amplitudes(gamma_yeh: &Matrix4<C64>) -> KernelResult<Amplitudes> {
    let _ = gamma_yeh;
    todo!("build_amplitudes not yet implemented")
}
