//! Stage 5, the eight Passler-Paarmann amplitude coefficients.
//!
//! Implements PP2017 Eq. (33) for `t_pp` (unchanged by erratum)
//! and PP2019 Eqs. (34*), (35*), (36*) for `t_ss`, `t_ps`, `t_sp`
//! (sign-flipped per erratum). The four reflection coefficients
//! `r_pp, r_ss, r_ps, r_sp` follow PP2017 Eqs. (29) through (32),
//! untouched by the erratum. See
//! `docs/theory/reflection_transmission.md`.

use crate::error::{KernelError, KernelResult};
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

const DEN_TOL: f64 = 1.0e-18;

fn xi_denominator(g: &Matrix4<C64>) -> KernelResult<C64> {
    let d = g[(0, 0)] * g[(2, 2)] - g[(0, 2)] * g[(2, 0)];
    if d.norm() < DEN_TOL {
        return Err(KernelError::SingularCoefficientDenominator(d.norm()));
    }
    Ok(d)
}

/// Solve for the four reflection coefficients per PP2017
/// Eqs. (29)-(32). Reads the Yeh-layout `Gamma_N` matrix.
pub fn build_r_kl(gamma_yeh: &Matrix4<C64>) -> KernelResult<(C64, C64, C64, C64)> {
    let den = xi_denominator(gamma_yeh)?;
    let g = gamma_yeh;
    let r_pp = (g[(1, 0)] * g[(2, 2)] - g[(1, 2)] * g[(2, 0)]) / den;
    let r_ss = (g[(0, 0)] * g[(3, 2)] - g[(3, 0)] * g[(0, 2)]) / den;
    let r_ps = (g[(3, 0)] * g[(2, 2)] - g[(3, 2)] * g[(2, 0)]) / den;
    let r_sp = (g[(0, 0)] * g[(1, 2)] - g[(1, 0)] * g[(0, 2)]) / den;
    Ok((r_pp, r_ss, r_ps, r_sp))
}

/// Solve for `t_pp` per PP2017 Eq. (33). Untouched by the
/// erratum.
pub fn build_t_pp(gamma_yeh: &Matrix4<C64>) -> KernelResult<C64> {
    let den = xi_denominator(gamma_yeh)?;
    Ok(gamma_yeh[(2, 2)] / den)
}

/// Solve for `t_ss` per PP2019 Eq. (34*). Sign flipped relative
/// to the PP2017 form.
pub fn build_t_ss(gamma_yeh: &Matrix4<C64>) -> KernelResult<C64> {
    let den = xi_denominator(gamma_yeh)?;
    Ok(gamma_yeh[(0, 0)] / den)
}

/// Solve for `t_ps` per PP2019 Eq. (35*). Sign flipped relative
/// to the PP2017 form.
pub fn build_t_ps(gamma_yeh: &Matrix4<C64>) -> KernelResult<C64> {
    let den = xi_denominator(gamma_yeh)?;
    Ok(-gamma_yeh[(2, 0)] / den)
}

/// Solve for `t_sp` per PP2019 Eq. (36*). Sign flipped relative
/// to the PP2017 form.
pub fn build_t_sp(gamma_yeh: &Matrix4<C64>) -> KernelResult<C64> {
    let den = xi_denominator(gamma_yeh)?;
    Ok(-gamma_yeh[(0, 2)] / den)
}

/// Compose all eight amplitudes from the Yeh-layout matrix.
pub fn build_amplitudes(gamma_yeh: &Matrix4<C64>) -> KernelResult<Amplitudes> {
    let (r_pp, r_ss, r_ps, r_sp) = build_r_kl(gamma_yeh)?;
    Ok(Amplitudes {
        r_pp,
        r_ss,
        r_ps,
        r_sp,
        t_pp: build_t_pp(gamma_yeh)?,
        t_ss: build_t_ss(gamma_yeh)?,
        t_ps: build_t_ps(gamma_yeh)?,
        t_sp: build_t_sp(gamma_yeh)?,
    })
}
