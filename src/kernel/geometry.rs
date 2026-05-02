//! Stage 1, the Snell invariant.
//!
//! See PP2017 Eq. (1). The dimensionless tangential wave vector
//! `xi = sqrt(eps_inc) sin(theta)` is shared across every layer.

use crate::error::{KernelError, KernelResult};
use crate::types::scalar::C64;

fn principal_sqrt_eps(z: C64) -> KernelResult<C64> {
    if !z.re.is_finite() || !z.im.is_finite() {
        return Err(KernelError::InvalidGeometry(
            "incident permittivity must be finite".into(),
        ));
    }
    let mut s = z.sqrt();
    if s.im < 0.0 {
        s = -s;
    }
    Ok(s)
}

/// Compute the Snell invariant `xi = sqrt(eps_inc) sin(theta)`.
/// The kernel uses the principal branch of the complex square root
/// so that `Im(xi) >= 0` for passive incident media.
///
/// Implements PP2017 Eq. (1).
pub fn tangential_xi(eps_inc: C64, theta_rad: f64) -> KernelResult<C64> {
    if !theta_rad.is_finite() {
        return Err(KernelError::InvalidGeometry(
            "theta_rad must be finite".into(),
        ));
    }
    let root = principal_sqrt_eps(eps_inc)?;
    let xi = root * theta_rad.sin();
    Ok(xi)
}
