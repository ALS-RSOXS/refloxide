//! Stage 1, the Snell invariant.
//!
//! See PP2017 Eq. (1). The dimensionless tangential wave vector
//! `xi = sqrt(eps_inc) sin(theta)` is shared across every layer.

use crate::error::KernelResult;
use crate::types::scalar::C64;

/// Compute the Snell invariant `xi = sqrt(eps_inc) sin(theta)`.
/// The kernel uses the principal branch of the complex square root
/// so that `Im(xi) >= 0` for passive incident media.
///
/// Implements PP2017 Eq. (1).
pub fn tangential_xi(eps_inc: C64, theta_rad: f64) -> KernelResult<C64> {
    let _ = (eps_inc, theta_rad);
    todo!("tangential_xi not yet implemented")
}
