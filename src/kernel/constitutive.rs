//! Stage 1, the constitutive matrix `M`.
//!
//! See PP2017 Eq. (4). The constitutive map `C = M G` with
//! `C = (D, B)^T` and `G = (E, H)^T` in the lab frame is the
//! 6x6 block matrix
//!
//! ```text
//!     M = [ bar_epsilon   bar_rho_1 ]
//!         [ bar_rho_2     bar_mu    ]
//! ```
//!
//! `refloxide` does not currently treat the optical-rotation
//! tensors `bar_rho_{1,2}`, so they are zero by construction. The
//! permeability `bar_mu` defaults to the identity for non-magnetic
//! media.

use crate::error::KernelResult;
use crate::types::scalar::C64;
use crate::types::tensor::LabTensor;
use nalgebra::Matrix6;

/// Type alias for the Berreman 6x6 constitutive matrix.
pub type MMatrix = Matrix6<C64>;

/// Build the 6x6 constitutive matrix from the lab-frame
/// `bar_epsilon` and `bar_mu`. Implements PP2017 Eq. (4).
///
/// The rotation tensors are not currently exposed and are set to
/// zero. A future revision can add them as optional inputs.
pub fn build_m(epsilon_lab: LabTensor, mu_lab: LabTensor) -> KernelResult<MMatrix> {
    let _ = (epsilon_lab, mu_lab);
    todo!("build_m not yet implemented")
}
