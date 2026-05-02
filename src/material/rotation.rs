//! Euler-angle rotation of principal-frame tensors into the lab
//! frame.
//!
//! The convention is the Passler `z, x', z''` convention of PP2017
//! Eq. (2). The rotation matrix `Omega` is constructed from the
//! three angles, and the lab-frame tensor is
//! `bar_epsilon_lab = Omega bar_epsilon_principal Omega^T`.
//!
//! For graded interfaces with tilted optic axes, the kernel uses
//! quaternion slerp to interpolate between two Euler triples. The
//! [`slerp`] function implements that interpolation.

use crate::error::KernelResult;
use crate::types::scalar::C64;
use crate::types::tensor::{LabTensor, PrincipalDiag};

/// Three Euler angles in radians under the `z, x', z''`
/// (Passler / PP2017 Eq. (2)) convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EulerAnglesZxz {
    pub phi: f64,
    pub theta: f64,
    pub psi: f64,
}

impl EulerAnglesZxz {
    /// Zero Euler angles, equivalent to the identity rotation.
    pub fn zero() -> Self {
        Self {
            phi: 0.0,
            theta: 0.0,
            psi: 0.0,
        }
    }

    /// Construct from three angles in radians.
    pub fn from_radians(phi: f64, theta: f64, psi: f64) -> Self {
        Self { phi, theta, psi }
    }

    /// Construct from three angles in degrees.
    pub fn from_degrees(phi_deg: f64, theta_deg: f64, psi_deg: f64) -> Self {
        Self {
            phi: phi_deg.to_radians(),
            theta: theta_deg.to_radians(),
            psi: psi_deg.to_radians(),
        }
    }

    /// Build the 3x3 real-valued rotation matrix `Omega` per
    /// PP2017 Eq. (2). The matrix is then promoted to complex
    /// entries for application to `bar_epsilon`.
    pub fn rotation_matrix(&self) -> nalgebra::Matrix3<C64> {
        todo!("EulerAnglesZxz::rotation_matrix not yet implemented")
    }
}

/// Apply the Euler rotation to a principal-frame diagonal tensor,
/// returning the full lab-frame tensor.
pub fn rotate_principal(diag: PrincipalDiag, angles: EulerAnglesZxz) -> KernelResult<LabTensor> {
    let _ = (diag, angles);
    todo!("rotate_principal not yet implemented")
}

/// Spherical linear interpolation between two Euler triples,
/// performed in quaternion space for shortest-path geodesic
/// behavior. Used by the graded-interface module to smoothly
/// rotate the optic axis across a rough boundary.
pub fn slerp(angles_above: EulerAnglesZxz, angles_below: EulerAnglesZxz, s: f64) -> EulerAnglesZxz {
    let _ = (angles_above, angles_below, s);
    todo!("slerp not yet implemented")
}
