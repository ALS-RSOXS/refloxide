//! Principal-frame and lab-frame complex 3x3 tensor wrappers.

use crate::error::{KernelError, KernelResult};
use crate::types::scalar::C64;
use nalgebra::{Matrix3, Vector3};

/// A 3x3 complex tensor in the lab frame, used for both
/// `bar_epsilon` and `bar_mu`. Wrapping `Matrix3<C64>` lets the
/// kernel attach trait impls and conversions without leaking the
/// nalgebra type through the public API.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabTensor(pub Matrix3<C64>);

impl LabTensor {
    /// Construct from a raw `nalgebra::Matrix3<C64>`.
    #[inline]
    pub fn from_matrix(m: Matrix3<C64>) -> Self {
        Self(m)
    }

    /// Identity tensor.
    pub fn identity() -> Self {
        Self(Matrix3::identity())
    }

    /// Return the underlying matrix view.
    #[inline]
    pub fn as_matrix(&self) -> &Matrix3<C64> {
        &self.0
    }

    /// Return the underlying matrix by value.
    #[inline]
    pub fn into_matrix(self) -> Matrix3<C64> {
        self.0
    }

    /// Verify Hermitian symmetry within numerical tolerance. Used
    /// by the validation pass on user input. Loss-bearing tensors
    /// are not Hermitian, so this check is informational only.
    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return false;
        }
        let a = &self.0;
        let mut acc = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let d = a[(i, j)] - a[(j, i)].conj();
                acc += d.norm_sqr();
            }
        }
        acc.sqrt() <= tolerance
    }
}

impl From<Matrix3<C64>> for LabTensor {
    fn from(m: Matrix3<C64>) -> Self {
        Self(m)
    }
}

impl From<LabTensor> for Matrix3<C64> {
    fn from(t: LabTensor) -> Self {
        t.0
    }
}

/// Diagonal principal-frame tensor stored as the three principal
/// values. Used for the input parameterization where the user
/// specifies the dielectric tensor along its own principal axes,
/// before the Euler rotation applied by
/// [`crate::material::rotation`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrincipalDiag {
    pub xx: C64,
    pub yy: C64,
    pub zz: C64,
}

impl PrincipalDiag {
    /// Convert to a full lab-frame tensor by placing the principal
    /// values on the diagonal. No rotation applied.
    pub fn to_lab(&self) -> LabTensor {
        LabTensor(Matrix3::from_diagonal(&Vector3::new(
            self.xx, self.yy, self.zz,
        )))
    }

    /// Validate that all three principal values are finite and that
    /// `Im` is non-negative for passive media.
    pub fn validate_passive(&self) -> KernelResult<()> {
        const IM_EPS: f64 = 1.0e-14;
        for (axis, z) in [("xx", self.xx), ("yy", self.yy), ("zz", self.zz)] {
            if !z.re.is_finite() || !z.im.is_finite() {
                return Err(KernelError::InvalidGeometry(format!(
                    "principal {axis} value must be finite (got {z})"
                )));
            }
            if z.im < -IM_EPS {
                return Err(KernelError::InvalidGeometry(format!(
                    "principal {axis} imaginary part must be non-negative for passive media (Im={})",
                    z.im
                )));
            }
        }
        Ok(())
    }
}
