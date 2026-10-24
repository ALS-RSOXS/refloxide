//! Floating-point scalar abstraction for the transfer-matrix kernels.
//!
//! [`Real`] is implemented for `f32` and `f64` only. It does not carry
//! complex or tensor types.

use num_traits::{Float, FloatConst, NumAssign};

/// Floating-point scalar accepted by the kernel (`f32` or `f64`).
pub trait Real: Float + FloatConst + NumAssign + Send + Sync + std::fmt::Debug + 'static {
    /// Rounds an `f64` literal to this precision.
    fn lit(x: f64) -> Self;
    /// Widens this value to `f64`.
    fn to_f64_lossless(self) -> f64;
}

impl Real for f32 {
    #[inline]
    fn lit(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn to_f64_lossless(self) -> f64 {
        f64::from(self)
    }
}

impl Real for f64 {
    #[inline]
    fn lit(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_f64_lossless(self) -> f64 {
        self
    }
}
