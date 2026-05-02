//! Shared scalar and tensor types.
//!
//! - [`scalar`] aliases the complex scalar `C64` and exposes
//!   physical constants used by the parameterization conversions.
//! - [`tensor`] provides the principal-frame and lab-frame tensor
//!   wrappers around `nalgebra::Matrix3<C64>`.
//! - [`parameterization`] enumerates the four supported optical
//!   index parameterizations and their conversions to the canonical
//!   relative-permittivity form.

pub mod parameterization;
pub mod scalar;
pub mod tensor;
