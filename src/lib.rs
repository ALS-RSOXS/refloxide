//! refloxide: 4x4 transfer matrix reflectometry kernels in Rust.
//!
//! This crate is the native backbone of the `refloxide` Python package. The
//! uniaxial-z kernel lives in [`uniaxial`] and is exposed through PyO3 as
//! `refloxide.rust` when the `python` feature is enabled. Errors are typed
//! through [`error::RefloxideError`] so callers can recover or surface
//! diagnostics. Future backends (biaxial, off-axis tilt, magneto-optic)
//! should land alongside the uniaxial module under `src/` rather than
//! inside the existing kernel.

pub mod error;
pub mod uniaxial;

#[cfg(feature = "python")]
mod python;

pub use error::{RefloxideError, Result};
pub use uniaxial::{uniaxial_reflectivity, Layer, UniaxialOutput};
