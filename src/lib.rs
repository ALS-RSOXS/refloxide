//! refloxide: 4x4 transfer matrix reflectometry kernels in Rust.
//!
//! This crate is the native backbone of the `refloxide` Python package. The
//! uniaxial-z kernel lives in [`uniaxial`] and is exposed through PyO3 as
//! `refloxide.rust` when the `python` feature is enabled. Errors are typed
//! through [`error::RefloxideError`] so callers can recover or surface
//! diagnostics. Shared numerics used by stratified solvers live in [`math`].

mod c4x4;
pub mod error;
pub mod math;
pub mod uniaxial;

#[cfg(feature = "python")]
mod python;

pub use error::{RefloxideError, Result};
pub use uniaxial::{uniaxial_reflectivity, Layer, UniaxialOutput};
