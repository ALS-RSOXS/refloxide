//! refloxide: 4x4 transfer matrix reflectometry kernels in Rust.
//!
//! This crate is the native backbone of the `refloxide` Python package. The
//! uniaxial-z kernel lives in [`uniaxial`] and is exposed through PyO3 as
//! `refloxide.rust` when the `python` feature is enabled. Errors are typed
//! through [`error::RefloxideError`] so callers can recover or surface
//! diagnostics. Shared numerics used by stratified solvers live in [`math`].

pub mod bookended;
mod c4x4;
pub mod error;
pub mod math;
pub mod optics;
pub mod sld;
pub mod uniaxial;

#[cfg(feature = "python")]
mod python;

pub use bookended::{
    adaptive_microslab_thicknesses, bookended_uniaxial_reflectivity, build_bookended_film_stack,
    density_profile_bookended, orientation_profile_bookended, BookendedParams,
};
pub use sld::{
    isotropic_lab_tensor, molecular_index, molecular_index_at_ooc, tensor_to_slab_row,
    uniaxial_lab_tensor,
};
pub use error::{RefloxideError, Result};
pub use uniaxial::{
    uniaxial_reflectivity, uniaxial_reflectivity_batch, Layer, UniaxialBatchOutput,
    UniaxialOutput,
};
