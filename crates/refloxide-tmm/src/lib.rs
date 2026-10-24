//! Transfer-matrix reflectivity kernels and CPU drivers.
//!
//! [`kernel`] holds the backend-agnostic per-point solvers (uniaxial-z
//! recursion, R-matrix, full general 4x4, forward-mode tangents),
//! [`uniaxial`] and [`general_stack`] validate inputs and drive them over
//! q-points, energies, and stacks with optional rayon parallelism, and
//! [`bookended`] builds graded book-ended film stacks and solves them.
//!
//! It does not own shared types ([`refloxide_core`]), optical-constant
//! handling ([`refloxide_optics`]), the GPU backend, or any Python binding.

pub mod bookended;
pub mod general_stack;
pub mod kernel;
pub mod uniaxial;

pub use bookended::{
    adaptive_microslab_thicknesses, bookended_uniaxial_reflectivity, build_bookended_film_stack,
    density_profile_bookended, orientation_profile_bookended, BookendedParams,
};
pub use general_stack::general_reflectivity;
pub use uniaxial::{
    uniaxial_reflectivity, uniaxial_reflectivity_batch, uniaxial_reflectivity_points, Layer,
    UniaxialBatchOutput, UniaxialOutput,
};
