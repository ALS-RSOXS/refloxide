//! Reusable numerical kernels for transfer-matrix solvers.
//!
//! This module has no dependency on any solver.

mod exact_inv4;

pub use exact_inv4::exact_inv_4x4;
pub use exact_inv4::exact_inv_4x4_generic;
