//! Reusable numerical kernels for transfer-matrix solvers.
//!
//! This module must not depend on [`crate::matrix`]. Stratified code in
//! [`crate::matrix`] may import from here.

mod exact_inv4;

pub use exact_inv4::exact_inv_4x4;
