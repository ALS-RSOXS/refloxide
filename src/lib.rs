//! `refloxide`, the Rust kernel of the 4x4 transfer matrix method.
//!
//! Crate layout follows the plan at
//! `.cursor/plan/01_rust_kernel_implementation.md`. The pure-Rust
//! algorithm lives under [`kernel`], typed inputs live under
//! [`material`] and [`stack`], the staged builder lives under
//! [`solver`], and the pyo3 surface lives under [`ffi`].
//!
//! The plan and the algorithm-audit traceability matrix refer to
//! the algorithm modules under the path `core::*`. The Rust crate
//! exposes them as [`kernel`] to avoid the implicit shadowing of
//! the standard library `core` crate. Public-facing identifiers
//! and equation citations reference `kernel::*` accordingly.
//!
//! Equation references in module docstrings point to PP2017
//! (see `docs/theory/foundations.md`) for the Berreman reduction,
//! PP2019 erratum for the corrected `gamma_i13` and `gamma_i33`
//! components, and `docs/theory/algorithm_audit.md` for the
//! per-equation traceability matrix.

#![allow(clippy::module_inception)]
#![allow(dead_code)]

pub mod error;
#[cfg(feature = "python")]
pub mod ffi;
pub mod kernel;
pub mod material;
pub mod solver;
pub mod stack;
pub mod types;

pub use error::{KernelError, KernelResult};
pub use solver::pipeline::{Amplitudes, FieldProfile, Pipeline, Polarization};
pub use stack::{Layer, Stack};
pub use types::parameterization::{OpticalIndex, PrincipalTensor};
pub use types::scalar::C64;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// pyo3 module entry point. Must match `[tool.maturin] module-name = "refloxide._core"`
/// so the extension exports `PyInit__core`.
#[cfg(feature = "python")]
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ffi::register_module(module)
}
