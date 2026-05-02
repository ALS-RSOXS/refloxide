//! pyo3 boundary.
//!
//! The pyo3 surface is intentionally narrow. Only the one-shot
//! amplitude and field helpers are exposed, plus a constructor
//! for [`Stack`]. The staged [`Pipeline`] remains Rust-only.

pub mod pyfunctions;

mod stack_repr;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::error::KernelError;

/// Register the kernel functions and types with the pyo3 module.
pub fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        pyfunctions::compute_amplitudes_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(pyfunctions::compute_field_py, module)?)?;
    module.add_function(wrap_pyfunction!(pyfunctions::hello_from_rust, module)?)?;
    Ok(())
}

/// Convert a [`KernelError`] into a pyo3 exception. Input
/// validation errors map to `ValueError`, numerical-runtime errors
/// map to `RuntimeError`. Documented in the plan.
impl From<KernelError> for PyErr {
    fn from(err: KernelError) -> PyErr {
        use pyo3::exceptions::{PyRuntimeError, PyValueError};
        let message = err.to_string();
        match err {
            KernelError::SingularConstitutive(_)
            | KernelError::EigenSolveFailure { .. }
            | KernelError::AmbiguousPartition { .. }
            | KernelError::AmbiguousLiSort { .. }
            | KernelError::SingularCoefficientDenominator(_)
            | KernelError::GradedConvergenceFailure { .. } => PyRuntimeError::new_err(message),
            KernelError::InvalidGeometry(_)
            | KernelError::UnsupportedConversion(_)
            | KernelError::VacuumSubstrate
            | KernelError::InvalidEulerAngle(_)
            | KernelError::RoughnessOutOfValidity(_) => PyValueError::new_err(message),
        }
    }
}
