//! pyo3 function surface.
//!
//! The Python-visible API is intentionally narrow. The Rust-side
//! [`crate::Pipeline`] staged builder is not exposed.

use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Smoke-test entry point retained from the original stub.
#[pyfunction]
pub fn hello_from_rust() -> String {
    "Hello from refloxide core".to_string()
}

/// One-shot amplitude entry point. The Python side passes the
/// stack as a sequence of layer dicts and receives the eight
/// amplitudes as a tuple. The exact signature will be finalized
/// when the type infrastructure pass lands.
#[pyfunction]
pub fn compute_amplitudes_py(
    _py: Python<'_>,
    _stack_repr: Py<PyAny>,
    _omega_rad_per_s: f64,
    _theta_rad: f64,
) -> PyResult<Py<PyAny>> {
    todo!("compute_amplitudes_py not yet implemented")
}

/// One-shot field-reconstruction entry point.
#[pyfunction]
pub fn compute_field_py(
    _py: Python<'_>,
    _stack_repr: Py<PyAny>,
    _omega_rad_per_s: f64,
    _theta_rad: f64,
    _layer_index: usize,
    _z_nm: f64,
    _polarization: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    todo!("compute_field_py not yet implemented")
}
