//! pyo3 function surface.
//!
//! The Python-visible API is intentionally narrow. The Rust-side
//! [`crate::Pipeline`] staged builder is not exposed.

use super::stack_repr::stack_from_py;
use crate::solver::pipeline::{compute_amplitudes, compute_field};
use crate::solver::Polarization;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::IntoPyObject;
use pyo3::Py;

fn polarization_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Polarization> {
    let s: String = obj.extract()?;
    match s.to_ascii_lowercase().as_str() {
        "p" | "pp" => Ok(Polarization::P),
        "s" | "ss" => Ok(Polarization::S),
        _ => Err(PyValueError::new_err(
            "polarization must be 'p' or 's' (or pp/ss)",
        )),
    }
}

/// Smoke-test entry point retained from the original stub.
#[pyfunction]
pub fn hello_from_rust() -> String {
    "Hello from refloxide core".to_string()
}

/// One-shot amplitude entry point.
///
/// Parameters
/// ----------
/// stack_repr : dict
///     ``incident``, ``layers``, ``substrate``, optional ``roughness`` list.
///     Each layer is ``thickness_nm`` and ``material`` (``epsilon_principal``,
///     optional ``mu_principal``, optional ``euler_zxz_rad``). Optical indices
///     use discriminated dicts: ``nk``, ``delta_beta``, ``sld``,
///     ``scattering_factor``, or ``epsilon``.
/// omega_rad_per_s : float
///     Angular frequency in rad/s.
/// theta_rad : float
///     Incidence angle in radians (internal convention matches ``kernel`` docs).
///
/// Returns
/// -------
/// tuple
///     Eight Python complex numbers ``r_pp, r_ss, r_ps, r_sp, t_pp, t_ss, t_ps, t_sp``.
#[pyfunction]
pub fn compute_amplitudes_py(
    py: Python<'_>,
    stack_repr: Bound<'_, PyAny>,
    omega_rad_per_s: f64,
    theta_rad: f64,
) -> PyResult<Py<PyAny>> {
    let stack = stack_from_py(stack_repr.as_ref())?;
    let a = compute_amplitudes(&stack, omega_rad_per_s, theta_rad).map_err(PyErr::from)?;
    Ok((
        a.r_pp, a.r_ss, a.r_ps, a.r_sp, a.t_pp, a.t_ss, a.t_ps, a.t_sp,
    )
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

/// One-shot field-reconstruction entry point at depth ``z_nm`` inside layer ``layer_index``.
#[pyfunction]
pub fn compute_field_py(
    py: Python<'_>,
    stack_repr: Bound<'_, PyAny>,
    omega_rad_per_s: f64,
    theta_rad: f64,
    layer_index: usize,
    z_nm: f64,
    polarization: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let stack = stack_from_py(stack_repr.as_ref())?;
    let pol = polarization_from_py(polarization.as_ref())?;
    let f = compute_field(
        &stack,
        omega_rad_per_s,
        theta_rad,
        layer_index,
        z_nm,
        pol,
    )
    .map_err(PyErr::from)?;
    Ok((f.e_x, f.e_y, f.e_z)
        .into_pyobject(py)?
        .into_any()
        .unbind())
}
