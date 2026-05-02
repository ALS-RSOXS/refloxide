//! Roughness-model auto-dispatcher.
//!
//! Implements the decision matrix from
//! `docs/theory/roughness_selection_guide.md`. The dispatcher
//! consumes a [`crate::stack::RoughnessSpec`] per interface, the
//! local geometry, and the probe wavelength, and returns the
//! resolved model along with the audit log.

use crate::error::KernelResult;
use crate::stack::{RoughnessChoice, RoughnessSpec};
use crate::types::scalar::C64;
use crate::types::tensor::LabTensor;

/// Per-interface dispatch outcome with the diagnostic
/// dimensionless quantities recorded for the audit log.
#[derive(Debug, Clone)]
pub struct RoughnessInterfaceLog {
    pub model: RoughnessChoice,
    pub alpha: f64,
    pub beta: f64,
    pub contrast: f64,
    pub warnings: Vec<String>,
}

/// Stack-level dispatch outcome, one [`RoughnessInterfaceLog`]
/// per interface.
#[derive(Debug, Clone)]
pub struct RoughnessDispatchResult {
    pub interfaces: Vec<RoughnessInterfaceLog>,
}

/// Auto-select the roughness model per the decision matrix in
/// `docs/theory/roughness_selection_guide.md`. The function
/// resolves only the per-interface model. Construction of the
/// model object itself happens at the call site, which has access
/// to the layer geometry needed by the model constructors.
pub fn auto_select(
    spec: &RoughnessSpec,
    eps_above: &LabTensor,
    eps_below: &LabTensor,
    wavelength_m: f64,
    theta_rad: f64,
) -> KernelResult<RoughnessInterfaceLog> {
    let _ = (spec, eps_above, eps_below, wavelength_m, theta_rad);
    todo!("auto_select not yet implemented")
}

/// Compute the small-roughness parameter
/// `alpha = sigma * |k_z|`.
pub fn small_roughness_alpha(sigma_nm: f64, k_z: C64) -> f64 {
    let _ = (sigma_nm, k_z);
    todo!("small_roughness_alpha not yet implemented")
}

/// Compute the correlation-length ratio
/// `beta = xi / xi_diff`.
pub fn correlation_ratio_beta(xi_nm: f64, lambda_nm: f64, theta_rad: f64) -> f64 {
    let _ = (xi_nm, lambda_nm, theta_rad);
    todo!("correlation_ratio_beta not yet implemented")
}
