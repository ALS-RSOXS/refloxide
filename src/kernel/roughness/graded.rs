//! Graded-interface structural roughness model.
//!
//! See `docs/theory/roughness_graded_interface.md` for the two
//! implementation strategies and the convergence criteria.

use crate::error::KernelResult;
use crate::kernel::roughness::traits::RoughnessModel;
use crate::stack::roughness_spec::ProfileShape;
use crate::stack::Layer;
use crate::types::scalar::C64;
use crate::types::tensor::LabTensor;
use nalgebra::Matrix4;

/// Graded-interface parameters.
#[derive(Debug, Clone)]
pub struct Graded {
    pub sigma_nm: f64,
    pub profile: ProfileShape,
    pub sublayer_count: usize,
    pub cutoff_sigmas: f64,
    pub merge_tolerance: f64,
    pub convergence_check: bool,
    pub convergence_tolerance: f64,
}

impl Graded {
    /// Default constructor with the kernel's recommended numerical
    /// knobs from the plan.
    pub fn new(sigma_nm: f64, profile: ProfileShape, sublayer_count: usize) -> Self {
        Self {
            sigma_nm,
            profile,
            sublayer_count,
            cutoff_sigmas: 3.0,
            merge_tolerance: 1e-12,
            convergence_check: false,
            convergence_tolerance: 1e-6,
        }
    }
}

impl RoughnessModel for Graded {
    fn validate(
        &self,
        wavelength_m: f64,
        eps_above: &LabTensor,
        eps_below: &LabTensor,
        theta_rad: f64,
    ) -> KernelResult<()> {
        let _ = (wavelength_m, eps_above, eps_below, theta_rad);
        todo!("Graded::validate not yet implemented")
    }

    fn correct_interface(
        &self,
        _a_above: &mut Matrix4<C64>,
        _a_below: &mut Matrix4<C64>,
        _kz_above: C64,
        _kz_below: C64,
    ) -> KernelResult<()> {
        // Graded model is structural, not multiplicative.
        Ok(())
    }

    fn discretize_interface(
        &self,
        layer_above: &Layer,
        layer_below: &Layer,
    ) -> KernelResult<Vec<Layer>> {
        let _ = (layer_above, layer_below);
        todo!("Graded::discretize_interface not yet implemented")
    }
}

/// Profile-function evaluators used by the graded model.
pub mod profiles {
    /// Gaussian distribution evaluated at `z` with width `sigma`.
    pub fn gaussian(z: f64, sigma: f64) -> f64 {
        let _ = (z, sigma);
        todo!("profiles::gaussian not yet implemented")
    }

    /// Error-function profile, the antiderivative of [`gaussian`].
    pub fn erf_profile(z: f64, sigma: f64) -> f64 {
        let _ = (z, sigma);
        todo!("profiles::erf_profile not yet implemented")
    }

    /// Linear profile transitioning from 0 to 1 across the rough
    /// region.
    pub fn linear(z: f64, sigma: f64) -> f64 {
        let _ = (z, sigma);
        todo!("profiles::linear not yet implemented")
    }

    /// Sine profile.
    pub fn sine(z: f64, sigma: f64) -> f64 {
        let _ = (z, sigma);
        todo!("profiles::sine not yet implemented")
    }

    /// Tanh profile, the antiderivative of `sech^2`.
    pub fn tanh_profile(z: f64, sigma: f64) -> f64 {
        let _ = (z, sigma);
        todo!("profiles::tanh_profile not yet implemented")
    }
}

/// Merge neighboring sublayers whose dielectric tensors agree to
/// within `tolerance`.
pub fn merge_layers(layers: Vec<Layer>, tolerance: f64) -> Vec<Layer> {
    let _ = (layers, tolerance);
    todo!("merge_layers not yet implemented")
}
