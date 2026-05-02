//! Debye-Waller multiplicative roughness factor.
//!
//! See `docs/theory/roughness_debye_waller.md`.

use crate::error::KernelResult;
use crate::kernel::roughness::traits::RoughnessModel;
use crate::stack::Layer;
use crate::types::scalar::C64;
use crate::types::tensor::LabTensor;
use nalgebra::Matrix4;

/// Debye-Waller parameters.
#[derive(Debug, Clone, Copy)]
pub struct DebyeWaller {
    pub sigma_nm: f64,
    pub correlation_length_nm: f64,
}

impl RoughnessModel for DebyeWaller {
    fn validate(
        &self,
        wavelength_m: f64,
        eps_above: &LabTensor,
        eps_below: &LabTensor,
        theta_rad: f64,
    ) -> KernelResult<()> {
        let _ = (wavelength_m, eps_above, eps_below, theta_rad);
        todo!("DebyeWaller::validate not yet implemented")
    }

    fn correct_interface(
        &self,
        a_above: &mut Matrix4<C64>,
        a_below: &mut Matrix4<C64>,
        kz_above: C64,
        kz_below: C64,
    ) -> KernelResult<()> {
        let _ = (a_above, a_below, kz_above, kz_below);
        todo!("DebyeWaller::correct_interface not yet implemented")
    }

    fn discretize_interface(
        &self,
        _layer_above: &Layer,
        _layer_below: &Layer,
    ) -> KernelResult<Vec<Layer>> {
        Ok(Vec::new())
    }
}
