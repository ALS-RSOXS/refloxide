//! Nevot-Croce multiplicative roughness factor.
//!
//! See `docs/theory/roughness_nevot_croce.md`.

use crate::error::KernelResult;
use crate::kernel::roughness::traits::RoughnessModel;
use crate::stack::Layer;
use crate::types::scalar::C64;
use crate::types::tensor::LabTensor;
use nalgebra::Matrix4;

/// Nevot-Croce parameters.
#[derive(Debug, Clone, Copy)]
pub struct NevotCroce {
    pub sigma_nm: f64,
}

impl RoughnessModel for NevotCroce {
    fn validate(
        &self,
        wavelength_m: f64,
        eps_above: &LabTensor,
        eps_below: &LabTensor,
        theta_rad: f64,
    ) -> KernelResult<()> {
        let _ = (wavelength_m, eps_above, eps_below, theta_rad);
        todo!("NevotCroce::validate not yet implemented")
    }

    fn correct_interface(
        &self,
        a_above: &mut Matrix4<C64>,
        a_below: &mut Matrix4<C64>,
        kz_above: C64,
        kz_below: C64,
    ) -> KernelResult<()> {
        let _ = (a_above, a_below, kz_above, kz_below);
        todo!("NevotCroce::correct_interface not yet implemented")
    }

    fn discretize_interface(
        &self,
        _layer_above: &Layer,
        _layer_below: &Layer,
    ) -> KernelResult<Vec<Layer>> {
        Ok(Vec::new())
    }
}
