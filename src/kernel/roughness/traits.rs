//! Shared trait for roughness models.
//!
//! See `docs/theory/roughness_framework.md` for the architectural
//! placement of this trait and the dispatcher.

use crate::error::KernelResult;
use crate::stack::Layer;
use crate::types::scalar::C64;
use crate::types::tensor::LabTensor;
use nalgebra::Matrix4;

/// Trait satisfied by every roughness model. A given model
/// implements either [`Self::correct_interface`] (multiplicative
/// models) or [`Self::discretize_interface`] (structural models),
/// never both. The default implementations make the unused method
/// a no-op.
pub trait RoughnessModel {
    /// Validate the model parameters against the local stack
    /// geometry and the probe wavelength. Returns `Ok(())` when
    /// the model is in its documented validity region. Returns
    /// [`crate::KernelError::RoughnessOutOfValidity`] otherwise.
    fn validate(
        &self,
        wavelength_m: f64,
        eps_above: &LabTensor,
        eps_below: &LabTensor,
        theta_rad: f64,
    ) -> KernelResult<()>;

    /// Modify the per-interface matrix product before it enters
    /// the propagation chain. Multiplicative models override this.
    fn correct_interface(
        &self,
        a_above: &mut Matrix4<C64>,
        a_below: &mut Matrix4<C64>,
        kz_above: C64,
        kz_below: C64,
    ) -> KernelResult<()> {
        let _ = (a_above, a_below, kz_above, kz_below);
        Ok(())
    }

    /// Replace a rough interface by a sequence of thin sublayers
    /// before the per-layer eigensolve. Structural models
    /// override this.
    fn discretize_interface(
        &self,
        layer_above: &Layer,
        layer_below: &Layer,
    ) -> KernelResult<Vec<Layer>> {
        let _ = (layer_above, layer_below);
        Ok(Vec::new())
    }
}
