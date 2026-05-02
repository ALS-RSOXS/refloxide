//! Stage 6, the layer-resolved electric-field reconstruction.
//!
//! Implements PP2019 Eq. (37*) for the per-layer amplitude vector
//! `vec_E_i` from the eight amplitudes, and PP2019 Eq. (E2) for
//! the field reconstruction inside layer `i` at depth `z`. See
//! `docs/theory/electric_field_distribution.md`.

use crate::error::KernelResult;
use crate::kernel::coefficients::Amplitudes;
use crate::kernel::modes::LayerModes;
use crate::types::scalar::C64;
use nalgebra::{Matrix4, Vector3, Vector4};

/// Polarization channel selector for the field reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarization {
    P,
    S,
}

/// Three complex components of the electric field at a single
/// `(layer, z)` evaluation point.
#[derive(Debug, Clone, Copy)]
pub struct FieldProfile {
    pub e_x: C64,
    pub e_y: C64,
    pub e_z: C64,
}

/// Compute the layer amplitude vector `vec_E_i` per PP2019
/// Eq. (37*).
pub fn layer_amplitudes(
    amplitudes: &Amplitudes,
    interface_chain: &[Matrix4<C64>],
    polarization: Polarization,
) -> KernelResult<Vector4<C64>> {
    let _ = (amplitudes, interface_chain, polarization);
    todo!("layer_amplitudes not yet implemented")
}

/// Reconstruct the electric field at depth `z_nm` inside layer
/// `layer_index` per PP2019 Eq. (E2). The longitudinal `E_z` is
/// recovered through [`crate::kernel::modes::longitudinal_components`].
pub fn reconstruct(
    layer_index: usize,
    z_nm: f64,
    layer_amplitudes: &Vector4<C64>,
    modes: &LayerModes,
    a3n: &[C64; 6],
    a6n: &[C64; 6],
    omega_rad_per_s: f64,
) -> KernelResult<FieldProfile> {
    let _ = (
        layer_index,
        z_nm,
        layer_amplitudes,
        modes,
        a3n,
        a6n,
        omega_rad_per_s,
    );
    todo!("reconstruct not yet implemented")
}

impl FieldProfile {
    /// Convert to a `Vector3<C64>`.
    pub fn as_vector(&self) -> Vector3<C64> {
        Vector3::new(self.e_x, self.e_y, self.e_z)
    }
}
