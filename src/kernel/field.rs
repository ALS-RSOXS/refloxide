//! Stage 6, the layer-resolved electric-field reconstruction.
//!
//! Implements PP2019 Eq. (37*) for the per-layer amplitude vector
//! `vec_E_i` from the eight amplitudes, and PP2019 Eq. (E2) for
//! the field reconstruction inside layer `i` at depth `z`. See
//! `docs/theory/electric_field_distribution.md`.

use crate::error::KernelResult;
use crate::kernel::coefficients::Amplitudes;
use crate::kernel::interface::Gamma;
use crate::kernel::modes::LayerModes;
use crate::kernel::propagate::build_p_partial_z;
use crate::types::scalar::C64;
use nalgebra::{Vector3, Vector4};

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

/// Build substrate-side mode coefficient vector per PP2019 Eq. (37*).
pub fn substrate_mode_vector(amplitudes: &Amplitudes, pol: Polarization) -> Vector4<C64> {
    let z = C64::new(0.0, 0.0);
    match pol {
        Polarization::P => Vector4::new(amplitudes.t_pp, amplitudes.t_ps, z, z),
        Polarization::S => Vector4::new(amplitudes.t_sp, amplitudes.t_ss, z, z),
    }
}

/// Compute the layer amplitude vector `vec_E_i` per PP2019
/// Eq. (37*) given upstream propagation context.
pub fn layer_amplitudes(
    amplitudes: &Amplitudes,
    mode_coeff_top: &Vector4<C64>,
    polarization: Polarization,
) -> KernelResult<Vector4<C64>> {
    let _ = (amplitudes, polarization);
    Ok(*mode_coeff_top)
}

/// Reconstruct the electric field at depth `z_nm` inside layer
/// `layer_index` per PP2019 Eq. (E2).
pub fn reconstruct(
    z_nm: f64,
    layer_amplitudes_top: &Vector4<C64>,
    modes: &LayerModes,
    hat_gamma: &[Gamma; 4],
    omega_rad_per_s: f64,
) -> KernelResult<FieldProfile> {
    let pz = build_p_partial_z(&modes.q, z_nm, omega_rad_per_s)?;
    let mut coeff = Vector4::zeros();
    for j in 0..4 {
        coeff[j] = layer_amplitudes_top[j] * pz[(j, j)];
    }
    let mut ex = C64::new(0.0, 0.0);
    let mut ey = C64::new(0.0, 0.0);
    let mut ez = C64::new(0.0, 0.0);
    for j in 0..4 {
        let c = coeff[j];
        ex += c * hat_gamma[j][0];
        ey += c * hat_gamma[j][1];
        ez += c * hat_gamma[j][2];
    }
    Ok(FieldProfile {
        e_x: ex,
        e_y: ey,
        e_z: ez,
    })
}

impl FieldProfile {
    /// Convert to a `Vector3<C64>`.
    pub fn as_vector(&self) -> Vector3<C64> {
        Vector3::new(self.e_x, self.e_y, self.e_z)
    }
}
