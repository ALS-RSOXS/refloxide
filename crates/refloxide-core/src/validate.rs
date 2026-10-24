//! Shape, energy, and tangent validation shared by the CPU and GPU backends.
//!
//! Each check returns a typed [`RefloxideError`] and never panics. It does
//! not build solver inputs beyond the [`LayerCoeffs`] packing in
//! [`jvp_coeffs`].

use nalgebra::Matrix3;
use num_complex::Complex;

use crate::error::{RefloxideError, Result};
use crate::layer::{layer_coeffs, Layer, LayerCoeffs};
use crate::output::PointsTangents;

/// Complex alias used throughout the kernel.
type C = Complex<f64>;

/// Validated value and tangent layer records for the JVP entry points.
#[allow(clippy::type_complexity)]
pub fn jvp_coeffs(
    q: &[f64],
    stack_of: &[usize],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
    tangents: &PointsTangents,
) -> Result<(Vec<Vec<LayerCoeffs<f64>>>, Vec<Vec<Vec<LayerCoeffs<f64>>>>)> {
    validate_points(q, stack_of, layers, tensor, energies_ev)?;
    let n_dirs = tangents.dq.len();
    let shape_err =
        |what: &str| RefloxideError::InvalidShape(format!("tangent {what} shape mismatch"));
    if tangents.dlayers.len() != n_dirs || tangents.dtensor.len() != n_dirs {
        return Err(shape_err("direction count"));
    }
    for dir in 0..n_dirs {
        if tangents.dq[dir].len() != q.len() {
            return Err(shape_err("dq"));
        }
        let (dl, dt) = (&tangents.dlayers[dir], &tangents.dtensor[dir]);
        if dl.len() != layers.len() || dt.len() != layers.len() {
            return Err(shape_err("stack count"));
        }
        for si in 0..layers.len() {
            if dl[si].len() != layers[si].len() || dt[si].len() != layers[si].len() {
                return Err(shape_err("layer count"));
            }
        }
    }
    let coeffs = layers
        .iter()
        .zip(tensor)
        .map(|(l, t)| layer_coeffs(l, t))
        .collect();
    let dcoeffs = tangents
        .dlayers
        .iter()
        .zip(&tangents.dtensor)
        .map(|(dl, dt)| dl.iter().zip(dt).map(|(l, t)| layer_coeffs(l, t)).collect())
        .collect();
    Ok((coeffs, dcoeffs))
}

/// Checks shared by the CPU and GPU point entry points.
pub fn validate_points(
    q: &[f64],
    stack_of: &[usize],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
) -> Result<()> {
    validate_batch(layers, tensor, energies_ev)?;
    if stack_of.len() != q.len() {
        return Err(RefloxideError::InvalidShape(format!(
            "stack_of length {} must match q length {}",
            stack_of.len(),
            q.len()
        )));
    }
    if let Some(&bad) = stack_of.iter().find(|&&s| s >= energies_ev.len()) {
        return Err(RefloxideError::InvalidShape(format!(
            "stack index {bad} out of range for {} stacks",
            energies_ev.len()
        )));
    }
    Ok(())
}

/// Shape and energy checks shared by the batched CPU and GPU entry points.
pub fn validate_batch(
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
) -> Result<()> {
    let n_e = energies_ev.len();
    if layers.len() != n_e || tensor.len() != n_e {
        return Err(RefloxideError::InvalidShape(format!(
            "layers and tensor batch length must match energies_ev ({n_e}), got layers={}, tensor={}",
            layers.len(),
            tensor.len()
        )));
    }
    if n_e == 0 {
        return Err(RefloxideError::InvalidShape(
            "uniaxial_reflectivity_batch requires at least one energy".into(),
        ));
    }
    for (ei, energy_ev) in energies_ev.iter().enumerate() {
        if !energy_ev.is_finite() || *energy_ev <= 0.0 {
            return Err(RefloxideError::InvalidEnergy(*energy_ev));
        }
        if layers[ei].len() != tensor[ei].len() {
            return Err(RefloxideError::LayerCountMismatch {
                layers: layers[ei].len(),
                tensor: tensor[ei].len(),
            });
        }
        if layers[ei].len() < 2 {
            return Err(RefloxideError::InsufficientLayers(layers[ei].len()));
        }
    }
    Ok(())
}
