//! Book-ended graded film profiles and fused uniaxial reflectivity evaluation.
//!
//! Ports the adaptive microslab mesh and orientation/density profiles from
//! `refloxide.pxr.energy.bookended`, assembles laboratory-frame tensors, and
//! runs the existing [`crate::uniaxial::uniaxial_reflectivity`] kernel without
//! per-evaluation Python overhead.

use nalgebra::Matrix3;
use num_complex::Complex;

use crate::optics::{interpolate_ooc_linear, lab_diagonal_uniaxial};
use crate::uniaxial::{uniaxial_reflectivity, Layer, UniaxialOutput};

type C = Complex<f64>;

/// Scalar book-ended profile parameters (angles in radians).
#[derive(Debug, Clone, Copy)]
pub struct BookendedParams {
    pub total_thick: f64,
    pub surface_roughness: f64,
    pub tau_si: f64,
    pub tau_vac: f64,
    pub alpha_bulk: f64,
    pub alpha_si: f64,
    pub alpha_vac: f64,
    pub density_bulk: f64,
    pub density_si: f64,
    pub density_vac: f64,
    pub num_slabs: usize,
    pub mesh_constant: f64,
}

/// Orientation angle (rad) at depth ``z`` from the vacuum interface.
pub fn orientation_profile_bookended(
    total_thick: f64,
    z: f64,
    tau_si: f64,
    tau_vac: f64,
    alpha_bulk: f64,
    alpha_si: f64,
    alpha_vac: f64,
) -> f64 {
    let term_vac = (alpha_vac - alpha_bulk) * (-z / tau_vac).exp();
    let term_si = (alpha_si - alpha_bulk) * (-(total_thick - z) / tau_si).exp();
    alpha_bulk + term_vac + term_si
}

/// Mass density at depth ``z`` (same functional form as orientation).
pub fn density_profile_bookended(
    total_thick: f64,
    z: f64,
    tau_si: f64,
    tau_vac: f64,
    rho_bulk: f64,
    rho_si: f64,
    rho_vac: f64,
) -> f64 {
    let term_vac = (rho_vac - rho_bulk) * (-z / tau_vac).exp();
    let term_si = (rho_si - rho_bulk) * (-(total_thick - z) / tau_si).exp();
    rho_bulk + term_vac + term_si
}

/// Symmetric refining mesh that sums to ``total_thick``.
pub fn adaptive_microslab_thicknesses(
    total_thick: f64,
    num_slabs: usize,
    mesh_constant: f64,
) -> Vec<f64> {
    if num_slabs <= 1 {
        return vec![total_thick];
    }
    let n_half = num_slabs / 2;
    let half_thick = total_thick / 2.0;
    let r = mesh_constant.powf(1.0 / n_half as f64);
    let mut mesh = if num_slabs.is_multiple_of(2) {
        let a = half_thick * (r - 1.0) / (r.powi(n_half as i32) - 1.0);
        let mesh_half: Vec<f64> = (0..n_half).map(|i| a * r.powi(i as i32)).collect();
        let mut left = mesh_half.clone();
        left.reverse();
        left.extend(mesh_half);
        left
    } else {
        let center_share = total_thick / num_slabs as f64;
        let half_sum = (total_thick - center_share) / 2.0;
        let a = half_sum * (r - 1.0) / (r.powi(n_half as i32) - 1.0);
        let mesh_half: Vec<f64> = (0..n_half).map(|i| a * r.powi(i as i32)).collect();
        let mut left = mesh_half.clone();
        left.reverse();
        let center = total_thick - 2.0 * mesh_half.iter().sum::<f64>();
        left.push(center);
        left.extend(mesh_half);
        left
    };
    let remainder = total_thick - mesh.iter().sum::<f64>();
    if !mesh.is_empty() {
        mesh[0] += remainder;
    }
    mesh
}

/// Depth of each microslab center from the vacuum interface (angstrom).
pub fn mid_points_from_thicknesses(thicknesses: &[f64]) -> Vec<f64> {
    let mut cumulative = 0.0;
    thicknesses
        .iter()
        .map(|&d| {
            cumulative += d;
            cumulative - d / 2.0
        })
        .collect()
}

fn diagonal_n_tensor(n_o: C, n_e: C) -> Matrix3<C> {
    let z = C::new(0.0, 0.0);
    Matrix3::new(n_o, z, z, z, n_o, z, z, z, n_e)
}

fn isotropic_tensor_from_delta_beta(delta: f64, beta: f64) -> Matrix3<C> {
    let t = C::new(delta, beta);
    diagonal_n_tensor(t, t)
}

fn layer_row_to_parts(row: [f64; 4]) -> (Layer, Matrix3<C>) {
    let layer = Layer::new(row[0], row[1], row[2], row[3]);
    let tensor = isotropic_tensor_from_delta_beta(row[1], row[2]);
    (layer, tensor)
}

/// Build graded-film layers and tensors at one photon energy.
pub fn build_bookended_film_stack(
    energy_ev: &[f64],
    n_xx: &[f64],
    n_ixx: &[f64],
    n_zz: &[f64],
    n_izz: &[f64],
    query_ev: f64,
    params: &BookendedParams,
) -> (Vec<Layer>, Vec<Matrix3<C>>) {
    let ooc = interpolate_ooc_linear(energy_ev, n_xx, n_ixx, n_zz, n_izz, query_ev);
    let n_mol_xx_base = C::new(ooc[0], ooc[1]);
    let n_mol_zz_base = C::new(ooc[2], ooc[3]);

    let thicknesses =
        adaptive_microslab_thicknesses(params.total_thick, params.num_slabs, params.mesh_constant);
    let mid = mid_points_from_thicknesses(&thicknesses);
    let mut layers = Vec::with_capacity(params.num_slabs);
    let mut tensors = Vec::with_capacity(params.num_slabs);
    for (i, (&d, &z)) in thicknesses.iter().zip(mid.iter()).enumerate() {
        let alpha = orientation_profile_bookended(
            params.total_thick,
            z,
            params.tau_si,
            params.tau_vac,
            params.alpha_bulk,
            params.alpha_si,
            params.alpha_vac,
        );
        let rho = density_profile_bookended(
            params.total_thick,
            z,
            params.tau_si,
            params.tau_vac,
            params.density_bulk,
            params.density_si,
            params.density_vac,
        );
        let n_mol_xx = n_mol_xx_base * rho;
        let n_mol_zz = n_mol_zz_base * rho;
        let diag = lab_diagonal_uniaxial(n_mol_xx, n_mol_zz, alpha);
        let n_o = diag[0];
        let n_e = diag[2];
        let iso = n_o + n_o + n_e;
        let delta = iso.re;
        let beta = iso.im;
        let sigma = if i == 0 {
            params.surface_roughness
        } else {
            0.0
        };
        layers.push(Layer::new(d, delta, beta, sigma));
        tensors.push(diagonal_n_tensor(n_o, n_e));
    }
    (layers, tensors)
}

/// Assemble a full stratified stack and evaluate uniaxial reflectivity.
///
/// ``fronting`` and ``backing`` are refnx-style rows ``[d, delta, beta, sigma]``.
/// ``backing`` may contain multiple substrate rows (oxide + bulk Si).
#[allow(clippy::too_many_arguments)]
pub fn bookended_uniaxial_reflectivity(
    q: &[f64],
    energy_ev: &[f64],
    n_xx: &[f64],
    n_ixx: &[f64],
    n_zz: &[f64],
    n_izz: &[f64],
    query_ev: f64,
    params: &BookendedParams,
    fronting: [f64; 4],
    backing: &[[f64; 4]],
    parallel: bool,
) -> crate::error::Result<UniaxialOutput> {
    let (film_layers, film_tensors) =
        build_bookended_film_stack(energy_ev, n_xx, n_ixx, n_zz, n_izz, query_ev, params);
    let (front_layer, front_tensor) = layer_row_to_parts(fronting);
    let mut layers = vec![front_layer];
    let mut tensors = vec![front_tensor];
    layers.extend(film_layers);
    tensors.extend(film_tensors);
    for row in backing {
        let (layer, tensor) = layer_row_to_parts(*row);
        layers.push(layer);
        tensors.push(tensor);
    }
    uniaxial_reflectivity(q, &layers, &tensors, query_ev, parallel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn adaptive_mesh_sums_to_total() {
        let mesh = adaptive_microslab_thicknesses(100.0, 100, 0.1);
        assert_eq!(mesh.len(), 100);
        assert_relative_eq!(mesh.iter().sum::<f64>(), 100.0, epsilon = 1e-9);
    }

    #[test]
    fn orientation_matches_python_reference_at_midpoint() {
        let p = BookendedParams {
            total_thick: 50.0,
            surface_roughness: 3.0,
            tau_si: 5.0,
            tau_vac: 5.0,
            alpha_bulk: 0.5,
            alpha_si: 0.8,
            alpha_vac: 0.2,
            density_bulk: 1.0,
            density_si: 1.0,
            density_vac: 1.0,
            num_slabs: 10,
            mesh_constant: 0.1,
        };
        let z = 25.0;
        let rust = orientation_profile_bookended(
            p.total_thick,
            z,
            p.tau_si,
            p.tau_vac,
            p.alpha_bulk,
            p.alpha_si,
            p.alpha_vac,
        );
        let vac = (p.alpha_vac - p.alpha_bulk) * (-z / p.tau_vac).exp();
        let si = (p.alpha_si - p.alpha_bulk) * (-(p.total_thick - z) / p.tau_si).exp();
        let py = p.alpha_bulk + vac + si;
        assert_relative_eq!(rust, py, epsilon = 1e-12);
    }

    #[test]
    fn fused_stack_layer_count() {
        let e = [250.0, 300.0];
        let xx = [1.5, 1.6];
        let ix = [0.01, 0.02];
        let zz = [1.55, 1.65];
        let iz = [0.01, 0.02];
        let params = BookendedParams {
            total_thick: 80.0,
            surface_roughness: 4.0,
            tau_si: 8.0,
            tau_vac: 6.0,
            alpha_bulk: 0.4,
            alpha_si: 0.6,
            alpha_vac: 0.1,
            density_bulk: 1.2,
            density_si: 1.0,
            density_vac: 0.8,
            num_slabs: 20,
            mesh_constant: 0.1,
        };
        let (layers, tensors) = build_bookended_film_stack(&e, &xx, &ix, &zz, &iz, 275.0, &params);
        assert_eq!(layers.len(), 20);
        assert_eq!(tensors.len(), 20);
        assert_relative_eq!(layers[0].sigma, 4.0, epsilon = 1e-12);
    }
}
