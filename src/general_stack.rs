//! General anisotropic stack reflectivity over a q-grid.
//!
//! Thin driver around [`crate::kernel::solve_point_general`] that accepts the
//! same Henke-style dispersion-tensor packing as [`crate::uniaxial`]
//! (`eps = conj(I - 2 tensor)`, `chi = eps - I`) and returns the polarized
//! power-reflectance block at every q, including cross-polarized channels.

use nalgebra::Matrix3;
use num_complex::Complex;
use rayon::prelude::*;

use crate::error::{RefloxideError, Result};
use crate::kernel::{solve_point_general, GeneralLayer};
use crate::uniaxial::{wavenumber, Layer};

type C = Complex<f64>;

/// Polarized power reflectance for a general anisotropic multilayer.
///
/// # Parameters
/// - `q`: scattering wavevectors in `1/Angstrom`.
/// - `layers`: per-slab rows; first and last are fronting/backing.
/// - `tensor`: per-slab 3x3 dispersion tensors (`delta + i beta` packing).
/// - `energy_ev`: photon energy in eV (strictly positive).
/// - `parallel`: distribute q-points across rayon when true.
///
/// # Returns
/// Power reflectance with shape `(n_q, 2, 2)` in the kernel packing
/// `[[R_pp, R_sp], [R_ps, R_ss]]`.
///
/// # Errors
/// Shape mismatches, fewer than two layers, non-positive energy, or a
/// singular dynamic matrix at some q-point.
pub fn general_reflectivity(
    q: &[f64],
    layers: &[Layer],
    tensor: &[Matrix3<C>],
    energy_ev: f64,
    parallel: bool,
) -> Result<Vec<[[f64; 2]; 2]>> {
    if layers.len() != tensor.len() {
        return Err(RefloxideError::InvalidShape(format!(
            "layers length {} must match tensor length {}",
            layers.len(),
            tensor.len()
        )));
    }
    if layers.len() < 2 {
        return Err(RefloxideError::InvalidShape(
            "need at least fronting and backing layers".into(),
        ));
    }
    if !energy_ev.is_finite() || energy_ev <= 0.0 {
        return Err(RefloxideError::InvalidShape(format!(
            "energy_ev must be finite and positive, got {energy_ev}"
        )));
    }

    let k0 = wavenumber(energy_ev);
    let stack: Vec<GeneralLayer<f64>> = layers
        .iter()
        .zip(tensor)
        .map(|(layer, t)| GeneralLayer {
            chi: tensor_to_chi(t),
            thickness: layer.thickness,
            sigma: layer.sigma,
        })
        .collect();

    let solve_one = |qi: f64| -> Result<[[f64; 2]; 2]> { solve_point_general(qi, k0, &stack) };

    if parallel {
        q.par_iter().map(|&qi| solve_one(qi)).collect()
    } else {
        q.iter().map(|&qi| solve_one(qi)).collect()
    }
}

fn tensor_to_chi(t: &Matrix3<C>) -> [[C; 3]; 3] {
    let mut chi = [[C::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            chi[i][j] = (t[(i, j)] * C::new(-2.0, 0.0)).conj();
        }
    }
    chi
}
