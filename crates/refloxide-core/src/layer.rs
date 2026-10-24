//! Slab descriptions and the packing of dispersion tensors into kernel records.
//!
//! Owns the plain-old-data types shared by every solver backend ([`Layer`],
//! [`LayerCoeffs`]) and the wavelength conversion. It does not solve
//! transfer chains or validate whole stacks; see [`crate::validate`] and the
//! `refloxide-tmm` crate.

use nalgebra::Matrix3;
use num_complex::Complex;

use crate::real::Real;

/// Complex alias used throughout the kernel.
type C = Complex<f64>;

/// Photon energy to wavelength conversion constant in `eV * Angstrom`.
pub const HC_EV_ANGSTROM: f64 = 12398.4193;

/// One slab in a stratified medium.
///
/// `thickness` and `sigma` are in Angstroms. `sld_re` and `sld_im` carry the
/// Henke-style `delta` and `beta` for the isotropic legacy path; the per-axis
/// dispersion comes in through the `uniaxial_reflectivity` `tensor`
/// argument.
#[derive(Debug, Clone, Copy)]
pub struct Layer {
    /// Slab thickness in Angstroms. Ignored for the fronting and backing rows.
    pub thickness: f64,
    /// Real part of the isotropic SLD in `1e-6 * Angstrom^-2`.
    pub sld_re: f64,
    /// Imaginary part of the isotropic SLD in `1e-6 * Angstrom^-2`.
    pub sld_im: f64,
    /// Nevot-Croce roughness sigma in Angstroms.
    pub sigma: f64,
}

impl Layer {
    /// Builds a layer from its four scalar parameters.
    pub fn new(thickness: f64, sld_re: f64, sld_im: f64, sigma: f64) -> Self {
        Self {
            thickness,
            sld_re,
            sld_im,
            sigma,
        }
    }
}

impl From<[f64; 4]> for Layer {
    fn from(row: [f64; 4]) -> Self {
        Self::new(row[0], row[1], row[2], row[3])
    }
}

/// Per-slab coefficients consumed by the uniaxial kernel.
///
/// This is the device-uploadable layer record: only the ordinary and
/// extraordinary Berreman components of a uniaxial-z slab enter the
/// closed-form eigenstructure, so the full 3x3 tensor is not carried.
/// Components are stored as susceptibilities `chi = eps - 1` rather than
/// `eps` because soft X-ray `|chi|` is `O(1e-3)`: packing `1 + chi` into a
/// single-precision word would discard roughly three significant digits of
/// the optical constants before any arithmetic happens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerCoeffs<T: Real> {
    /// Ordinary susceptibility `eps_xx - 1 = eps_yy - 1` (dimensionless).
    pub chi_o: Complex<T>,
    /// Extraordinary susceptibility `eps_zz - 1` (dimensionless).
    pub chi_e: Complex<T>,
    /// Slab thickness in Angstroms. Ignored for the fronting and backing rows.
    pub thickness: T,
    /// Nevot-Croce roughness of the interface above this slab, in Angstroms.
    pub sigma: T,
}

/// Vacuum wavenumber `2 pi / lambda` in `1/Angstrom` for a photon energy in eV.
pub fn wavenumber(energy_ev: f64) -> f64 {
    2.0 * std::f64::consts::PI / (HC_EV_ANGSTROM / energy_ev)
}

/// Packs per-slab rows and dispersion tensors into kernel [`LayerCoeffs`].
///
/// Reads the `xx` and `zz` tensor entries as the ordinary and extraordinary
/// axes of a uniaxial-z slab; thickness and roughness come from `layers`.
pub fn layer_coeffs(layers: &[Layer], tensor: &[Matrix3<C>]) -> Vec<LayerCoeffs<f64>> {
    layers
        .iter()
        .zip(tensor)
        .map(|(layer, t)| LayerCoeffs {
            chi_o: berreman_susceptibility(t[(0, 0)]),
            chi_e: berreman_susceptibility(t[(2, 2)]),
            thickness: layer.thickness,
            sigma: layer.sigma,
        })
        .collect()
}

/// Diagonal susceptibility `eps - 1 = conj(-2 t)` of `eps = conj(I - 2 t)`.
fn berreman_susceptibility(t: C) -> C {
    (t * C::new(-2.0, 0.0)).conj()
}
