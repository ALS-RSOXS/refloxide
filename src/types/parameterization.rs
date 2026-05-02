//! Optical-index parameterization layer.
//!
//! The kernel canonical form is the relative dielectric tensor
//! `bar_epsilon`. Users may specify each layer in any of five
//! parameterizations through the [`OpticalIndex`] enum, and the
//! kernel collapses them onto `bar_epsilon` at the entry to the
//! pipeline.
//!
//! See `docs/api_examples/material_parameterizations.md` for the
//! user-facing specification of the supported parameterizations,
//! and `.cursor/plan/01_rust_kernel_implementation.md` Section
//! "Index-of-refraction parameterization layer" for the design
//! rationale.

use crate::error::{KernelError, KernelResult};
use crate::types::scalar::{C64, CLASSICAL_ELECTRON_RADIUS_M};

/// User-facing optical index parameterization for a single
/// principal-axis component. The kernel converts to the canonical
/// relative-permittivity form via [`OpticalIndex::to_epsilon`].
#[derive(Debug, Clone, Copy)]
pub enum OpticalIndex {
    /// Refractive index `n + i k`, with `tilde_n = n + i k` and
    /// `epsilon = tilde_n^2`.
    NK { n: C64, k: C64 },

    /// X-ray and EUV form `tilde_n = 1 - delta + i beta`. The two
    /// components carry the same sign convention as the Henke
    /// scattering tables.
    DeltaBeta { delta: C64, beta: C64 },

    /// Scattering-length density. The conversion to epsilon uses
    /// the Born-approximation formula
    /// `epsilon = 1 - (lambda^2 / pi) rho_SLD`.
    Sld { rho: C64, wavelength_m: f64 },

    /// Atomic scattering factors `f1 + i f2` with the species
    /// number density and the probe wavelength. Converts to
    /// `(delta, beta)` first, then to `epsilon`.
    ScatteringFactor {
        f1: C64,
        f2: C64,
        number_density_per_m3: f64,
        wavelength_m: f64,
    },

    /// Direct relative permittivity. Used internally and exposed
    /// for users who already have epsilon in hand.
    Epsilon(C64),
}

impl OpticalIndex {
    /// Convert any variant to the canonical relative-permittivity
    /// form. Returns [`crate::KernelError::UnsupportedConversion`]
    /// when a required field is missing.
    pub fn to_epsilon(&self) -> KernelResult<C64> {
        let i = C64::i();
        match self {
            OpticalIndex::NK { n, k } => {
                if !n.re.is_finite() || !n.im.is_finite() || !k.re.is_finite() || !k.im.is_finite()
                {
                    return Err(KernelError::UnsupportedConversion(
                        "NK parameterization requires finite n and k".into(),
                    ));
                }
                let tilde_n = *n + i * *k;
                Ok(tilde_n * tilde_n)
            }
            OpticalIndex::DeltaBeta { delta, beta } => {
                if !delta.re.is_finite()
                    || !delta.im.is_finite()
                    || !beta.re.is_finite()
                    || !beta.im.is_finite()
                {
                    return Err(KernelError::UnsupportedConversion(
                        "DeltaBeta parameterization requires finite delta and beta".into(),
                    ));
                }
                let tilde_n = C64::new(1.0, 0.0) - *delta + i * *beta;
                Ok(tilde_n * tilde_n)
            }
            OpticalIndex::Sld { rho, wavelength_m } => {
                if !wavelength_m.is_finite() || *wavelength_m <= 0.0 {
                    return Err(KernelError::UnsupportedConversion(
                        "SLD parameterization requires a finite, positive wavelength_m".into(),
                    ));
                }
                if !rho.re.is_finite() || !rho.im.is_finite() {
                    return Err(KernelError::UnsupportedConversion(
                        "SLD parameterization requires finite rho".into(),
                    ));
                }
                let coeff = wavelength_m * wavelength_m / std::f64::consts::PI;
                Ok(C64::new(1.0, 0.0) - *rho * coeff)
            }
            OpticalIndex::ScatteringFactor {
                f1,
                f2,
                number_density_per_m3,
                wavelength_m,
            } => {
                if !wavelength_m.is_finite() || *wavelength_m <= 0.0 {
                    return Err(KernelError::UnsupportedConversion(
                        "scattering-factor parameterization requires a finite, positive wavelength_m"
                            .into(),
                    ));
                }
                if !number_density_per_m3.is_finite() {
                    return Err(KernelError::UnsupportedConversion(
                        "scattering-factor parameterization requires finite number_density_per_m3"
                            .into(),
                    ));
                }
                if !f1.re.is_finite()
                    || !f1.im.is_finite()
                    || !f2.re.is_finite()
                    || !f2.im.is_finite()
                {
                    return Err(KernelError::UnsupportedConversion(
                        "scattering-factor parameterization requires finite f1 and f2".into(),
                    ));
                }
                let pref = CLASSICAL_ELECTRON_RADIUS_M * wavelength_m * wavelength_m
                    / (2.0 * std::f64::consts::PI)
                    * number_density_per_m3;
                let delta = pref * *f1;
                let beta = pref * *f2;
                let tilde_n = C64::new(1.0, 0.0) - delta + i * beta;
                Ok(tilde_n * tilde_n)
            }
            OpticalIndex::Epsilon(eps) => {
                if !eps.re.is_finite() || !eps.im.is_finite() {
                    return Err(KernelError::UnsupportedConversion(
                        "Epsilon parameterization requires a finite value".into(),
                    ));
                }
                Ok(*eps)
            }
        }
    }

    /// Return the equivalent complex refractive index
    /// `tilde_n = sqrt(epsilon)` with the principal-branch choice
    /// that places `Im(tilde_n) >= 0` for passive media.
    pub fn to_complex_index(&self) -> KernelResult<C64> {
        let eps = self.to_epsilon()?;
        let mut n = eps.sqrt();
        if n.im < 0.0 {
            n = -n;
        }
        if n.re < 0.0 {
            n = -n;
        }
        if n.im < 0.0 {
            n = -n;
        }
        Ok(n)
    }

    /// Round-trip constructor for the [`OpticalIndex::Epsilon`]
    /// variant.
    #[inline]
    pub fn from_epsilon(eps: C64) -> Self {
        OpticalIndex::Epsilon(eps)
    }
}

/// Three-axis principal-frame dielectric (or magnetic) tensor with
/// independent parameterizations along each axis. The flexible
/// per-axis parameterization is needed when the dielectric tensor
/// is sourced from disparate measurements (for example x-ray
/// scattering factors on one axis and tabulated `(n, k)` on the
/// orthogonal axes).
#[derive(Debug, Clone, Copy)]
pub struct PrincipalTensor {
    pub xx: OpticalIndex,
    pub yy: OpticalIndex,
    pub zz: OpticalIndex,
}

impl PrincipalTensor {
    /// Isotropic principal tensor with the same value on all three
    /// principal axes.
    pub fn isotropic(idx: OpticalIndex) -> Self {
        Self {
            xx: idx,
            yy: idx,
            zz: idx,
        }
    }

    /// Uniaxial principal tensor with `ordinary` along the in-plane
    /// axes and `extraordinary` along the optic axis.
    pub fn uniaxial(ordinary: OpticalIndex, extraordinary: OpticalIndex) -> Self {
        Self {
            xx: ordinary,
            yy: ordinary,
            zz: extraordinary,
        }
    }

    /// Resolve all three principal-axis parameterizations to the
    /// canonical relative-permittivity diagonal.
    pub fn to_epsilon_principal_diag(&self) -> KernelResult<crate::types::tensor::PrincipalDiag> {
        Ok(crate::types::tensor::PrincipalDiag {
            xx: self.xx.to_epsilon()?,
            yy: self.yy.to_epsilon()?,
            zz: self.zz.to_epsilon()?,
        })
    }
}
