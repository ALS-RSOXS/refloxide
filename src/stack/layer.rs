//! Single layer specification.

use crate::material::builder::Material;
use crate::material::rotation::EulerAnglesZxz;
use crate::types::parameterization::PrincipalTensor;

/// A single homogeneous layer.
///
/// The `thickness_nm` field is ignored for the two cladding media
/// because they are half-infinite. The `material` field carries the
/// principal-frame tensors and the Euler rotation, ready for
/// `crate::kernel::constitutive` to consume.
#[derive(Debug, Clone, Copy)]
pub struct Layer {
    pub thickness_nm: f64,
    pub material: Material,
}

impl Layer {
    /// Convenience constructor for a non-magnetic isotropic layer.
    pub fn isotropic(thickness_nm: f64, epsilon: PrincipalTensor) -> Self {
        Self {
            thickness_nm,
            material: Material {
                epsilon_principal: epsilon,
                mu_principal: None,
                euler_zxz_rad: EulerAnglesZxz::zero(),
            },
        }
    }

    /// Half-infinite cladding constructor. The thickness is set to
    /// `f64::INFINITY` for clarity in debug output.
    pub fn cladding(epsilon: PrincipalTensor) -> Self {
        Self {
            thickness_nm: f64::INFINITY,
            material: Material {
                epsilon_principal: epsilon,
                mu_principal: None,
                euler_zxz_rad: EulerAnglesZxz::zero(),
            },
        }
    }
}
