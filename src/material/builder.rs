//! Fluent builder for [`Material`] specifications.

use crate::error::KernelResult;
use crate::material::rotation::EulerAnglesZxz;
use crate::types::parameterization::PrincipalTensor;

/// Concrete material specification, ready to enter a layer.
///
/// `epsilon_principal` carries the principal-frame dielectric
/// tensor, `mu_principal` is `None` for non-magnetic media,
/// and `euler_zxz_rad` rotates the principal frame into the lab
/// frame per the Passler convention.
#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub epsilon_principal: PrincipalTensor,
    pub mu_principal: Option<PrincipalTensor>,
    pub euler_zxz_rad: EulerAnglesZxz,
}

/// Fluent builder for [`Material`].
///
/// Each setter consumes self and returns the builder, so a
/// material is constructed by a single chained expression.
#[derive(Debug, Clone, Copy)]
pub struct MaterialBuilder {
    epsilon_principal: Option<PrincipalTensor>,
    mu_principal: Option<PrincipalTensor>,
    euler_zxz_rad: EulerAnglesZxz,
}

impl MaterialBuilder {
    /// Empty builder. The `epsilon_principal` field must be set
    /// before `build` is called.
    pub fn new() -> Self {
        Self {
            epsilon_principal: None,
            mu_principal: None,
            euler_zxz_rad: EulerAnglesZxz::zero(),
        }
    }

    /// Set the principal-frame relative permittivity tensor.
    pub fn epsilon(mut self, epsilon_principal: PrincipalTensor) -> Self {
        self.epsilon_principal = Some(epsilon_principal);
        self
    }

    /// Set the principal-frame relative permeability tensor. Omit
    /// for non-magnetic media.
    pub fn mu(mut self, mu_principal: PrincipalTensor) -> Self {
        self.mu_principal = Some(mu_principal);
        self
    }

    /// Set the Euler angles rotating the principal frame into the
    /// lab frame.
    pub fn euler(mut self, euler_zxz_rad: EulerAnglesZxz) -> Self {
        self.euler_zxz_rad = euler_zxz_rad;
        self
    }

    /// Finalize the builder.
    pub fn build(self) -> KernelResult<Material> {
        let epsilon_principal = self
            .epsilon_principal
            .ok_or_else(|| crate::KernelError::InvalidGeometry("missing epsilon_principal".into()))?;
        Ok(Material {
            epsilon_principal,
            mu_principal: self.mu_principal,
            euler_zxz_rad: self.euler_zxz_rad,
        })
    }
}

impl Default for MaterialBuilder {
    fn default() -> Self {
        Self::new()
    }
}
