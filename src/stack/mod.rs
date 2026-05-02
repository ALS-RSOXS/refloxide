//! Layer and stack data structures.

pub mod builder;
pub mod layer;
pub mod roughness_spec;

pub use builder::StackBuilder;
pub use layer::Layer;
pub use roughness_spec::{ProfileShape, RoughnessChoice, RoughnessSpec};

use crate::error::KernelResult;

/// A stratified medium consisting of two half-infinite cladding
/// media and an ordered sequence of finite interior layers.
///
/// Layer ordering runs from the incident side to the substrate
/// side. The interior `layers` vector is empty for a bare
/// substrate. Roughness is stored as one [`RoughnessSpec`] per
/// internal interface, with `roughness.len() == layers.len() + 1`.
#[derive(Debug, Clone)]
pub struct Stack {
    pub incident: Layer,
    pub layers: Vec<Layer>,
    pub substrate: Layer,
    pub roughness: Vec<RoughnessSpec>,
}

impl Stack {
    /// Total number of internal interfaces, including the surface
    /// (incident-to-first-layer) and the substrate-side boundary.
    pub fn interface_count(&self) -> usize {
        self.layers.len() + 1
    }

    /// Validate the stack against the kernel's structural
    /// invariants. Used by [`StackBuilder::build`] and re-runnable
    /// after manual mutation.
    pub fn validate(&self) -> KernelResult<()> {
        if self.roughness.len() != self.interface_count() {
            return Err(crate::KernelError::InvalidGeometry(format!(
                "roughness specs must match interface_count {} (got {})",
                self.interface_count(),
                self.roughness.len()
            )));
        }
        let validate_layer = |label: &str, layer: &crate::stack::Layer| -> KernelResult<()> {
            if !(layer.thickness_nm.is_finite() || layer.thickness_nm.is_infinite()) {
                return Err(crate::KernelError::InvalidGeometry(format!(
                    "{label} thickness_nm must be finite or infinite"
                )));
            }
            let diag = layer
                .material
                .epsilon_principal
                .to_epsilon_principal_diag()?;
            diag.validate_passive()?;
            if let Some(mu) = layer.material.mu_principal {
                let mud = mu.to_epsilon_principal_diag()?;
                mud.validate_passive()?;
            }
            Ok(())
        };
        validate_layer("incident", &self.incident)?;
        for (k, layer) in self.layers.iter().enumerate() {
            if !(layer.thickness_nm.is_finite() && layer.thickness_nm > 0.0) {
                return Err(crate::KernelError::InvalidGeometry(format!(
                    "interior layer {k} thickness_nm must be finite and positive"
                )));
            }
            validate_layer(&format!("layer[{k}]"), layer)?;
        }
        validate_layer("substrate", &self.substrate)?;
        Ok(())
    }
}
