//! Fluent builder for [`Stack`].

use crate::error::KernelResult;
use crate::stack::{Layer, RoughnessSpec, Stack};

/// Fluent builder for [`Stack`].
///
/// Each setter consumes self and returns the builder, so a stack
/// is constructed by a single chained expression. Validation runs
/// in [`StackBuilder::build`] and returns
/// [`crate::KernelError::InvalidGeometry`] for structural failures.
#[derive(Debug, Clone)]
pub struct StackBuilder {
    incident: Option<Layer>,
    layers: Vec<Layer>,
    substrate: Option<Layer>,
    roughness: Vec<RoughnessSpec>,
}

impl StackBuilder {
    /// Empty builder.
    pub fn new() -> Self {
        Self {
            incident: None,
            layers: Vec::new(),
            substrate: None,
            roughness: Vec::new(),
        }
    }

    /// Set the half-infinite incident medium.
    pub fn incident(mut self, layer: Layer) -> Self {
        self.incident = Some(layer);
        self
    }

    /// Append an interior layer in the order it physically appears
    /// from the incident side toward the substrate.
    pub fn add_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Append an interior layer with an associated roughness spec
    /// for the interface above it.
    pub fn add_layer_with_roughness(
        mut self,
        layer: Layer,
        roughness_above: RoughnessSpec,
    ) -> Self {
        self.layers.push(layer);
        self.roughness.push(roughness_above);
        self
    }

    /// Set the half-infinite substrate medium and the roughness of
    /// the substrate-side interface.
    pub fn substrate(mut self, layer: Layer, roughness_above: RoughnessSpec) -> Self {
        self.substrate = Some(layer);
        self.roughness.push(roughness_above);
        self
    }

    /// Finalize the builder, running structural validation.
    pub fn build(self) -> KernelResult<Stack> {
        let incident = self
            .incident
            .ok_or_else(|| crate::KernelError::InvalidGeometry("missing incident medium".into()))?;
        let substrate = self
            .substrate
            .ok_or_else(|| crate::KernelError::InvalidGeometry("missing substrate medium".into()))?;
        let roughness = if self.roughness.is_empty() {
            vec![RoughnessSpec::sharp(); self.layers.len() + 1]
        } else if self.roughness.len() == self.layers.len() + 1 {
            self.roughness
        } else {
            return Err(crate::KernelError::InvalidGeometry(format!(
                "expected {} roughness entries for {} interior layers (got {})",
                self.layers.len() + 1,
                self.layers.len(),
                self.roughness.len()
            )));
        };
        let stack = Stack {
            incident,
            layers: self.layers,
            substrate,
            roughness,
        };
        stack.validate()?;
        Ok(stack)
    }
}

impl Default for StackBuilder {
    fn default() -> Self {
        Self::new()
    }
}
