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
        todo!("Stack::validate not yet implemented")
    }
}
