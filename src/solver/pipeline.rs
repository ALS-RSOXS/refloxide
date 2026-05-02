//! Phantom-state Pipeline composing the six pipeline stages.
//!
//! See `.cursor/plan/01_rust_kernel_implementation.md`,
//! "Method chaining and the staged pipeline".
//!
//! The phantom-state parameter `S` enforces stage ordering at the
//! type level. A caller cannot ask for amplitudes before
//! propagation because the method does not exist on
//! `Pipeline<Reduced>`. The chain consumes self at each step.

use crate::error::KernelResult;
use crate::kernel::constitutive::MMatrix;
use crate::kernel::delta::DeltaMatrix;
use crate::kernel::modes::LayerModes;
use crate::stack::Stack;
use crate::types::scalar::C64;
use nalgebra::Matrix4;
use std::marker::PhantomData;

pub use crate::kernel::coefficients::Amplitudes;
pub use crate::kernel::field::{FieldProfile, Polarization};

/// The pipeline marker type, parameterized by its stage.
#[derive(Debug)]
pub struct Pipeline<S> {
    stack: Stack,
    omega_rad_per_s: f64,
    theta_rad: f64,
    state: S,
    _marker: PhantomData<S>,
}

/// Stage 0, freshly constructed pipeline.
#[derive(Debug, Default)]
pub struct Init;

/// Stage 1a, constitutive matrix `M` per layer.
#[derive(Debug)]
pub struct Constituted {
    pub m_matrices: Vec<MMatrix>,
    pub xi: C64,
}

/// Stage 1b, Berreman `Delta` per layer.
#[derive(Debug)]
pub struct Reduced {
    pub delta_matrices: Vec<DeltaMatrix>,
    pub xi: C64,
}

/// Stage 2, sorted eigenmodes per layer.
#[derive(Debug)]
pub struct Eigenmoded {
    pub modes: Vec<LayerModes>,
    pub xi: C64,
}

/// Stage 3, interface matrices per layer.
#[derive(Debug)]
pub struct Interfaced {
    pub a_matrices: Vec<Matrix4<C64>>,
    pub xi: C64,
}

/// Stage 4, stack-level `Gamma_N` matrix.
#[derive(Debug)]
pub struct Propagated {
    pub gamma: Matrix4<C64>,
}

/// Stage 5/6, eight amplitudes ready, fields available on demand.
#[derive(Debug)]
pub struct Solved {
    pub amplitudes: Amplitudes,
}

impl Pipeline<Init> {
    /// Construct a fresh pipeline from a stack and a probe geometry.
    pub fn new(stack: Stack, omega_rad_per_s: f64, theta_rad: f64) -> Self {
        Self {
            stack,
            omega_rad_per_s,
            theta_rad,
            state: Init,
            _marker: PhantomData,
        }
    }

    /// Stage 1a, build the constitutive matrix `M` per layer.
    pub fn build_constitutive(self) -> KernelResult<Pipeline<Constituted>> {
        let _ = self;
        todo!("Pipeline<Init>::build_constitutive not yet implemented")
    }
}

impl Pipeline<Constituted> {
    /// Stage 1b, reduce to the Berreman 4x4 `Delta` per layer.
    pub fn reduce_to_delta(self) -> KernelResult<Pipeline<Reduced>> {
        let _ = self;
        todo!("Pipeline<Constituted>::reduce_to_delta not yet implemented")
    }
}

impl Pipeline<Reduced> {
    /// Stage 2, solve and sort eigenmodes per layer.
    pub fn solve_eigenmodes(self) -> KernelResult<Pipeline<Eigenmoded>> {
        let _ = self;
        todo!("Pipeline<Reduced>::solve_eigenmodes not yet implemented")
    }
}

impl Pipeline<Eigenmoded> {
    /// Stage 3, build `A_i` per layer.
    pub fn build_interfaces(self) -> KernelResult<Pipeline<Interfaced>> {
        let _ = self;
        todo!("Pipeline<Eigenmoded>::build_interfaces not yet implemented")
    }
}

impl Pipeline<Interfaced> {
    /// Stage 4, assemble the stack-level `Gamma_N`.
    pub fn propagate(self) -> KernelResult<Pipeline<Propagated>> {
        let _ = self;
        todo!("Pipeline<Interfaced>::propagate not yet implemented")
    }
}

impl Pipeline<Propagated> {
    /// Stage 5, solve for the eight amplitudes.
    pub fn solve_amplitudes(self) -> KernelResult<Pipeline<Solved>> {
        let _ = self;
        todo!("Pipeline<Propagated>::solve_amplitudes not yet implemented")
    }
}

impl Pipeline<Solved> {
    /// Read the eight amplitudes.
    pub fn amplitudes(&self) -> &Amplitudes {
        &self.state.amplitudes
    }

    /// Stage 6, reconstruct the electric field at one
    /// `(layer, z)` evaluation point.
    pub fn reconstruct_field_at(
        &self,
        layer_index: usize,
        z_nm: f64,
        polarization: Polarization,
    ) -> KernelResult<FieldProfile> {
        let _ = (layer_index, z_nm, polarization);
        todo!("Pipeline<Solved>::reconstruct_field_at not yet implemented")
    }
}

/// One-shot helper that runs the full chain end to end. The pyo3
/// surface in [`crate::ffi`] exposes this entry point and keeps
/// the staged pipeline as a Rust-side debugging aid.
pub fn compute_amplitudes(
    stack: &Stack,
    omega_rad_per_s: f64,
    theta_rad: f64,
) -> KernelResult<Amplitudes> {
    let _ = (stack, omega_rad_per_s, theta_rad);
    todo!("compute_amplitudes not yet implemented")
}

/// One-shot helper for the field reconstruction.
pub fn compute_field(
    stack: &Stack,
    omega_rad_per_s: f64,
    theta_rad: f64,
    layer_index: usize,
    z_nm: f64,
    polarization: Polarization,
) -> KernelResult<FieldProfile> {
    let _ = (
        stack,
        omega_rad_per_s,
        theta_rad,
        layer_index,
        z_nm,
        polarization,
    );
    todo!("compute_field not yet implemented")
}
