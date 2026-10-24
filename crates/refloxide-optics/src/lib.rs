//! Optical constants and laboratory-frame tensors for uniaxial stacks.
//!
//! [`optics`] interpolates tabulated optical constants and assembles
//! diagonal uniaxial tensors; [`sld`] wraps single-point density scaling,
//! tensor construction, and slab-row packing. Tensor entries carry
//! `delta + i beta` per principal axis, the convention the transfer-matrix
//! kernels expect.
//!
//! It does not solve reflectivity, validate stacks, or depend on Python or
//! the GPU.

pub mod optics;
pub mod sld;

pub use sld::{
    isotropic_lab_tensor, molecular_index, molecular_index_at_ooc, tensor_to_slab_row,
    uniaxial_lab_tensor,
};
