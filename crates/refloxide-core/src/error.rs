//! Error type and result alias for the refloxide kernel.
//!
//! All fallible operations in the public Rust API return [`Result<T>`].
//! [`RefloxideError`] carries enough context for callers to recover or to
//! surface a typed Python exception in the bindings crate.

use thiserror::Error;

fn format_energy_index(energy_index: &Option<usize>) -> String {
    match energy_index {
        Some(index) => format!(", energy-index {index}"),
        None => String::new(),
    }
}

/// Errors returned by the refloxide kernel.
#[derive(Debug, Error, Clone)]
pub enum RefloxideError {
    /// Layer rows and tensor rows do not agree on the number of slabs.
    #[error("layer count mismatch: layers has {layers} rows, tensor has {tensor} rows")]
    LayerCountMismatch {
        /// Number of rows seen in the layers buffer.
        layers: usize,
        /// Number of slabs seen in the tensor buffer.
        tensor: usize,
    },

    /// The stack does not carry both a fronting and a backing row.
    #[error("stack requires at least 2 layers (fronting + backing), got {0}")]
    InsufficientLayers(usize),

    /// Photon energy is not strictly positive.
    #[error("invalid photon energy {0} eV (must be positive)")]
    InvalidEnergy(f64),

    /// Dynamic matrix could not be inverted at a particular layer and q-point.
    #[error("dynamic matrix is singular at layer {layer}, q-index {q_index}{}", format_energy_index(.energy_index))]
    SingularDynamicMatrix {
        /// Index of the offending slab.
        layer: usize,
        /// Index in the q-vector at which the inversion failed.
        q_index: usize,
        /// Batch energy index when applicable.
        energy_index: Option<usize>,
    },

    /// An input array did not have the expected shape at the FFI boundary.
    #[error("invalid input shape: {0}")]
    InvalidShape(String),

    /// A peak parameter is outside its valid domain: non-finite; Gaussian `width <= 0`;
    /// Lorentzian `gamma <= 0`; Voigt `sigma <= 0` or `gamma < 0`; pseudo-Voigt
    /// `fwhm <= 0` or `eta` outside `[0, 1]`.
    #[error("invalid peak parameter: {0}")]
    InvalidParameter(String),

    /// The GPU backend could not acquire a device or failed during dispatch.
    #[error("gpu backend error: {0}")]
    Gpu(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, RefloxideError>;
