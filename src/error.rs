//! Error type and result alias for the refloxide kernel.
//!
//! All fallible operations in the public Rust API return [`Result<T>`].
//! [`RefloxideError`] carries enough context for callers to recover or to
//! surface a typed Python exception when the PyO3 boundary is crossed.

use thiserror::Error;

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
    #[error("dynamic matrix is singular at layer {layer}, q-index {q_index}")]
    SingularDynamicMatrix {
        /// Index of the offending slab.
        layer: usize,
        /// Index in the q-vector at which the inversion failed.
        q_index: usize,
    },

    /// An input array did not have the expected shape at the FFI boundary.
    #[error("invalid input shape: {0}")]
    InvalidShape(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, RefloxideError>;

#[cfg(feature = "python")]
impl From<RefloxideError> for pyo3::PyErr {
    fn from(err: RefloxideError) -> pyo3::PyErr {
        match err {
            RefloxideError::InvalidShape(_)
            | RefloxideError::LayerCountMismatch { .. }
            | RefloxideError::InsufficientLayers(_)
            | RefloxideError::InvalidEnergy(_) => {
                pyo3::exceptions::PyValueError::new_err(err.to_string())
            }
            RefloxideError::SingularDynamicMatrix { .. } => {
                pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
            }
        }
    }
}
