//! Shared foundation for the refloxide transfer-matrix crates.
//!
//! Holds the scalar abstraction ([`Real`]), the slab and kernel-record
//! types ([`Layer`], [`LayerCoeffs`]), result containers, input validation,
//! fixed-size 4x4 complex helpers ([`c4x4`]), the exact 4x4 inverse
//! ([`math`]), and the typed [`RefloxideError`].
//!
//! It contains no solver loops, no threading, and no Python or GPU
//! dependencies. Solvers live in `refloxide-tmm`; optical-constant handling
//! lives in `refloxide-optics`.

pub mod c4x4;
pub mod error;
pub mod layer;
pub mod math;
pub mod output;
pub mod real;
pub mod validate;

pub use error::{RefloxideError, Result};
pub use layer::{layer_coeffs, wavenumber, Layer, LayerCoeffs, HC_EV_ANGSTROM};
pub use output::{PointsJvpOutput, PointsTangents, UniaxialBatchOutput, UniaxialOutput};
pub use real::Real;
pub use validate::{jvp_coeffs, validate_batch, validate_points};
