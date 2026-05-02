//! Kernel error model.
//!
//! [`KernelError`] is the single failure type returned across the
//! kernel API. Each variant maps onto a documented failure mode in
//! `docs/theory/algorithm_audit.md`. The FFI layer in
//! [`crate::ffi`] converts each variant to the appropriate pyo3
//! exception type using the convention that input-validation
//! failures map to `ValueError` and numerical-runtime failures map
//! to `RuntimeError`.

use crate::types::scalar::C64;

/// All errors returned by the kernel.
///
/// The enum is intentionally finite and exhaustive. The kernel
/// does not return `Box<dyn Error>` or `anyhow::Error`. This forces
/// every call site to handle each failure mode explicitly and lets
/// the FFI boundary deliver actionable diagnostics to the Python
/// side.
#[derive(thiserror::Error, Debug)]
pub enum KernelError {
    /// PP2017 Eq. (10) scalar `b = M_33 M_66 - M_36 M_63` vanished,
    /// so the longitudinal elimination of `E_z` and `H_z` is
    /// singular at the requested geometry.
    #[error("longitudinal elimination singular: b = M33 M66 - M36 M63 = {0}")]
    SingularConstitutive(C64),

    /// The dense eigensolver in [`crate::kernel::modes`] returned a
    /// non-convergent result for layer index `layer_index`. The
    /// `message` field carries the underlying error string from the
    /// linear-algebra backend.
    #[error("eigensolve failed at layer index {layer_index}: {message}")]
    EigenSolveFailure { layer_index: usize, message: String },

    /// PP2017 Eq. (12) cannot decide forward vs. backward
    /// classification because two eigenvalues have equal `Im(q)`
    /// magnitudes within numerical noise.
    #[error("forward/backward partition ambiguous at layer {layer_index}: equal Im(q) magnitudes")]
    AmbiguousPartition { layer_index: usize },

    /// PP2017 Eq. (14) Li-Sullivan-Parsons sort cannot decide the
    /// within-pair ordering because the two `C(q)` functionals are
    /// equal within `threshold`. Falling back to the Poynting
    /// criterion is the recommended action.
    #[error("Li projection ambiguous at layer {layer_index}: |C(q1) - C(q2)| < {threshold}")]
    AmbiguousLiSort { layer_index: usize, threshold: f64 },

    /// PP2019 Eq. (33) coefficient denominator
    /// `Gamma_11 Gamma_33 - Gamma_13 Gamma_31` vanished in the
    /// substrate-side amplitude assembly.
    #[error("xi denominator vanished in coefficient assembly: |Gamma11 Gamma33 - Gamma13 Gamma31| = {0}")]
    SingularCoefficientDenominator(f64),

    /// User-facing geometry or input validation failure with a
    /// descriptive message.
    #[error("invalid layer geometry: {0}")]
    InvalidGeometry(String),

    /// User requested a parameterization conversion that the
    /// kernel cannot perform, typically because a missing field
    /// (wavelength, number density) was not supplied.
    #[error("unsupported optical parameterization conversion: {0}")]
    UnsupportedConversion(String),

    /// User asked for transmission coefficients into a vacuum
    /// substrate, which is not physically meaningful.
    #[error("substrate must be non-vacuum to compute transmission into a measurable medium")]
    VacuumSubstrate,

    /// Euler angle outside the documented range.
    #[error("Euler angle out of range: {0}")]
    InvalidEulerAngle(f64),

    /// Roughness model parameters violate the documented validity
    /// region for that model. The `message` field carries the
    /// specific violated constraint.
    #[error("roughness model out of validity: {0}")]
    RoughnessOutOfValidity(String),

    /// Graded-interface convergence check failed at the user's
    /// requested tolerance.
    #[error("graded-interface convergence check failed: |delta| = {discrepancy} > tolerance {tolerance}")]
    GradedConvergenceFailure { discrepancy: f64, tolerance: f64 },
}

/// Convenience alias for `Result<T, KernelError>`.
pub type KernelResult<T> = Result<T, KernelError>;
