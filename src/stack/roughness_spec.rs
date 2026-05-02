//! Per-interface roughness specification.
//!
//! See `docs/theory/roughness_framework.md` for the architectural
//! placement of these types in the pipeline.

use std::sync::Arc;

/// Roughness specification attached to a single interface.
#[derive(Debug, Clone)]
pub struct RoughnessSpec {
    pub sigma_nm: f64,
    pub correlation_length_nm: Option<f64>,
    pub model: RoughnessChoice,
    pub profile: ProfileShape,
}

impl RoughnessSpec {
    /// Sharp interface, no roughness applied.
    pub fn sharp() -> Self {
        Self {
            sigma_nm: 0.0,
            correlation_length_nm: None,
            model: RoughnessChoice::Sharp,
            profile: ProfileShape::Gaussian,
        }
    }

    /// Nevot-Croce factor with the given rms roughness in
    /// nanometers.
    pub fn nevot_croce(sigma_nm: f64) -> Self {
        Self {
            sigma_nm,
            correlation_length_nm: None,
            model: RoughnessChoice::NevotCroce,
            profile: ProfileShape::Gaussian,
        }
    }

    /// Debye-Waller factor with the given rms roughness and
    /// correlation length in nanometers.
    pub fn debye_waller(sigma_nm: f64, correlation_length_nm: f64) -> Self {
        Self {
            sigma_nm,
            correlation_length_nm: Some(correlation_length_nm),
            model: RoughnessChoice::DebyeWaller,
            profile: ProfileShape::Gaussian,
        }
    }

    /// Graded-interface discretization with the given sublayer
    /// count and profile shape.
    pub fn graded(sigma_nm: f64, profile: ProfileShape, sublayer_count: usize) -> Self {
        Self {
            sigma_nm,
            correlation_length_nm: None,
            model: RoughnessChoice::Graded { sublayer_count },
            profile,
        }
    }

    /// Auto-dispatch through the heuristic in
    /// `docs/theory/roughness_selection_guide.md`.
    pub fn auto(sigma_nm: f64, correlation_length_nm: f64) -> Self {
        Self {
            sigma_nm,
            correlation_length_nm: Some(correlation_length_nm),
            model: RoughnessChoice::Auto,
            profile: ProfileShape::Gaussian,
        }
    }
}

/// User-selected or auto-dispatched roughness model.
#[derive(Debug, Clone, Copy)]
pub enum RoughnessChoice {
    Sharp,
    NevotCroce,
    DebyeWaller,
    Graded { sublayer_count: usize },
    Auto,
}

/// Profile family for the graded-interface implementation. Custom
/// profiles are wrapped in [`Arc`] for cheap clone of the
/// [`RoughnessSpec`] container.
#[derive(Clone)]
pub enum ProfileShape {
    Gaussian,
    Linear,
    Sine,
    TanhSech2,
    Custom(Arc<dyn Fn(f64) -> f64 + Send + Sync>),
}

impl std::fmt::Debug for ProfileShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileShape::Gaussian => f.write_str("Gaussian"),
            ProfileShape::Linear => f.write_str("Linear"),
            ProfileShape::Sine => f.write_str("Sine"),
            ProfileShape::TanhSech2 => f.write_str("TanhSech2"),
            ProfileShape::Custom(_) => f.write_str("Custom(<fn>)"),
        }
    }
}
