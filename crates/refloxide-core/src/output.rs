//! Result containers returned by the CPU uniaxial drivers.
//!
//! Plain data only: no solver logic and no backend dependencies.

use nalgebra::Matrix3;
use num_complex::Complex;

use crate::layer::Layer;

/// Complex alias used throughout the kernel.
type C = Complex<f64>;

/// Polarized reflectance and amplitude transmission for a single energy.
///
/// Index layout mirrors the Python reference: `refl[i][k][l]` and
/// `tran[i][k][l]` with `(0, 0) = ss`, `(1, 1) = pp`, `(0, 1) = sp`,
/// `(1, 0) = ps`.
#[derive(Debug, Clone)]
pub struct UniaxialOutput {
    /// Power reflectance with shape `(numpnts, 2, 2)`.
    pub refl: Vec<[[f64; 2]; 2]>,
    /// Complex amplitude transmission with shape `(numpnts, 2, 2)`.
    pub tran: Vec<[[C; 2]; 2]>,
}

/// Batched polarized reflectance and transmission over energies and q-points.
#[derive(Debug, Clone)]
pub struct UniaxialBatchOutput {
    /// Power reflectance with shape `(n_energies, n_q, 2, 2)`.
    pub refl: Vec<Vec<[[f64; 2]; 2]>>,
    /// Complex amplitude transmission with the same layout.
    pub tran: Vec<Vec<[[C; 2]; 2]>>,
}

/// Reflectance and Jacobian-vector products for `(q, stack)` points.
#[derive(Debug, Clone)]
pub struct PointsJvpOutput {
    /// `[R_pp, R_ss]` per point, length `n_points`.
    pub refl: Vec<[f64; 2]>,
    /// Row-major `(n_dirs, n_points)` derivatives `[dR_pp, dR_ss]`.
    pub jac: Vec<[f64; 2]>,
}

/// Direction-major tangent inputs for `uniaxial_reflectivity_points_jvp`.
#[derive(Debug, Clone)]
pub struct PointsTangents {
    /// `(n_dirs, n_points)` tangent of `q`.
    pub dq: Vec<Vec<f64>>,
    /// `(n_dirs, n_stacks, n_layers)` tangent of the slab rows.
    pub dlayers: Vec<Vec<Vec<Layer>>>,
    /// `(n_dirs, n_stacks, n_layers)` tangent of the dispersion tensors.
    pub dtensor: Vec<Vec<Vec<Matrix3<C>>>>,
}
