//! Kernel algorithm modules.
//!
//! Each submodule corresponds to one stage of the six-stage
//! Passler-Paarmann pipeline documented in `docs/theory/`. The
//! per-equation traceability matrix in
//! `docs/theory/algorithm_audit.md` lists which equations land in
//! which submodule.

pub mod coefficients;
pub mod constitutive;
pub mod delta;
pub mod field;
pub mod geometry;
pub mod interface;
pub mod modes;
pub mod propagate;
pub mod roughness;

/// Berreman state-vector ordering, `Psi = (E_x, H_y, E_y, -H_x)`,
/// per PP2017 Eq. (7).
pub const PSI_BASIS_ORDER: [&str; 4] = ["E_x", "H_y", "E_y", "-H_x"];

/// Interface-matrix row ordering used by [`interface::build_a`],
/// `(E_x, E_y, H_y, -H_x)`, per PP2017 Eq. (22). Distinct from
/// [`PSI_BASIS_ORDER`] by the swap of rows 2 and 3, and the kernel
/// converts between the two via an explicit permutation rather than
/// by relying on row-index arithmetic.
pub const INTERFACE_BASIS_ORDER: [&str; 4] = ["E_x", "E_y", "H_y", "-H_x"];

/// Threshold for the Xu degenerate-branch dispatch in
/// [`interface`]. Branches are considered degenerate when
/// `|q_i1 - q_i2| < XU_DEGENERATE_THRESHOLD`.
pub const XU_DEGENERATE_THRESHOLD: f64 = 1e-12;
