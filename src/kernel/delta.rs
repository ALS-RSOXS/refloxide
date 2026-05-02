//! Stage 1, the Berreman 4x4 matrix `Delta`.
//!
//! Implements PP2017 Eq. (8) for the sixteen `Delta_ij` entries,
//! Eq. (9) for the auxiliary coefficients `a_3n` and `a_6n`, and
//! Eq. (10) for the scalar `b`. The full tabulation is reproduced
//! in `docs/theory/foundations.md`.

use crate::error::KernelResult;
use crate::kernel::constitutive::MMatrix;
use crate::types::scalar::C64;
use nalgebra::Matrix4;

/// Type alias for the Berreman 4x4 matrix.
pub type DeltaMatrix = Matrix4<C64>;

/// Compute the auxiliary scalar `b = M_33 M_66 - M_36 M_63` of
/// PP2017 Eq. (10). Returns
/// [`crate::KernelError::SingularConstitutive`] when `b` vanishes
/// within numerical tolerance.
pub fn compute_b(m_matrix: &MMatrix) -> KernelResult<C64> {
    let _ = m_matrix;
    todo!("compute_b not yet implemented")
}

/// Compute the six-element column vector `a_3n` of PP2017 Eq. (9).
/// The `xi`-dependent entries pick up shifted `M` entries per the
/// tabulation in `docs/theory/foundations.md`.
pub fn compute_a3n(m_matrix: &MMatrix, b: C64, xi: C64) -> KernelResult<[C64; 6]> {
    let _ = (m_matrix, b, xi);
    todo!("compute_a3n not yet implemented")
}

/// Compute the six-element column vector `a_6n` of PP2017 Eq. (9).
pub fn compute_a6n(m_matrix: &MMatrix, b: C64, xi: C64) -> KernelResult<[C64; 6]> {
    let _ = (m_matrix, b, xi);
    todo!("compute_a6n not yet implemented")
}

/// Build the 4x4 `Delta` matrix per PP2017 Eq. (8). The entry
/// pattern follows the tabulation in `docs/theory/foundations.md`,
/// with `xi` appearing additively in rows 1 and 4 only.
pub fn build_delta(m_matrix: &MMatrix, xi: C64) -> KernelResult<DeltaMatrix> {
    let _ = (m_matrix, xi);
    todo!("build_delta not yet implemented")
}
