//! Stage 1, the Berreman 4x4 matrix `Delta`.
//!
//! Implements PP2017 Eq. (8) for the sixteen `Delta_ij` entries,
//! Eq. (9) for the auxiliary coefficients `a_3n` and `a_6n`, and
//! Eq. (10) for the scalar `b`. The full tabulation is reproduced
//! in `docs/theory/foundations.md`.

use crate::error::{KernelError, KernelResult};
use crate::kernel::constitutive::MMatrix;
use crate::types::scalar::C64;
use nalgebra::Matrix4;

/// Type alias for the Berreman 4x4 matrix.
pub type DeltaMatrix = Matrix4<C64>;

#[inline]
fn m(m: &MMatrix, i: usize, j: usize) -> C64 {
    m[(i - 1, j - 1)]
}

const B_TOL: f64 = 1.0e-18;

/// Compute the auxiliary scalar `b = M_33 M_66 - M_36 M_63` of
/// PP2017 Eq. (10). Returns
/// [`crate::KernelError::SingularConstitutive`] when `b` vanishes
/// within numerical tolerance.
pub fn compute_b(m_matrix: &MMatrix) -> KernelResult<C64> {
    let b = m(m_matrix, 3, 3) * m(m_matrix, 6, 6) - m(m_matrix, 3, 6) * m(m_matrix, 6, 3);
    if b.norm() <= B_TOL {
        return Err(KernelError::SingularConstitutive(b));
    }
    Ok(b)
}

/// Compute the six-element column vector `a_3n` of PP2017 Eq. (9).
/// The `xi`-dependent entries pick up shifted `M` entries per the
/// tabulation in `docs/theory/foundations.md`.
pub fn compute_a3n(m_matrix: &MMatrix, b: C64, xi: C64) -> KernelResult<[C64; 6]> {
    let _ = xi;
    let b_inv = C64::new(1.0, 0.0) / b;
    let a31 = (m(m_matrix, 6, 1) * m(m_matrix, 3, 6) - m(m_matrix, 3, 1) * m(m_matrix, 6, 6)) * b_inv;
    let a32 = ((m(m_matrix, 6, 2) - xi) * m(m_matrix, 3, 6) - m(m_matrix, 3, 2) * m(m_matrix, 6, 6))
        * b_inv;
    let a34 = (m(m_matrix, 6, 4) * m(m_matrix, 3, 6) - m(m_matrix, 3, 4) * m(m_matrix, 6, 6)) * b_inv;
    let a35 = (m(m_matrix, 6, 5) * m(m_matrix, 3, 6)
        - (m(m_matrix, 3, 5) + xi) * m(m_matrix, 6, 6))
        * b_inv;
    Ok([a31, a32, C64::new(0.0, 0.0), a34, a35, C64::new(0.0, 0.0)])
}

/// Compute the six-element column vector `a_6n` of PP2017 Eq. (9).
pub fn compute_a6n(m_matrix: &MMatrix, b: C64, xi: C64) -> KernelResult<[C64; 6]> {
    let b_inv = C64::new(1.0, 0.0) / b;
    let a61 = (m(m_matrix, 6, 3) * m(m_matrix, 3, 1) - m(m_matrix, 3, 3) * m(m_matrix, 6, 1)) * b_inv;
    let a62 = (m(m_matrix, 6, 3) * m(m_matrix, 3, 2)
        - m(m_matrix, 3, 3) * (m(m_matrix, 6, 2) - xi))
        * b_inv;
    let a64 = (m(m_matrix, 6, 3) * m(m_matrix, 3, 4) - m(m_matrix, 3, 3) * m(m_matrix, 6, 4)) * b_inv;
    let a65 = (m(m_matrix, 6, 3) * (m(m_matrix, 3, 5) + xi)
        - m(m_matrix, 3, 3) * m(m_matrix, 6, 5))
        * b_inv;
    Ok([a61, a62, C64::new(0.0, 0.0), a64, a65, C64::new(0.0, 0.0)])
}

/// Build the 4x4 `Delta` matrix per PP2017 Eq. (8). The entry
/// pattern follows the tabulation in `docs/theory/foundations.md`,
/// with `xi` appearing additively in rows 1 and 4 only.
pub fn build_delta(m_matrix: &MMatrix, xi: C64) -> KernelResult<DeltaMatrix> {
    let b = compute_b(m_matrix)?;
    let a3 = compute_a3n(m_matrix, b, xi)?;
    let a6 = compute_a6n(m_matrix, b, xi)?;
    let a31 = a3[0];
    let a32 = a3[1];
    let a34 = a3[3];
    let a35 = a3[4];
    let a61 = a6[0];
    let a62 = a6[1];
    let a64 = a6[3];
    let a65 = a6[4];
    let mut d = DeltaMatrix::zeros();
    d[(0, 0)] = m(m_matrix, 5, 1) + (m(m_matrix, 5, 3) + xi) * a31 + m(m_matrix, 5, 6) * a61;
    d[(0, 1)] = m(m_matrix, 5, 5) + (m(m_matrix, 5, 3) + xi) * a35 + m(m_matrix, 5, 6) * a65;
    d[(0, 2)] = m(m_matrix, 5, 2) + (m(m_matrix, 5, 3) + xi) * a32 + m(m_matrix, 5, 6) * a62;
    d[(0, 3)] = -m(m_matrix, 5, 4) - (m(m_matrix, 5, 3) + xi) * a34 - m(m_matrix, 5, 6) * a64;
    d[(1, 0)] = m(m_matrix, 1, 1) + m(m_matrix, 1, 3) * a31 + m(m_matrix, 1, 6) * a61;
    d[(1, 1)] = m(m_matrix, 1, 5) + m(m_matrix, 1, 3) * a35 + m(m_matrix, 1, 6) * a65;
    d[(1, 2)] = m(m_matrix, 1, 2) + m(m_matrix, 1, 3) * a32 + m(m_matrix, 1, 6) * a62;
    d[(1, 3)] = -m(m_matrix, 1, 4) - m(m_matrix, 1, 3) * a34 - m(m_matrix, 1, 6) * a64;
    d[(2, 0)] = -m(m_matrix, 4, 1) - m(m_matrix, 4, 3) * a31 - m(m_matrix, 4, 6) * a61;
    d[(2, 1)] = -m(m_matrix, 4, 5) - m(m_matrix, 4, 3) * a35 - m(m_matrix, 4, 6) * a65;
    d[(2, 2)] = -m(m_matrix, 4, 2) - m(m_matrix, 4, 3) * a32 - m(m_matrix, 4, 6) * a62;
    d[(2, 3)] = m(m_matrix, 4, 4) + m(m_matrix, 4, 3) * a34 + m(m_matrix, 4, 6) * a64;
    d[(3, 0)] = m(m_matrix, 2, 1) + m(m_matrix, 2, 3) * a31 + (m(m_matrix, 2, 6) - xi) * a61;
    d[(3, 1)] = m(m_matrix, 2, 5) + m(m_matrix, 2, 3) * a35 + (m(m_matrix, 2, 6) - xi) * a65;
    d[(3, 2)] = m(m_matrix, 2, 2) + m(m_matrix, 2, 3) * a32 + (m(m_matrix, 2, 6) - xi) * a62;
    d[(3, 3)] = -m(m_matrix, 2, 4) - m(m_matrix, 2, 3) * a34 - (m(m_matrix, 2, 6) - xi) * a64;
    Ok(d)
}
