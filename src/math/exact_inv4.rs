//! Closed-form inversion of a 4×4 complex matrix (Dietze / pyGTM `exact_inv`).
//!
//! Transpose convention matches [pyGTM `GTMcore.exact_inv`](https://github.com/pyMatJ/pyGTM/blob/master/GTM/GTMcore.py):
//! internal algebra uses `A = Mᵀ` in row–column index sense (`A[r,c] = M[c,r]`).

use nalgebra::Matrix4;
use num_complex::Complex;

use crate::c4x4::{zero, Mat4};
use crate::kernel::Real;

type C = Complex<f64>;

/// Inverse of a 4x4 complex matrix via the closed-form adjugate.
///
/// # Parameters
/// - `m`: matrix to invert.
///
/// # Returns
/// `None` when the analytic determinant magnitude falls below
/// `f64::EPSILON**2` (upstream pyGTM falls back to `pinv` there).
pub fn exact_inv_4x4(m: &Matrix4<C>) -> Option<Matrix4<C>> {
    let mut arr = [[zero::<f64>(); 4]; 4];
    for (r, row) in arr.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = m[(r, c)];
        }
    }
    exact_inv_4x4_generic(&arr).map(|inv| Matrix4::from_fn(|r, c| inv[r][c]))
}

/// Closed-form inverse on row-major arrays, generic over precision.
///
/// Returns `None` when the analytic determinant is exactly zero (upstream uses `pinv` then).
pub(crate) fn exact_inv_4x4_generic<T: Real>(m: &Mat4<T>) -> Option<Mat4<T>> {
    let a = |r: usize, c: usize| m[c][r];

    let mut det_a = a(0, 0) * a(1, 1) * a(2, 2) * a(3, 3)
        + a(0, 0) * a(1, 2) * a(2, 3) * a(3, 1)
        + a(0, 0) * a(1, 3) * a(2, 1) * a(3, 2);
    det_a += a(0, 1) * a(1, 0) * a(2, 3) * a(3, 2)
        + a(0, 1) * a(1, 2) * a(2, 0) * a(3, 3)
        + a(0, 1) * a(1, 3) * a(2, 2) * a(3, 0);
    det_a += a(0, 2) * a(1, 0) * a(2, 1) * a(3, 3)
        + a(0, 2) * a(1, 1) * a(2, 3) * a(3, 0)
        + a(0, 2) * a(1, 3) * a(2, 0) * a(3, 1);
    det_a += a(0, 3) * a(1, 0) * a(2, 2) * a(3, 1)
        + a(0, 3) * a(1, 1) * a(2, 0) * a(3, 2)
        + a(0, 3) * a(1, 2) * a(2, 1) * a(3, 0);

    det_a -= a(0, 0) * a(1, 1) * a(2, 3) * a(3, 2)
        + a(0, 0) * a(1, 2) * a(2, 1) * a(3, 3)
        + a(0, 0) * a(1, 3) * a(2, 2) * a(3, 1);
    det_a -= a(0, 1) * a(1, 0) * a(2, 2) * a(3, 3)
        + a(0, 1) * a(1, 2) * a(2, 3) * a(3, 0)
        + a(0, 1) * a(1, 3) * a(2, 0) * a(3, 2);
    det_a -= a(0, 2) * a(1, 0) * a(2, 3) * a(3, 1)
        + a(0, 2) * a(1, 1) * a(2, 0) * a(3, 3)
        + a(0, 2) * a(1, 3) * a(2, 1) * a(3, 0);
    det_a -= a(0, 3) * a(1, 0) * a(2, 1) * a(3, 2)
        + a(0, 3) * a(1, 1) * a(2, 2) * a(3, 0)
        + a(0, 3) * a(1, 2) * a(2, 0) * a(3, 1);

    if det_a.norm() < T::epsilon() * T::epsilon() {
        return None;
    }

    let mut b = [[zero::<T>(); 4]; 4];
    b[0][0] =
        a(1, 1) * a(2, 2) * a(3, 3) + a(1, 2) * a(2, 3) * a(3, 1) + a(1, 3) * a(2, 1) * a(3, 2)
            - a(1, 1) * a(2, 3) * a(3, 2)
            - a(1, 2) * a(2, 1) * a(3, 3)
            - a(1, 3) * a(2, 2) * a(3, 1);
    b[0][1] =
        a(0, 1) * a(2, 3) * a(3, 2) + a(0, 2) * a(2, 1) * a(3, 3) + a(0, 3) * a(2, 2) * a(3, 1)
            - a(0, 1) * a(2, 2) * a(3, 3)
            - a(0, 2) * a(2, 3) * a(3, 1)
            - a(0, 3) * a(2, 1) * a(3, 2);
    b[0][2] =
        a(0, 1) * a(1, 2) * a(3, 3) + a(0, 2) * a(1, 3) * a(3, 1) + a(0, 3) * a(1, 1) * a(3, 2)
            - a(0, 1) * a(1, 3) * a(3, 2)
            - a(0, 2) * a(1, 1) * a(3, 3)
            - a(0, 3) * a(1, 2) * a(3, 1);
    b[0][3] =
        a(0, 1) * a(1, 3) * a(2, 2) + a(0, 2) * a(1, 1) * a(2, 3) + a(0, 3) * a(1, 2) * a(2, 1)
            - a(0, 1) * a(1, 2) * a(2, 3)
            - a(0, 2) * a(1, 3) * a(2, 1)
            - a(0, 3) * a(1, 1) * a(2, 2);

    b[1][0] =
        a(1, 0) * a(2, 3) * a(3, 2) + a(1, 2) * a(2, 0) * a(3, 3) + a(1, 3) * a(2, 2) * a(3, 0)
            - a(1, 0) * a(2, 2) * a(3, 3)
            - a(1, 2) * a(2, 3) * a(3, 0)
            - a(1, 3) * a(2, 0) * a(3, 2);
    b[1][1] =
        a(0, 0) * a(2, 2) * a(3, 3) + a(0, 2) * a(2, 3) * a(3, 0) + a(0, 3) * a(2, 0) * a(3, 2)
            - a(0, 0) * a(2, 3) * a(3, 2)
            - a(0, 2) * a(2, 0) * a(3, 3)
            - a(0, 3) * a(2, 2) * a(3, 0);
    b[1][2] =
        a(0, 0) * a(1, 3) * a(3, 2) + a(0, 2) * a(1, 0) * a(3, 3) + a(0, 3) * a(1, 2) * a(3, 0)
            - a(0, 0) * a(1, 2) * a(3, 3)
            - a(0, 2) * a(1, 3) * a(3, 0)
            - a(0, 3) * a(1, 0) * a(3, 2);
    b[1][3] =
        a(0, 0) * a(1, 2) * a(2, 3) + a(0, 2) * a(1, 3) * a(2, 0) + a(0, 3) * a(1, 0) * a(2, 2)
            - a(0, 0) * a(1, 3) * a(2, 2)
            - a(0, 2) * a(1, 0) * a(2, 3)
            - a(0, 3) * a(1, 2) * a(2, 0);

    b[2][0] =
        a(1, 0) * a(2, 1) * a(3, 3) + a(1, 1) * a(2, 3) * a(3, 0) + a(1, 3) * a(2, 0) * a(3, 1)
            - a(1, 0) * a(2, 3) * a(3, 1)
            - a(1, 1) * a(2, 0) * a(3, 3)
            - a(1, 3) * a(2, 1) * a(3, 0);
    b[2][1] =
        a(0, 0) * a(2, 3) * a(3, 1) + a(0, 1) * a(2, 0) * a(3, 3) + a(0, 3) * a(2, 1) * a(3, 0)
            - a(0, 0) * a(2, 1) * a(3, 3)
            - a(0, 1) * a(2, 3) * a(3, 0)
            - a(0, 3) * a(2, 0) * a(3, 1);
    b[2][2] =
        a(0, 0) * a(1, 1) * a(3, 3) + a(0, 1) * a(1, 3) * a(3, 0) + a(0, 3) * a(1, 0) * a(3, 1)
            - a(0, 0) * a(1, 3) * a(3, 1)
            - a(0, 1) * a(1, 0) * a(3, 3)
            - a(0, 3) * a(1, 1) * a(3, 0);
    b[2][3] =
        a(0, 0) * a(1, 3) * a(2, 1) + a(0, 1) * a(1, 0) * a(2, 3) + a(0, 3) * a(1, 1) * a(2, 0)
            - a(0, 0) * a(1, 1) * a(2, 3)
            - a(0, 1) * a(1, 3) * a(2, 0)
            - a(0, 3) * a(1, 0) * a(2, 1);

    b[3][0] =
        a(1, 0) * a(2, 2) * a(3, 1) + a(1, 1) * a(2, 0) * a(3, 2) + a(1, 2) * a(2, 1) * a(3, 0)
            - a(1, 0) * a(2, 1) * a(3, 2)
            - a(1, 1) * a(2, 2) * a(3, 0)
            - a(1, 2) * a(2, 0) * a(3, 1);
    b[3][1] =
        a(0, 0) * a(2, 1) * a(3, 2) + a(0, 1) * a(2, 2) * a(3, 0) + a(0, 2) * a(2, 0) * a(3, 1)
            - a(0, 0) * a(2, 2) * a(3, 1)
            - a(0, 1) * a(2, 0) * a(3, 2)
            - a(0, 2) * a(2, 1) * a(3, 0);
    b[3][2] =
        a(0, 0) * a(1, 2) * a(3, 1) + a(0, 1) * a(1, 0) * a(3, 2) + a(0, 2) * a(1, 1) * a(3, 0)
            - a(0, 0) * a(1, 1) * a(3, 2)
            - a(0, 1) * a(1, 2) * a(3, 0)
            - a(0, 2) * a(1, 0) * a(3, 1);
    b[3][3] =
        a(0, 0) * a(1, 1) * a(2, 2) + a(0, 1) * a(1, 2) * a(2, 0) + a(0, 2) * a(1, 0) * a(2, 1)
            - a(0, 0) * a(1, 2) * a(2, 1)
            - a(0, 1) * a(1, 0) * a(2, 2)
            - a(0, 2) * a(1, 1) * a(2, 0);

    let inv_det = Complex::new(T::one(), T::zero()) / det_a;
    let mut out = [[zero::<T>(); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = b[j][i] * inv_det;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Matrix4;

    #[test]
    fn exact_inv_matches_nalgebra_random() {
        let m = Matrix4::new(
            C::new(1.2, 0.1),
            C::new(0.3, -0.2),
            C::new(0.0, 0.5),
            C::new(0.7, 0.0),
            C::new(-0.1, 0.4),
            C::new(1.0, 0.0),
            C::new(0.2, 0.2),
            C::new(0.0, -0.3),
            C::new(0.4, -0.1),
            C::new(0.1, 0.1),
            C::new(1.5, 0.2),
            C::new(0.0, 0.1),
            C::new(0.0, 0.2),
            C::new(0.3, -0.4),
            C::new(-0.2, 0.1),
            C::new(0.9, 0.0),
        );
        let inv_e = m.try_inverse().expect("well-conditioned");
        let inv_a = exact_inv_4x4(&m).expect("non-singular");
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(inv_e[(i, j)].re, inv_a[(i, j)].re, epsilon = 1e-9);
                assert_relative_eq!(inv_e[(i, j)].im, inv_a[(i, j)].im, epsilon = 1e-9);
            }
        }
    }
}
