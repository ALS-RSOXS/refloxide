//! Fixed-size 4x4 complex kernels for the uniaxial transfer chain hot path.

use nalgebra::Matrix4;
use num_complex::Complex;

type C = Complex<f64>;

/// Stack-allocated 4x4 complex matrix in row-major `[row][col]` layout.
pub(crate) type Mat4 = [[C; 4]; 4];

#[inline]
pub(crate) fn identity() -> Mat4 {
    let z = C::new(0.0, 0.0);
    let o = C::new(1.0, 0.0);
    [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, o]]
}

#[inline]
pub(crate) fn from_nalgebra(m: &Matrix4<C>) -> Mat4 {
    let mut out = identity();
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = m[(r, c)];
        }
    }
    out
}

#[inline]
pub(crate) fn mul(a: &Mat4, b: &Mat4, out: &mut Mat4) {
    for i in 0..4 {
        for j in 0..4 {
            let mut s = C::new(0.0, 0.0);
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
}

#[inline]
pub(crate) fn mul_assign(acc: &mut Mat4, b: &Mat4) {
    let left = *acc;
    mul(&left, b, acc);
}

#[inline]
pub(crate) fn hadamard(a: &Mat4, b: &Mat4, out: &mut Mat4) {
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][j] * b[i][j];
        }
    }
}

/// Writes `diag(exp(-i kz d))` as column scaling factors.
#[inline]
pub(crate) fn propagation_diag(kz: &[C; 4], thickness: f64, out: &mut [C; 4]) {
    let d = C::new(thickness, 0.0);
    let minus_i = C::new(0.0, -1.0);
    for s in 0..4 {
        out[s] = (minus_i * kz[s] * d).exp();
    }
}

/// Fills the Nevot-Croce roughness matrix at an interface.
#[inline]
pub(crate) fn fill_w(kz_prev: &[C; 4], kz_curr: &[C; 4], sigma: f64, out: &mut Mat4) {
    let r2_half = C::new(sigma * sigma * 0.5, 0.0);
    let mut eplus = [C::new(0.0, 0.0); 4];
    let mut eminus = [C::new(0.0, 0.0); 4];
    for s in 0..4 {
        let plus = kz_curr[s] + kz_prev[s];
        let minus = kz_curr[s] - kz_prev[s];
        eplus[s] = (-plus * plus * r2_half).exp();
        eminus[s] = (-minus * minus * r2_half).exp();
    }
    for row in 0..4 {
        for col in 0..4 {
            out[row][col] = if (row + col) % 2 == 0 {
                eminus[col]
            } else {
                eplus[col]
            };
        }
    }
}

/// Fused `(prev_di * d).hadamard(w)` optionally followed by column scaling `* diag(p)`.
#[inline]
pub(crate) fn fused_interface_kernel(
    prev_di: &Mat4,
    d: &Mat4,
    w: &Mat4,
    p_diag: Option<&[C; 4]>,
    scratch: &mut Mat4,
    out: &mut Mat4,
) {
    mul(prev_di, d, scratch);
    hadamard(scratch, w, out);
    if let Some(p) = p_diag {
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] *= p[j];
            }
        }
    }
}
