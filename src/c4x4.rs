//! Fixed-size 4x4 complex kernels for the uniaxial transfer chain hot path.
//!
//! Generic over [`Real`] so the same arithmetic can be evaluated in `f32`
//! for precision studies and ported one-to-one to GPU backends.

use num_complex::Complex;

use crate::kernel::Real;

/// Stack-allocated 4x4 complex matrix in row-major `[row][col]` layout.
pub(crate) type Mat4<T> = [[Complex<T>; 4]; 4];

#[inline]
pub(crate) fn zero<T: Real>() -> Complex<T> {
    Complex::new(T::zero(), T::zero())
}

#[inline]
pub(crate) fn identity<T: Real>() -> Mat4<T> {
    let z = zero();
    let o = Complex::new(T::one(), T::zero());
    [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, o]]
}

#[inline]
pub(crate) fn mul<T: Real>(a: &Mat4<T>, b: &Mat4<T>, out: &mut Mat4<T>) {
    for i in 0..4 {
        for j in 0..4 {
            let mut s = zero();
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
}

#[inline]
pub(crate) fn mul_assign<T: Real>(acc: &mut Mat4<T>, b: &Mat4<T>) {
    let left = *acc;
    mul(&left, b, acc);
}

#[inline]
pub(crate) fn hadamard<T: Real>(a: &Mat4<T>, b: &Mat4<T>, out: &mut Mat4<T>) {
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][j] * b[i][j];
        }
    }
}

/// Writes `diag(exp(-i kz d))` as column scaling factors.
#[inline]
pub(crate) fn propagation_diag<T: Real>(
    kz: &[Complex<T>; 4],
    thickness: T,
    out: &mut [Complex<T>; 4],
) {
    let d = Complex::new(thickness, T::zero());
    let minus_i = Complex::new(T::zero(), -T::one());
    for s in 0..4 {
        out[s] = (minus_i * kz[s] * d).exp();
    }
}

/// Fills the Nevot-Croce roughness matrix at an interface.
#[inline]
pub(crate) fn fill_w<T: Real>(
    kz_prev: &[Complex<T>; 4],
    kz_curr: &[Complex<T>; 4],
    sigma: T,
    out: &mut Mat4<T>,
) {
    let r2_half = Complex::new(sigma * sigma * T::lit(0.5), T::zero());
    let mut eplus = [zero(); 4];
    let mut eminus = [zero(); 4];
    for s in 0..4 {
        let plus = kz_curr[s] + kz_prev[s];
        let minus = kz_curr[s] - kz_prev[s];
        eplus[s] = (-plus * plus * r2_half).exp();
        eminus[s] = (-minus * minus * r2_half).exp();
    }
    for (row, out_row) in out.iter_mut().enumerate().take(4) {
        for col in 0..4 {
            out_row[col] = if (row + col) % 2 == 0 {
                eminus[col]
            } else {
                eplus[col]
            };
        }
    }
}

/// Fused `(prev_di * d).hadamard(w)` optionally followed by column scaling `* diag(p)`.
#[inline]
pub(crate) fn fused_interface_kernel<T: Real>(
    prev_di: &Mat4<T>,
    d: &Mat4<T>,
    w: &Mat4<T>,
    p_diag: Option<&[Complex<T>; 4]>,
    scratch: &mut Mat4<T>,
    out: &mut Mat4<T>,
) {
    mul(prev_di, d, scratch);
    hadamard(scratch, w, out);
    if let Some(p) = p_diag {
        for out_row in out.iter_mut().take(4) {
            for j in 0..4 {
                out_row[j] *= p[j];
            }
        }
    }
}
