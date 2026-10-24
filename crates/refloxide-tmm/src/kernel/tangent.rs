//! Forward-mode derivative of the decoupled uniaxial recursion.
//!
//! [`solve_point_recursive_jvp`] evaluates [`super::solve_point_recursive`]
//! together with its directional derivative along a tangent of the layer
//! records and `q` (a Jacobian-vector product). Every complex intermediate
//! carries `(value, tangent)` through exact derivative rules, including the
//! cancellation-free Fresnel differences, so derivatives keep the relative
//! accuracy of the value in `f32` rather than inheriting finite-difference
//! cancellation. This is what gradient-based optimizers need from the
//! single-precision GPU path.
//!
//! The clamp `R <= 1` is treated as inactive (its derivative is not
//! propagated), matching the value path wherever `R < 1`.

use std::ops::{Add, Div, Mul, Neg, Sub};

use num_complex::Complex;

use super::{LayerCoeffs, Real};

/// Complex dual number `v + eps t` with `eps^2 = 0`.
#[derive(Debug, Clone, Copy)]
struct Dual<T: Real> {
    v: Complex<T>,
    t: Complex<T>,
}

impl<T: Real> Dual<T> {
    fn new(v: Complex<T>, t: Complex<T>) -> Self {
        Self { v, t }
    }

    fn constant(v: Complex<T>) -> Self {
        Self::new(v, Complex::new(T::zero(), T::zero()))
    }

    fn real(v: T) -> Self {
        Self::constant(Complex::new(v, T::zero()))
    }

    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        Self::new(s, self.t / (s * T::lit(2.0)))
    }

    fn exp(self) -> Self {
        let e = self.v.exp();
        Self::new(e, e * self.t)
    }

    fn scale(self, k: T) -> Self {
        Self::new(self.v * k, self.t * k)
    }

    /// `|v|^2` and its derivative `2 Re(conj(v) t)`.
    fn norm_sqr(self) -> (T, T) {
        (
            self.v.norm_sqr(),
            T::lit(2.0) * (self.v.re * self.t.re + self.v.im * self.t.im),
        )
    }
}

impl<T: Real> Add for Dual<T> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.v + o.v, self.t + o.t)
    }
}

impl<T: Real> Sub for Dual<T> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.v - o.v, self.t - o.t)
    }
}

impl<T: Real> Mul for Dual<T> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(self.v * o.v, self.t * o.v + self.v * o.t)
    }
}

impl<T: Real> Div for Dual<T> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let q = self.v / o.v;
        Self::new(q, (self.t - q * o.t) / o.v)
    }
}

impl<T: Real> Neg for Dual<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.v, -self.t)
    }
}

struct Modes<T: Real> {
    chi_o: Dual<T>,
    chi_e: Dual<T>,
    k_o: Dual<T>,
    k_e: Dual<T>,
    z: Dual<T>,
    x_e: Dual<T>,
    pi: Dual<T>,
}

fn modes<T: Real>(l: &LayerCoeffs<T>, dl: &LayerCoeffs<T>, k0sq: T, neg_kzv2: Dual<T>) -> Modes<T> {
    let one = Dual::real(T::one());
    let chi_o = Dual::new(l.chi_o, dl.chi_o);
    let chi_e = Dual::new(l.chi_e, dl.chi_e);
    let e_o = one + chi_o;
    let one_nu = one + (chi_e - chi_o) / e_o;
    let x_e = chi_e.scale(k0sq) - neg_kzv2;
    let k_o = (chi_o.scale(k0sq) - neg_kzv2).sqrt();
    let k_e = (one_nu * x_e).sqrt() / one_nu;
    Modes {
        chi_o,
        chi_e,
        k_o,
        k_e,
        z: k_e / e_o,
        x_e,
        pi: chi_o + chi_e + chi_o * chi_e,
    }
}

fn fresnel<T: Real>(a: &Modes<T>, b: &Modes<T>, sigma: Dual<T>, k0sq: T) -> (Dual<T>, Dual<T>) {
    let one = Dual::real(T::one());
    let s2 = (sigma * sigma).scale(T::lit(-2.0));
    let sum_k = a.k_o + b.k_o;
    let dk = (a.chi_o - b.chi_o).scale(k0sq) / sum_k;
    let r_s = dk / sum_k * (s2 * a.k_o * b.k_o).exp();
    let num = (a.chi_e - b.chi_e).scale(k0sq) + a.x_e * b.pi - b.x_e * a.pi;
    let sum_z = a.z + b.z;
    let dz = num / ((one + a.pi) * (one + b.pi) * sum_z);
    let r_p = dz / sum_z * (s2 * a.k_e * b.k_e).exp();
    (r_s, r_p)
}

/// Reflectance and its directional derivative at one q-point.
///
/// # Parameters
/// - `q`, `k0`, `layers`: as in [`super::solve_point_recursive`].
/// - `dq`: tangent of `q` in `1/Angstrom` per unit step along the direction.
/// - `dlayers`: tangent of every layer record (same length as `layers`);
///   `chi_o`, `chi_e`, `thickness`, and `sigma` carry the per-unit-step
///   change of the corresponding field. `k0` is held fixed.
///
/// # Returns
/// `([R_pp, R_ss], [dR_pp, dR_ss])`.
pub fn solve_point_recursive_jvp<T: Real>(
    q: T,
    dq: T,
    k0: T,
    layers: &[LayerCoeffs<T>],
    dlayers: &[LayerCoeffs<T>],
) -> ([T; 2], [T; 2]) {
    let n = layers.len();
    let half = T::lit(0.5);
    let clamped = (q * half).abs() > k0;
    let kz_vac = (q * half).max(-k0).min(k0);
    let dkz_vac = if clamped { T::zero() } else { dq * half };
    let neg_kzv2 = Dual::new(
        Complex::new(-(kz_vac * kz_vac), T::zero()),
        Complex::new(T::lit(-2.0) * kz_vac * dkz_vac, T::zero()),
    );
    let k0sq = k0 * k0;
    let one = Dual::real(T::one());
    let two_i = Dual::constant(Complex::new(T::zero(), T::lit(2.0)));
    let sig = |j: usize| {
        Dual::new(
            Complex::new(layers[j].sigma, T::zero()),
            Complex::new(dlayers[j].sigma, T::zero()),
        )
    };

    let mut below = modes(&layers[n - 1], &dlayers[n - 1], k0sq, neg_kzv2);
    let above = modes(&layers[n - 2], &dlayers[n - 2], k0sq, neg_kzv2);
    let (mut x_s, mut x_p) = fresnel(&above, &below, sig(n - 1), k0sq);
    below = above;
    for j in (1..n - 1).rev() {
        let above = modes(&layers[j - 1], &dlayers[j - 1], k0sq, neg_kzv2);
        let (r_s, r_p) = fresnel(&above, &below, sig(j), k0sq);
        let d = Dual::new(
            Complex::new(layers[j].thickness, T::zero()),
            Complex::new(dlayers[j].thickness, T::zero()),
        );
        let ys = x_s * (two_i * below.k_o * d).exp();
        let yp = x_p * (two_i * below.k_e * d).exp();
        x_s = (r_s + ys) / (one + r_s * ys);
        x_p = (r_p + yp) / (one + r_p * yp);
        below = above;
    }
    let (r_pp, dr_pp) = x_p.norm_sqr();
    let (r_ss, dr_ss) = x_s.norm_sqr();
    let clip = |r: T, dr: T| {
        if r > T::one() {
            (T::one(), T::zero())
        } else {
            (r, dr)
        }
    };
    let (r_pp, dr_pp) = clip(r_pp, dr_pp);
    let (r_ss, dr_ss) = clip(r_ss, dr_ss);
    ([r_pp, r_ss], [dr_pp, dr_ss])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::solve_point_recursive;

    fn stack(t: f64) -> Vec<LayerCoeffs<f64>> {
        let l = |co: (f64, f64), ce: (f64, f64), d: f64, s: f64| LayerCoeffs {
            chi_o: Complex::new(co.0, co.1),
            chi_e: Complex::new(ce.0, ce.1),
            thickness: d,
            sigma: s,
        };
        vec![
            l((0.0, 0.0), (0.0, 0.0), 0.0, 0.0),
            l(
                (-4e-3 * (1.0 + t), 2e-3),
                (-2e-3, 5e-3 * (1.0 - t)),
                180.0 + 40.0 * t,
                4.0 + t,
            ),
            l((-3e-3, 2e-4), (-3e-3, 2e-4), 15.0, 3.0 - t),
            l((-2.4e-3, 3e-4), (-2.4e-3, 3e-4), 0.0, 2.0),
        ]
    }

    #[test]
    fn jvp_matches_central_difference() {
        let k0 = 0.144;
        let base = stack(0.0);
        let h = 1e-6;
        let (plus, minus) = (stack(h), stack(-h));
        let dl: Vec<LayerCoeffs<f64>> = plus
            .iter()
            .zip(&minus)
            .map(|(p, m)| LayerCoeffs {
                chi_o: (p.chi_o - m.chi_o) / (2.0 * h),
                chi_e: (p.chi_e - m.chi_e) / (2.0 * h),
                thickness: (p.thickness - m.thickness) / (2.0 * h),
                sigma: (p.sigma - m.sigma) / (2.0 * h),
            })
            .collect();
        for q in [0.01, 0.05, 0.12, 0.25] {
            let dq = 0.3 * q;
            let (r, dr) = solve_point_recursive_jvp(q, dq, k0, &base, &dl);
            let rp = solve_point_recursive(q + dq * h, k0, &plus);
            let rm = solve_point_recursive(q - dq * h, k0, &minus);
            let r0 = solve_point_recursive(q, k0, &base);
            for (c, idx) in [(0usize, 0usize), (1, 1)] {
                assert!((r[c] - r0[idx][idx]).abs() <= 1e-15 * r0[idx][idx].abs().max(1e-300));
                let fd = (rp[idx][idx] - rm[idx][idx]) / (2.0 * h);
                let rel = ((dr[c] - fd) / fd).abs();
                assert!(
                    rel < 1e-5,
                    "q={q} channel {c}: jvp {} fd {fd} rel {rel:e}",
                    dr[c]
                );
            }
        }
    }
}
