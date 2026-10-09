//! Eigenmodes of an arbitrary dielectric tensor for the 4x4 reflection-matrix engine.
//!
//! For incidence in the `xz` plane (`ky = 0`, `zeta = kx / k0` fixed by the
//! isotropic fronting) and a full complex tensor `eps` (non-magnetic),
//! eliminating `Ez` and `Hz` from Maxwell's equations gives the Berreman
//! system `q psi = Delta psi` for `psi = [Ex, Hy, Ey, Hx]` (`H` in units of
//! `k0`) and `q = kz / k0`, with
//! `Ez = -(eps_zx Ex + eps_zy Ey + zeta Hy) / eps_zz`. This module solves that
//! system per layer and hands [`super::rmatrix::reflect_chain`] ordered,
//! labeled modes:
//!
//! * `q` are the roots of the characteristic quartic (Faddeev-LeVerrier
//!   coefficients, Ferrari's closed form, Newton polish);
//! * roots are refined by the two-sided Rayleigh quotient of `Delta`, so
//!   their accuracy follows the matrix eigenvalue conditioning rather than
//!   the (much worse) conditioning of polynomial roots near a crossing;
//! * eigenvectors pin `Hy = 1` (p-like) or `Ey = 1` (s-like) and solve the
//!   reduced system with `Hx = -q Ey` exactly; a forward (or backward) pair
//!   uses the explicit p (`Ey = 0`) and s (`Ex = 0`) basis instead only when
//!   that basis is an eigenspace to within rounding at the pair's mean root
//!   (as in isotropic media, whose double roots Ferrari splits by about
//!   `sqrt(eps)`), judged by residual rather than root gap. Near-degenerate
//!   pairs keep individual eigenvectors: their mixing error lies along a
//!   partner mode that propagates almost identically, so it does not reach
//!   the reflectance;
//! * modes are split into forward/backward by `Im q` (decaying toward the
//!   backing) or, for lossless propagating modes, by the Poynting flux
//!   `Re(Ex conj(Hy) - Ey conj(Hx))`, and labeled p-like/s-like by
//!   `|Ex|^2 / (|Ex|^2 + |Ey|^2)` following Passler and Paarmann (2017).
//!
//! Differences `D_b - D_a` are formed by subtraction, so unlike the
//! closed-form uniaxial provider this path carries the usual cancellation in
//! the small interface blocks; in `f64` that costs roughly `1e-11`
//! relative in `R`. Everything is algebraic in `T`, so the same code runs in
//! single precision and with dual numbers.

use num_complex::Complex;

use super::rmatrix::{reflect_chain, Eigenmodes};
use super::Real;
use crate::c4x4::{zero, Mat4};
use crate::error::Result;

type C<T> = Complex<T>;

/// One layer of a general anisotropic stack.
///
/// `chi` is the susceptibility tensor `eps - I` (dimensionless, row-major
/// `[i][j] = chi_ij`); the fronting row must be isotropic so incident and
/// reflected s/p modes are well defined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralLayer<T: Real> {
    /// Susceptibility tensor `eps - I`.
    pub chi: [[C<T>; 3]; 3],
    /// Slab thickness in Angstroms. Ignored for the fronting and backing rows.
    pub thickness: T,
    /// Nevot-Croce roughness of the interface above this slab, in Angstroms.
    pub sigma: T,
}

/// Solved, ordered modes of one layer at fixed `(zeta, k0)`.
struct Modes<T: Real> {
    q: [C<T>; 4],
    d: Mat4<T>,
    ez: [C<T>; 4],
    k0: T,
}

impl<T: Real> Eigenmodes<T> for Modes<T> {
    fn kz(&self) -> [C<T>; 4] {
        self.q.map(|q| q * self.k0)
    }

    fn dynamic(&self) -> Mat4<T> {
        self.d
    }

    fn e_norms(&self) -> [T; 4] {
        let mut out = [T::zero(); 4];
        for (j, v) in out.iter_mut().enumerate() {
            *v = (self.d[0][j].norm_sqr() + self.d[2][j].norm_sqr() + self.ez[j].norm_sqr()).sqrt();
        }
        out
    }

    fn diff(&self, below: &Self) -> (Mat4<T>, [C<T>; 4]) {
        let mut dd = [[zero::<T>(); 4]; 4];
        for (r, row) in dd.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = below.d[r][c] - self.d[r][c];
            }
        }
        let mut dkz = [zero::<T>(); 4];
        for (j, v) in dkz.iter_mut().enumerate() {
            *v = (below.q[j] - self.q[j]) * self.k0;
        }
        (dd, dkz)
    }
}

/// Berreman matrix and the `Ez` coefficients `(a_x, a_y, a_H)`.
fn berreman<T: Real>(chi: &[[C<T>; 3]; 3], zeta: T) -> (Mat4<T>, [C<T>; 3]) {
    let one = C::new(T::one(), T::zero());
    let z = C::new(zeta, T::zero());
    let e = |i: usize, j: usize| if i == j { one + chi[i][j] } else { chi[i][j] };
    let ezz = e(2, 2);
    let a_x = -e(2, 0) / ezz;
    let a_y = -e(2, 1) / ezz;
    let a_h = -z / ezz;
    let zero_c = zero::<T>();
    let delta = [
        [z * a_x, one + z * a_h, z * a_y, zero_c],
        [
            e(0, 0) + e(0, 2) * a_x,
            e(0, 2) * a_h,
            e(0, 1) + e(0, 2) * a_y,
            zero_c,
        ],
        [zero_c, zero_c, zero_c, -one],
        [
            -(e(1, 0) + e(1, 2) * a_x),
            -(e(1, 2) * a_h),
            z * z - e(1, 1) - e(1, 2) * a_y,
            zero_c,
        ],
    ];
    (delta, [a_x, a_y, a_h])
}

fn matmul<T: Real>(a: &Mat4<T>, b: &Mat4<T>) -> Mat4<T> {
    let mut out = [[zero::<T>(); 4]; 4];
    crate::c4x4::mul(a, b, &mut out);
    out
}

/// Monic characteristic polynomial `q^4 + c[0] q^3 + c[1] q^2 + c[2] q + c[3]`.
fn char_poly<T: Real>(delta: &Mat4<T>) -> [C<T>; 4] {
    let trace = |m: &Mat4<T>| m[0][0] + m[1][1] + m[2][2] + m[3][3];
    let shift = |m: &Mat4<T>, c: C<T>| {
        let mut out = *m;
        for (i, row) in out.iter_mut().enumerate() {
            row[i] += c;
        }
        out
    };
    let mut coeffs = [zero::<T>(); 4];
    let mut m = *delta;
    for k in 1..=4usize {
        if k > 1 {
            m = matmul(delta, &shift(&m, coeffs[k - 2]));
        }
        coeffs[k - 1] = -trace(&m) / C::new(T::lit(k as f64), T::zero());
    }
    coeffs
}

fn eval_poly<T: Real>(c: &[C<T>; 4], x: C<T>) -> (C<T>, C<T>) {
    let four = C::new(T::lit(4.0), T::zero());
    let three = C::new(T::lit(3.0), T::zero());
    let two = C::new(T::lit(2.0), T::zero());
    let f = (((x + c[0]) * x + c[1]) * x + c[2]) * x + c[3];
    let df = ((four * x + three * c[0]) * x + two * c[1]) * x + c[2];
    (f, df)
}

/// Principal complex cube root.
fn cbrt<T: Real>(z: C<T>) -> C<T> {
    if z.norm() == T::zero() {
        return z;
    }
    (z.ln() / C::new(T::lit(3.0), T::zero())).exp()
}

/// Roots of the monic quartic by Ferrari's method, each Newton-polished.
fn quartic_roots<T: Real>(c: &[C<T>; 4]) -> [C<T>; 4] {
    let lit = |x: f64| C::new(T::lit(x), T::zero());
    let (a, b, cc, d) = (c[0], c[1], c[2], c[3]);
    let a2 = a * a;
    let p = b - lit(3.0 / 8.0) * a2;
    let r1 = cc - a * b * lit(0.5) + a2 * a * lit(1.0 / 8.0);
    let r0 = d - a * cc * lit(0.25) + a2 * b * lit(1.0 / 16.0) - lit(3.0 / 256.0) * a2 * a2;
    let shift = -a * lit(0.25);
    let scale = (p.norm() + r0.norm().sqrt() + r1.norm().powf(T::lit(1.0 / 3.0))).max(T::epsilon());

    let mut roots = if r1.norm() <= T::epsilon() * scale * scale * scale {
        let disc = (p * p - lit(4.0) * r0).sqrt();
        let y1 = ((-p + disc) * lit(0.5)).sqrt();
        let y2 = ((-p - disc) * lit(0.5)).sqrt();
        [y1, -y1, y2, -y2]
    } else {
        // Resolvent 8 m^3 + 8 p m^2 + (2 p^2 - 8 r0) m - r1^2 = 0, monic form.
        let ca = p;
        let cb = p * p * lit(0.25) - r0;
        let cd = -r1 * r1 * lit(1.0 / 8.0);
        let m = cubic_root_nonzero(ca, cb, cd);
        let s = (m * lit(2.0)).sqrt();
        let t = r1 / (s * lit(2.0));
        let half_p_m = p * lit(0.5) + m;
        let q1 = (s * s - lit(4.0) * (half_p_m + t)).sqrt();
        let q2 = (s * s - lit(4.0) * (half_p_m - t)).sqrt();
        [
            (s + q1) * lit(0.5),
            (s - q1) * lit(0.5),
            (-s + q2) * lit(0.5),
            (-s - q2) * lit(0.5),
        ]
    };
    for r in &mut roots {
        *r += shift;
        for _ in 0..8 {
            let (f, df) = eval_poly(c, *r);
            if df.norm() <= T::epsilon().sqrt() * (r.norm() + T::one()).powi(3) {
                break;
            }
            *r -= f / df;
        }
    }
    roots
}

/// A root of `m^3 + a m^2 + b m + d` with the largest magnitude (Cardano).
fn cubic_root_nonzero<T: Real>(a: C<T>, b: C<T>, d: C<T>) -> C<T> {
    let lit = |x: f64| C::new(T::lit(x), T::zero());
    let p = b - a * a * lit(1.0 / 3.0);
    let q = a * a * a * lit(2.0 / 27.0) - a * b * lit(1.0 / 3.0) + d;
    let disc = (q * q * lit(0.25) + p * p * p * lit(1.0 / 27.0)).sqrt();
    let u_plus = -q * lit(0.5) + disc;
    let u_minus = -q * lit(0.5) - disc;
    let u = if u_plus.norm() >= u_minus.norm() {
        cbrt(u_plus)
    } else {
        cbrt(u_minus)
    };
    let omega = C::new(T::lit(-0.5), T::lit(0.75f64.sqrt()));
    let shift = -a * lit(1.0 / 3.0);
    let mut best = shift;
    let mut best_norm = -T::one();
    let mut w = C::new(T::one(), T::zero());
    for _ in 0..3 {
        let uk = u * w;
        let root = if uk.norm() > T::zero() {
            uk - p / (uk * lit(3.0)) + shift
        } else {
            shift
        };
        if root.norm() > best_norm {
            best_norm = root.norm();
            best = root;
        }
        w *= omega;
    }
    best
}

fn det3<T: Real>(m: &[[C<T>; 3]; 3]) -> C<T> {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Cofactor matrix `C[i][j] = (-1)^(i+j) det(minor_ij)` of a 4x4 matrix.
fn cofactors<T: Real>(a: &Mat4<T>) -> Mat4<T> {
    let mut out = [[zero::<T>(); 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            let mut minor = [[zero::<T>(); 3]; 3];
            for (mr, r) in (0..4).filter(|&r| r != i).enumerate() {
                for (mc, c) in (0..4).filter(|&c| c != j).enumerate() {
                    minor[mr][mc] = a[r][c];
                }
            }
            let sign = if (i + j) % 2 == 0 {
                T::one()
            } else {
                -T::one()
            };
            *v = det3(&minor) * sign;
        }
    }
    out
}

fn norm4<T: Real>(v: &[C<T>; 4]) -> T {
    v.iter()
        .map(|x| x.norm_sqr())
        .fold(T::zero(), |s, x| s + x)
        .sqrt()
}

/// Right and left null vectors of a rank-3 matrix `a`.
///
/// `A adj(A) = adj(A) A = det(A) I = 0`, so adjugate columns are right and
/// adjugate rows left null vectors; the largest of each is returned.
fn null_vectors<T: Real>(a: &Mat4<T>) -> ([C<T>; 4], [C<T>; 4]) {
    let cof = cofactors(a);
    let pick = |vs: [[C<T>; 4]; 4]| {
        let mut best = vs[0];
        for v in vs {
            if norm4(&v) > norm4(&best) {
                best = v;
            }
        }
        best
    };
    let rows = cof;
    let mut cols = [[zero::<T>(); 4]; 4];
    for (i, row) in cof.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            cols[j][i] = *v;
        }
    }
    (pick(rows), pick(cols))
}

/// Mode vector at eigenvalue `q` with `Hy` (`pin = 1`) or `Ey` (`pin = 2`) set to 1.
///
/// The Berreman `Ey` row is exactly `q Ey = -Hx`, so `Hx = -q Ey` for every
/// mode; substituting it leaves the 3x3 system `B(q) [Ex, Hy, Ey] = 0` from
/// the `Ex`, `Hy`, and `Hx` rows. Pinning one component reduces that to a
/// 2x2 solve over the best-conditioned pair of rows. When a polarization
/// block does not couple to the pinned component its right-hand side is
/// exactly zero, so decoupled media keep exactly unmixed modes and
/// `Hx = -q` holds to rounding, instead of dividing by the near-singular
/// determinant of the other block. Returns `None` when every row pair is
/// singular to within rounding (a degenerate eigenspace), in which case the
/// caller uses the explicit p/s basis.
fn mode_vector<T: Real>(delta: &Mat4<T>, q: C<T>, pin: usize) -> Option<[C<T>; 4]> {
    let b = [
        [delta[0][0] - q, delta[0][1], delta[0][2]],
        [delta[1][0], delta[1][1] - q, delta[1][2]],
        [delta[3][0], delta[3][1], delta[3][2] + q * q],
    ];
    let unknowns: [usize; 2] = if pin == 1 { [0, 2] } else { [0, 1] };
    let mut best: Option<(usize, usize, C<T>)> = None;
    for (r1, r2) in [(0usize, 1usize), (0, 2), (1, 2)] {
        let det = b[r1][unknowns[0]] * b[r2][unknowns[1]] - b[r1][unknowns[1]] * b[r2][unknowns[0]];
        if best.is_none_or(|(_, _, d)| det.norm() > d.norm()) {
            best = Some((r1, r2, det));
        }
    }
    let (r1, r2, det) = best?;
    let b_scale: T = b.iter().flatten().map(|x| x.norm()).fold(T::zero(), T::max);
    if det.norm() <= T::lit(1e3) * T::epsilon() * b_scale * b_scale {
        return None;
    }
    let (rhs1, rhs2) = (-b[r1][pin], -b[r2][pin]);
    let u0 = (rhs1 * b[r2][unknowns[1]] - b[r1][unknowns[1]] * rhs2) / det;
    let u1 = (b[r1][unknowns[0]] * rhs2 - rhs1 * b[r2][unknowns[0]]) / det;
    let mut reduced = [zero::<T>(); 3];
    reduced[pin] = C::new(T::one(), T::zero());
    reduced[unknowns[0]] = u0;
    reduced[unknowns[1]] = u1;
    Some([reduced[0], reduced[1], reduced[2], -q * reduced[2]])
}

/// Two-sided Rayleigh quotient `w^T Delta v / w^T v`, refining a root to the
/// eigenvalue's own (matrix) conditioning; `None` if `w^T v` is negligible.
fn rayleigh<T: Real>(delta: &Mat4<T>, v: &[C<T>; 4], w: &[C<T>; 4]) -> Option<C<T>> {
    let mut num = zero::<T>();
    let mut den = zero::<T>();
    for i in 0..4 {
        let mut dv = zero::<T>();
        for j in 0..4 {
            dv += delta[i][j] * v[j];
        }
        num += w[i] * dv;
        den += w[i] * v[i];
    }
    if den.norm() <= T::epsilon().sqrt() * norm4(v) * norm4(w) {
        None
    } else {
        Some(num / den)
    }
}

/// p (`Ey = 0`, `Ex = 1`) and s (`Ex = 0`, `Ey = 1`) vectors of a
/// two-dimensional eigenspace of `a = Delta - q I`.
fn degenerate_pair<T: Real>(a: &Mat4<T>) -> ([C<T>; 4], [C<T>; 4]) {
    let mut best = (0usize, 1usize);
    let mut best_det = -T::one();
    for r1 in 0..4 {
        for r2 in (r1 + 1)..4 {
            let det = (a[r1][1] * a[r2][3] - a[r1][3] * a[r2][1]).norm();
            if det > best_det {
                best_det = det;
                best = (r1, r2);
            }
        }
    }
    let (r1, r2) = best;
    let solve = |fixed: usize| {
        let det = a[r1][1] * a[r2][3] - a[r1][3] * a[r2][1];
        let b1 = -a[r1][fixed];
        let b2 = -a[r2][fixed];
        let hy = (b1 * a[r2][3] - a[r1][3] * b2) / det;
        let hx = (a[r1][1] * b2 - b1 * a[r2][1]) / det;
        let mut v = [zero::<T>(); 4];
        v[fixed] = C::new(T::one(), T::zero());
        v[1] = hy;
        v[3] = hx;
        v
    };
    (solve(0), solve(2))
}

/// Relative eigen-residual `|A v| / |v|`.
fn residual<T: Real>(a: &Mat4<T>, v: &[C<T>; 4]) -> T {
    let mut r = [zero::<T>(); 4];
    for (i, ri) in r.iter_mut().enumerate() {
        for (j, vj) in v.iter().enumerate() {
            *ri += a[i][j] * *vj;
        }
    }
    norm4(&r) / norm4(v)
}

fn shifted<T: Real>(delta: &Mat4<T>, q: C<T>) -> Mat4<T> {
    let mut a = *delta;
    for (i, row) in a.iter_mut().enumerate() {
        row[i] -= q;
    }
    a
}

/// Scales a mode so component `primary` is 1 (`Hy` for p-like, `Ey` for
/// s-like), matching the closed-form uniaxial gauge.
///
/// Neighboring layers with similar tensors then have nearly equal columns,
/// so `D_b - D_a` is small and the interface blocks avoid cancellation.
/// Falls back to `secondary` when `primary` is negligible (strongly mixed
/// modes).
fn gauge<T: Real>(v: &[C<T>; 4], primary: usize, secondary: usize) -> [C<T>; 4] {
    let norm: T = v
        .iter()
        .map(|x| x.norm_sqr())
        .fold(T::zero(), |s, x| s + x)
        .sqrt();
    let pivot = if v[primary].norm() >= T::lit(1e-3) * norm {
        v[primary]
    } else {
        v[secondary]
    };
    v.map(|x| x / pivot)
}

fn p_likeness<T: Real>(v: &[C<T>; 4]) -> T {
    let ex = v[0].norm_sqr();
    let ey = v[2].norm_sqr();
    ex / (ex + ey + T::epsilon() * T::epsilon())
}

impl<T: Real> Modes<T> {
    fn new(layer: &GeneralLayer<T>, zeta: T, k0: T) -> Self {
        let (delta, a) = berreman(&layer.chi, zeta);
        let roots = quartic_roots(&char_poly(&delta));
        let scale = roots.iter().map(|r| r.norm()).fold(T::one(), T::max);
        let deg_tol = T::lit(1e3) * T::epsilon() * scale;
        let im_tol = T::epsilon().sqrt() * scale;

        let a_scale: T = delta
            .iter()
            .flatten()
            .map(|x| x.norm())
            .fold(T::one(), T::max);
        let floor = T::epsilon().sqrt() * a_scale * a_scale * a_scale;
        let mut roots = roots;
        for r in &mut roots {
            for _ in 0..2 {
                let (v, w) = null_vectors(&shifted(&delta, *r));
                if norm4(&v) <= floor {
                    break;
                }
                match rayleigh(&delta, &v, &w) {
                    Some(refined) => *r = refined,
                    None => break,
                }
            }
        }
        let vec_of = |q: C<T>| {
            let a_q = shifted(&delta, q);
            let v = null_vectors(&a_q).0;
            if norm4(&v) <= floor {
                degenerate_pair(&a_q).0
            } else {
                v
            }
        };
        let poynting = |v: &[C<T>; 4]| (v[0] * v[1].conj() - v[2] * v[3].conj()).re;
        let mut forward = [false; 4];
        for (i, &q) in roots.iter().enumerate() {
            forward[i] = if q.im.abs() > im_tol {
                q.im > T::zero()
            } else {
                poynting(&vec_of(q)) > T::zero()
            };
        }
        let mut order: [usize; 4] = [0, 1, 2, 3];
        order.sort_by(|&i, &j| {
            (forward[j], roots[j].im)
                .partial_cmp(&(forward[i], roots[i].im))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let fwd = [order[0], order[1]];
        let bwd = [order[2], order[3]];

        let mut q = [zero::<T>(); 4];
        let mut d = [[zero::<T>(); 4]; 4];
        for (pair, (p_slot, s_slot)) in [(fwd, (0usize, 2usize)), (bwd, (1usize, 3usize))] {
            let (qa, qb) = (roots[pair[0]], roots[pair[1]]);
            let qm = (qa + qb) * C::new(T::lit(0.5), T::zero());
            let a_m = shifted(&delta, qm);
            let (dp, ds) = degenerate_pair(&a_m);
            let eigenspace = residual(&a_m, &dp).max(residual(&a_m, &ds)) <= deg_tol;
            let individual = if eigenspace {
                None
            } else {
                let (va, vb) = (vec_of(qa), vec_of(qb));
                let (vp0, qp, vs0, qs) = if p_likeness(&va) >= p_likeness(&vb) {
                    (va, qa, vb, qb)
                } else {
                    (vb, qb, va, qa)
                };
                let pinned = |q: C<T>, v: [C<T>; 4], pin: usize| {
                    if v[pin].norm() >= T::lit(1e-3) * norm4(&v) {
                        mode_vector(&delta, q, pin)
                    } else {
                        Some(v)
                    }
                };
                match (pinned(qp, vp0, 1), pinned(qs, vs0, 2)) {
                    (Some(vp), Some(vs)) => Some((vp, qp, vs, qs)),
                    _ => None,
                }
            };
            let (vp, qp, vs, qs) = individual.unwrap_or((dp, qm, ds, qm));
            let (vp, vs) = (gauge(&vp, 1, 0), gauge(&vs, 2, 3));
            q[p_slot] = qp;
            q[s_slot] = qs;
            for r in 0..4 {
                d[r][p_slot] = vp[r];
                d[r][s_slot] = vs[r];
            }
        }
        let mut ez = [zero::<T>(); 4];
        for (j, v) in ez.iter_mut().enumerate() {
            *v = a[0] * d[0][j] + a[1] * d[2][j] + a[2] * d[1][j];
        }
        Self { q, d, ez, k0 }
    }
}

/// Polarized reflectance at one q-point for a general anisotropic stack.
///
/// Solves every layer's Berreman eigenmodes (see the module documentation)
/// and runs the 4x4 reflection-matrix recursion, including Nevot-Croce
/// roughness applied per labeled mode. Uniaxial-z stacks reproduce
/// [`super::solve_point_rmatrix`]; tilted, rotated, and biaxial tensors
/// produce nonzero cross-polarized reflectance.
///
/// # Parameters
/// - `q`: scattering vector in `1/Angstrom`, with `|q / (2 k0)| <= 1` after clamping.
/// - `k0`: vacuum wavenumber `2 pi / lambda` in `1/Angstrom`, positive.
/// - `layers`: fronting (isotropic), interior slabs, backing; length at least two.
///
/// # Returns
/// Reflectance `[[R_pp, R_sp], [R_ps, R_ss]]` in the [`super::PointResult`]
/// packing, where `R_sp` is s reflected from p incident.
///
/// # Errors
/// [`crate::RefloxideError::SingularDynamicMatrix`] when a layer's mode
/// matrix or the recursion denominator is singular.
pub fn solve_point_general<T: Real>(
    q: T,
    k0: T,
    layers: &[GeneralLayer<T>],
) -> Result<[[T; 2]; 2]> {
    let kz_vac = (q * T::lit(0.5)).max(-k0).min(k0);
    let zeta = (k0 * k0 - kz_vac * kz_vac).sqrt() / k0;
    reflect_chain(
        layers.len(),
        |j| Modes::new(&layers[j], zeta, k0),
        |j| (layers[j].thickness, layers[j].sigma),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{solve_point_rmatrix, LayerCoeffs};

    fn diag_layer(
        o: C<f64>,
        e: C<f64>,
        thickness: f64,
        sigma: f64,
    ) -> (GeneralLayer<f64>, LayerCoeffs<f64>) {
        let z = C::new(0.0, 0.0);
        (
            GeneralLayer {
                chi: [[o, z, z], [z, o, z], [z, z, e]],
                thickness,
                sigma,
            },
            LayerCoeffs {
                chi_o: o,
                chi_e: e,
                thickness,
                sigma,
            },
        )
    }

    #[test]
    fn general_matches_closed_form_uniaxial_and_keeps_in_plane_symmetry() {
        let (vac_g, vac_u) = diag_layer(C::new(0.0, 0.0), C::new(0.0, 0.0), 0.0, 0.0);
        let (film_g, film_u) = diag_layer(C::new(-4e-3, 2.4e-3), C::new(-2e-3, 5e-3), 200.0, 4.0);
        let (ox_g, ox_u) = diag_layer(C::new(-3e-3, 4e-4), C::new(-3e-3, 4e-4), 15.0, 3.0);
        let (si_g, si_u) = diag_layer(C::new(-2.4e-3, 3e-4), C::new(-2.4e-3, 3e-4), 0.0, 2.0);
        let general = [vac_g, film_g, ox_g, si_g];
        let uniaxial = [vac_u, film_u, ox_u, si_u];
        let k0 = 0.1444;
        for q in [0.01, 0.08, 0.2, 0.283] {
            let a = solve_point_rmatrix(q, k0, &uniaxial).unwrap();
            let b = solve_point_general(q, k0, &general).unwrap();
            for c in 0..2 {
                let rel = (b[c][c] / a[c][c] - 1.0).abs();
                assert!(rel < 1e-10, "q={q} channel {c}: rel {rel:e}");
            }
            assert_eq!(b[0][1], 0.0);
            assert_eq!(b[1][0], 0.0);
        }

        // Optic axis tilted within the plane of incidence keeps y -> -y
        // mirror symmetry, so s and p stay exactly decoupled.
        let (ct, st) = (0.7f64.cos(), 0.7f64.sin());
        let (o, e) = (C::new(-4e-3, 2.4e-3), C::new(-2e-3, 5e-3));
        let z = C::new(0.0, 0.0);
        let tilted = [
            [o * ct * ct + e * st * st, z, (e - o) * ct * st],
            [z, o, z],
            [(e - o) * ct * st, z, o * st * st + e * ct * ct],
        ];
        let stack = [
            vac_g,
            GeneralLayer {
                chi: tilted,
                thickness: 200.0,
                sigma: 4.0,
            },
            ox_g,
            si_g,
        ];
        for q in [0.02, 0.15, 0.28] {
            let r = solve_point_general(q, k0, &stack).unwrap();
            assert_eq!(r[0][1], 0.0, "q={q}");
            assert_eq!(r[1][0], 0.0, "q={q}");
            assert!(r[0][0] > 0.0 && r[1][1] > 0.0);
        }
    }
}
