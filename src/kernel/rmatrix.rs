//! Reflection-matrix 4x4 recursion with cancellation-free interface matrices.
//!
//! The transfer-matrix product of [`super::solve_point`] recovers `r` as a
//! ratio of `O(1)` matrix entries, so absolute rounding near machine epsilon
//! becomes a large relative error once `|r|` is small, and growing
//! propagators overflow the dynamic range of thick absorbing films. This
//! module keeps the full 4x4 mode description but instead carries the 2x2
//! reflection matrix `G` (up-going amplitudes `= G` times down-going
//! amplitudes) from the backing to the fronting:
//!
//! `G_above = (K_-+ + K_-- G)(K_++ + K_+- G)^-1`, with `K = (D_a^-1 D_b) o W`,
//!
//! and forms `K = I + D_a^-1 (D_b - D_a)` from analytic eigenvector
//! differences so the small off-diagonal blocks that determine `r` are never
//! obtained by subtracting `O(1)` numbers. Propagation of `G` across a slab
//! only multiplies by decaying exponentials.
//!
//! The recursion is generic over an [`Eigenmodes`] provider: the
//! closed-form uniaxial-z provider here supplies analytic, cancellation-free
//! differences, and [`super::general`] supplies numerically solved modes of
//! an arbitrary dielectric tensor.

use num_complex::Complex;

use super::{mode_kz, LayerCoeffs, Real};
use crate::c4x4::{zero, Mat4};
use crate::error::{RefloxideError, Result};
use crate::math::exact_inv_4x4_generic;

type C<T> = Complex<T>;
type Mat2<T> = [[C<T>; 2]; 2];

/// Down-going mode columns `[e+, o+]` and up-going columns `[e-, o-]`.
const DOWN: [usize; 2] = [0, 2];
const UP: [usize; 2] = [1, 3];

/// Mode data of one layer at fixed `(q, k0)`, unnormalized.
///
/// Columns follow the kernel ordering `[e+, e-, o+, o-]` and rows
/// `[Ex, Hy, Ey, Hx]`: `e+- = [+-z/k0, 1, 0, 0]` with `z = kz_e / eps_o`,
/// `o+- = [0, 0, 1, -+k_o/k0]`.
struct Medium<T: Real> {
    chi_o: C<T>,
    chi_e: C<T>,
    k0: T,
    kx: T,
    k_o: C<T>,
    k_e: C<T>,
    z: C<T>,
    x_e: C<T>,
    pi: C<T>,
}

/// Mode data consumed by the reflection-matrix recursion.
///
/// Columns of [`Eigenmodes::dynamic`] follow `[p+, p-, s+, s-]` (the
/// kernel's `[e+, e-, o+, o-]`), rows `[Ex, Hy, Ey, Hx]` with `H` in units of
/// `k0`; `+` modes carry energy toward the backing. Column normalization is
/// arbitrary except in the fronting, where [`Eigenmodes::e_norms`] rescales
/// to unit `|E|` so reflection amplitudes refer to unit-field modes.
pub(crate) trait Eigenmodes<T: Real> {
    /// Mode `kz` in `1/Angstrom`, ordered like the dynamic-matrix columns.
    fn kz(&self) -> [C<T>; 4];
    /// Dynamic matrix of field components per mode.
    fn dynamic(&self) -> Mat4<T>;
    /// Electric-field magnitude of each dynamic-matrix column.
    fn e_norms(&self) -> [T; 4];
    /// `(D_below - D_self, kz_below - kz_self)`, as accurately as the
    /// provider can form them.
    fn diff(&self, below: &Self) -> (Mat4<T>, [C<T>; 4]);
}

/// Cancellation-free differences `b - a` of the mode quantities.
struct Delta<T: Real> {
    k_o: C<T>,
    k_e: C<T>,
    z: C<T>,
}

impl<T: Real> Medium<T> {
    fn new(l: &LayerCoeffs<T>, k0: T, kx: T, neg_kzv2: C<T>) -> Self {
        let one = C::new(T::one(), T::zero());
        let (k_o, k_e, x_e) = mode_kz(l.chi_o, l.chi_e, k0 * k0, neg_kzv2);
        Self {
            chi_o: l.chi_o,
            chi_e: l.chi_e,
            k0,
            kx,
            k_o,
            k_e,
            z: k_e / (one + l.chi_o),
            x_e,
            pi: l.chi_o + l.chi_e + l.chi_o * l.chi_e,
        }
    }

    /// `below - self` for `k_o`, `k_e`, and `z`, written in terms of
    /// susceptibility differences.
    fn delta(&self, below: &Self) -> Delta<T> {
        let one = C::new(T::one(), T::zero());
        let k0sq = self.k0 * self.k0;
        let k_o = (below.chi_o - self.chi_o) * k0sq / (self.k_o + below.k_o);
        let dz2 = ((below.chi_e - self.chi_e) * k0sq + below.x_e * self.pi - self.x_e * below.pi)
            / ((one + self.pi) * (one + below.pi));
        let z = dz2 / (self.z + below.z);
        let cross_b = (below.chi_o + self.chi_e + below.chi_o * self.chi_e) * below.x_e;
        let cross_a = (self.chi_o + below.chi_e + self.chi_o * below.chi_e) * self.x_e;
        let dk_e2 = ((below.chi_e - self.chi_e) * k0sq + cross_b - cross_a)
            / ((one + self.chi_e) * (one + below.chi_e));
        let k_e = dk_e2 / (self.k_e + below.k_e);
        Delta { k_o, k_e, z }
    }
}

impl<T: Real> Eigenmodes<T> for Medium<T> {
    fn kz(&self) -> [C<T>; 4] {
        [self.k_e, -self.k_e, self.k_o, -self.k_o]
    }

    fn dynamic(&self) -> Mat4<T> {
        let z0 = zero();
        let one = C::new(T::one(), T::zero());
        let a = self.z / self.k0;
        let b = self.k_o / self.k0;
        [
            [a, -a, z0, z0],
            [one, one, z0, z0],
            [z0, z0, one, one],
            [z0, z0, -b, b],
        ]
    }

    fn e_norms(&self) -> [T; 4] {
        let e_e = C::new(T::one(), T::zero()) + self.chi_e;
        let ez = C::new(self.kx, T::zero()) / (e_e * self.k0);
        let ex = self.z / self.k0;
        let p = (ex.norm_sqr() + ez.norm_sqr()).sqrt() + T::epsilon();
        [p, p, T::one(), T::one()]
    }

    fn diff(&self, below: &Self) -> (Mat4<T>, [C<T>; 4]) {
        let z0 = zero();
        let d = self.delta(below);
        let dz = d.z / self.k0;
        let dk = d.k_o / self.k0;
        let dd = [
            [dz, -dz, z0, z0],
            [z0, z0, z0, z0],
            [z0, z0, z0, z0],
            [z0, z0, -dk, dk],
        ];
        (dd, [d.k_e, -d.k_e, d.k_o, -d.k_o])
    }
}

/// Interface matrix `K = (I + D_a^-1 dD) o W` between `a` (above) and `b` (below).
fn interface<T: Real>(
    di_a: &Mat4<T>,
    kz_a: &[C<T>; 4],
    kz_b: &[C<T>; 4],
    dd: &Mat4<T>,
    dkz: &[C<T>; 4],
    sigma: T,
) -> Mat4<T> {
    let z0 = zero();
    let mut k = [[z0; 4]; 4];
    for (i, row) in k.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            let mut s = z0;
            for m in 0..4 {
                s += di_a[i][m] * dd[m][j];
            }
            if i == j {
                s += C::new(T::one(), T::zero());
            }
            *v = s;
        }
    }
    if sigma != T::zero() {
        let r2_half = C::new(sigma * sigma * T::lit(0.5), T::zero());
        for s in 0..4 {
            let plus = kz_b[s] + kz_a[s];
            let eplus = (-plus * plus * r2_half).exp();
            let eminus = (-dkz[s] * dkz[s] * r2_half).exp();
            for (row, k_row) in k.iter_mut().enumerate() {
                k_row[s] *= if (row + s) % 2 == 0 { eminus } else { eplus };
            }
        }
    }
    k
}

fn block<T: Real>(k: &Mat4<T>, rows: [usize; 2], cols: [usize; 2]) -> Mat2<T> {
    [
        [k[rows[0]][cols[0]], k[rows[0]][cols[1]]],
        [k[rows[1]][cols[0]], k[rows[1]][cols[1]]],
    ]
}

fn mul2<T: Real>(a: &Mat2<T>, b: &Mat2<T>) -> Mat2<T> {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn add2<T: Real>(a: &Mat2<T>, b: &Mat2<T>) -> Mat2<T> {
    [
        [a[0][0] + b[0][0], a[0][1] + b[0][1]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1]],
    ]
}

/// `num * den^-1` for 2x2 matrices; `None` when `den` is singular.
fn right_divide<T: Real>(num: &Mat2<T>, den: &Mat2<T>) -> Option<Mat2<T>> {
    let det = den[0][0] * den[1][1] - den[0][1] * den[1][0];
    let det_norm = det.norm();
    if det_norm.is_nan() || det_norm == T::zero() {
        return None;
    }
    let inv = [
        [den[1][1] / det, -den[0][1] / det],
        [-den[1][0] / det, den[0][0] / det],
    ];
    Some(mul2(num, &inv))
}

/// Polarized reflectance at one q-point from the 4x4 reflection-matrix recursion.
///
/// Agrees with [`super::solve_point`] in exact arithmetic (including
/// cross-polarized terms and the fronting unit-`|E|` mode normalization) and
/// is accurate to near working precision in `f32` and `f64`; see the module
/// documentation for the formulation. Transmission is not computed.
///
/// # Parameters
/// Same as [`super::solve_point`].
///
/// # Returns
/// Reflectance in the [`super::PointResult`] packing: `(0, 0) = R_pp`,
/// `(1, 1) = R_ss`, off-diagonals cross-polarized.
///
/// # Errors
/// [`RefloxideError::SingularDynamicMatrix`] with the failing layer index
/// and `q_index = usize::MAX` when a dynamic matrix or the 2x2 recursion
/// denominator is singular.
pub fn solve_point_rmatrix<T: Real>(q: T, k0: T, layers: &[LayerCoeffs<T>]) -> Result<[[T; 2]; 2]> {
    let kz_vac = (q * T::lit(0.5)).max(-k0).min(k0);
    let neg_kzv2 = C::new(-(kz_vac * kz_vac), T::zero());
    let kx = (k0 * k0 - kz_vac * kz_vac).sqrt();
    reflect_chain(
        layers.len(),
        |j| Medium::new(&layers[j], k0, kx, neg_kzv2),
        |j| (layers[j].thickness, layers[j].sigma),
    )
}

/// Reflection-matrix recursion over `n` layers built by `medium(j)`.
///
/// `slab(j)` returns `(thickness, sigma)` of layer `j`, with `sigma` the
/// roughness of the interface above it. Returns reflectance in the
/// [`super::PointResult`] packing, normalized to unit-`|E|` fronting modes.
pub(crate) fn reflect_chain<T: Real, M: Eigenmodes<T>>(
    n: usize,
    medium: impl Fn(usize) -> M,
    slab: impl Fn(usize) -> (T, T),
) -> Result<[[T; 2]; 2]> {
    let singular = |layer| RefloxideError::SingularDynamicMatrix {
        layer,
        q_index: usize::MAX,
        energy_index: None,
    };
    let mut below = medium(n - 1);
    let mut g: Mat2<T> = [[zero(); 2]; 2];
    let mut front = None;
    for j in (1..n).rev() {
        let above = medium(j - 1);
        let (thickness, sigma) = slab(j);
        if j < n - 1 {
            let kz = below.kz();
            let i_d = C::new(T::zero(), thickness);
            for (a, &u) in UP.iter().enumerate() {
                for (b, &dn) in DOWN.iter().enumerate() {
                    g[a][b] *= (-(kz[u] - kz[dn]) * i_d).exp();
                }
            }
        }
        let di_a = exact_inv_4x4_generic(&above.dynamic()).ok_or_else(|| singular(j - 1))?;
        let (dd, dkz) = above.diff(&below);
        let k = interface(&di_a, &above.kz(), &below.kz(), &dd, &dkz, sigma);
        let num = add2(&block(&k, UP, DOWN), &mul2(&block(&k, UP, UP), &g));
        let den = add2(&block(&k, DOWN, DOWN), &mul2(&block(&k, DOWN, UP), &g));
        g = right_divide(&num, &den).ok_or_else(|| singular(j - 1))?;
        if j == 1 {
            front = Some(above);
            break;
        }
        below = above;
    }
    let norms = front.ok_or_else(|| singular(0))?.e_norms();
    let refl = |a: usize, b: usize| {
        (g[a][b] * (norms[UP[a]] / norms[DOWN[b]]))
            .norm_sqr()
            .min(T::one())
    };
    Ok([[refl(0, 0), refl(1, 0)], [refl(0, 1), refl(1, 1)]])
}
