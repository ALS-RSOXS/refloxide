//! Backend-agnostic uniaxial transfer-matrix core.
//!
//! This module owns the per-(q, stack) inner loop of the uniaxial-z 4x4
//! solver in a form that is portable to accelerator backends: no heap
//! allocation, no `nalgebra`, plain `[..; 4]` arrays, and a flat
//! plain-old-data layer description ([`LayerCoeffs`]). Arithmetic is generic
//! over [`Real`] so the identical code path can be evaluated in `f32` to
//! quantify single-precision error before committing to GPU precision modes.
//!
//! It does not validate shapes or energies, build dielectric tensors, or
//! choose a threading strategy; those live in [`crate::uniaxial`]. Callers
//! must pass at least two layers (fronting and backing) and a strictly
//! positive `k0`.

use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign};
use rayon::prelude::*;

use crate::c4x4::{self, zero, Mat4};
use crate::error::{RefloxideError, Result};
use crate::math::exact_inv_4x4_generic;

mod general;
mod rmatrix;
mod tangent;

pub use general::{solve_point_general, GeneralLayer};
pub use rmatrix::solve_point_rmatrix;
pub use tangent::solve_point_recursive_jvp;

/// Floating-point scalar accepted by the kernel (`f32` or `f64`).
pub trait Real: Float + FloatConst + NumAssign + Send + Sync + std::fmt::Debug + 'static {
    /// Rounds an `f64` literal to this precision.
    fn lit(x: f64) -> Self;
    /// Widens this value to `f64`.
    fn to_f64_lossless(self) -> f64;
}

impl Real for f32 {
    #[inline]
    fn lit(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn to_f64_lossless(self) -> f64 {
        f64::from(self)
    }
}

impl Real for f64 {
    #[inline]
    fn lit(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_f64_lossless(self) -> f64 {
        self
    }
}

/// Per-slab coefficients consumed by the uniaxial kernel.
///
/// This is the device-uploadable layer record: only the ordinary and
/// extraordinary Berreman components of a uniaxial-z slab enter the
/// closed-form eigenstructure, so the full 3x3 tensor is not carried.
/// Components are stored as susceptibilities `chi = eps - 1` rather than
/// `eps` because soft X-ray `|chi|` is `O(1e-3)`: packing `1 + chi` into a
/// single-precision word would discard roughly three significant digits of
/// the optical constants before any arithmetic happens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerCoeffs<T: Real> {
    /// Ordinary susceptibility `eps_xx - 1 = eps_yy - 1` (dimensionless).
    pub chi_o: Complex<T>,
    /// Extraordinary susceptibility `eps_zz - 1` (dimensionless).
    pub chi_e: Complex<T>,
    /// Slab thickness in Angstroms. Ignored for the fronting and backing rows.
    pub thickness: T,
    /// Nevot-Croce roughness of the interface above this slab, in Angstroms.
    pub sigma: T,
}

/// Polarized power reflectance and amplitude transmission at one q-point.
///
/// Index layout follows the Fresnel-validated kernel packing:
/// `(0, 0) = R_pp` and `(1, 1) = R_ss`, with the off-diagonals holding the
/// cross-polarized terms.
pub type PointResult<T> = ([[T; 2]; 2], [[Complex<T>; 2]; 2]);

struct LayerSnapshot<T: Real> {
    kz: [Complex<T>; 4],
    di: Mat4<T>,
}

/// Solves the uniaxial transfer chain for one scattering vector.
///
/// # Parameters
/// - `q`: scattering vector in `1/Angstrom`; clamped so that
///   `|q / (2 k0)| <= 1`.
/// - `k0`: vacuum wavenumber `2 pi / lambda` in `1/Angstrom`, positive.
/// - `layers`: fronting, interior slabs, backing; length at least two.
///
/// # Errors
/// [`RefloxideError::SingularDynamicMatrix`] with the failing layer index and
/// `q_index = usize::MAX`; callers annotate the q and energy indices.
pub fn solve_point<T: Real>(q: T, k0: T, layers: &[LayerCoeffs<T>]) -> Result<PointResult<T>> {
    let nlayers = layers.len();
    let kz_vac = (q * T::lit(0.5)).max(-k0).min(k0);
    let kz_vac_sq = kz_vac * kz_vac;
    let kx = (k0 * k0 - kz_vac_sq).sqrt();
    let wave = Wave { kx, kz_vac_sq, k0 };

    let mut m = c4x4::identity::<T>();
    let mut tmp = c4x4::identity::<T>();
    let mut kernel = c4x4::identity::<T>();
    let mut w = c4x4::identity::<T>();
    let mut p_diag = [zero::<T>(); 4];

    let (kz0, d0) = eigenstructure(&layers[0], &wave);
    let mut prev = LayerSnapshot {
        kz: kz0,
        di: invert_dynamic(0, &d0)?,
    };

    for (j, layer) in layers.iter().enumerate().take(nlayers - 1).skip(1) {
        let (kz, d) = eigenstructure(layer, &wave);
        let di = invert_dynamic(j, &d)?;
        c4x4::fill_w(&prev.kz, &kz, layer.sigma, &mut w);
        c4x4::propagation_diag(&kz, layer.thickness, &mut p_diag);
        c4x4::fused_interface_kernel(&prev.di, &d, &w, Some(&p_diag), &mut tmp, &mut kernel);
        c4x4::mul_assign(&mut m, &kernel);
        prev = LayerSnapshot { kz, di };
    }

    let last = &layers[nlayers - 1];
    let (kz, d) = eigenstructure(last, &wave);
    invert_dynamic(nlayers - 1, &d)?;
    c4x4::fill_w(&prev.kz, &kz, last.sigma, &mut w);
    c4x4::fused_interface_kernel(&prev.di, &d, &w, None, &mut tmp, &mut kernel);
    c4x4::mul_assign(&mut m, &kernel);

    Ok(extract_rt(&m))
}

/// Polarized power reflectance at one q-point via decoupled Parratt recursion.
///
/// For uniaxial-z media at `ky = 0` the 4x4 chain is block-diagonal: s
/// couples only to ordinary modes and p only to extraordinary modes, so
/// `R_sp = R_ps = 0` exactly and each channel reduces to a scalar recursion.
/// Interface coefficients `r_s = (k_i - k_j)/(k_i + k_j)` and
/// `r_p = (z_i - z_j)/(z_i + z_j)` with `z = kz_e / eps_o` are formed from
/// susceptibility differences (for example `k_i - k_j = k0^2 (chi_i - chi_j)
/// / (k_i + k_j)`), so the small quantities that set `R` are never recovered
/// by subtracting `O(1)` numbers. Only decaying propagators
/// `exp(2 i kz d)` enter, which keeps thick absorbing films stable. Nevot-Croce
/// roughness multiplies each `r` by `exp(-2 k_i k_j sigma^2)`.
///
/// This is the preferred reduced-precision and accelerator path. It returns
/// reflectance only; amplitude transmission in the 4x4 normalization comes
/// from [`solve_point`].
///
/// # Parameters
/// Same as [`solve_point`].
///
/// # Returns
/// Reflectance in the [`PointResult`] packing, `[[R_pp, 0], [0, R_ss]]`.
pub fn solve_point_recursive<T: Real>(q: T, k0: T, layers: &[LayerCoeffs<T>]) -> [[T; 2]; 2] {
    let n = layers.len();
    let kz_vac = (q * T::lit(0.5)).max(-k0).min(k0);
    let neg_kzv2 = Complex::new(-(kz_vac * kz_vac), T::zero());
    let k0sq = k0 * k0;
    let one = Complex::new(T::one(), T::zero());
    let two_i = Complex::new(T::zero(), T::lit(2.0));

    let modes = |l: &LayerCoeffs<T>| {
        let e_o = one + l.chi_o;
        let (k_o, k_e, x_e) = mode_kz(l.chi_o, l.chi_e, k0sq, neg_kzv2);
        let pi = l.chi_o + l.chi_e + l.chi_o * l.chi_e;
        RecursiveModes {
            k_o,
            k_e,
            z: k_e / e_o,
            x_e,
            pi,
        }
    };
    let fresnel = |a: &LayerCoeffs<T>,
                   ma: &RecursiveModes<T>,
                   b: &LayerCoeffs<T>,
                   mb: &RecursiveModes<T>,
                   sigma: T| {
        let s2 = Complex::new(T::lit(-2.0) * sigma * sigma, T::zero());
        let dk = (a.chi_o - b.chi_o) * k0sq / (ma.k_o + mb.k_o);
        let r_s = dk / (ma.k_o + mb.k_o) * (s2 * ma.k_o * mb.k_o).exp();
        let num = (a.chi_e - b.chi_e) * k0sq + ma.x_e * mb.pi - mb.x_e * ma.pi;
        let dz = num / ((one + ma.pi) * (one + mb.pi) * (ma.z + mb.z));
        let r_p = dz / (ma.z + mb.z) * (s2 * ma.k_e * mb.k_e).exp();
        (r_s, r_p)
    };

    let mut below = modes(&layers[n - 1]);
    let above = modes(&layers[n - 2]);
    let (mut x_s, mut x_p) = fresnel(
        &layers[n - 2],
        &above,
        &layers[n - 1],
        &below,
        layers[n - 1].sigma,
    );
    below = above;
    for j in (1..n - 1).rev() {
        let above = modes(&layers[j - 1]);
        let (r_s, r_p) = fresnel(&layers[j - 1], &above, &layers[j], &below, layers[j].sigma);
        let d = Complex::new(layers[j].thickness, T::zero());
        let ph_s = (two_i * below.k_o * d).exp();
        let ph_p = (two_i * below.k_e * d).exp();
        x_s = (r_s + x_s * ph_s) / (one + r_s * x_s * ph_s);
        x_p = (r_p + x_p * ph_p) / (one + r_p * x_p * ph_p);
        below = above;
    }
    [
        [x_p.norm_sqr().min(T::one()), T::zero()],
        [T::zero(), x_s.norm_sqr().min(T::one())],
    ]
}

struct RecursiveModes<T: Real> {
    k_o: Complex<T>,
    k_e: Complex<T>,
    z: Complex<T>,
    x_e: Complex<T>,
    pi: Complex<T>,
}

/// Ordinary and extraordinary `kz` plus `X_e = chi_e k0^2 + (q/2)^2`.
///
/// Sums with the vacuum term are written as subtractions of `-(q/2)^2` so an
/// input `-0` imaginary part survives and lossless evanescent media keep the
/// same square-root branch as the legacy `eps k0^2 - kx^2` form.
#[inline]
fn mode_kz<T: Real>(
    chi_o: Complex<T>,
    chi_e: Complex<T>,
    k0sq: T,
    neg_kzv2: Complex<T>,
) -> (Complex<T>, Complex<T>, Complex<T>) {
    let one = Complex::new(T::one(), T::zero());
    let one_nu = one + (chi_e - chi_o) / (one + chi_o);
    let x_e = chi_e * k0sq - neg_kzv2;
    let k_o = (chi_o * k0sq - neg_kzv2).sqrt();
    let k_e = (one_nu * x_e).sqrt() / one_nu;
    (k_o, k_e, x_e)
}

/// Solves many stacks of equal depth against a shared q-grid.
///
/// This is the batched entry point shaped for accelerator dispatch: inputs
/// are flat, row-major buffers and every `(stack, q)` pair is independent.
///
/// # Parameters
/// - `q`: scattering vectors in `1/Angstrom`, length `n_q`.
/// - `k0`: vacuum wavenumber per stack in `1/Angstrom`, length `n_stacks`.
/// - `layers`: row-major `(n_stacks, n_layers)` buffer of layer records.
/// - `n_layers`: depth of every stack, at least two.
/// - `parallel`: distribute `(stack, q)` pairs across rayon when true.
///
/// # Returns
/// Row-major `(n_stacks, n_q)` results.
///
/// # Errors
/// [`RefloxideError::InvalidShape`] when `layers.len() != n_stacks * n_layers`
/// or `n_layers < 2`; [`RefloxideError::SingularDynamicMatrix`] with
/// `energy_index` set to the stack index and `q_index` to the q index.
pub fn solve_flat<T: Real>(
    q: &[T],
    k0: &[T],
    layers: &[LayerCoeffs<T>],
    n_layers: usize,
    parallel: bool,
) -> Result<Vec<PointResult<T>>> {
    check_flat(k0.len(), layers.len(), n_layers)?;
    let n_q = q.len();
    let solve = |idx: usize| {
        let (si, qi) = (idx / n_q, idx % n_q);
        let stack = &layers[si * n_layers..(si + 1) * n_layers];
        solve_point(q[qi], k0[si], stack).map_err(|err| match err {
            RefloxideError::SingularDynamicMatrix { layer, .. } => {
                RefloxideError::SingularDynamicMatrix {
                    layer,
                    q_index: qi,
                    energy_index: Some(si),
                }
            }
            other => other,
        })
    };
    let total = k0.len() * n_q;
    if parallel {
        (0..total).into_par_iter().map(solve).collect()
    } else {
        (0..total).map(solve).collect()
    }
}

/// Batched [`solve_point_recursive`] over equal-depth stacks and a shared q-grid.
///
/// CPU counterpart of the GPU dispatch with identical buffer layout; see
/// [`solve_flat`] for the parameter contract.
///
/// # Returns
/// Row-major `(n_stacks, n_q)` pairs `[R_pp, R_ss]`.
///
/// # Errors
/// [`RefloxideError::InvalidShape`] or [`RefloxideError::InsufficientLayers`]
/// for inconsistent buffer lengths.
pub fn solve_flat_recursive<T: Real>(
    q: &[T],
    k0: &[T],
    layers: &[LayerCoeffs<T>],
    n_layers: usize,
    parallel: bool,
) -> Result<Vec<[T; 2]>> {
    check_flat(k0.len(), layers.len(), n_layers)?;
    let n_q = q.len();
    let solve = |idx: usize| {
        let (si, qi) = (idx / n_q, idx % n_q);
        let r = solve_point_recursive(q[qi], k0[si], &layers[si * n_layers..(si + 1) * n_layers]);
        [r[0][0], r[1][1]]
    };
    let total = k0.len() * n_q;
    Ok(if parallel {
        (0..total).into_par_iter().map(solve).collect()
    } else {
        (0..total).map(solve).collect()
    })
}

/// Batched [`solve_point_rmatrix`] over equal-depth stacks and a shared q-grid.
///
/// Same buffer layout as [`solve_flat`]; see it for the parameter contract.
///
/// # Returns
/// Row-major `(n_stacks, n_q)` reflectance blocks in the [`PointResult`]
/// packing.
///
/// # Errors
/// Shape errors as in [`solve_flat`]; [`RefloxideError::SingularDynamicMatrix`]
/// with `energy_index` set to the stack index and `q_index` to the q index.
pub fn solve_flat_rmatrix<T: Real>(
    q: &[T],
    k0: &[T],
    layers: &[LayerCoeffs<T>],
    n_layers: usize,
    parallel: bool,
) -> Result<Vec<[[T; 2]; 2]>> {
    check_flat(k0.len(), layers.len(), n_layers)?;
    let n_q = q.len();
    let solve = |idx: usize| {
        let (si, qi) = (idx / n_q, idx % n_q);
        let stack = &layers[si * n_layers..(si + 1) * n_layers];
        solve_point_rmatrix(q[qi], k0[si], stack).map_err(|err| match err {
            RefloxideError::SingularDynamicMatrix { layer, .. } => {
                RefloxideError::SingularDynamicMatrix {
                    layer,
                    q_index: qi,
                    energy_index: Some(si),
                }
            }
            other => other,
        })
    };
    let total = k0.len() * n_q;
    if parallel {
        (0..total).into_par_iter().map(solve).collect()
    } else {
        (0..total).map(solve).collect()
    }
}

fn check_flat(n_stacks: usize, n_records: usize, n_layers: usize) -> Result<()> {
    if n_layers < 2 {
        return Err(RefloxideError::InsufficientLayers(n_layers));
    }
    if n_records != n_stacks * n_layers {
        return Err(RefloxideError::InvalidShape(format!(
            "layers buffer length {n_records} != n_stacks ({n_stacks}) * n_layers ({n_layers})"
        )));
    }
    Ok(())
}

/// In-plane and vacuum-normal wavevector components shared by every layer.
struct Wave<T: Real> {
    kx: T,
    kz_vac_sq: T,
    k0: T,
}

/// Mode wavevectors and dynamic matrix of one uniaxial-z layer.
///
/// `kz^2` is formed as `chi k0^2 + (q/2)^2` using the identity
/// `kx^2 = k0^2 - (q/2)^2`, which avoids the catastrophic cancellation of
/// `eps k0^2 - kx^2` at grazing incidence.
fn eigenstructure<T: Real>(layer: &LayerCoeffs<T>, wave: &Wave<T>) -> ([Complex<T>; 4], Mat4<T>) {
    let one = Complex::new(T::one(), T::zero());
    let c = |x: T| Complex::new(x, T::zero());
    let (kx, k0) = (wave.kx, wave.k0);
    let k0sq = k0 * k0;
    let neg_kzv2 = c(-wave.kz_vac_sq);
    let e_o = one + layer.chi_o;
    let nu = (layer.chi_e - layer.chi_o) / e_o;
    let (kz_ord, kz_ext, _) = mode_kz(layer.chi_o, layer.chi_e, k0sq, neg_kzv2);
    let kz = [kz_ext, -kz_ext, kz_ord, -kz_ord];

    let optic_z = one;
    let inv_k0 = T::one() / k0;
    let mut d = [[zero::<T>(); 4]; 4];
    for s in 0..4 {
        let k = [c(kx), zero(), kz[s]];
        let kmag = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
        let kn = [k[0] / kmag, k[1] / kmag, k[2] / kmag];
        let kpol = kn[2] * optic_z;

        let dv = if s >= 2 {
            [-kn[1], kn[0], zero()]
        } else {
            let scale = ((one + nu) / (one + nu * kpol * kpol)) * kpol;
            [
                zero::<T>() - kn[0] * scale,
                zero::<T>() - kn[1] * scale,
                optic_z - kn[2] * scale,
            ]
        };
        let mag2 = dv[0].norm_sqr() + dv[1].norm_sqr() + dv[2].norm_sqr();
        let inv_norm = T::one() / (mag2.sqrt() + T::epsilon());
        let dn = [dv[0] * inv_norm, dv[1] * inv_norm, dv[2] * inv_norm];

        let hx = k[1] * dn[2] - k[2] * dn[1];
        let hy = k[2] * dn[0] - k[0] * dn[2];

        d[0][s] = dn[0];
        d[1][s] = hy * inv_k0;
        d[2][s] = dn[1];
        d[3][s] = hx * inv_k0;
    }
    (kz, d)
}

fn invert_dynamic<T: Real>(layer_idx: usize, d: &Mat4<T>) -> Result<Mat4<T>> {
    exact_inv_4x4_generic(d).ok_or(RefloxideError::SingularDynamicMatrix {
        layer: layer_idx,
        q_index: usize::MAX,
        energy_index: None,
    })
}

fn extract_rt<T: Real>(m: &Mat4<T>) -> PointResult<T> {
    let eps = T::epsilon();
    let mut denom = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    if denom.norm() < eps {
        denom += Complex::new(eps, T::zero());
    }
    let r_ss = (m[1][0] * m[2][2] - m[1][2] * m[2][0]) / denom;
    let r_sp = (m[3][0] * m[2][2] - m[3][2] * m[2][0]) / denom;
    let r_ps = (m[0][0] * m[1][2] - m[1][0] * m[0][2]) / denom;
    let r_pp = (m[0][0] * m[3][2] - m[3][0] * m[0][2]) / denom;
    let t_ss = m[2][2] / denom;
    let t_sp = -m[2][0] / denom;
    let t_ps = -m[0][2] / denom;
    let t_pp = m[0][0] / denom;

    let refl = [
        [r_ss.norm_sqr().min(T::one()), r_sp.norm_sqr().min(T::one())],
        [r_ps.norm_sqr().min(T::one()), r_pp.norm_sqr().min(T::one())],
    ];
    (refl, [[t_ss, t_sp], [t_ps, t_pp]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recursive_and_rmatrix_match_4x4_in_packing_order() {
        let l = |co: (f64, f64), ce: (f64, f64), d: f64, s: f64| LayerCoeffs {
            chi_o: Complex::new(co.0, co.1),
            chi_e: Complex::new(ce.0, ce.1),
            thickness: d,
            sigma: s,
        };
        let stack = [
            l((0.0, 0.0), (0.0, 0.0), 0.0, 0.0),
            l((-4e-3, 2e-3), (-2e-3, 5e-3), 180.0, 4.0),
            l((-3e-3, 2e-4), (-3e-3, 2e-4), 15.0, 3.0),
            l((-2.4e-3, 3e-4), (-2.4e-3, 3e-4), 0.0, 2.0),
        ];
        for q in [0.01, 0.05, 0.12, 0.25] {
            let tmm = solve_point(q, 0.144, &stack).unwrap().0;
            let rec = solve_point_recursive(q, 0.144, &stack);
            let rmx = solve_point_rmatrix(q, 0.144, &stack).unwrap();
            for c in 0..2 {
                let rel = ((rec[c][c] - tmm[c][c]) / tmm[c][c]).abs();
                assert!(rel < 1e-9, "q={q} channel {c}: rel err {rel:e}");
                let rel = ((rmx[c][c] - rec[c][c]) / rec[c][c]).abs();
                assert!(rel < 1e-12, "q={q} channel {c}: rmatrix rel err {rel:e}");
            }
            assert_eq!(rmx[0][1], 0.0);
            assert_eq!(rmx[1][0], 0.0);
        }
    }
}
