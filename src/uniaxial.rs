//! Uniaxial 4x4 transfer matrix kernel.
//!
//! Port of `refloxide.pxr.tjf4x4.uniaxial_reflectivity`. The implementation
//! is streaming, holding only the previous-layer `(kz, Di)` snapshot across
//! iterations of the transfer chain, and parallel across q-points via
//! [`rayon`]. The closed form `kz` and `(D, H)` polarization eigenvectors
//! for uniaxial-z with optic axis along `z` are written directly to avoid an
//! eigensolve per layer.

use nalgebra::{Matrix3, Matrix4, Vector3};
use num_complex::Complex;
use rayon::prelude::*;

use crate::c4x4;
use crate::c4x4::Mat4;
use crate::error::{RefloxideError, Result};
use crate::math::exact_inv_4x4;

/// Complex alias used throughout the kernel.
type C = Complex<f64>;

/// Polarized 2x2 reflectance and transmission blocks for one q-point.
type PolBlock = ([[f64; 2]; 2], [[C; 2]; 2]);

/// Photon energy to wavelength conversion constant in `eV * Angstrom`.
const HC_EV_ANGSTROM: f64 = 12398.4193;

/// One slab in a stratified medium.
///
/// `thickness` and `sigma` are in Angstroms. `sld_re` and `sld_im` carry the
/// Henke-style `delta` and `beta` for the isotropic legacy path; the per-axis
/// dispersion comes in through the [`uniaxial_reflectivity`] `tensor`
/// argument.
#[derive(Debug, Clone, Copy)]
pub struct Layer {
    /// Slab thickness in Angstroms. Ignored for the fronting and backing rows.
    pub thickness: f64,
    /// Real part of the isotropic SLD in `1e-6 * Angstrom^-2`.
    pub sld_re: f64,
    /// Imaginary part of the isotropic SLD in `1e-6 * Angstrom^-2`.
    pub sld_im: f64,
    /// Nevot-Croce roughness sigma in Angstroms.
    pub sigma: f64,
}

impl Layer {
    /// Builds a layer from its four scalar parameters.
    pub fn new(thickness: f64, sld_re: f64, sld_im: f64, sigma: f64) -> Self {
        Self {
            thickness,
            sld_re,
            sld_im,
            sigma,
        }
    }
}

impl From<[f64; 4]> for Layer {
    fn from(row: [f64; 4]) -> Self {
        Self::new(row[0], row[1], row[2], row[3])
    }
}

/// Polarized reflectance and amplitude transmission for a single energy.
///
/// Index layout mirrors the Python reference: `refl[i][k][l]` and
/// `tran[i][k][l]` with `(0, 0) = ss`, `(1, 1) = pp`, `(0, 1) = sp`,
/// `(1, 0) = ps`.
#[derive(Debug, Clone)]
pub struct UniaxialOutput {
    /// Power reflectance with shape `(numpnts, 2, 2)`.
    pub refl: Vec<[[f64; 2]; 2]>,
    /// Complex amplitude transmission with shape `(numpnts, 2, 2)`.
    pub tran: Vec<[[C; 2]; 2]>,
}

/// Snapshot of per-layer state retained across the streaming chain.
#[derive(Debug, Clone, Copy)]
struct LayerSnapshot {
    /// Mode-ordered z-component wavevectors: `[extraord+, extraord-, ord+, ord-]`.
    kz: [C; 4],
    /// Inverse dynamic matrix for the layer.
    di: Mat4,
}

/// Reusable scratch for one q-point solve (avoids per-slab allocations).
struct QScratch {
    m: Mat4,
    tmp: Mat4,
    kernel: Mat4,
    w: Mat4,
    d: Mat4,
    p_diag: [C; 4],
}

impl QScratch {
    fn new() -> Self {
        Self {
            m: c4x4::identity(),
            tmp: c4x4::identity(),
            kernel: c4x4::identity(),
            w: c4x4::identity(),
            d: c4x4::identity(),
            p_diag: [C::new(0.0, 0.0); 4],
        }
    }
}

/// Computes polarized reflectance and transmission for a uniaxial multilayer.
///
/// # Parameters
/// - `q`: scattering wavevectors in `1/Angstrom`.
/// - `layers`: per-slab parameters, length `nlayers`. The first and last
///   entries describe the fronting and the backing.
/// - `tensor`: per-slab 3x3 dispersion tensor, length `nlayers`. The
///   Berreman dielectric is built as `eps = conj(I - 2 * tensor)`.
/// - `energy`: photon energy in eV. Must be strictly positive.
/// - `parallel`: when true, distribute q-points across rayon's global thread
///   pool. When false, run sequentially. Callers driving the kernel from a
///   Python fitting routine that is itself multi-threaded or multi-process
///   should pass `false` to avoid CPU oversubscription. The rayon pool size
///   is controlled by the `RAYON_NUM_THREADS` environment variable when set.
///
/// # Errors
/// Returns [`RefloxideError`] for shape mismatches, fewer than two slabs,
/// non-finite or non-positive energies, and for singular dynamic matrices
/// at any (q, layer).
pub fn uniaxial_reflectivity(
    q: &[f64],
    layers: &[Layer],
    tensor: &[Matrix3<C>],
    energy: f64,
    parallel: bool,
) -> Result<UniaxialOutput> {
    if layers.len() != tensor.len() {
        return Err(RefloxideError::LayerCountMismatch {
            layers: layers.len(),
            tensor: tensor.len(),
        });
    }
    if layers.len() < 2 {
        return Err(RefloxideError::InsufficientLayers(layers.len()));
    }
    if !energy.is_finite() || energy <= 0.0 {
        return Err(RefloxideError::InvalidEnergy(energy));
    }

    let wl = HC_EV_ANGSTROM / energy;
    let k0 = 2.0 * std::f64::consts::PI / wl;

    let eps: Vec<Matrix3<C>> = tensor.iter().map(berreman_dielectric).collect();

    let solve = |(i, qi): (usize, f64)| solve_q(qi, layers, &eps, k0).map_err(|e| annotate(e, i));
    let solved: Vec<PolBlock> = if parallel {
        q.par_iter()
            .copied()
            .enumerate()
            .map(solve)
            .collect::<Result<Vec<_>>>()?
    } else {
        q.iter()
            .copied()
            .enumerate()
            .map(solve)
            .collect::<Result<Vec<_>>>()?
    };

    let mut refl = Vec::with_capacity(q.len());
    let mut tran = Vec::with_capacity(q.len());
    for (r, t) in solved {
        refl.push(r);
        tran.push(t);
    }

    Ok(UniaxialOutput { refl, tran })
}

fn annotate(err: RefloxideError, q_index: usize) -> RefloxideError {
    match err {
        RefloxideError::SingularDynamicMatrix { layer, .. } => {
            RefloxideError::SingularDynamicMatrix { layer, q_index }
        }
        other => other,
    }
}

fn solve_q(qi: f64, layers: &[Layer], eps: &[Matrix3<C>], k0: f64) -> Result<PolBlock> {
    let nlayers = layers.len();
    let s = (qi / (2.0 * k0)).clamp(-1.0, 1.0);
    let theta = std::f64::consts::FRAC_PI_2 - s.asin();
    let kx = k0 * theta.sin();
    let ky = 0.0_f64;

    let mut scratch = QScratch::new();
    scratch.m = c4x4::identity();

    let mut prev = build_snapshot(0, &eps[0], kx, ky, k0)?;

    for j in 1..nlayers - 1 {
        let layer = &layers[j];
        let snap = build_snapshot_with_d(j, &eps[j], kx, ky, k0, &mut scratch.d)?;
        c4x4::fill_w(&prev.kz, &snap.kz, layer.sigma, &mut scratch.w);
        c4x4::propagation_diag(&snap.kz, layer.thickness, &mut scratch.p_diag);
        c4x4::fused_interface_kernel(
            &prev.di,
            &scratch.d,
            &scratch.w,
            Some(&scratch.p_diag),
            &mut scratch.tmp,
            &mut scratch.kernel,
        );
        c4x4::mul_assign(&mut scratch.m, &scratch.kernel);
        prev = snap;
    }

    let snap_last =
        build_snapshot_with_d(nlayers - 1, &eps[nlayers - 1], kx, ky, k0, &mut scratch.d)?;
    c4x4::fill_w(
        &prev.kz,
        &snap_last.kz,
        layers[nlayers - 1].sigma,
        &mut scratch.w,
    );
    c4x4::fused_interface_kernel(
        &prev.di,
        &scratch.d,
        &scratch.w,
        None,
        &mut scratch.tmp,
        &mut scratch.kernel,
    );
    c4x4::mul_assign(&mut scratch.m, &scratch.kernel);

    Ok(extract_rt(&scratch.m))
}

fn berreman_dielectric(t: &Matrix3<C>) -> Matrix3<C> {
    let two = C::new(2.0, 0.0);
    let scaled = t.map(|v| v * two);
    let mut m = Matrix3::<C>::identity() - scaled;
    for v in m.iter_mut() {
        *v = v.conj();
    }
    m
}

fn compute_eigenstructure(eps: &Matrix3<C>, kx: f64, ky: f64, k0: f64) -> ([C; 4], Matrix4<C>) {
    let one = C::new(1.0, 0.0);
    let e_o = eps[(0, 0)];
    let e_e = eps[(2, 2)];
    let nu = (e_e - e_o) / e_o;
    let kpar2 = C::new(kx * kx + ky * ky, 0.0);
    let k0sq = C::new(k0 * k0, 0.0);
    let kz_ord = (e_o * k0sq - kpar2).sqrt();
    let radicand = e_o * k0sq * (one + nu) * (one + nu) - kpar2 * (one + nu);
    let kz_ext = radicand.sqrt() / (one + nu);
    let kz = [kz_ext, -kz_ext, kz_ord, -kz_ord];

    let optic_z = one;
    let mut d = Matrix4::<C>::zeros();
    for s in 0..4 {
        let kvec = Vector3::new(C::new(kx, 0.0), C::new(ky, 0.0), kz[s]);
        let kdotk = kvec.x * kvec.x + kvec.y * kvec.y + kvec.z * kvec.z;
        let kmag = kdotk.sqrt();
        let knorm = Vector3::new(kvec.x / kmag, kvec.y / kmag, kvec.z / kmag);
        let kpol = knorm.z * optic_z;

        let dvec = if s >= 2 {
            Vector3::new(-knorm.y, knorm.x, C::new(0.0, 0.0))
        } else {
            let scale = ((one + nu) / (one + nu * kpol * kpol)) * kpol;
            Vector3::new(
                C::new(0.0, 0.0) - knorm.x * scale,
                C::new(0.0, 0.0) - knorm.y * scale,
                optic_z - knorm.z * scale,
            )
        };
        let mag2 = dvec.x.norm_sqr() + dvec.y.norm_sqr() + dvec.z.norm_sqr();
        let inv_norm = 1.0 / (mag2.sqrt() + f64::EPSILON);
        let dnorm = Vector3::new(dvec.x * inv_norm, dvec.y * inv_norm, dvec.z * inv_norm);

        let h = kvec.cross(&dnorm);
        let inv_k0 = 1.0 / k0;
        let hpol = Vector3::new(h.x * inv_k0, h.y * inv_k0, h.z * inv_k0);

        d[(0, s)] = dnorm.x;
        d[(1, s)] = hpol.y;
        d[(2, s)] = dnorm.y;
        d[(3, s)] = hpol.x;
    }

    (kz, d)
}

fn invert_dynamic(layer_idx: usize, d: &Matrix4<C>) -> Result<Mat4> {
    match exact_inv_4x4(d) {
        Some(inv) => Ok(c4x4::from_nalgebra(&inv)),
        None => Err(RefloxideError::SingularDynamicMatrix {
            layer: layer_idx,
            q_index: usize::MAX,
        }),
    }
}

fn build_snapshot(
    layer_idx: usize,
    eps: &Matrix3<C>,
    kx: f64,
    ky: f64,
    k0: f64,
) -> Result<LayerSnapshot> {
    let (kz, d) = compute_eigenstructure(eps, kx, ky, k0);
    let di = invert_dynamic(layer_idx, &d)?;
    Ok(LayerSnapshot { kz, di })
}

fn build_snapshot_with_d(
    layer_idx: usize,
    eps: &Matrix3<C>,
    kx: f64,
    ky: f64,
    k0: f64,
    d_out: &mut Mat4,
) -> Result<LayerSnapshot> {
    let (kz, d) = compute_eigenstructure(eps, kx, ky, k0);
    *d_out = c4x4::from_nalgebra(&d);
    let di = invert_dynamic(layer_idx, &d)?;
    Ok(LayerSnapshot { kz, di })
}

fn extract_rt(m: &Mat4) -> ([[f64; 2]; 2], [[C; 2]; 2]) {
    let mut denom = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    if denom.norm() < f64::EPSILON {
        denom += C::new(f64::EPSILON, 0.0);
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
        [r_ss.norm_sqr().min(1.0), r_sp.norm_sqr().min(1.0)],
        [r_ps.norm_sqr().min(1.0), r_pp.norm_sqr().min(1.0)],
    ];
    let tran = [[t_ss, t_sp], [t_ps, t_pp]];
    (refl, tran)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn c4_mul_matches_nalgebra() {
        let a = Matrix4::new(
            C::new(1.0, 0.1),
            C::new(0.2, 0.0),
            C::new(0.0, 0.3),
            C::new(0.1, -0.1),
            C::new(0.3, 0.0),
            C::new(1.1, 0.2),
            C::new(0.4, 0.0),
            C::new(0.0, 0.2),
            C::new(0.0, 0.1),
            C::new(0.2, -0.1),
            C::new(0.9, 0.0),
            C::new(0.3, 0.1),
            C::new(0.1, 0.0),
            C::new(0.0, 0.2),
            C::new(0.2, 0.1),
            C::new(1.2, -0.1),
        );
        let b = Matrix4::new(
            C::new(0.8, 0.0),
            C::new(0.1, 0.2),
            C::new(0.0, 0.0),
            C::new(0.3, 0.0),
            C::new(0.0, 0.1),
            C::new(1.0, 0.0),
            C::new(0.2, 0.1),
            C::new(0.0, 0.0),
            C::new(0.4, -0.1),
            C::new(0.0, 0.0),
            C::new(0.7, 0.2),
            C::new(0.1, 0.0),
            C::new(0.0, 0.0),
            C::new(0.2, 0.0),
            C::new(0.0, 0.1),
            C::new(0.5, 0.0),
        );
        let aa = c4x4::from_nalgebra(&a);
        let bb = c4x4::from_nalgebra(&b);
        let mut out = c4x4::identity();
        c4x4::mul(&aa, &bb, &mut out);
        let ref_prod = a * b;
        for r in 0..4 {
            for c in 0..4 {
                assert_relative_eq!(out[r][c].re, ref_prod[(r, c)].re, epsilon = 1e-12);
                assert_relative_eq!(out[r][c].im, ref_prod[(r, c)].im, epsilon = 1e-12);
            }
        }
    }
}
