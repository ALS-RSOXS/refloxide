//! Minimal Rust port of `refloxide.pxr.tjf4x4.uniaxial_reflectivity` exposed
//! through PyO3 as `refloxide.rust.uniaxial_reflectivity`.
//!
//! The kernel mirrors the 4x4 transfer matrix pipeline of the Python
//! reference. Per layer the ordinary and extraordinary `kz` and their
//! polarization eigenvectors `(D, H)` are written in closed form for a
//! uniaxial dielectric with optic axis along z, the dynamic matrix D, its
//! inverse, the Berreman propagation matrix P, and the Nevot-Croce W are
//! assembled, and the layer transfer is chained over all interior slabs.
//! The Berreman extraction yields `r_kl` and `t_kl` for `(s, p)` pairings.
//! Outputs are returned as `numpy.ndarray` with shape `(numpnts, 2, 2)`
//! so that downstream Python code can keep its existing indexing.

use nalgebra::{Matrix3, Matrix4, Vector3};
use num_complex::Complex;

#[cfg(feature = "python")]
use numpy::ndarray::Array3;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Complex alias used throughout the kernel.
type C = Complex<f64>;

/// Photon energy to wavelength conversion constant in eV * Angstroms.
const HC_EV_ANGSTROM: f64 = 12_398.4193;

/// Bundle returned by the core kernel.
#[derive(Debug, Clone)]
pub struct UniaxialOutput {
    /// Power reflectance with layout `[q][k][l]` matching Python `refl[:, k, l]`,
    /// where `(k, l) = (0, 0)` is r_ss, `(1, 1)` is r_pp, `(0, 1)` is r_sp,
    /// `(1, 0)` is r_ps.
    pub refl: Vec<[[f64; 2]; 2]>,
    /// Complex amplitude transmission with the same indexing convention.
    pub tran: Vec<[[C; 2]; 2]>,
}

/// Computes the polarized reflectance and transmission of a uniaxial
/// multilayer at a single photon energy.
///
/// # Parameters
/// - `q`: scattering wavevectors in `1/Angstrom`.
/// - `layers`: per-layer rows `[d, sld_real, sld_imag, sigma]`, where the
///   first and last rows describe the fronting and the backing. Thickness
///   of the fronting and backing rows is ignored.
/// - `tensor`: per-layer 3x3 dispersion tensor of length `layers.len()`.
///   The lab-frame dielectric is built as `eps = conj(I - 2 * tensor)`,
///   matching the Python convention so that diagonal entries
///   `delta + i beta` recover the Henke-style linearization
///   `n = 1 - delta + i beta` along each principal axis.
/// - `energy`: photon energy in eV.
pub fn uniaxial_reflectivity_core(
    q: &[f64],
    layers: &[[f64; 4]],
    tensor: &[Matrix3<C>],
    energy: f64,
) -> UniaxialOutput {
    assert_eq!(
        layers.len(),
        tensor.len(),
        "layers and tensor must agree on layer count"
    );
    assert!(
        layers.len() >= 2,
        "stack requires at least fronting and backing rows"
    );

    let nlayers = layers.len();
    let numpnts = q.len();
    let one = C::new(1.0, 0.0);
    let two = C::new(2.0, 0.0);
    let optic_z = one;

    let wl = HC_EV_ANGSTROM / energy;
    let k0 = 2.0 * std::f64::consts::PI / wl;
    let k0sq = C::new(k0 * k0, 0.0);

    // eps_layer = conj(I - 2 * tensor_layer).
    let eps: Vec<Matrix3<C>> = tensor
        .iter()
        .map(|t| {
            let scaled = t.map(|v| v * two);
            let mut m = Matrix3::<C>::identity() - scaled;
            for v in m.iter_mut() {
                *v = v.conj();
            }
            m
        })
        .collect();

    // theta = pi/2 - asin(q / 2k0). phi = 0, so kx = k0 sin(theta), ky = 0.
    let mut kx = vec![0.0_f64; numpnts];
    let ky = vec![0.0_f64; numpnts];
    for (i, &qi) in q.iter().enumerate() {
        let s = (qi / (2.0 * k0)).clamp(-1.0, 1.0);
        let theta = std::f64::consts::FRAC_PI_2 - s.asin();
        kx[i] = k0 * theta.sin();
    }

    // Allocations for per (q, layer) eigenstructure. Mode order is
    // [extraord+, extraord-, ord+, ord-] to match the reference layout.
    let mut kz = vec![vec![[C::new(0.0, 0.0); 4]; nlayers]; numpnts];
    let mut dpol = vec![vec![[Vector3::<C>::zeros(); 4]; nlayers]; numpnts];
    let mut hpol = vec![vec![[Vector3::<C>::zeros(); 4]; nlayers]; numpnts];

    for j in 0..nlayers {
        let e_o = eps[j][(0, 0)];
        let e_e = eps[j][(2, 2)];
        let nu = (e_e - e_o) / e_o;

        for i in 0..numpnts {
            let kpar2 = C::new(kx[i] * kx[i] + ky[i] * ky[i], 0.0);
            let kz_ord = (e_o * k0sq - kpar2).sqrt();
            // na = 1, la = 0 (optic axis along z, kvec in x-z plane).
            let radicand = e_o * k0sq * (one + nu) * (one + nu) - kpar2 * (one + nu);
            let kz_ext = radicand.sqrt() / (one + nu);

            kz[i][j][0] = kz_ext;
            kz[i][j][1] = -kz_ext;
            kz[i][j][2] = kz_ord;
            kz[i][j][3] = -kz_ord;

            for s in 0..4 {
                let kvec = Vector3::new(C::new(kx[i], 0.0), C::new(ky[i], 0.0), kz[i][j][s]);
                let kdotk = kvec.x * kvec.x + kvec.y * kvec.y + kvec.z * kvec.z;
                let kmag = kdotk.sqrt();
                let knorm =
                    Vector3::new(kvec.x / kmag, kvec.y / kmag, kvec.z / kmag);
                let kpol = knorm.z * optic_z;

                let dvec = if s >= 2 {
                    // Ordinary: opticaxis x knorm.
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
                let dnorm = Vector3::new(
                    dvec.x * inv_norm,
                    dvec.y * inv_norm,
                    dvec.z * inv_norm,
                );
                dpol[i][j][s] = dnorm;

                let h = kvec.cross(&dnorm);
                let inv_k0 = 1.0 / k0;
                hpol[i][j][s] = Vector3::new(h.x * inv_k0, h.y * inv_k0, h.z * inv_k0);
            }
        }
    }

    // Dynamic matrix D[(numpnts, nlayers, 4, 4)] and its inverse.
    let mut d_all = vec![vec![Matrix4::<C>::zeros(); nlayers]; numpnts];
    let mut di_all = vec![vec![Matrix4::<C>::zeros(); nlayers]; numpnts];
    for i in 0..numpnts {
        for j in 0..nlayers {
            let mut d = Matrix4::<C>::zeros();
            for s in 0..4 {
                d[(0, s)] = dpol[i][j][s].x;
                d[(1, s)] = hpol[i][j][s].y;
                d[(2, s)] = dpol[i][j][s].y;
                d[(3, s)] = hpol[i][j][s].x;
            }
            let di = d.try_inverse().unwrap_or_else(|| Matrix4::<C>::zeros());
            d_all[i][j] = d;
            di_all[i][j] = di;
        }
    }

    // Propagation matrix diag(exp(-i kz d)).
    let mut p_all = vec![vec![Matrix4::<C>::zeros(); nlayers]; numpnts];
    for i in 0..numpnts {
        for j in 0..nlayers {
            let dj = layers[j][0];
            let mut p = Matrix4::<C>::zeros();
            for s in 0..4 {
                let arg = C::new(0.0, -1.0) * kz[i][j][s] * C::new(dj, 0.0);
                p[(s, s)] = arg.exp();
            }
            p_all[i][j] = p;
        }
    }

    // Nevot-Croce W matrix mirroring np.roll(kz, 1, axis=1) on the second array.
    let mut w_all = vec![vec![Matrix4::<C>::zeros(); nlayers]; numpnts];
    for i in 0..numpnts {
        for j in 0..nlayers {
            let prev = if j == 0 { nlayers - 1 } else { j - 1 };
            let r = layers[j][3];
            let r2_half = C::new(r * r * 0.5, 0.0);
            let mut eplus = [C::new(0.0, 0.0); 4];
            let mut eminus = [C::new(0.0, 0.0); 4];
            for s in 0..4 {
                let plus = kz[i][j][s] + kz[i][prev][s];
                let minus = kz[i][j][s] - kz[i][prev][s];
                eplus[s] = (-plus * plus * r2_half).exp();
                eminus[s] = (-minus * minus * r2_half).exp();
            }
            let mut w = Matrix4::<C>::zeros();
            for row in 0..4 {
                for col in 0..4 {
                    w[(row, col)] = if (row + col) % 2 == 0 {
                        eminus[col]
                    } else {
                        eplus[col]
                    };
                }
            }
            w_all[i][j] = w;
        }
    }

    // Transfer matrix product: M = prod_{j=1..N-1} (Di[j-1] D[j] o W[j]) P[j]
    // then right-multiplied by (Di[N-2] D[N-1] o W[N-1]).
    let mut m_full = vec![Matrix4::<C>::identity(); numpnts];
    for i in 0..numpnts {
        let mut m = Matrix4::<C>::identity();
        for j in 1..nlayers - 1 {
            let a = di_all[i][j - 1] * d_all[i][j];
            let b = a.component_mul(&w_all[i][j]);
            let c = b * p_all[i][j];
            m = m * c;
        }
        let aa = di_all[i][nlayers - 2] * d_all[i][nlayers - 1];
        let bb = aa.component_mul(&w_all[i][nlayers - 1]);
        m_full[i] = m * bb;
    }

    // Berreman extraction of `r_kl`, `t_kl` from the assembled 4x4 product.
    let mut refl = vec![[[0.0_f64; 2]; 2]; numpnts];
    let mut tran = vec![[[C::new(0.0, 0.0); 2]; 2]; numpnts];
    for i in 0..numpnts {
        let m = &m_full[i];
        let mut denom = m[(0, 0)] * m[(2, 2)] - m[(0, 2)] * m[(2, 0)];
        if denom.norm() < f64::EPSILON {
            denom += C::new(f64::EPSILON, 0.0);
        }
        let r_ss = (m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)]) / denom;
        let r_sp = (m[(3, 0)] * m[(2, 2)] - m[(3, 2)] * m[(2, 0)]) / denom;
        let r_ps = (m[(0, 0)] * m[(1, 2)] - m[(1, 0)] * m[(0, 2)]) / denom;
        let r_pp = (m[(0, 0)] * m[(3, 2)] - m[(3, 0)] * m[(0, 2)]) / denom;
        let t_ss = m[(2, 2)] / denom;
        let t_sp = -m[(2, 0)] / denom;
        let t_ps = -m[(0, 2)] / denom;
        let t_pp = m[(0, 0)] / denom;

        refl[i][0][0] = r_ss.norm_sqr().min(1.0);
        refl[i][0][1] = r_sp.norm_sqr().min(1.0);
        refl[i][1][0] = r_ps.norm_sqr().min(1.0);
        refl[i][1][1] = r_pp.norm_sqr().min(1.0);
        tran[i][0][0] = t_ss;
        tran[i][0][1] = t_sp;
        tran[i][1][0] = t_ps;
        tran[i][1][1] = t_pp;
    }

    UniaxialOutput { refl, tran }
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (q, layers, tensor, energy))]
fn uniaxial_reflectivity<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    layers: PyReadonlyArray2<'py, f64>,
    tensor: PyReadonlyArray3<'py, C>,
    energy: f64,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<C>>)> {
    let q_view = q.as_array();
    let layers_view = layers.as_array();
    let tensor_view = tensor.as_array();

    let nlayers = layers_view.nrows();
    if layers_view.ncols() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "layers must have shape (N, 4)",
        ));
    }
    if tensor_view.shape() != [nlayers, 3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tensor must have shape (N, 3, 3) with the same N as layers",
        ));
    }

    let q_vec: Vec<f64> = q_view.iter().copied().collect();
    let mut layers_rust: Vec<[f64; 4]> = Vec::with_capacity(nlayers);
    for i in 0..nlayers {
        layers_rust.push([
            layers_view[[i, 0]],
            layers_view[[i, 1]],
            layers_view[[i, 2]],
            layers_view[[i, 3]],
        ]);
    }
    let mut tensor_rust: Vec<Matrix3<C>> = Vec::with_capacity(nlayers);
    for i in 0..nlayers {
        let mut m = Matrix3::<C>::zeros();
        for r in 0..3 {
            for c in 0..3 {
                m[(r, c)] = tensor_view[[i, r, c]];
            }
        }
        tensor_rust.push(m);
    }

    let out = uniaxial_reflectivity_core(&q_vec, &layers_rust, &tensor_rust, energy);
    let numpnts = out.refl.len();

    let mut refl_arr = Array3::<f64>::zeros((numpnts, 2, 2));
    let mut tran_arr = Array3::<C>::zeros((numpnts, 2, 2));
    for i in 0..numpnts {
        for r in 0..2 {
            for c in 0..2 {
                refl_arr[[i, r, c]] = out.refl[i][r][c];
                tran_arr[[i, r, c]] = out.tran[i][r][c];
            }
        }
    }

    Ok((refl_arr.into_pyarray(py), tran_arr.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uniaxial_reflectivity, m)?)?;
    Ok(())
}
