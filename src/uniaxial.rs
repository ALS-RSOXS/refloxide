//! Uniaxial 4x4 transfer matrix kernel.
//!
//! Port of `refloxide.pxr.tjf4x4.uniaxial_reflectivity`. The implementation
//! is streaming, holding only the previous-layer `(kz, Di)` snapshot across
//! iterations of the transfer chain, and parallel across q-points via
//! [`rayon`]. This module validates inputs and packs tensors into
//! [`LayerCoeffs`]; the per-q transfer chain itself lives in
//! [`crate::kernel`].

use nalgebra::Matrix3;
use num_complex::Complex;
use rayon::prelude::*;

use crate::error::{RefloxideError, Result};
use crate::kernel::{self, LayerCoeffs};

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

/// Batched polarized reflectance and transmission over energies and q-points.
#[derive(Debug, Clone)]
pub struct UniaxialBatchOutput {
    /// Power reflectance with shape `(n_energies, n_q, 2, 2)`.
    pub refl: Vec<Vec<[[f64; 2]; 2]>>,
    /// Complex amplitude transmission with the same layout.
    pub tran: Vec<Vec<[[C; 2]; 2]>>,
}

/// Computes polarized reflectance and transmission for a uniaxial multilayer.
///
/// # Parameters
/// - `q`: scattering wavevectors in `1/Angstrom`.
/// - `layers`: per-slab parameters, length `nlayers`. The first and last
///   entries describe the fronting and the backing.
/// - `tensor`: per-slab 3x3 dispersion tensor, length `nlayers`. The
///   Berreman dielectric is built as `eps = conj(I - 2 * tensor)`.
/// - `energy_ev`: photon energy in eV. Must be strictly positive.
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
    energy_ev: f64,
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
    if !energy_ev.is_finite() || energy_ev <= 0.0 {
        return Err(RefloxideError::InvalidEnergy(energy_ev));
    }

    let wl = HC_EV_ANGSTROM / energy_ev;
    let k0 = 2.0 * std::f64::consts::PI / wl;

    let coeffs = layer_coeffs(layers, tensor);

    let solve = |(i, qi): (usize, f64)| solve_q(qi, &coeffs, k0).map_err(|e| annotate(e, i));
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

/// Polarized reflectance only, for callers that discard transmission.
///
/// Same inputs, validation, and `(0, 0) = R_pp`, `(1, 1) = R_ss` packing as
/// [`uniaxial_reflectivity`], evaluated with the decoupled uniaxial-z
/// recursion ([`crate::kernel::solve_point_recursive`]). In this scope the
/// recursion equals the 4x4 chain in exact arithmetic and is more accurate
/// in floating point (about `1e-14` relative against a 50-digit reference,
/// versus `1e-11` for the transfer-matrix product), stable for thick
/// absorbing films, and several times faster. Cross-polarized entries are
/// exactly zero.
///
/// # Errors
/// Shape and energy errors as for [`uniaxial_reflectivity`]; the
/// recursion has no singular dynamic matrix.
pub fn uniaxial_reflectance(
    q: &[f64],
    layers: &[Layer],
    tensor: &[Matrix3<C>],
    energy_ev: f64,
    parallel: bool,
) -> Result<Vec<[[f64; 2]; 2]>> {
    if layers.len() != tensor.len() {
        return Err(RefloxideError::LayerCountMismatch {
            layers: layers.len(),
            tensor: tensor.len(),
        });
    }
    if layers.len() < 2 {
        return Err(RefloxideError::InsufficientLayers(layers.len()));
    }
    if !energy_ev.is_finite() || energy_ev <= 0.0 {
        return Err(RefloxideError::InvalidEnergy(energy_ev));
    }
    let k0 = wavenumber(energy_ev);
    let coeffs = layer_coeffs(layers, tensor);
    let solve = |&qi: &f64| kernel::solve_point_recursive(qi, k0, &coeffs);
    Ok(if parallel {
        q.par_iter().map(solve).collect()
    } else {
        q.iter().map(solve).collect()
    })
}

/// Computes polarized reflectance for many energies sharing one q-grid.
///
/// # Parameters
/// - `q`: scattering wavevectors in `1/Angstrom`, length `n_q`.
/// - `layers`: per-energy slab rows with shape conceptually `(n_E, N, 4)`.
/// - `tensor`: per-energy tensors with shape `(n_E, N, 3, 3)`.
/// - `energies_ev`: photon energies in eV, length `n_E`.
/// - `parallel`: distribute flattened `(energy, q)` pairs across rayon.
///
/// # Errors
/// Same contract as [`uniaxial_reflectivity`], with energy index included in
/// singularity messages.
pub fn uniaxial_reflectivity_batch(
    q: &[f64],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
    parallel: bool,
) -> Result<UniaxialBatchOutput> {
    validate_batch(layers, tensor, energies_ev)?;
    let n_e = energies_ev.len();

    let coeffs: Vec<Vec<LayerCoeffs<f64>>> = layers
        .iter()
        .zip(tensor)
        .map(|(l, t)| layer_coeffs(l, t))
        .collect();
    let k0: Vec<f64> = energies_ev.iter().map(|&e| wavenumber(e)).collect();

    let pairs: Vec<(usize, usize)> = (0..n_e)
        .flat_map(|ei| (0..q.len()).map(move |qi| (ei, qi)))
        .collect();

    let solve = |(ei, qi): (usize, usize)| {
        let result = solve_q(q[qi], &coeffs[ei], k0[ei]);
        result.map_err(|err| annotate_batch(err, ei, qi))
    };

    let solved: Vec<PolBlock> = if parallel {
        pairs
            .par_iter()
            .copied()
            .map(solve)
            .collect::<Result<Vec<_>>>()?
    } else {
        pairs
            .iter()
            .copied()
            .map(solve)
            .collect::<Result<Vec<_>>>()?
    };

    let n_q = q.len();
    let mut refl = vec![vec![[[0.0; 2]; 2]; n_q]; n_e];
    let mut tran = vec![vec![[[C::new(0.0, 0.0); 2]; 2]; n_q]; n_e];
    for (idx, (ei, qi)) in pairs.iter().enumerate() {
        let (r, t) = solved[idx];
        refl[*ei][*qi] = r;
        tran[*ei][*qi] = t;
    }

    Ok(UniaxialBatchOutput { refl, tran })
}

/// Polarized reflectance and transmission for independent `(q, stack)` points.
///
/// Generalizes [`uniaxial_reflectivity_batch`] to ragged q-grids: point `i`
/// is evaluated at `q[i]` against stack `stack_of[i]`, whose layers, tensors,
/// and photon energy are `layers[s]`, `tensor[s]`, and `energies_ev[s]`.
/// Typical use is one stack per photon energy with each energy's measured
/// q-grid (and per-polarization theta offsets) concatenated into `q`.
///
/// # Parameters
/// - `q`: scattering vector per point in `1/Angstrom`.
/// - `stack_of`: stack index per point, same length as `q`.
/// - `layers`, `tensor`, `energies_ev`: per-stack inputs as in
///   [`uniaxial_reflectivity_batch`].
/// - `parallel`: distribute points across rayon when true.
///
/// # Returns
/// [`UniaxialOutput`] with one entry per point, in input order.
///
/// # Errors
/// Same contract as [`uniaxial_reflectivity_batch`], plus
/// [`RefloxideError::InvalidShape`] for a length mismatch or out-of-range
/// stack index; singularity errors report the stack as `energy_index` and
/// the point as `q_index`.
pub fn uniaxial_reflectivity_points(
    q: &[f64],
    stack_of: &[usize],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
    parallel: bool,
) -> Result<UniaxialOutput> {
    validate_points(q, stack_of, layers, tensor, energies_ev)?;
    let coeffs: Vec<Vec<LayerCoeffs<f64>>> = layers
        .iter()
        .zip(tensor)
        .map(|(l, t)| layer_coeffs(l, t))
        .collect();
    let k0: Vec<f64> = energies_ev.iter().map(|&e| wavenumber(e)).collect();
    let solve = |i: usize| {
        let si = stack_of[i];
        solve_q(q[i], &coeffs[si], k0[si]).map_err(|err| annotate_batch(err, si, i))
    };
    let solved: Vec<PolBlock> = if parallel {
        (0..q.len())
            .into_par_iter()
            .map(solve)
            .collect::<Result<_>>()?
    } else {
        (0..q.len()).map(solve).collect::<Result<_>>()?
    };
    let (refl, tran) = solved.into_iter().unzip();
    Ok(UniaxialOutput { refl, tran })
}

/// Reflectance and Jacobian-vector products for `(q, stack)` points.
#[derive(Debug, Clone)]
pub struct PointsJvpOutput {
    /// `[R_pp, R_ss]` per point, length `n_points`.
    pub refl: Vec<[f64; 2]>,
    /// Row-major `(n_dirs, n_points)` derivatives `[dR_pp, dR_ss]`.
    pub jac: Vec<[f64; 2]>,
}

/// Direction-major tangent inputs for [`uniaxial_reflectivity_points_jvp`].
#[derive(Debug, Clone)]
pub struct PointsTangents {
    /// `(n_dirs, n_points)` tangent of `q`.
    pub dq: Vec<Vec<f64>>,
    /// `(n_dirs, n_stacks, n_layers)` tangent of the slab rows.
    pub dlayers: Vec<Vec<Vec<Layer>>>,
    /// `(n_dirs, n_stacks, n_layers)` tangent of the dispersion tensors.
    pub dtensor: Vec<Vec<Vec<Matrix3<C>>>>,
}

/// Reflectance and directional derivatives for independent `(q, stack)` points.
///
/// Evaluates the decoupled uniaxial-z recursion (equal to the 4x4 kernel in
/// this scope) with forward-mode derivatives, see
/// [`crate::kernel::solve_point_recursive_jvp`]. Direction `k` moves `q` by
/// `tangents.dq[k]` and the stack inputs by `tangents.dlayers[k]` /
/// `tangents.dtensor[k]`; tangents enter through the same linear packing as
/// the values, so any parameterization whose host-side Jacobian is known
/// (analytically or by finite differences of the `f64` materialization)
/// maps onto exact kernel derivatives.
///
/// # Parameters
/// - `q`, `stack_of`, `layers`, `tensor`, `energies_ev`: as in
///   [`uniaxial_reflectivity_points`].
/// - `tangents`: per-direction tangents with matching shapes.
/// - `parallel`: distribute `(direction, point)` pairs across rayon.
///
/// # Errors
/// Shape and energy errors as for [`uniaxial_reflectivity_points`], plus
/// [`RefloxideError::InvalidShape`] for tangent shape mismatches.
pub fn uniaxial_reflectivity_points_jvp(
    q: &[f64],
    stack_of: &[usize],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
    tangents: &PointsTangents,
    parallel: bool,
) -> Result<PointsJvpOutput> {
    let (coeffs, dcoeffs) = jvp_coeffs(q, stack_of, layers, tensor, energies_ev, tangents)?;
    let k0: Vec<f64> = energies_ev.iter().map(|&e| wavenumber(e)).collect();
    let n_points = q.len();
    let n_dirs = tangents.dq.len();
    let solve = |idx: usize| {
        let (dir, i) = (idx / n_points, idx % n_points);
        let si = stack_of[i];
        crate::kernel::solve_point_recursive_jvp(
            q[i],
            tangents.dq[dir][i],
            k0[si],
            &coeffs[si],
            &dcoeffs[dir][si],
        )
    };
    let total = n_dirs * n_points;
    let solved: Vec<([f64; 2], [f64; 2])> = if parallel {
        (0..total).into_par_iter().map(solve).collect()
    } else {
        (0..total).map(solve).collect()
    };
    let refl = if n_dirs == 0 {
        (0..n_points)
            .map(|i| {
                let r = crate::kernel::solve_point_recursive(
                    q[i],
                    k0[stack_of[i]],
                    &coeffs[stack_of[i]],
                );
                [r[0][0], r[1][1]]
            })
            .collect()
    } else {
        solved[..n_points].iter().map(|(r, _)| *r).collect()
    };
    Ok(PointsJvpOutput {
        refl,
        jac: solved.into_iter().map(|(_, d)| d).collect(),
    })
}

/// Validated value and tangent layer records for the JVP entry points.
#[allow(clippy::type_complexity)]
pub(crate) fn jvp_coeffs(
    q: &[f64],
    stack_of: &[usize],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
    tangents: &PointsTangents,
) -> Result<(Vec<Vec<LayerCoeffs<f64>>>, Vec<Vec<Vec<LayerCoeffs<f64>>>>)> {
    validate_points(q, stack_of, layers, tensor, energies_ev)?;
    let n_dirs = tangents.dq.len();
    let shape_err =
        |what: &str| RefloxideError::InvalidShape(format!("tangent {what} shape mismatch"));
    if tangents.dlayers.len() != n_dirs || tangents.dtensor.len() != n_dirs {
        return Err(shape_err("direction count"));
    }
    for dir in 0..n_dirs {
        if tangents.dq[dir].len() != q.len() {
            return Err(shape_err("dq"));
        }
        let (dl, dt) = (&tangents.dlayers[dir], &tangents.dtensor[dir]);
        if dl.len() != layers.len() || dt.len() != layers.len() {
            return Err(shape_err("stack count"));
        }
        for si in 0..layers.len() {
            if dl[si].len() != layers[si].len() || dt[si].len() != layers[si].len() {
                return Err(shape_err("layer count"));
            }
        }
    }
    let coeffs = layers
        .iter()
        .zip(tensor)
        .map(|(l, t)| layer_coeffs(l, t))
        .collect();
    let dcoeffs = tangents
        .dlayers
        .iter()
        .zip(&tangents.dtensor)
        .map(|(dl, dt)| dl.iter().zip(dt).map(|(l, t)| layer_coeffs(l, t)).collect())
        .collect();
    Ok((coeffs, dcoeffs))
}

/// Checks shared by the CPU and GPU point entry points.
pub(crate) fn validate_points(
    q: &[f64],
    stack_of: &[usize],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
) -> Result<()> {
    validate_batch(layers, tensor, energies_ev)?;
    if stack_of.len() != q.len() {
        return Err(RefloxideError::InvalidShape(format!(
            "stack_of length {} must match q length {}",
            stack_of.len(),
            q.len()
        )));
    }
    if let Some(&bad) = stack_of.iter().find(|&&s| s >= energies_ev.len()) {
        return Err(RefloxideError::InvalidShape(format!(
            "stack index {bad} out of range for {} stacks",
            energies_ev.len()
        )));
    }
    Ok(())
}

fn annotate(err: RefloxideError, q_index: usize) -> RefloxideError {
    match err {
        RefloxideError::SingularDynamicMatrix { layer, .. } => {
            RefloxideError::SingularDynamicMatrix {
                layer,
                q_index,
                energy_index: None,
            }
        }
        other => other,
    }
}

fn annotate_batch(err: RefloxideError, energy_index: usize, q_index: usize) -> RefloxideError {
    match err {
        RefloxideError::SingularDynamicMatrix { layer, .. } => {
            RefloxideError::SingularDynamicMatrix {
                layer,
                q_index,
                energy_index: Some(energy_index),
            }
        }
        other => other,
    }
}

fn solve_q(qi: f64, coeffs: &[LayerCoeffs<f64>], k0: f64) -> Result<PolBlock> {
    kernel::solve_point(qi, k0, coeffs)
}

/// Shape and energy checks shared by the batched CPU and GPU entry points.
pub(crate) fn validate_batch(
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
) -> Result<()> {
    let n_e = energies_ev.len();
    if layers.len() != n_e || tensor.len() != n_e {
        return Err(RefloxideError::InvalidShape(format!(
            "layers and tensor batch length must match energies_ev ({n_e}), got layers={}, tensor={}",
            layers.len(),
            tensor.len()
        )));
    }
    if n_e == 0 {
        return Err(RefloxideError::InvalidShape(
            "uniaxial_reflectivity_batch requires at least one energy".into(),
        ));
    }
    for (ei, energy_ev) in energies_ev.iter().enumerate() {
        if !energy_ev.is_finite() || *energy_ev <= 0.0 {
            return Err(RefloxideError::InvalidEnergy(*energy_ev));
        }
        if layers[ei].len() != tensor[ei].len() {
            return Err(RefloxideError::LayerCountMismatch {
                layers: layers[ei].len(),
                tensor: tensor[ei].len(),
            });
        }
        if layers[ei].len() < 2 {
            return Err(RefloxideError::InsufficientLayers(layers[ei].len()));
        }
    }
    Ok(())
}

/// Vacuum wavenumber `2 pi / lambda` in `1/Angstrom` for a photon energy in eV.
pub(crate) fn wavenumber(energy_ev: f64) -> f64 {
    2.0 * std::f64::consts::PI / (HC_EV_ANGSTROM / energy_ev)
}

pub(crate) fn layer_coeffs(layers: &[Layer], tensor: &[Matrix3<C>]) -> Vec<LayerCoeffs<f64>> {
    layers
        .iter()
        .zip(tensor)
        .map(|(layer, t)| LayerCoeffs {
            chi_o: berreman_susceptibility(t[(0, 0)]),
            chi_e: berreman_susceptibility(t[(2, 2)]),
            thickness: layer.thickness,
            sigma: layer.sigma,
        })
        .collect()
}

/// Diagonal susceptibility `eps - 1 = conj(-2 t)` of `eps = conj(I - 2 t)`.
fn berreman_susceptibility(t: C) -> C {
    (t * C::new(-2.0, 0.0)).conj()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn batch_matches_sequential_uniaxial() {
        use nalgebra::Vector3;

        let q = vec![0.02, 0.05, 0.1];
        let layers_a = vec![
            Layer::new(0.0, 0.0, 0.0, 0.0),
            Layer::new(100.0, 1e-6, 1e-8, 2.0),
            Layer::new(0.0, 2e-6, 0.0, 0.0),
        ];
        let n = Complex::new(1.0 - 1e-6, -1e-8);
        let tensor_a = vec![
            Matrix3::from_diagonal(&Vector3::new(n, n, n)),
            Matrix3::from_diagonal(&Vector3::new(
                Complex::new(1.0 - 2e-6, -2e-8),
                Complex::new(1.0 - 2e-6, -2e-8),
                Complex::new(1.0 - 3e-6, -3e-8),
            )),
            Matrix3::from_diagonal(&Vector3::new(
                Complex::new(1.0 - 2e-6, 0.0),
                Complex::new(1.0 - 2e-6, 0.0),
                Complex::new(1.0 - 2e-6, 0.0),
            )),
        ];
        let energies_ev = [250.0, 284.4];
        let single: Vec<_> = energies_ev
            .iter()
            .map(|&energy_ev| {
                uniaxial_reflectivity(&q, &layers_a, &tensor_a, energy_ev, false).unwrap()
            })
            .collect();
        let batch = uniaxial_reflectivity_batch(
            &q,
            &[layers_a.clone(), layers_a],
            &[tensor_a.clone(), tensor_a],
            &energies_ev,
            false,
        )
        .unwrap();
        for (ei, one) in single.iter().enumerate() {
            for qi in 0..q.len() {
                for r in 0..2 {
                    for c in 0..2 {
                        assert_relative_eq!(
                            batch.refl[ei][qi][r][c],
                            one.refl[qi][r][c],
                            epsilon = 1e-12
                        );
                    }
                }
            }
        }
    }
}
