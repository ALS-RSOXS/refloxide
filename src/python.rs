//! PyO3 bindings for the refloxide kernel.
//!
//! Exposes [`crate::uniaxial::uniaxial_reflectivity`] as a Python function on
//! the `refloxide.rust` module. The GIL is released during the kernel call so
//! that rayon can actually parallelize across the available threads.

use nalgebra::Matrix3;
use num_complex::Complex;
use numpy::ndarray::{Array2, Array3, Array4};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4, PyReadonlyArrayDyn,
};
use pyo3::prelude::*;

use crate::bookended::{bookended_uniaxial_reflectivity as core_bookended, BookendedParams};
use crate::error::{RefloxideError, Result};
use crate::optics::{
    interpolate_ooc_linear, isotropic_tensor, lab_diagonal_uniaxial_batch, pack_diagonal_tensors,
};
use crate::sld::{
    molecular_index_at_ooc as core_molecular_index_at_ooc,
    tensor_to_slab_row as core_tensor_to_slab_row, uniaxial_lab_tensor as core_uniaxial_lab_tensor,
};
use crate::general_stack::general_reflectivity as core_general_solve;
use crate::uniaxial::{
    uniaxial_reflectivity as core_solve, uniaxial_reflectivity_batch as core_solve_batch, Layer,
};

type C = Complex<f64>;

type UniaxialPyArrays<'py> = (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<C>>);

type UniaxialBatchPyArrays<'py> = (Bound<'py, PyArray4<f64>>, Bound<'py, PyArray4<C>>);

type PointsJvpPyArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<f64>>);

type UnpackedInputs = (Vec<f64>, Vec<Layer>, Vec<Matrix3<C>>);

type UnpackedBatchInputs = (Vec<f64>, Vec<Vec<Layer>>, Vec<Vec<Matrix3<C>>>, Vec<f64>);

/// Computes uniaxial reflection and transmission.
///
/// See [`crate::uniaxial::uniaxial_reflectivity`] for the contract. The
/// numpy arrays are copied into owned Rust buffers before the kernel runs,
/// after which the GIL is released for the duration of the solve.
///
/// `parallel` defaults to `True`. Callers driving the function from a Python
/// fitting routine that is itself multi-threaded or multi-process (for
/// example refnx fitters with worker pools, emcee with parallel walkers, or
/// `multiprocessing.Pool`) should pass `parallel=False` to keep the rayon
/// pool from oversubscribing the CPU. The same effect can be obtained
/// process-wide by setting the environment variable `RAYON_NUM_THREADS=1`
/// before importing `refloxide`.
///
/// `device` selects `"cpu"` (default; `f64` 4x4 with transmission) or
/// `"gpu"` (`f32` recursion on the shared GPU context, `NaN` transmission);
/// see [`crate::gpu::uniaxial_reflectivity_batch`]. `parallel` is ignored on
/// the GPU.
#[pyfunction]
#[pyo3(signature = (q, layers, tensor, energy_ev, parallel = true, device = "cpu"))]
fn uniaxial_reflectivity<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    layers: PyReadonlyArray2<'py, f64>,
    tensor: PyReadonlyArray3<'py, C>,
    energy_ev: f64,
    parallel: bool,
    device: &str,
) -> PyResult<UniaxialPyArrays<'py>> {
    let use_gpu = parse_device(device)?;
    let (q_vec, layers_rust, tensor_rust) =
        unpack_inputs(&q, &layers, &tensor).map_err(PyErr::from)?;

    if use_gpu {
        let out = py
            .detach(|| {
                gpu_batch(
                    &q_vec,
                    std::slice::from_ref(&layers_rust),
                    std::slice::from_ref(&tensor_rust),
                    &[energy_ev],
                )
            })
            .map_err(PyErr::from)?;
        let single = crate::uniaxial::UniaxialOutput {
            refl: out.refl.into_iter().next().unwrap_or_default(),
            tran: out.tran.into_iter().next().unwrap_or_default(),
        };
        return pack_uniaxial_output(py, &single);
    }

    let out = py
        .detach(|| core_solve(&q_vec, &layers_rust, &tensor_rust, energy_ev, parallel))
        .map_err(PyErr::from)?;
    pack_uniaxial_output(py, &out)
}

/// Batched uniaxial reflectivity over shared ``q`` and many energies.
///
/// `device` behaves as in [`uniaxial_reflectivity`].
#[pyfunction]
#[pyo3(signature = (q, layers, tensor, energies_ev, parallel = true, device = "cpu"))]
fn uniaxial_reflectivity_batch<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    layers: PyReadonlyArray3<'py, f64>,
    tensor: PyReadonlyArray4<'py, C>,
    energies_ev: PyReadonlyArray1<'py, f64>,
    parallel: bool,
    device: &str,
) -> PyResult<UniaxialBatchPyArrays<'py>> {
    let use_gpu = parse_device(device)?;
    let (q_vec, layers_rust, tensor_rust, energies_rust) =
        unpack_batch_inputs(&q, &layers, &tensor, &energies_ev).map_err(PyErr::from)?;

    if use_gpu {
        let out = py
            .detach(|| gpu_batch(&q_vec, &layers_rust, &tensor_rust, &energies_rust))
            .map_err(PyErr::from)?;
        return pack_uniaxial_batch_output(py, &out);
    }

    let out = py
        .detach(|| core_solve_batch(&q_vec, &layers_rust, &tensor_rust, &energies_rust, parallel))
        .map_err(PyErr::from)?;
    pack_uniaxial_batch_output(py, &out)
}

/// Uniaxial reflectivity for independent `(q, stack)` points.
///
/// See [`crate::uniaxial::uniaxial_reflectivity_points`]; `stack_index`
/// must be non-negative. `device` behaves as in [`uniaxial_reflectivity`],
/// with the whole call issued as one GPU dispatch.
#[pyfunction]
#[pyo3(signature = (q, stack_index, layers, tensor, energies_ev, parallel = true, device = "cpu"))]
#[allow(clippy::too_many_arguments)]
fn uniaxial_reflectivity_points<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    stack_index: PyReadonlyArray1<'py, i64>,
    layers: PyReadonlyArray3<'py, f64>,
    tensor: PyReadonlyArray4<'py, C>,
    energies_ev: PyReadonlyArray1<'py, f64>,
    parallel: bool,
    device: &str,
) -> PyResult<UniaxialPyArrays<'py>> {
    let use_gpu = parse_device(device)?;
    let (q_vec, layers_rust, tensor_rust, energies_rust) =
        unpack_batch_inputs(&q, &layers, &tensor, &energies_ev).map_err(PyErr::from)?;
    let stack_of = stack_index
        .as_array()
        .iter()
        .map(|&s| usize::try_from(s))
        .collect::<std::result::Result<Vec<usize>, _>>()
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("stack_index entries must be non-negative")
        })?;

    let out = py
        .detach(|| {
            if use_gpu {
                gpu_points(
                    &q_vec,
                    &stack_of,
                    &layers_rust,
                    &tensor_rust,
                    &energies_rust,
                )
            } else {
                crate::uniaxial::uniaxial_reflectivity_points(
                    &q_vec,
                    &stack_of,
                    &layers_rust,
                    &tensor_rust,
                    &energies_rust,
                    parallel,
                )
            }
        })
        .map_err(PyErr::from)?;
    pack_uniaxial_output(py, &out)
}

#[cfg(feature = "gpu")]
fn gpu_points(
    q: &[f64],
    stack_of: &[usize],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
) -> Result<crate::uniaxial::UniaxialOutput> {
    crate::gpu::uniaxial_reflectivity_points(q, stack_of, layers, tensor, energies_ev)
}

#[cfg(not(feature = "gpu"))]
fn gpu_points(
    _q: &[f64],
    _stack_of: &[usize],
    _layers: &[Vec<Layer>],
    _tensor: &[Vec<Matrix3<C>>],
    _energies_ev: &[f64],
) -> Result<crate::uniaxial::UniaxialOutput> {
    Err(RefloxideError::Gpu(
        "refloxide was built without the `gpu` feature".into(),
    ))
}

/// Reflectance and Jacobian-vector products for independent `(q, stack)` points.
///
/// See [`crate::uniaxial::uniaxial_reflectivity_points_jvp`]. Returns
/// `(refl, jac)` with `refl` shaped `(n_points, 2)` holding `[R_pp, R_ss]`
/// and `jac` shaped `(n_dirs, n_points, 2)` holding `[dR_pp, dR_ss]`.
#[pyfunction]
#[pyo3(signature = (q, stack_index, layers, tensor, energies_ev, dq, dlayers, dtensor, parallel = true, device = "cpu"))]
#[allow(clippy::too_many_arguments)]
fn uniaxial_reflectivity_points_jvp<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    stack_index: PyReadonlyArray1<'py, i64>,
    layers: PyReadonlyArray3<'py, f64>,
    tensor: PyReadonlyArray4<'py, C>,
    energies_ev: PyReadonlyArray1<'py, f64>,
    dq: PyReadonlyArray2<'py, f64>,
    dlayers: PyReadonlyArray4<'py, f64>,
    dtensor: PyReadonlyArrayDyn<'py, C>,
    parallel: bool,
    device: &str,
) -> PyResult<PointsJvpPyArrays<'py>> {
    let use_gpu = parse_device(device)?;
    let (q_vec, layers_rust, tensor_rust, energies_rust) =
        unpack_batch_inputs(&q, &layers, &tensor, &energies_ev).map_err(PyErr::from)?;
    let stack_of = stack_index
        .as_array()
        .iter()
        .map(|&s| usize::try_from(s))
        .collect::<std::result::Result<Vec<usize>, _>>()
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("stack_index entries must be non-negative")
        })?;
    let tangents = unpack_tangents(&dq, &dlayers, &dtensor, &layers_rust).map_err(PyErr::from)?;
    let n_points = q_vec.len();
    let n_dirs = tangents.dq.len();

    let out = py
        .detach(|| {
            if use_gpu {
                gpu_points_jvp(
                    &q_vec,
                    &stack_of,
                    &layers_rust,
                    &tensor_rust,
                    &energies_rust,
                    &tangents,
                )
            } else {
                crate::uniaxial::uniaxial_reflectivity_points_jvp(
                    &q_vec,
                    &stack_of,
                    &layers_rust,
                    &tensor_rust,
                    &energies_rust,
                    &tangents,
                    parallel,
                )
            }
        })
        .map_err(PyErr::from)?;
    let mut refl = Array2::<f64>::zeros((n_points, 2));
    for (i, r) in out.refl.iter().enumerate() {
        refl[[i, 0]] = r[0];
        refl[[i, 1]] = r[1];
    }
    let mut jac = Array3::<f64>::zeros((n_dirs, n_points, 2));
    for (idx, d) in out.jac.iter().enumerate() {
        let (k, i) = (idx / n_points.max(1), idx % n_points.max(1));
        jac[[k, i, 0]] = d[0];
        jac[[k, i, 1]] = d[1];
    }
    Ok((refl.into_pyarray(py), jac.into_pyarray(py)))
}

/// Validates tangent shapes against the value stacks and copies them.
fn unpack_tangents(
    dq: &PyReadonlyArray2<'_, f64>,
    dlayers: &PyReadonlyArray4<'_, f64>,
    dtensor: &PyReadonlyArrayDyn<'_, C>,
    layers: &[Vec<Layer>],
) -> Result<crate::uniaxial::PointsTangents> {
    let dq_view = dq.as_array();
    let dl_view = dlayers.as_array();
    let dt_view = dtensor.as_array();
    let n_dirs = dq_view.shape()[0];
    let n_stacks = layers.len();
    let n_layers = layers.first().map_or(0, Vec::len);
    if dl_view.shape() != [n_dirs, n_stacks, n_layers, 4] {
        return Err(RefloxideError::InvalidShape(format!(
            "dlayers must have shape (n_dirs, n_stacks, N, 4) = ({n_dirs}, {n_stacks}, {n_layers}, 4), got {:?}",
            dl_view.shape()
        )));
    }
    if dt_view.shape() != [n_dirs, n_stacks, n_layers, 3, 3] {
        return Err(RefloxideError::InvalidShape(format!(
            "dtensor must have shape (n_dirs, n_stacks, N, 3, 3), got {:?}",
            dt_view.shape()
        )));
    }
    let dq_rows = (0..n_dirs).map(|k| dq_view.row(k).to_vec()).collect();
    let mut dlayers_rust = Vec::with_capacity(n_dirs);
    let mut dtensor_rust = Vec::with_capacity(n_dirs);
    for k in 0..n_dirs {
        let mut ls = Vec::with_capacity(n_stacks);
        let mut ts = Vec::with_capacity(n_stacks);
        for si in 0..n_stacks {
            let mut lrow = Vec::with_capacity(n_layers);
            let mut trow = Vec::with_capacity(n_layers);
            for li in 0..n_layers {
                lrow.push(Layer::new(
                    dl_view[[k, si, li, 0]],
                    dl_view[[k, si, li, 1]],
                    dl_view[[k, si, li, 2]],
                    dl_view[[k, si, li, 3]],
                ));
                let mut m = Matrix3::<C>::zeros();
                for r in 0..3 {
                    for c in 0..3 {
                        m[(r, c)] = dt_view[[k, si, li, r, c]];
                    }
                }
                trow.push(m);
            }
            ls.push(lrow);
            ts.push(trow);
        }
        dlayers_rust.push(ls);
        dtensor_rust.push(ts);
    }
    Ok(crate::uniaxial::PointsTangents {
        dq: dq_rows,
        dlayers: dlayers_rust,
        dtensor: dtensor_rust,
    })
}

#[cfg(feature = "gpu")]
fn gpu_points_jvp(
    q: &[f64],
    stack_of: &[usize],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
    tangents: &crate::uniaxial::PointsTangents,
) -> Result<crate::uniaxial::PointsJvpOutput> {
    crate::gpu::uniaxial_reflectivity_points_jvp(q, stack_of, layers, tensor, energies_ev, tangents)
}

#[cfg(not(feature = "gpu"))]
fn gpu_points_jvp(
    _q: &[f64],
    _stack_of: &[usize],
    _layers: &[Vec<Layer>],
    _tensor: &[Vec<Matrix3<C>>],
    _energies_ev: &[f64],
    _tangents: &crate::uniaxial::PointsTangents,
) -> Result<crate::uniaxial::PointsJvpOutput> {
    Err(RefloxideError::Gpu(
        "refloxide was built without the `gpu` feature".into(),
    ))
}

/// Maps the `device` keyword to `true` for GPU and `false` for CPU.
fn parse_device(device: &str) -> PyResult<bool> {
    match device {
        "cpu" => Ok(false),
        "gpu" => Ok(true),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "device must be 'cpu' or 'gpu', got {other:?}"
        ))),
    }
}

#[cfg(feature = "gpu")]
fn gpu_batch(
    q: &[f64],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<C>>],
    energies_ev: &[f64],
) -> Result<crate::uniaxial::UniaxialBatchOutput> {
    crate::gpu::uniaxial_reflectivity_batch(q, layers, tensor, energies_ev)
}

#[cfg(not(feature = "gpu"))]
fn gpu_batch(
    _q: &[f64],
    _layers: &[Vec<Layer>],
    _tensor: &[Vec<Matrix3<C>>],
    _energies_ev: &[f64],
) -> Result<crate::uniaxial::UniaxialBatchOutput> {
    Err(RefloxideError::Gpu(
        "refloxide was built without the `gpu` feature".into(),
    ))
}

/// Validates batch input shapes and copies them into owned Rust buffers.
fn unpack_batch_inputs(
    q: &PyReadonlyArray1<'_, f64>,
    layers: &PyReadonlyArray3<'_, f64>,
    tensor: &PyReadonlyArray4<'_, C>,
    energies_ev: &PyReadonlyArray1<'_, f64>,
) -> Result<UnpackedBatchInputs> {
    let layers_view = layers.as_array();
    let tensor_view = tensor.as_array();
    let n_e = layers_view.shape()[0];
    let nlayers = layers_view.shape()[1];
    if layers_view.ndim() != 3 || layers_view.shape()[2] != 4 {
        return Err(RefloxideError::InvalidShape(format!(
            "layers must have shape (n_E, N, 4), got {:?}",
            layers_view.shape()
        )));
    }
    if tensor_view.shape() != [n_e, nlayers, 3, 3] {
        return Err(RefloxideError::InvalidShape(format!(
            "tensor must have shape (n_E, N, 3, 3), got {:?}",
            tensor_view.shape()
        )));
    }
    let energies_vec: Vec<f64> = energies_ev.as_array().iter().copied().collect();
    if energies_vec.len() != n_e {
        return Err(RefloxideError::InvalidShape(format!(
            "energies_ev length {n} must match layers batch {n_e}",
            n = energies_vec.len()
        )));
    }

    let q_vec: Vec<f64> = q.as_array().iter().copied().collect();
    let mut layers_rust: Vec<Vec<Layer>> = Vec::with_capacity(n_e);
    let mut tensor_rust: Vec<Vec<Matrix3<C>>> = Vec::with_capacity(n_e);
    for ei in 0..n_e {
        let mut layer_rows = Vec::with_capacity(nlayers);
        let mut tensor_rows = Vec::with_capacity(nlayers);
        for li in 0..nlayers {
            layer_rows.push(Layer::new(
                layers_view[[ei, li, 0]],
                layers_view[[ei, li, 1]],
                layers_view[[ei, li, 2]],
                layers_view[[ei, li, 3]],
            ));
            let mut matrix = Matrix3::<C>::zeros();
            for r in 0..3 {
                for c in 0..3 {
                    matrix[(r, c)] = tensor_view[[ei, li, r, c]];
                }
            }
            tensor_rows.push(matrix);
        }
        layers_rust.push(layer_rows);
        tensor_rust.push(tensor_rows);
    }

    Ok((q_vec, layers_rust, tensor_rust, energies_vec))
}

/// Validates input shapes and copies them into owned Rust buffers.
fn unpack_inputs(
    q: &PyReadonlyArray1<'_, f64>,
    layers: &PyReadonlyArray2<'_, f64>,
    tensor: &PyReadonlyArray3<'_, C>,
) -> Result<UnpackedInputs> {
    let layers_view = layers.as_array();
    let tensor_view = tensor.as_array();
    let nlayers = layers_view.nrows();

    if layers_view.ncols() != 4 {
        return Err(RefloxideError::InvalidShape(format!(
            "layers must have shape (N, 4), got (N, {})",
            layers_view.ncols()
        )));
    }
    if tensor_view.shape() != [nlayers, 3, 3] {
        return Err(RefloxideError::InvalidShape(format!(
            "tensor must have shape (N, 3, 3) matching layers, got {:?}",
            tensor_view.shape()
        )));
    }

    let q_vec: Vec<f64> = q.as_array().iter().copied().collect();
    let layers_rust: Vec<Layer> = (0..nlayers)
        .map(|i| {
            Layer::new(
                layers_view[[i, 0]],
                layers_view[[i, 1]],
                layers_view[[i, 2]],
                layers_view[[i, 3]],
            )
        })
        .collect();
    let tensor_rust: Vec<Matrix3<C>> = (0..nlayers)
        .map(|i| {
            let mut m = Matrix3::<C>::zeros();
            for r in 0..3 {
                for c in 0..3 {
                    m[(r, c)] = tensor_view[[i, r, c]];
                }
            }
            m
        })
        .collect();

    Ok((q_vec, layers_rust, tensor_rust))
}

/// Linear OOC interpolation at one photon energy (eV).
#[pyfunction]
fn interp_ooc_linear<'py>(
    py: Python<'py>,
    energy_ev: PyReadonlyArray1<'py, f64>,
    n_xx: PyReadonlyArray1<'py, f64>,
    n_ixx: PyReadonlyArray1<'py, f64>,
    n_zz: PyReadonlyArray1<'py, f64>,
    n_izz: PyReadonlyArray1<'py, f64>,
    query_ev: f64,
) -> [f64; 4] {
    let _ = py;
    interpolate_ooc_linear(
        &energy_ev.as_array().to_vec(),
        &n_xx.as_array().to_vec(),
        &n_ixx.as_array().to_vec(),
        &n_zz.as_array().to_vec(),
        &n_izz.as_array().to_vec(),
        query_ev,
    )
}

/// Batch laboratory `(n_o, n_o, n_e)` diagonals from uniaxial molecular constants.
#[pyfunction]
fn lab_tensor_diagonals_batch<'py>(
    py: Python<'py>,
    n_mol_xx: C,
    n_mol_zz: C,
    orientations_rad: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray3<C>> {
    let diags =
        lab_diagonal_uniaxial_batch(n_mol_xx, n_mol_zz, &orientations_rad.as_array().to_vec());
    let packed = pack_diagonal_tensors(&diags);
    let n = packed.len();
    let mut arr = Array3::<C>::zeros((n, 3, 3));
    for (i, layer) in packed.iter().enumerate() {
        for r in 0..3 {
            for c in 0..3 {
                arr[[i, r, c]] = layer[r][c];
            }
        }
    }
    arr.into_pyarray(py)
}

/// Isotropic `(3, 3)` tensor for one scalar index of refraction.
#[pyfunction]
fn isotropic_lab_tensor(n: C) -> [[C; 3]; 3] {
    isotropic_tensor(n)
}

/// Density-scaled molecular indices from a linear OOC table at one energy (eV).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn molecular_index_at_ooc<'py>(
    py: Python<'py>,
    energy_ev: PyReadonlyArray1<'py, f64>,
    n_xx: PyReadonlyArray1<'py, f64>,
    n_ixx: PyReadonlyArray1<'py, f64>,
    n_zz: PyReadonlyArray1<'py, f64>,
    n_izz: PyReadonlyArray1<'py, f64>,
    query_ev: f64,
    density: f64,
) -> (C, C) {
    let _ = py;
    core_molecular_index_at_ooc(
        &energy_ev.as_array().to_vec(),
        &n_xx.as_array().to_vec(),
        &n_ixx.as_array().to_vec(),
        &n_zz.as_array().to_vec(),
        &n_izz.as_array().to_vec(),
        query_ev,
        density,
    )
}

/// Laboratory `(3, 3)` tensor for one uniaxial orientation (radians).
#[pyfunction]
fn uniaxial_lab_tensor(n_mol_xx: C, n_mol_zz: C, orientation_rad: f64) -> [[C; 3]; 3] {
    core_uniaxial_lab_tensor(n_mol_xx, n_mol_zz, orientation_rad)
}

/// Refnx slab row ``[d, delta, beta, sigma]`` from a laboratory `(3, 3)` tensor.
#[pyfunction]
fn tensor_to_slab_row(thickness: f64, roughness: f64, tensor: [[C; 3]; 3]) -> [f64; 4] {
    core_tensor_to_slab_row(thickness, roughness, &tensor)
}

/// Fused book-ended graded film + substrate stack reflectivity (GIL released).
#[pyfunction]
#[pyo3(signature = (
    q,
    energy_ev,
    n_xx,
    n_ixx,
    n_zz,
    n_izz,
    query_ev,
    wavelength_ev,
    total_thick,
    surface_roughness,
    tau_si,
    tau_vac,
    alpha_bulk,
    alpha_si,
    alpha_vac,
    density_bulk,
    density_si,
    density_vac,
    num_slabs,
    mesh_constant,
    fronting,
    backing,
    parallel = false,
    with_transmission = true,
))]
#[allow(clippy::too_many_arguments)]
fn bookended_uniaxial_reflectivity<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    energy_ev: PyReadonlyArray1<'py, f64>,
    n_xx: PyReadonlyArray1<'py, f64>,
    n_ixx: PyReadonlyArray1<'py, f64>,
    n_zz: PyReadonlyArray1<'py, f64>,
    n_izz: PyReadonlyArray1<'py, f64>,
    query_ev: f64,
    wavelength_ev: f64,
    total_thick: f64,
    surface_roughness: f64,
    tau_si: f64,
    tau_vac: f64,
    alpha_bulk: f64,
    alpha_si: f64,
    alpha_vac: f64,
    density_bulk: f64,
    density_si: f64,
    density_vac: f64,
    num_slabs: usize,
    mesh_constant: f64,
    fronting: [f64; 4],
    backing: PyReadonlyArray2<'py, f64>,
    parallel: bool,
    with_transmission: bool,
) -> PyResult<UniaxialPyArrays<'py>> {
    let backing_view = backing.as_array();
    if backing_view.ncols() != 4 {
        return Err(PyErr::from(RefloxideError::InvalidShape(format!(
            "backing must have shape (N, 4), got (N, {})",
            backing_view.ncols()
        ))));
    }
    let backing_rows: Vec<[f64; 4]> = (0..backing_view.nrows())
        .map(|i| {
            [
                backing_view[[i, 0]],
                backing_view[[i, 1]],
                backing_view[[i, 2]],
                backing_view[[i, 3]],
            ]
        })
        .collect();
    let params = BookendedParams {
        total_thick,
        surface_roughness,
        tau_si,
        tau_vac,
        alpha_bulk,
        alpha_si,
        alpha_vac,
        density_bulk,
        density_si,
        density_vac,
        num_slabs,
        mesh_constant,
    };
    let q_vec: Vec<f64> = q.as_array().iter().copied().collect();
    let e_vec = energy_ev.as_array().to_vec();
    let n_xx_v = n_xx.as_array().to_vec();
    let n_ixx_v = n_ixx.as_array().to_vec();
    let n_zz_v = n_zz.as_array().to_vec();
    let n_izz_v = n_izz.as_array().to_vec();

    let out = py
        .detach(|| {
            if with_transmission {
                return core_bookended(
                    &q_vec,
                    &e_vec,
                    &n_xx_v,
                    &n_ixx_v,
                    &n_zz_v,
                    &n_izz_v,
                    query_ev,
                    wavelength_ev,
                    &params,
                    fronting,
                    &backing_rows,
                    parallel,
                );
            }
            let refl = crate::bookended::bookended_uniaxial_reflectance(
                &q_vec,
                &e_vec,
                &n_xx_v,
                &n_ixx_v,
                &n_zz_v,
                &n_izz_v,
                query_ev,
                wavelength_ev,
                &params,
                fronting,
                &backing_rows,
                parallel,
            )?;
            let nan = C::new(f64::NAN, f64::NAN);
            let tran = vec![[[nan; 2]; 2]; refl.len()];
            Ok(crate::uniaxial::UniaxialOutput { refl, tran })
        })
        .map_err(PyErr::from)?;

    pack_uniaxial_output(py, &out)
}

fn pack_uniaxial_output<'py>(
    py: Python<'py>,
    out: &crate::uniaxial::UniaxialOutput,
) -> PyResult<UniaxialPyArrays<'py>> {
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

fn pack_uniaxial_batch_output<'py>(
    py: Python<'py>,
    out: &crate::uniaxial::UniaxialBatchOutput,
) -> PyResult<UniaxialBatchPyArrays<'py>> {
    let n_e = out.refl.len();
    let n_q = if n_e == 0 { 0 } else { out.refl[0].len() };
    let mut refl_arr = Array4::<f64>::zeros((n_e, n_q, 2, 2));
    let mut tran_arr = Array4::<C>::zeros((n_e, n_q, 2, 2));
    for ei in 0..n_e {
        for qi in 0..n_q {
            for r in 0..2 {
                for c in 0..2 {
                    refl_arr[[ei, qi, r, c]] = out.refl[ei][qi][r][c];
                    tran_arr[[ei, qi, r, c]] = out.tran[ei][qi][r][c];
                }
            }
        }
    }
    Ok((refl_arr.into_pyarray(py), tran_arr.into_pyarray(py)))
}

/// General anisotropic reflectivity (includes cross-polarized channels).
///
/// See [`crate::general_stack::general_reflectivity`]. Returns power
/// reflectance only (no transmission), shape ``(n_q, 2, 2)`` with
/// ``[:,0,0]=R_pp``, ``[:,0,1]=R_sp``, ``[:,1,0]=R_ps``, ``[:,1,1]=R_ss``.
#[pyfunction]
#[pyo3(signature = (q, layers, tensor, energy_ev, parallel = true))]
fn general_reflectivity<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    layers: PyReadonlyArray2<'py, f64>,
    tensor: PyReadonlyArray3<'py, C>,
    energy_ev: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let (q_vec, layers_rust, tensor_rust) =
        unpack_inputs(&q, &layers, &tensor).map_err(PyErr::from)?;
    let refl = py
        .detach(|| {
            core_general_solve(&q_vec, &layers_rust, &tensor_rust, energy_ev, parallel)
        })
        .map_err(PyErr::from)?;
    let numpnts = refl.len();
    let mut refl_arr = Array3::<f64>::zeros((numpnts, 2, 2));
    for i in 0..numpnts {
        for r in 0..2 {
            for c in 0..2 {
                refl_arr[[i, r, c]] = refl[i][r][c];
            }
        }
    }
    Ok(refl_arr.into_pyarray(py))
}

/// Registers the Python-facing module `refloxide.rust`.
#[pymodule]
pub fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uniaxial_reflectivity, m)?)?;
    m.add_function(wrap_pyfunction!(uniaxial_reflectivity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(uniaxial_reflectivity_points, m)?)?;
    m.add_function(wrap_pyfunction!(uniaxial_reflectivity_points_jvp, m)?)?;
    m.add_function(wrap_pyfunction!(general_reflectivity, m)?)?;
    m.add_function(wrap_pyfunction!(bookended_uniaxial_reflectivity, m)?)?;
    m.add_function(wrap_pyfunction!(interp_ooc_linear, m)?)?;
    m.add_function(wrap_pyfunction!(lab_tensor_diagonals_batch, m)?)?;
    m.add_function(wrap_pyfunction!(isotropic_lab_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(molecular_index_at_ooc, m)?)?;
    m.add_function(wrap_pyfunction!(uniaxial_lab_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_to_slab_row, m)?)?;
    Ok(())
}
