//! PyO3 bindings for the refloxide kernel.
//!
//! Exposes [`crate::uniaxial::uniaxial_reflectivity`] as a Python function on
//! the `refloxide.rust` module. The GIL is released during the kernel call so
//! that rayon can actually parallelize across the available threads.

use nalgebra::Matrix3;
use num_complex::Complex;
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::bookended::{bookended_uniaxial_reflectivity as core_bookended, BookendedParams};
use crate::error::{RefloxideError, Result};
use crate::optics::{
    interpolate_ooc_linear, isotropic_tensor, lab_diagonal_uniaxial_batch, pack_diagonal_tensors,
};
use crate::uniaxial::{uniaxial_reflectivity as core_solve, Layer};

type C = Complex<f64>;

type UniaxialPyArrays<'py> = (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<C>>);

type UnpackedInputs = (Vec<f64>, Vec<Layer>, Vec<Matrix3<C>>);

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
#[pyfunction]
#[pyo3(signature = (q, layers, tensor, energy, parallel = true))]
fn uniaxial_reflectivity<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    layers: PyReadonlyArray2<'py, f64>,
    tensor: PyReadonlyArray3<'py, C>,
    energy: f64,
    parallel: bool,
) -> PyResult<UniaxialPyArrays<'py>> {
    let (q_vec, layers_rust, tensor_rust) =
        unpack_inputs(&q, &layers, &tensor).map_err(PyErr::from)?;

    let out = py
        .detach(|| core_solve(&q_vec, &layers_rust, &tensor_rust, energy, parallel))
        .map_err(PyErr::from)?;
    pack_uniaxial_output(py, &out)
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
    let diags = lab_diagonal_uniaxial_batch(
        n_mol_xx,
        n_mol_zz,
        &orientations_rad.as_array().to_vec(),
    );
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
))]
fn bookended_uniaxial_reflectivity<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    energy_ev: PyReadonlyArray1<'py, f64>,
    n_xx: PyReadonlyArray1<'py, f64>,
    n_ixx: PyReadonlyArray1<'py, f64>,
    n_zz: PyReadonlyArray1<'py, f64>,
    n_izz: PyReadonlyArray1<'py, f64>,
    query_ev: f64,
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
            core_bookended(
                &q_vec,
                &e_vec,
                &n_xx_v,
                &n_ixx_v,
                &n_zz_v,
                &n_izz_v,
                query_ev,
                &params,
                fronting,
                &backing_rows,
                parallel,
            )
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

/// Registers the Python-facing module `refloxide.rust`.
#[pymodule]
pub fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uniaxial_reflectivity, m)?)?;
    m.add_function(wrap_pyfunction!(bookended_uniaxial_reflectivity, m)?)?;
    m.add_function(wrap_pyfunction!(interp_ooc_linear, m)?)?;
    m.add_function(wrap_pyfunction!(lab_tensor_diagonals_batch, m)?)?;
    m.add_function(wrap_pyfunction!(isotropic_lab_tensor, m)?)?;
    Ok(())
}
