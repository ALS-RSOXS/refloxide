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

use crate::error::{RefloxideError, Result};
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

/// Registers the Python-facing module `refloxide.rust`.
#[pymodule]
pub fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uniaxial_reflectivity, m)?)?;
    Ok(())
}
