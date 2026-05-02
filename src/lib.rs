use pyo3::prelude::*;

#[pyfunction]
fn hello_from_rust() -> String {
    "Hello from refloxide core".to_string()
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hello_from_rust, module)?)?;
    Ok(())
}
