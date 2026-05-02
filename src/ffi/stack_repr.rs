//! Deserialize [`crate::stack::Stack`] from Python ``dict`` mappings.

use crate::material::builder::Material;
use crate::material::rotation::EulerAnglesZxz;
use crate::stack::roughness_spec::{ProfileShape, RoughnessSpec};
use crate::stack::{Layer, Stack};
use crate::types::parameterization::{OpticalIndex, PrincipalTensor};
use crate::types::scalar::C64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyComplex, PyComplexMethods, PyDict, PyDictMethods, PyList};

pub(crate) fn stack_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Stack> {
    let dict = obj.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("stack_repr must be a dict with incident, layers, substrate")
    })?;
    let incident = layer_from_py(
        dict.get_item("incident")?
            .ok_or_else(|| PyValueError::new_err("missing incident"))?
            .as_ref(),
        true,
    )?;
    let layers_any = dict
        .get_item("layers")?
        .ok_or_else(|| PyValueError::new_err("missing layers"))?;
    let layers_list = layers_any.downcast::<PyList>().map_err(|_| {
        PyValueError::new_err("layers must be a list of layer dicts")
    })?;
    let mut layers = Vec::with_capacity(layers_list.len());
    for item in layers_list.iter() {
        layers.push(layer_from_py(item.as_ref(), false)?);
    }
    let substrate = layer_from_py(
        dict.get_item("substrate")?
            .ok_or_else(|| PyValueError::new_err("missing substrate"))?
            .as_ref(),
        true,
    )?;
    let n_if = layers.len() + 1;
    let roughness = if let Some(r) = dict.get_item("roughness")? {
        roughness_vec_from_py(r.as_ref(), n_if)?
    } else {
        vec![RoughnessSpec::sharp(); n_if]
    };
    if roughness.len() != n_if {
        return Err(PyValueError::new_err(format!(
            "roughness must have length {} (interfaces), got {}",
            n_if,
            roughness.len()
        )));
    }
    Ok(Stack {
        incident,
        layers,
        substrate,
        roughness,
    })
}

fn py_to_c64(obj: &Bound<'_, PyAny>) -> PyResult<C64> {
    if let Ok(x) = obj.extract::<f64>() {
        if x.is_finite() {
            return Ok(C64::new(x, 0.0));
        }
        return Err(PyValueError::new_err("expected finite scalar"));
    }
    if let Ok([re, im]) = obj.extract::<[f64; 2]>() {
        return Ok(C64::new(re, im));
    }
    if let Ok(c) = obj.downcast::<PyComplex>() {
        return Ok(C64::new(c.real(), c.imag()));
    }
    Err(PyValueError::new_err(
        "expected complex number, length-2 sequence, or float",
    ))
}

fn optical_index_from_py(obj: &Bound<'_, PyAny>) -> PyResult<OpticalIndex> {
    let dict = obj.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("optical index must be a dict with one variant key")
    })?;
    if let Some(inner) = dict.get_item("nk")? {
        let d = inner.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("nk optical index requires an inner dict")
        })?;
        let n = py_to_c64(
            d.get_item("n")?
                .ok_or_else(|| PyValueError::new_err("nk.n"))?
                .as_ref(),
        )?;
        let k = py_to_c64(
            d.get_item("k")?
                .ok_or_else(|| PyValueError::new_err("nk.k"))?
                .as_ref(),
        )?;
        return Ok(OpticalIndex::NK { n, k });
    }
    if let Some(inner) = dict.get_item("delta_beta")? {
        let d = inner.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("delta_beta optical index requires an inner dict")
        })?;
        let delta = py_to_c64(
            d.get_item("delta")?
                .ok_or_else(|| PyValueError::new_err("delta_beta.delta"))?
                .as_ref(),
        )?;
        let beta = py_to_c64(
            d.get_item("beta")?
                .ok_or_else(|| PyValueError::new_err("delta_beta.beta"))?
                .as_ref(),
        )?;
        return Ok(OpticalIndex::DeltaBeta { delta, beta });
    }
    if let Some(inner) = dict.get_item("sld")? {
        let d = inner.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("sld optical index requires an inner dict")
        })?;
        let rho = py_to_c64(
            d.get_item("rho")?
                .ok_or_else(|| PyValueError::new_err("sld.rho"))?
                .as_ref(),
        )?;
        let wavelength_m = d
            .get_item("wavelength_m")?
            .ok_or_else(|| PyValueError::new_err("sld.wavelength_m"))?
            .extract::<f64>()?;
        return Ok(OpticalIndex::Sld { rho, wavelength_m });
    }
    if let Some(inner) = dict.get_item("scattering_factor")? {
        let d = inner.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("scattering_factor optical index requires an inner dict")
        })?;
        let f1 = py_to_c64(
            d.get_item("f1")?
                .ok_or_else(|| PyValueError::new_err("scattering_factor.f1"))?
                .as_ref(),
        )?;
        let f2 = py_to_c64(
            d.get_item("f2")?
                .ok_or_else(|| PyValueError::new_err("scattering_factor.f2"))?
                .as_ref(),
        )?;
        let number_density_per_m3 = d
            .get_item("number_density_per_m3")?
            .ok_or_else(|| PyValueError::new_err("scattering_factor.number_density_per_m3"))?
            .extract::<f64>()?;
        let wavelength_m = d
            .get_item("wavelength_m")?
            .ok_or_else(|| PyValueError::new_err("scattering_factor.wavelength_m"))?
            .extract::<f64>()?;
        return Ok(OpticalIndex::ScatteringFactor {
            f1,
            f2,
            number_density_per_m3,
            wavelength_m,
        });
    }
    if let Some(inner) = dict.get_item("epsilon")? {
        return Ok(OpticalIndex::Epsilon(py_to_c64(inner.as_ref())?));
    }
    Err(PyValueError::new_err(
        "unknown optical index variant (expected nk, delta_beta, sld, scattering_factor, epsilon)",
    ))
}

fn principal_tensor_from_py(obj: &Bound<'_, PyAny>) -> PyResult<PrincipalTensor> {
    let dict = obj.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("principal tensor must be a dict with xx, yy, zz")
    })?;
    Ok(PrincipalTensor {
        xx: optical_index_from_py(
            dict.get_item("xx")?
                .ok_or_else(|| PyValueError::new_err("principal.xx"))?
                .as_ref(),
        )?,
        yy: optical_index_from_py(
            dict.get_item("yy")?
                .ok_or_else(|| PyValueError::new_err("principal.yy"))?
                .as_ref(),
        )?,
        zz: optical_index_from_py(
            dict.get_item("zz")?
                .ok_or_else(|| PyValueError::new_err("principal.zz"))?
                .as_ref(),
        )?,
    })
}

fn euler_from_py(obj: &Bound<'_, PyAny>) -> PyResult<EulerAnglesZxz> {
    if let Ok([a, b, c]) = obj.extract::<[f64; 3]>() {
        return Ok(EulerAnglesZxz::from_radians(a, b, c));
    }
    let dict = obj
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("euler_zxz_rad must be [phi,theta,psi] or dict"))?;
    let phi = dict
        .get_item("phi")?
        .ok_or_else(|| PyValueError::new_err("euler phi"))?
        .extract::<f64>()?;
    let theta = dict
        .get_item("theta")?
        .ok_or_else(|| PyValueError::new_err("euler theta"))?
        .extract::<f64>()?;
    let psi = dict
        .get_item("psi")?
        .ok_or_else(|| PyValueError::new_err("euler psi"))?
        .extract::<f64>()?;
    Ok(EulerAnglesZxz::from_radians(phi, theta, psi))
}

fn material_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Material> {
    let dict = obj.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("material must be a dict with epsilon_principal")
    })?;
    let epsilon_principal = principal_tensor_from_py(
        dict.get_item("epsilon_principal")?
            .ok_or_else(|| PyValueError::new_err("material.epsilon_principal"))?
            .as_ref(),
    )?;
    let mu_principal = match dict.get_item("mu_principal")? {
        Some(x) => Some(principal_tensor_from_py(x.as_ref())?),
        None => None,
    };
    let euler_zxz_rad = match dict.get_item("euler_zxz_rad")? {
        Some(x) => euler_from_py(x.as_ref())?,
        None => EulerAnglesZxz::zero(),
    };
    Ok(Material {
        epsilon_principal,
        mu_principal,
        euler_zxz_rad,
    })
}

fn layer_from_py(obj: &Bound<'_, PyAny>, allow_non_finite_thickness: bool) -> PyResult<Layer> {
    let dict = obj
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("layer must be a dict"))?;
    let thickness_nm = dict
        .get_item("thickness_nm")?
        .ok_or_else(|| PyValueError::new_err("layer.thickness_nm"))?
        .extract::<f64>()?;
    if !allow_non_finite_thickness && !(thickness_nm.is_finite() && thickness_nm > 0.0) {
        return Err(PyValueError::new_err(
            "interior layer thickness_nm must be finite and positive",
        ));
    }
    if allow_non_finite_thickness && !(thickness_nm.is_finite() || thickness_nm.is_infinite()) {
        return Err(PyValueError::new_err(
            "cladding thickness_nm must be finite or inf",
        ));
    }
    let material = material_from_py(
        dict.get_item("material")?
            .ok_or_else(|| PyValueError::new_err("layer.material"))?
            .as_ref(),
    )?;
    Ok(Layer {
        thickness_nm,
        material,
    })
}

fn profile_from_str(s: &str) -> PyResult<ProfileShape> {
    match s.to_ascii_lowercase().as_str() {
        "gaussian" => Ok(ProfileShape::Gaussian),
        "linear" => Ok(ProfileShape::Linear),
        "sine" => Ok(ProfileShape::Sine),
        "tanh_sech2" | "tanhsech2" => Ok(ProfileShape::TanhSech2),
        _ => Err(PyValueError::new_err(
            "profile must be gaussian, linear, sine, or tanh_sech2",
        )),
    }
}

fn roughness_from_py(obj: &Bound<'_, PyAny>) -> PyResult<RoughnessSpec> {
    let dict = obj.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("each roughness entry must be a dict")
    })?;
    let model = dict
        .get_item("model")?
        .ok_or_else(|| PyValueError::new_err("roughness.model"))?;
    let model_s: String = model.extract()?;
    let profile = match dict.get_item("profile")? {
        Some(p) => profile_from_str(p.extract::<String>()?.as_str())?,
        None => ProfileShape::Gaussian,
    };
    match model_s.to_ascii_lowercase().as_str() {
        "sharp" => Ok(RoughnessSpec::sharp()),
        "nevot_croce" | "nevot-croce" => {
            let sigma_nm = dict
                .get_item("sigma_nm")?
                .ok_or_else(|| PyValueError::new_err("nevot_croce.sigma_nm"))?
                .extract::<f64>()?;
            Ok(RoughnessSpec::nevot_croce(sigma_nm))
        }
        "debye_waller" | "debye-waller" => {
            let sigma_nm = dict
                .get_item("sigma_nm")?
                .ok_or_else(|| PyValueError::new_err("debye_waller.sigma_nm"))?
                .extract::<f64>()?;
            let correlation_length_nm = dict
                .get_item("correlation_length_nm")?
                .ok_or_else(|| PyValueError::new_err("debye_waller.correlation_length_nm"))?
                .extract::<f64>()?;
            Ok(RoughnessSpec::debye_waller(sigma_nm, correlation_length_nm))
        }
        "graded" => {
            let sigma_nm = dict
                .get_item("sigma_nm")?
                .ok_or_else(|| PyValueError::new_err("graded.sigma_nm"))?
                .extract::<f64>()?;
            let sublayer_count = dict
                .get_item("sublayer_count")?
                .ok_or_else(|| PyValueError::new_err("graded.sublayer_count"))?
                .extract::<usize>()?;
            Ok(RoughnessSpec::graded(sigma_nm, profile, sublayer_count))
        }
        "auto" => {
            let sigma_nm = dict
                .get_item("sigma_nm")?
                .ok_or_else(|| PyValueError::new_err("auto.sigma_nm"))?
                .extract::<f64>()?;
            let correlation_length_nm = dict
                .get_item("correlation_length_nm")?
                .ok_or_else(|| PyValueError::new_err("auto.correlation_length_nm"))?
                .extract::<f64>()?;
            Ok(RoughnessSpec::auto(sigma_nm, correlation_length_nm))
        }
        _ => Err(PyValueError::new_err(
            "roughness.model must be sharp, nevot_croce, debye_waller, graded, or auto",
        )),
    }
}

fn roughness_vec_from_py(obj: &Bound<'_, PyAny>, expected: usize) -> PyResult<Vec<RoughnessSpec>> {
    let list = obj.downcast::<PyList>().map_err(|_| {
        PyValueError::new_err("roughness must be a list of roughness dicts")
    })?;
    if list.len() != expected {
        return Err(PyValueError::new_err(format!(
            "roughness list length {} != expected {}",
            list.len(),
            expected
        )));
    }
    let mut out = Vec::with_capacity(expected);
    for item in list.iter() {
        out.push(roughness_from_py(item.as_ref())?);
    }
    Ok(out)
}
