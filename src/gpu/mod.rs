//! Portable GPU backend for the decoupled uniaxial recursion (feature `gpu`).
//!
//! Dispatches [`crate::kernel::solve_point_recursive`] as a WGSL compute
//! shader through `wgpu`, which targets Metal (Apple), Vulkan (NVIDIA, AMD,
//! Intel, Linux/Windows) and DX12 (Windows) from one source. Arithmetic is
//! single precision; the recursion's cancellation-free formulation keeps
//! relative reflectance error near `1e-5` against the double-precision 4x4
//! reference (see `examples/kernel_precision.rs`).
//!
//! [`GpuContext::reflectivity_points_general`] additionally runs the general
//! dielectric-tensor 4x4 engine ([`crate::kernel::solve_point_general`]).
//! It needs double precision to be accurate for tilted and biaxial layers,
//! which requires native `f64` shaders (wgpu `SHADER_F64`: Vulkan and DX12
//! adapters, not Metal); see [`GpuContext::supports_f64`].
//!
//! This module does not provide amplitude transmission, and it does not
//! validate physics inputs beyond buffer shapes. A [`GpuContext`] owns one
//! device and compiled pipelines and should be created once and reused:
//! device acquisition and shader compilation cost far more than a dispatch.

use std::sync::{mpsc, OnceLock};

use bytemuck::{Pod, Zeroable};
use nalgebra::Matrix3;
use num_complex::Complex;
use wgpu::util::DeviceExt;

use crate::error::{RefloxideError, Result};
use crate::kernel::{GeneralLayer, LayerCoeffs};
use crate::uniaxial::{
    jvp_coeffs, layer_coeffs, validate_batch, validate_points, wavenumber, Layer, PointsJvpOutput,
    PointsTangents, UniaxialBatchOutput, UniaxialOutput,
};

const WORKGROUP_SIZE: u32 = 64;

/// Double-single arithmetic library prepended to df64 shaders.
const DF64_WGSL: &str = include_str!("df64.wgsl");

/// General-tensor 4x4 shader body and its per-precision preludes.
const GENERAL_WGSL: &str = include_str!("general.wgsl");
const GENERAL_F32_WGSL: &str = include_str!("general_f32.wgsl");
const GENERAL_F64_WGSL: &str = include_str!("general_f64.wgsl");

/// Operation probed by [`GpuContext::df64_selftest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Df64Op {
    /// `a + b`.
    Add,
    /// `a * b`.
    Mul,
    /// `a / b`.
    Div,
    /// `sqrt(a)`.
    Sqrt,
    /// `exp(a)`.
    Exp,
    /// `sin(a)`.
    Sin,
    /// `cos(a)`.
    Cos,
    /// `ln(a)`.
    Log,
    /// Error term of the f32 two-sum of `a` and `b` (reassociation canary:
    /// identically zero if the compiler folds `(a - (s - v)) + (b - v)`).
    TwoSumError,
}

/// Numeric precision of the general-tensor shader build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneralPrecision {
    /// Native `f64` shader arithmetic; requires [`GpuContext::supports_f64`].
    F64,
    /// Single precision with hardware transcendentals. Runs on every
    /// adapter, but loses 4-48% of reflectance on tilted and biaxial stacks
    /// (see `examples/kernel_precision.rs`); intended for cross-checking the
    /// shader logic against the CPU kernel, not for fitting.
    F32,
}

/// Reals per general-shader layer record: 18 for `chi`, thickness, sigma.
const GENERAL_STRIDE: usize = 20;

/// Splits an `f64` into the `(hi, lo)` f32 pair of its df64 representation.
fn to_df64(x: f64) -> [f32; 2] {
    let hi = x as f32;
    [hi, (x - f64::from(hi)) as f32]
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLayer {
    chi_o: [f32; 2],
    chi_e: [f32; 2],
    thickness: f32,
    sigma: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n_points: u32,
    n_layers: u32,
    n_stacks: u32,
    row_pitch: u32,
}

/// df64 layer record: complex values as `[re.hi, re.lo, im.hi, im.lo]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLayerDf64 {
    chi_o: [f32; 4],
    chi_e: [f32; 4],
    thickness: [f32; 2],
    sigma: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct JvpParams {
    n_points: u32,
    n_layers: u32,
    n_stacks: u32,
    row_pitch: u32,
    n_dirs: u32,
    _pad: [u32; 3],
}

/// Device, queue, and compiled recursion pipelines for one GPU adapter.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    jvp_pipeline: wgpu::ComputePipeline,
    df64_selftest_pipeline: wgpu::ComputePipeline,
    df64_pipeline: wgpu::ComputePipeline,
    general_f32_pipeline: OnceLock<wgpu::ComputePipeline>,
    general_f64_pipeline: OnceLock<wgpu::ComputePipeline>,
    has_f64: bool,
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
}

fn compile_pipeline(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn gpu_layers(layers: &[LayerCoeffs<f32>]) -> Vec<GpuLayer> {
    layers
        .iter()
        .map(|l| GpuLayer {
            chi_o: [l.chi_o.re, l.chi_o.im],
            chi_e: [l.chi_e.re, l.chi_e.im],
            thickness: l.thickness,
            sigma: l.sigma,
        })
        .collect()
}

fn to_u32(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| RefloxideError::InvalidShape(format!("{what} exceeds u32")))
}

fn check_points(
    n_points: usize,
    stack_of: &[u32],
    n_stacks: usize,
    n_records: usize,
    n_layers: usize,
) -> Result<()> {
    if n_layers < 2 {
        return Err(RefloxideError::InsufficientLayers(n_layers));
    }
    if n_records != n_stacks * n_layers {
        return Err(RefloxideError::InvalidShape(format!(
            "layers buffer length {n_records} != n_stacks ({n_stacks}) * n_layers ({n_layers})"
        )));
    }
    if stack_of.len() != n_points {
        return Err(RefloxideError::InvalidShape(format!(
            "stack_of length {} != q length {n_points}",
            stack_of.len()
        )));
    }
    if let Some(&bad) = stack_of.iter().find(|&&s| s as usize >= n_stacks) {
        return Err(RefloxideError::InvalidShape(format!(
            "stack index {bad} out of range for {n_stacks} stacks"
        )));
    }
    Ok(())
}

impl GpuContext {
    /// Acquires the highest-performance adapter available and compiles the shaders.
    ///
    /// Backend selection honors the `WGPU_BACKEND` environment variable
    /// (for example `vulkan`, `metal`, `dx12`) through `wgpu`'s defaults.
    ///
    /// # Errors
    /// [`RefloxideError::Gpu`] when no adapter or device can be obtained.
    pub fn new() -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| RefloxideError::Gpu(format!("no compatible adapter: {e}")))?;
        let limits = adapter.limits();
        let has_f64 = adapter.features().contains(wgpu::Features::SHADER_F64);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("refloxide"),
            required_features: if has_f64 {
                wgpu::Features::SHADER_F64
            } else {
                wgpu::Features::empty()
            },
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| RefloxideError::Gpu(format!("device request failed: {e}")))?;
        let compile = |label: &str, source: &str| compile_pipeline(&device, label, source);
        let pipeline = compile("refloxide-recursive", include_str!("recursive.wgsl"));
        let jvp_pipeline = compile(
            "refloxide-recursive-jvp",
            include_str!("recursive_jvp.wgsl"),
        );
        let df64_selftest_pipeline = compile(
            "refloxide-df64-selftest",
            &[DF64_WGSL, include_str!("df64_selftest.wgsl")].concat(),
        );
        let df64_pipeline = compile(
            "refloxide-recursive-df64",
            &[DF64_WGSL, include_str!("recursive_df64.wgsl")].concat(),
        );
        Ok(Self {
            device,
            queue,
            pipeline,
            jvp_pipeline,
            df64_selftest_pipeline,
            df64_pipeline,
            general_f32_pipeline: OnceLock::new(),
            general_f64_pipeline: OnceLock::new(),
            has_f64,
            info: adapter.get_info(),
            limits,
        })
    }

    /// Human-readable adapter name and backend, for logs and benchmarks.
    pub fn describe(&self) -> String {
        format!("{} ({:?})", self.info.name, self.info.backend)
    }

    /// Whether the adapter provides native `f64` shader arithmetic.
    ///
    /// `true` on Vulkan and DX12 adapters that report `shaderFloat64`;
    /// `false` on Metal. Gates [`GeneralPrecision::F64`].
    pub fn supports_f64(&self) -> bool {
        self.has_f64
    }

    /// Single-precision reflectance for many equal-depth stacks on a shared q-grid.
    ///
    /// # Parameters
    /// - `q`: scattering vectors in `1/Angstrom`, length `n_q`.
    /// - `k0`: vacuum wavenumber per stack in `1/Angstrom`, length `n_stacks`.
    /// - `layers`: row-major `(n_stacks, n_layers)` layer records.
    /// - `n_layers`: depth of every stack, at least two.
    ///
    /// # Returns
    /// Row-major `(n_stacks, n_q)` pairs `[R_pp, R_ss]`, matching the
    /// `(0, 0)` and `(1, 1)` entries of [`crate::kernel::PointResult`].
    ///
    /// # Errors
    /// As for [`GpuContext::reflectivity_points`].
    pub fn reflectivity_flat(
        &self,
        q: &[f32],
        k0: &[f32],
        layers: &[LayerCoeffs<f32>],
        n_layers: usize,
    ) -> Result<Vec<[f32; 2]>> {
        let n_stacks = to_u32(k0.len(), "n_stacks")?;
        let q_points: Vec<f32> = (0..k0.len()).flat_map(|_| q.iter().copied()).collect();
        let stack_of: Vec<u32> = (0..n_stacks)
            .flat_map(|si| std::iter::repeat_n(si, q.len()))
            .collect();
        self.reflectivity_points(&q_points, &stack_of, k0, layers, n_layers)
    }

    /// Single-precision reflectance for independent `(q, stack)` points.
    ///
    /// Each point carries its own stack index, so stacks (for example one
    /// per photon energy) may be paired with ragged q-grids in one dispatch.
    ///
    /// # Parameters
    /// - `q`: scattering vector per point in `1/Angstrom`, length `n_points`.
    /// - `stack_of`: stack index per point, each `< n_stacks`.
    /// - `k0`: vacuum wavenumber per stack in `1/Angstrom`, length `n_stacks`.
    /// - `layers`: row-major `(n_stacks, n_layers)` layer records.
    /// - `n_layers`: depth of every stack, at least two.
    ///
    /// # Returns
    /// One `[R_pp, R_ss]` pair per point, in input order.
    ///
    /// # Errors
    /// [`RefloxideError::InvalidShape`] for inconsistent buffer lengths, an
    /// out-of-range stack index, or a batch exceeding device buffer limits;
    /// [`RefloxideError::Gpu`] when the readback fails.
    pub fn reflectivity_points(
        &self,
        q: &[f32],
        stack_of: &[u32],
        k0: &[f32],
        layers: &[LayerCoeffs<f32>],
        n_layers: usize,
    ) -> Result<Vec<[f32; 2]>> {
        check_points(q.len(), stack_of, k0.len(), layers.len(), n_layers)?;
        if q.is_empty() {
            return Ok(Vec::new());
        }
        let records = gpu_layers(layers);
        let (n_layers_u, n_stacks_u, n_points_u) = (
            to_u32(n_layers, "n_layers")?,
            to_u32(k0.len(), "n_stacks")?,
            to_u32(q.len(), "n_points")?,
        );
        self.dispatch(
            &self.pipeline,
            q.len(),
            |row_pitch| {
                bytemuck::bytes_of(&Params {
                    n_points: n_points_u,
                    n_layers: n_layers_u,
                    n_stacks: n_stacks_u,
                    row_pitch,
                })
                .to_vec()
            },
            &[
                (1, bytemuck::cast_slice(q)),
                (2, bytemuck::cast_slice(k0)),
                (3, bytemuck::cast_slice(&records)),
                (5, bytemuck::cast_slice(stack_of)),
            ],
        )
    }

    /// Single-precision reflectance and directional derivatives for `(q, stack)` points.
    ///
    /// Evaluates [`crate::kernel::solve_point_recursive_jvp`] for every
    /// `(direction, point)` pair in one dispatch. Direction `k` perturbs the
    /// layer records by `dlayers[k]` and the scattering vectors by `dq[k]`.
    ///
    /// # Parameters
    /// - `q`, `stack_of`, `k0`, `layers`, `n_layers`: as in
    ///   [`GpuContext::reflectivity_points`].
    /// - `dq`: row-major `(n_dirs, n_points)` tangent of `q`.
    /// - `dlayers`: row-major `(n_dirs, n_stacks, n_layers)` tangent records.
    ///
    /// # Returns
    /// Row-major `(n_dirs, n_points)` entries `[R_pp, R_ss, dR_pp, dR_ss]`;
    /// the value columns repeat for every direction.
    ///
    /// # Errors
    /// As for [`GpuContext::reflectivity_points`], plus
    /// [`RefloxideError::InvalidShape`] when the tangent buffers do not match.
    #[allow(clippy::too_many_arguments)]
    pub fn reflectivity_points_jvp(
        &self,
        q: &[f32],
        stack_of: &[u32],
        k0: &[f32],
        layers: &[LayerCoeffs<f32>],
        n_layers: usize,
        dq: &[f32],
        dlayers: &[LayerCoeffs<f32>],
    ) -> Result<Vec<[f32; 4]>> {
        check_points(q.len(), stack_of, k0.len(), layers.len(), n_layers)?;
        if q.is_empty() || dq.is_empty() {
            return Ok(Vec::new());
        }
        if !dq.len().is_multiple_of(q.len()) {
            return Err(RefloxideError::InvalidShape(format!(
                "dq length {} is not a multiple of n_points {}",
                dq.len(),
                q.len()
            )));
        }
        let n_dirs = dq.len() / q.len();
        if dlayers.len() != n_dirs * layers.len() {
            return Err(RefloxideError::InvalidShape(format!(
                "dlayers length {} != n_dirs ({n_dirs}) * n_stacks * n_layers ({})",
                dlayers.len(),
                layers.len()
            )));
        }
        let records = gpu_layers(layers);
        let d_records = gpu_layers(dlayers);
        let params = JvpParams {
            n_points: to_u32(q.len(), "n_points")?,
            n_layers: to_u32(n_layers, "n_layers")?,
            n_stacks: to_u32(k0.len(), "n_stacks")?,
            row_pitch: 0,
            n_dirs: to_u32(n_dirs, "n_dirs")?,
            _pad: [0; 3],
        };
        self.dispatch(
            &self.jvp_pipeline,
            n_dirs * q.len(),
            |row_pitch| {
                bytemuck::bytes_of(&JvpParams {
                    row_pitch,
                    ..params
                })
                .to_vec()
            },
            &[
                (1, bytemuck::cast_slice(q)),
                (2, bytemuck::cast_slice(k0)),
                (3, bytemuck::cast_slice(&records)),
                (5, bytemuck::cast_slice(stack_of)),
                (6, bytemuck::cast_slice(&d_records)),
                (7, bytemuck::cast_slice(dq)),
            ],
        )
    }

    /// Double-single (df64) reflectance for independent `(q, stack)` points.
    ///
    /// Same contract as [`GpuContext::reflectivity_points`], but inputs stay
    /// in `f64` and the recursion runs in df64 arithmetic (~48 mantissa
    /// bits, f32 exponent range), so reflectance tracks the CPU `f64`
    /// recursion instead of carrying single-precision error. For devices
    /// without native `f64` shaders (Metal); roughly an order of magnitude
    /// more ALU work per point than the `f32` kernel.
    ///
    /// # Errors
    /// As for [`GpuContext::reflectivity_points`].
    pub fn reflectivity_points_df64(
        &self,
        q: &[f64],
        stack_of: &[u32],
        k0: &[f64],
        layers: &[LayerCoeffs<f64>],
        n_layers: usize,
    ) -> Result<Vec<[f64; 2]>> {
        check_points(q.len(), stack_of, k0.len(), layers.len(), n_layers)?;
        if q.is_empty() {
            return Ok(Vec::new());
        }
        let cdf = |z: Complex<f64>| {
            let (re, im) = (to_df64(z.re), to_df64(z.im));
            [re[0], re[1], im[0], im[1]]
        };
        let records: Vec<GpuLayerDf64> = layers
            .iter()
            .map(|l| GpuLayerDf64 {
                chi_o: cdf(l.chi_o),
                chi_e: cdf(l.chi_e),
                thickness: to_df64(l.thickness),
                sigma: to_df64(l.sigma),
            })
            .collect();
        let qd: Vec<[f32; 2]> = q.iter().map(|&v| to_df64(v)).collect();
        let k0d: Vec<[f32; 2]> = k0.iter().map(|&v| to_df64(v)).collect();
        let header = [
            to_u32(q.len(), "n_points")?,
            to_u32(n_layers, "n_layers")?,
            to_u32(k0.len(), "n_stacks")?,
        ];
        let out: Vec<[f32; 4]> = self.dispatch(
            &self.df64_pipeline,
            q.len(),
            |row_pitch| {
                bytemuck::cast_slice(&[header[0], header[1], header[2], row_pitch, 0, 0, 0, 0])
                    .to_vec()
            },
            &[
                (1, bytemuck::cast_slice(&qd)),
                (2, bytemuck::cast_slice(&k0d)),
                (3, bytemuck::cast_slice(&records)),
                (5, bytemuck::cast_slice(stack_of)),
            ],
        )?;
        let join = |hi: f32, lo: f32| f64::from(hi) + f64::from(lo);
        Ok(out
            .iter()
            .map(|r| [join(r[0], r[1]), join(r[2], r[3])])
            .collect())
    }

    /// Reflectance of general dielectric-tensor stacks at independent `(q, stack)` points.
    ///
    /// Runs the shader port of [`crate::kernel::solve_point_general`]:
    /// Berreman eigenmodes of each layer's full susceptibility tensor and the
    /// 4x4 reflection-matrix recursion with Nevot-Croce roughness. Tilted,
    /// rotated, and biaxial layers produce cross-polarized reflectance.
    /// One invocation evaluates one point; inputs and outputs are `f64` on
    /// the host and are narrowed only for [`GeneralPrecision::F32`].
    ///
    /// # Parameters
    /// - `precision`: shader arithmetic. [`GeneralPrecision::F64`] needs
    ///   [`GpuContext::supports_f64`]; [`GeneralPrecision::F32`] is a
    ///   diagnostic build.
    /// - `q`: scattering vector per point in `1/Angstrom`.
    /// - `stack_of`: stack index per point, each `< n_stacks`.
    /// - `k0`: vacuum wavenumber per stack in `1/Angstrom`, positive.
    /// - `layers`: row-major `(n_stacks, n_layers)`; row 0 of a stack is the
    ///   isotropic fronting and the last row the backing.
    /// - `n_layers`: depth of every stack, at least two.
    ///
    /// # Returns
    /// One `[[R_pp, R_sp], [R_ps, R_ss]]` per point (the CPU packing), in
    /// input order.
    ///
    /// # Errors
    /// [`RefloxideError::Gpu`] when `F64` is requested on an adapter without
    /// `SHADER_F64`; [`RefloxideError::InvalidShape`] for inconsistent
    /// buffers or a batch exceeding device limits;
    /// [`RefloxideError::SingularDynamicMatrix`] (with the point's index
    /// and stack as `energy_index`) when a layer's mode matrix or the
    /// recursion denominator is singular.
    pub fn reflectivity_points_general(
        &self,
        precision: GeneralPrecision,
        q: &[f64],
        stack_of: &[u32],
        k0: &[f64],
        layers: &[GeneralLayer<f64>],
        n_layers: usize,
    ) -> Result<Vec<[[f64; 2]; 2]>> {
        check_points(q.len(), stack_of, k0.len(), layers.len(), n_layers)?;
        if precision == GeneralPrecision::F64 && !self.has_f64 {
            return Err(RefloxideError::Gpu(format!(
                "adapter {} has no native f64 shader support (wgpu SHADER_F64)",
                self.describe()
            )));
        }
        if q.is_empty() {
            return Ok(Vec::new());
        }
        let pipeline = match precision {
            GeneralPrecision::F64 => self.general_f64_pipeline.get_or_init(|| {
                compile_pipeline(
                    &self.device,
                    "refloxide-general-f64",
                    &[GENERAL_F64_WGSL, GENERAL_WGSL].concat(),
                )
            }),
            GeneralPrecision::F32 => self.general_f32_pipeline.get_or_init(|| {
                compile_pipeline(
                    &self.device,
                    "refloxide-general-f32",
                    &[GENERAL_F32_WGSL, GENERAL_WGSL].concat(),
                )
            }),
        };
        let mut records = Vec::with_capacity(layers.len() * GENERAL_STRIDE);
        for l in layers {
            for row in &l.chi {
                for c in row {
                    records.extend([c.re, c.im]);
                }
            }
            records.extend([l.thickness, l.sigma]);
        }
        let pack = |values: &[f64]| -> Vec<u8> {
            match precision {
                GeneralPrecision::F64 => bytemuck::cast_slice(values).to_vec(),
                GeneralPrecision::F32 => {
                    let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                    bytemuck::cast_slice(&narrow).to_vec()
                }
            }
        };
        let header = [
            to_u32(q.len(), "n_points")?,
            to_u32(n_layers, "n_layers")?,
            to_u32(k0.len(), "n_stacks")?,
        ];
        let (q_bytes, k0_bytes, record_bytes) = (pack(q), pack(k0), pack(&records));
        let params = |row_pitch| {
            bytemuck::cast_slice(&[header[0], header[1], header[2], row_pitch]).to_vec()
        };
        let inputs: [(u32, &[u8]); 4] = [
            (1, &q_bytes),
            (2, &k0_bytes),
            (3, &record_bytes),
            (5, bytemuck::cast_slice(stack_of)),
        ];
        let out: Vec<[f64; 4]> = match precision {
            GeneralPrecision::F64 => {
                self.dispatch::<[f64; 4]>(pipeline, q.len(), params, &inputs)?
            }
            GeneralPrecision::F32 => self
                .dispatch::<[f32; 4]>(pipeline, q.len(), params, &inputs)?
                .into_iter()
                .map(|r| r.map(f64::from))
                .collect(),
        };
        out.iter()
            .enumerate()
            .map(|(point, r)| {
                if r[0] < 0.0 {
                    Err(RefloxideError::SingularDynamicMatrix {
                        layer: (-r[0] - 1.0) as usize,
                        q_index: point,
                        energy_index: Some(stack_of[point] as usize),
                    })
                } else {
                    Ok([[r[0], r[1]], [r[2], r[3]]])
                }
            })
            .collect()
    }

    /// Evaluates one df64 operation elementwise on the device.
    ///
    /// Diagnostic for new adapters: compares against `f64` on the host to
    /// confirm that the shader compiler preserves the error-free
    /// transformations df64 relies on. Inputs are split into `(hi, lo)`
    /// pairs; results are recombined as `hi + lo`.
    ///
    /// # Errors
    /// [`RefloxideError::InvalidShape`] when `a` and `b` differ in length;
    /// [`RefloxideError::Gpu`] for device failures.
    pub fn df64_selftest(&self, a: &[f64], b: &[f64], op: Df64Op) -> Result<Vec<f64>> {
        if a.len() != b.len() {
            return Err(RefloxideError::InvalidShape(
                "df64 selftest inputs differ in length".into(),
            ));
        }
        if a.is_empty() {
            return Ok(Vec::new());
        }
        let da: Vec<[f32; 2]> = a.iter().map(|&x| to_df64(x)).collect();
        let db: Vec<[f32; 2]> = b.iter().map(|&x| to_df64(x)).collect();
        let n = to_u32(a.len(), "n")?;
        let op = op as u32;
        let out: Vec<[f32; 2]> = self.dispatch(
            &self.df64_selftest_pipeline,
            a.len(),
            |row_pitch| bytemuck::cast_slice(&[n, op, row_pitch, 0u32]).to_vec(),
            &[
                (1, bytemuck::cast_slice(&da)),
                (2, bytemuck::cast_slice(&db)),
            ],
        )?;
        Ok(out
            .iter()
            .map(|r| f64::from(r[0]) + f64::from(r[1]))
            .collect())
    }

    /// Uploads inputs, runs `threads` invocations of `pipeline`, and reads
    /// back binding 4 as `threads` values of `O`.
    fn dispatch<O: Pod>(
        &self,
        pipeline: &wgpu::ComputePipeline,
        threads: usize,
        params: impl FnOnce(u32) -> Vec<u8>,
        inputs: &[(u32, &[u8])],
    ) -> Result<Vec<O>> {
        let out_bytes = (threads * std::mem::size_of::<O>()) as u64;
        let max_binding = self.limits.max_storage_buffer_binding_size;
        let largest = inputs
            .iter()
            .map(|(_, b)| b.len() as u64)
            .chain(std::iter::once(out_bytes))
            .max()
            .unwrap_or(0);
        if largest > max_binding {
            return Err(RefloxideError::InvalidShape(format!(
                "batch needs {largest} bytes in one buffer, device binding limit is {max_binding}"
            )));
        }
        let groups = to_u32(threads.div_ceil(WORKGROUP_SIZE as usize), "workgroup count")?;
        let groups_x = groups.min(self.limits.max_compute_workgroups_per_dimension);
        let groups_y = groups.div_ceil(groups_x);

        let init = |label: &str, contents: &[u8], usage| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
        };
        let params_buf = init(
            "params",
            &params(groups_x * WORKGROUP_SIZE),
            wgpu::BufferUsages::UNIFORM,
        );
        let input_bufs: Vec<(u32, wgpu::Buffer)> = inputs
            .iter()
            .map(|(binding, bytes)| (*binding, init("input", bytes, wgpu::BufferUsages::STORAGE)))
            .collect();
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out_buf.as_entire_binding(),
            },
        ];
        entries.extend(
            input_bufs
                .iter()
                .map(|(binding, buf)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buf.as_entire_binding(),
                }),
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("refloxide"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("refloxide"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("refloxide"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        let (tx, rx) = mpsc::channel();
        read_buf.map_async(wgpu::MapMode::Read, .., move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| RefloxideError::Gpu(format!("device poll failed: {e}")))?;
        rx.recv()
            .map_err(|e| RefloxideError::Gpu(format!("readback channel closed: {e}")))?
            .map_err(|e| RefloxideError::Gpu(format!("readback map failed: {e}")))?;
        let view = read_buf
            .get_mapped_range(..)
            .map_err(|e| RefloxideError::Gpu(format!("readback view failed: {e}")))?;
        let out: Vec<O> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        read_buf.unmap();
        Ok(out)
    }
}

static SHARED: OnceLock<std::result::Result<GpuContext, String>> = OnceLock::new();

/// Process-wide [`GpuContext`], created on first use and reused thereafter.
///
/// # Errors
/// [`RefloxideError::Gpu`] when device acquisition failed; the failure is
/// cached so later calls fail fast instead of re-probing adapters.
pub fn shared_context() -> Result<&'static GpuContext> {
    SHARED
        .get_or_init(|| GpuContext::new().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| RefloxideError::Gpu(e.clone()))
}

/// GPU counterpart of [`crate::uniaxial::uniaxial_reflectivity_batch`].
///
/// Inputs are validated with the CPU contract, packed into susceptibility
/// records in `f64`, then explicitly narrowed to `f32` for the device. The
/// shader evaluates the decoupled uniaxial-z recursion, whose reflectance
/// agrees with the `f64` 4x4 kernel to about `1e-4` relative in the worst
/// validated case. Every stack must have the same number of layers.
///
/// Transmission is not computed: `tran` has the CPU layout but every entry
/// is `NaN + NaN i`, so accidental use propagates visibly rather than
/// silently. Cross-polarized reflectance entries are exactly zero.
///
/// # Errors
/// Shape and energy errors as for the CPU batch kernel,
/// [`RefloxideError::InvalidShape`] for ragged layer counts, and
/// [`RefloxideError::Gpu`] for device failures.
pub fn uniaxial_reflectivity_batch(
    q: &[f64],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<Complex<f64>>>],
    energies_ev: &[f64],
) -> Result<UniaxialBatchOutput> {
    validate_batch(layers, tensor, energies_ev)?;
    let n_e = energies_ev.len();
    let n_q = q.len();
    let q_points: Vec<f64> = (0..n_e).flat_map(|_| q.iter().copied()).collect();
    let stack_of: Vec<usize> = (0..n_e).flat_map(|e| std::iter::repeat_n(e, n_q)).collect();
    let flat = uniaxial_reflectivity_points(&q_points, &stack_of, layers, tensor, energies_ev)?;
    let mut refl = flat.refl.into_iter();
    let mut tran = flat.tran.into_iter();
    Ok(UniaxialBatchOutput {
        refl: (0..n_e)
            .map(|_| refl.by_ref().take(n_q).collect())
            .collect(),
        tran: (0..n_e)
            .map(|_| tran.by_ref().take(n_q).collect())
            .collect(),
    })
}

/// GPU counterpart of [`crate::uniaxial::uniaxial_reflectivity_points`].
///
/// Precision, transmission (`NaN`), and cross-polarized (zero) semantics are
/// those of [`uniaxial_reflectivity_batch`]; every stack must have the same
/// number of layers. All points run in a single dispatch.
///
/// # Errors
/// Shape and energy errors as for the CPU point kernel,
/// [`RefloxideError::InvalidShape`] for ragged layer counts, and
/// [`RefloxideError::Gpu`] for device failures.
pub fn uniaxial_reflectivity_points(
    q: &[f64],
    stack_of: &[usize],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<Complex<f64>>>],
    energies_ev: &[f64],
) -> Result<UniaxialOutput> {
    validate_points(q, stack_of, layers, tensor, energies_ev)?;
    let n_layers = layers[0].len();
    if layers.iter().any(|l| l.len() != n_layers) {
        return Err(RefloxideError::InvalidShape(
            "gpu kernels require the same layer count for every stack".into(),
        ));
    }
    let nan = Complex::new(f64::NAN, f64::NAN);
    let tran = vec![[[nan; 2]; 2]; q.len()];
    if q.is_empty() {
        return Ok(UniaxialOutput {
            refl: Vec::new(),
            tran,
        });
    }
    let narrow = |z: Complex<f64>| Complex::new(z.re as f32, z.im as f32);
    let records: Vec<LayerCoeffs<f32>> = layers
        .iter()
        .zip(tensor)
        .flat_map(|(l, t)| layer_coeffs(l, t))
        .map(|c| LayerCoeffs {
            chi_o: narrow(c.chi_o),
            chi_e: narrow(c.chi_e),
            thickness: c.thickness as f32,
            sigma: c.sigma as f32,
        })
        .collect();
    let q32: Vec<f32> = q.iter().map(|&v| v as f32).collect();
    let k0: Vec<f32> = energies_ev.iter().map(|&e| wavenumber(e) as f32).collect();
    let stack32 = stack_of
        .iter()
        .map(|&s| u32::try_from(s))
        .collect::<std::result::Result<Vec<u32>, _>>()
        .map_err(|_| RefloxideError::InvalidShape("stack index exceeds u32".into()))?;

    let out = shared_context()?.reflectivity_points(&q32, &stack32, &k0, &records, n_layers)?;
    let refl = out
        .iter()
        .map(|r| [[f64::from(r[0]), 0.0], [0.0, f64::from(r[1])]])
        .collect();
    Ok(UniaxialOutput { refl, tran })
}

/// GPU counterpart of [`crate::uniaxial::uniaxial_reflectivity_points_jvp`].
///
/// Values and tangents are narrowed to `f32` on the device; derivatives are
/// forward-mode exact for the single-precision recursion, so their relative
/// accuracy matches the reflectance (about `1e-4` worst case) instead of
/// suffering finite-difference cancellation. All `(direction, point)` pairs
/// run in one dispatch. Every stack must have the same number of layers.
///
/// # Errors
/// As for the CPU JVP kernel, [`RefloxideError::InvalidShape`] for ragged
/// layer counts, and [`RefloxideError::Gpu`] for device failures.
pub fn uniaxial_reflectivity_points_jvp(
    q: &[f64],
    stack_of: &[usize],
    layers: &[Vec<Layer>],
    tensor: &[Vec<Matrix3<Complex<f64>>>],
    energies_ev: &[f64],
    tangents: &PointsTangents,
) -> Result<PointsJvpOutput> {
    let (coeffs, dcoeffs) = jvp_coeffs(q, stack_of, layers, tensor, energies_ev, tangents)?;
    let n_layers = layers[0].len();
    if layers.iter().any(|l| l.len() != n_layers) {
        return Err(RefloxideError::InvalidShape(
            "gpu kernels require the same layer count for every stack".into(),
        ));
    }
    let n_points = q.len();
    let n_dirs = tangents.dq.len();
    if n_points == 0 || n_dirs == 0 {
        let refl = uniaxial_reflectivity_points(q, stack_of, layers, tensor, energies_ev)?
            .refl
            .iter()
            .map(|r| [r[0][0], r[1][1]])
            .collect();
        return Ok(PointsJvpOutput {
            refl,
            jac: Vec::new(),
        });
    }
    let narrow = |c: &LayerCoeffs<f64>| LayerCoeffs {
        chi_o: Complex::new(c.chi_o.re as f32, c.chi_o.im as f32),
        chi_e: Complex::new(c.chi_e.re as f32, c.chi_e.im as f32),
        thickness: c.thickness as f32,
        sigma: c.sigma as f32,
    };
    let records: Vec<LayerCoeffs<f32>> = coeffs.iter().flatten().map(narrow).collect();
    let d_records: Vec<LayerCoeffs<f32>> = dcoeffs.iter().flatten().flatten().map(narrow).collect();
    let q32: Vec<f32> = q.iter().map(|&v| v as f32).collect();
    let dq32: Vec<f32> = tangents.dq.iter().flatten().map(|&v| v as f32).collect();
    let k0: Vec<f32> = energies_ev.iter().map(|&e| wavenumber(e) as f32).collect();
    let stack32 = stack_of
        .iter()
        .map(|&s| u32::try_from(s))
        .collect::<std::result::Result<Vec<u32>, _>>()
        .map_err(|_| RefloxideError::InvalidShape("stack index exceeds u32".into()))?;
    let out = shared_context()?
        .reflectivity_points_jvp(&q32, &stack32, &k0, &records, n_layers, &dq32, &d_records)?;
    Ok(PointsJvpOutput {
        refl: out[..n_points]
            .iter()
            .map(|r| [f64::from(r[0]), f64::from(r[1])])
            .collect(),
        jac: out
            .iter()
            .map(|r| [f64::from(r[2]), f64::from(r[3])])
            .collect(),
    })
}
