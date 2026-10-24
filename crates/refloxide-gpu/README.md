# refloxide-gpu

wgpu compute backend for refloxide (Metal, Vulkan, DX12 from one WGSL source).

## Contents

- `GpuContext`: one device with compiled pipelines; create once and reuse, since device acquisition and shader compilation cost far more than a dispatch.
- `uniaxial_reflectivity_batch`, `uniaxial_reflectivity_points`, `uniaxial_reflectivity_points_jvp`: GPU counterparts of the `refloxide-tmm` entry points.
- Shaders: single-precision uniaxial recursion (the path Python uses), double-single (`df64`) experiments, and general-tensor shaders (native `f64` needs `SHADER_F64`: Vulkan and DX12, not Metal).

## Precision

Reflectance agrees with the double-precision CPU kernel to about 1e-4 relative. Amplitude transmission is not computed.

## Examples

```bash
cargo run -p refloxide-gpu --profile perf --example gpu_recursive
cargo run -p refloxide-gpu --example gpu_general_check   # naga validation, no GPU needed
```

## Boundaries

Inputs are validated with the shared checks from `refloxide-core`; the CPU kernels in `refloxide-tmm` remain the reference.
