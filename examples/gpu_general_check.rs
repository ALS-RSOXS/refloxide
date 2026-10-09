//! Offline check of the general-tensor shaders with naga (no GPU needed).
//!
//! Parses and validates the f32 and f64 builds of `src/gpu/general.wgsl` with
//! every capability enabled (the f64 build needs `SHADER_F64`, which Metal
//! adapters lack), then emits SPIR-V (Vulkan) and HLSL (DX12) for the f64
//! build to confirm those backends accept every operation used. Run with
//! `cargo run --example gpu_general_check`.

use naga::back::{hlsl, spv};
use naga::valid::{Capabilities, ValidationFlags, Validator};

fn check(label: &str, prelude: &str, emit: bool) -> bool {
    let source = [prelude, include_str!("../src/gpu/general.wgsl")].concat();
    let module = match naga::front::wgsl::parse_str(&source) {
        Ok(m) => m,
        Err(e) => {
            println!("{label}: PARSE ERROR\n{}", e.emit_to_string(&source));
            return false;
        }
    };
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = match validator.validate(&module) {
        Ok(i) => i,
        Err(e) => {
            println!("{label}: VALIDATION ERROR\n{}", e.emit_to_string(&source));
            return false;
        }
    };
    println!("{label}: parses and validates");
    if !emit {
        return true;
    }
    let mut ok = true;
    let options = spv::Options::default();
    match spv::write_vec(&module, &info, &options, None) {
        Ok(words) => println!("{label}: SPIR-V ok ({} words)", words.len()),
        Err(e) => {
            println!("{label}: SPIR-V ERROR {e}");
            ok = false;
        }
    }
    let mut hlsl_out = String::new();
    let hlsl_options = hlsl::Options::default();
    let pipeline_options = Default::default();
    let written = hlsl::Writer::new(&mut hlsl_out, &hlsl_options, &pipeline_options)
        .write(&module, &info, None);
    match written {
        Ok(_) => println!("{label}: HLSL ok ({} bytes)", hlsl_out.len()),
        Err(e) => {
            println!("{label}: HLSL ERROR {e}");
            ok = false;
        }
    }
    ok
}

fn main() {
    let ok32 = check(
        "general f32",
        include_str!("../src/gpu/general_f32.wgsl"),
        false,
    );
    let ok64 = check(
        "general f64",
        include_str!("../src/gpu/general_f64.wgsl"),
        true,
    );
    if !(ok32 && ok64) {
        std::process::exit(1);
    }
}
