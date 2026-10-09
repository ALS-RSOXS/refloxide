//! Device check of the general-tensor 4x4 shader against the CPU kernel.
//!
//! Evaluates tilted, rotated, biaxial, graded, and rough stacks on the GPU
//! and compares every reflectance channel with `solve_point_general` on the
//! CPU. The f32 shader build runs on every adapter and is compared with the
//! f32 CPU kernel (same algorithm, so this checks the shader logic) and with
//! the f64 CPU kernel (accuracy). The f64 build runs only where the adapter
//! reports `SHADER_F64` (Vulkan, DX12) and is compared with the f64 CPU
//! kernel. Run with
//! `cargo run --profile perf --no-default-features --features gpu --example gpu_general`.

use std::time::Instant;

use num_complex::Complex;
use refloxide::gpu::{GeneralPrecision, GpuContext};
use refloxide::kernel::{solve_point_general, GeneralLayer};

type C = Complex<f64>;

fn rot(theta: f64, phi: f64, psi: f64) -> [[f64; 3]; 3] {
    let (ct, st, cp, sp, cs, ss) = (
        theta.cos(),
        theta.sin(),
        phi.cos(),
        phi.sin(),
        psi.cos(),
        psi.sin(),
    );
    let rz = |c: f64, s: f64| [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    let ry = [[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]];
    let mul = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| {
        let mut o = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    o[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        o
    };
    mul(mul(rz(cp, sp), ry), rz(cs, ss))
}

fn tensor(diag: [C; 3], r: [[f64; 3]; 3]) -> [[C; 3]; 3] {
    let mut o = [[C::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                o[i][j] += diag[k] * r[i][k] * r[j][k];
            }
        }
    }
    o
}

fn chi(d: f64, b: f64) -> C {
    C::new(-2.0 * d, 2.0 * b)
}

fn iso(c: C, t: f64, sigma: f64) -> GeneralLayer<f64> {
    let z = C::new(0.0, 0.0);
    GeneralLayer {
        chi: [[c, z, z], [z, c, z], [z, z, c]],
        thickness: t,
        sigma,
    }
}

fn narrow(l: &GeneralLayer<f64>) -> GeneralLayer<f32> {
    GeneralLayer {
        chi: l
            .chi
            .map(|r| r.map(|c| Complex::new(c.re as f32, c.im as f32))),
        thickness: l.thickness as f32,
        sigma: l.sigma as f32,
    }
}

fn stacks() -> Vec<(&'static str, Vec<GeneralLayer<f64>>)> {
    let vac = iso(C::new(0.0, 0.0), 0.0, 0.0);
    let sio2 = |s: f64| iso(chi(1.5e-3, 2e-4), 15.0, s);
    let si = |s: f64| iso(chi(1.2e-3, 1.5e-4), 0.0, s);
    let uni = [chi(2e-3, 1.2e-3), chi(2e-3, 1.2e-3), chi(1e-3, 2.5e-3)];
    let bi = [
        chi(2.4e-3, 0.9e-3),
        chi(1.6e-3, 1.6e-3),
        chi(0.8e-3, 2.8e-3),
    ];
    let film = |t: [[C; 3]; 3], d: f64, s: f64| GeneralLayer {
        chi: t,
        thickness: d,
        sigma: s,
    };
    let graded = |sigma: f64| {
        let mut v = vec![vac];
        for j in 0..20 {
            let f = j as f64 / 19.0;
            v.push(film(
                tensor(uni, rot(0.2 + 1.1 * f, 0.3 * f, 0.0)),
                8.0,
                sigma,
            ));
        }
        v.push(film(tensor(uni, rot(0.0, 0.0, 0.0)), 30.0, sigma));
        v.push(sio2(sigma));
        v.push(si(sigma));
        v
    };
    vec![
        (
            "uniaxial-z",
            vec![
                vac,
                film(tensor(uni, rot(0.0, 0.0, 0.0)), 250.0, 0.0),
                sio2(0.0),
                si(0.0),
            ],
        ),
        (
            "tilted uniaxial 40deg, phi=0",
            vec![
                vac,
                film(tensor(uni, rot(0.7, 0.0, 0.0)), 250.0, 0.0),
                sio2(0.0),
                si(0.0),
            ],
        ),
        (
            "tilted uniaxial 55deg, phi=35deg",
            vec![
                vac,
                film(tensor(uni, rot(0.96, 0.61, 0.0)), 250.0, 0.0),
                sio2(0.0),
                si(0.0),
            ],
        ),
        (
            "biaxial, euler (0.4,0.9,1.3)",
            vec![
                vac,
                film(tensor(bi, rot(0.9, 0.4, 1.3)), 180.0, 0.0),
                sio2(0.0),
                si(0.0),
            ],
        ),
        (
            "biaxial, rough 4/3/2 A",
            vec![
                vac,
                film(tensor(bi, rot(0.9, 0.4, 1.3)), 180.0, 4.0),
                sio2(3.0),
                si(2.0),
            ],
        ),
        ("graded tilt 20 x 8 A", graded(0.0)),
        ("graded tilt 20 x 8 A, rough 1.5 A", graded(1.5)),
    ]
}

/// Largest `|a - b| / max(|b|, floor)` over paired entries of all channels.
fn max_rel(a: &[[[f64; 2]; 2]], b: &[[[f64; 2]; 2]], floor: f64) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| {
            (0..2).flat_map(move |i| {
                (0..2).map(move |j| {
                    let (u, v) = (x[i][j], y[i][j]);
                    if u.is_nan() || v.is_nan() {
                        f64::INFINITY
                    } else {
                        (u - v).abs() / v.abs().max(floor)
                    }
                })
            })
        })
        .fold(0.0, f64::max)
}

fn main() {
    let gpu = GpuContext::new().expect("gpu context");
    println!(
        "adapter: {}   native f64: {}",
        gpu.describe(),
        gpu.supports_f64()
    );
    let k0 = 2.0 * std::f64::consts::PI / (12398.4193 / 285.0);
    let n_q = 200;
    let q: Vec<f64> = (0..n_q)
        .map(|i| 0.01 + (0.98 * 2.0 * k0 - 0.01) * i as f64 / (n_q - 1) as f64)
        .collect();

    let all = stacks();
    let precisions: Vec<GeneralPrecision> = if gpu.supports_f64() {
        vec![GeneralPrecision::F32, GeneralPrecision::F64]
    } else {
        vec![GeneralPrecision::F32]
    };
    println!(
        "\nmax relative error over {n_q} q-points (all four channels, floor 1e-12)\n{:<36} {:>9} {:>14} {:>14}",
        "stack", "build", "vs cpu f32", "vs cpu f64"
    );
    for (name, layers) in &all {
        let n = layers.len();
        let cpu64: Vec<[[f64; 2]; 2]> = q
            .iter()
            .map(|&qi| solve_point_general(qi, k0, layers).unwrap_or([[f64::NAN; 2]; 2]))
            .collect();
        let l32: Vec<GeneralLayer<f32>> = layers.iter().map(narrow).collect();
        let cpu32: Vec<[[f64; 2]; 2]> = q
            .iter()
            .map(|&qi| {
                solve_point_general(qi as f32, k0 as f32, &l32)
                    .map(|r| r.map(|row| row.map(f64::from)))
                    .unwrap_or([[f64::NAN; 2]; 2])
            })
            .collect();
        let base = max_rel(&cpu32, &cpu64, 1e-12);
        println!("{name:<36} {:>9} {:>14} {:>14.2e}", "cpu f32", "-", base);
        for &precision in &precisions {
            let got: Vec<[[f64; 2]; 2]> = q
                .iter()
                .map(|&qi| {
                    gpu.reflectivity_points_general(precision, &[qi], &[0], &[k0], layers, n)
                        .map(|v| v[0])
                        .unwrap_or([[f64::NAN; 2]; 2])
                })
                .collect();
            let failed: Vec<usize> = (0..n_q).filter(|&i| got[i][0][0].is_nan()).collect();
            let cpu_failed = failed.iter().filter(|&&i| cpu32[i][0][0].is_nan()).count();
            let keep = |v: &[[[f64; 2]; 2]]| -> Vec<[[f64; 2]; 2]> {
                (0..n_q)
                    .filter(|i| !failed.contains(i))
                    .map(|i| v[i])
                    .collect()
            };
            let (g, c32, c64) = (keep(&got), keep(&cpu32), keep(&cpu64));
            println!(
                "{name:<36} {:>9} {:>14.2e} {:>14.2e}   singular on gpu: {} (cpu f32 also: {cpu_failed})",
                format!("{precision:?}"),
                max_rel(&g, &c32, 1e-12),
                max_rel(&g, &c64, 1e-12),
                failed.len()
            );
        }
    }

    println!("\nthroughput, rough biaxial 4-layer stack, 500 q per stack (best of 5, seconds)");
    println!("{:>8} {:>12} {:>12}", "stacks", "cpu f64", "gpu");
    let (_, layers) = &all[4];
    let n = layers.len();
    let qb: Vec<f64> = (0..500)
        .map(|i| 0.01 + (0.98 * 2.0 * k0 - 0.01) * i as f64 / 499.0)
        .collect();
    let precision = if gpu.supports_f64() {
        GeneralPrecision::F64
    } else {
        GeneralPrecision::F32
    };
    for n_stacks in [1usize, 16, 128] {
        let stacked: Vec<GeneralLayer<f64>> = (0..n_stacks).flat_map(|_| layers.clone()).collect();
        let k0s = vec![k0; n_stacks];
        let q_all: Vec<f64> = (0..n_stacks).flat_map(|_| qb.iter().copied()).collect();
        let stack_of: Vec<u32> = (0..n_stacks as u32)
            .flat_map(|s| std::iter::repeat_n(s, qb.len()))
            .collect();
        let best = |mut f: Box<dyn FnMut()>| {
            (0..5)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    t.elapsed().as_secs_f64()
                })
                .fold(f64::INFINITY, f64::min)
        };
        let cpu = best(Box::new(|| {
            for _ in 0..n_stacks {
                for &qi in &qb {
                    let _ = solve_point_general(qi, k0, layers);
                }
            }
        }));
        let dev = best(Box::new(|| {
            gpu.reflectivity_points_general(precision, &q_all, &stack_of, &k0s, &stacked, n)
                .expect("dispatch");
        }));
        println!("{n_stacks:>8} {cpu:>12.4e} {dev:>12.4e}");
    }
}
