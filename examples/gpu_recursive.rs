//! GPU recursion validation and throughput benchmark.
//!
//! Checks the WGSL recursion against the CPU `f32` recursion and the `f64`
//! 4x4 reference, then times batched `(n_stacks, n_q)` workloads shaped like
//! MCMC walker ensembles. Run with
//! `cargo run --profile perf --no-default-features --features gpu --example gpu_recursive`.

use std::time::Instant;

use num_complex::Complex;
use refloxide::gpu::GpuContext;
use refloxide::kernel::{solve_flat, solve_flat_recursive, solve_flat_rmatrix, LayerCoeffs};

const HC_EV_ANGSTROM: f64 = 12398.4193;

fn chi(delta: f64, beta: f64) -> Complex<f64> {
    Complex::new(-2.0 * delta, 2.0 * beta)
}

fn graded_stack(n_film: usize, dz: f64, perturb: f64) -> Vec<LayerCoeffs<f64>> {
    let iso = |d: f64, b: f64, t: f64, s: f64| LayerCoeffs {
        chi_o: chi(d, b),
        chi_e: chi(d, b),
        thickness: t,
        sigma: s,
    };
    let mut stack = vec![iso(0.0, 0.0, 0.0, 0.0)];
    for j in 0..n_film {
        let f = j as f64 / n_film as f64;
        stack.push(LayerCoeffs {
            chi_o: chi(2.0e-3 * (1.0 - 0.3 * f) * perturb, 1.2e-3 * (1.0 + 0.5 * f)),
            chi_e: chi(1.0e-3 * (1.0 + f), 2.5e-3 * (1.0 - 0.4 * f) * perturb),
            thickness: dz,
            sigma: if j == 0 { 4.0 } else { 0.0 },
        });
    }
    stack.push(iso(1.5e-3, 2.0e-4, 15.0, 3.0));
    stack.push(iso(1.2e-3, 1.5e-4, 0.0, 2.0));
    stack
}

fn to_f32(stack: &[LayerCoeffs<f64>]) -> Vec<LayerCoeffs<f32>> {
    stack
        .iter()
        .map(|l| LayerCoeffs {
            chi_o: Complex::new(l.chi_o.re as f32, l.chi_o.im as f32),
            chi_e: Complex::new(l.chi_e.re as f32, l.chi_e.im as f32),
            thickness: l.thickness as f32,
            sigma: l.sigma as f32,
        })
        .collect()
}

fn max_rel(reference: &[[f64; 2]], test: &[[f64; 2]]) -> f64 {
    reference
        .iter()
        .zip(test)
        .flat_map(|(a, b)| (0..2).map(move |c| ((b[c] - a[c]) / a[c]).abs()))
        .fold(0.0, |m, v| {
            if v.is_finite() {
                m.max(v)
            } else {
                f64::INFINITY
            }
        })
}

fn best_of<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let gpu = GpuContext::new().expect("gpu context");
    println!("adapter: {}", gpu.describe());

    let energy = 285.0;
    let k0 = 2.0 * std::f64::consts::PI / (HC_EV_ANGSTROM / energy);
    let q: Vec<f64> = (0..500)
        .map(|i| 0.005 + (0.98 * 2.0 * k0 - 0.005) * i as f64 / 499.0)
        .collect();
    let q32: Vec<f32> = q.iter().map(|&v| v as f32).collect();

    println!("\nvalidation (single stack, 500 q, max relative error over R_pp and R_ss)");
    for (n_film, dz) in [(1, 200.0), (1, 3000.0), (100, 2.0), (400, 2.5)] {
        let stack = graded_stack(n_film, dz, 1.0);
        let n = stack.len();
        let stack32 = to_f32(&stack);
        let reference: Vec<[f64; 2]> = solve_flat(&q, &[k0], &stack, n, false)
            .expect("f64 4x4")
            .iter()
            .map(|r| [r.0[0][0], r.0[1][1]])
            .collect();
        let widen = |v: Vec<[f32; 2]>| -> Vec<[f64; 2]> {
            v.iter()
                .map(|r| [f64::from(r[0]), f64::from(r[1])])
                .collect()
        };
        let cpu32 = widen(solve_flat_recursive(&q32, &[k0 as f32], &stack32, n, false).unwrap());
        let gpu32 = widen(
            gpu.reflectivity_flat(&q32, &[k0 as f32], &stack32, n)
                .unwrap(),
        );
        println!(
            "  {n_film:>4} x {dz:>6.1} A: gpu vs f64 4x4 {:.1e} | cpu f32 vs f64 4x4 {:.1e} | gpu vs cpu f32 {:.1e}",
            max_rel(&reference, &gpu32),
            max_rel(&reference, &cpu32),
            max_rel(&cpu32, &gpu32),
        );
    }

    println!("\nthroughput, 500 q x 104-layer stacks (best of 5, seconds)");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>9}",
        "stacks",
        "cpu 4x4 f64",
        "cpu rmx f64",
        "cpu rec f64",
        "cpu rec f32",
        "gpu rec f32",
        "speedup"
    );
    for n_stacks in [1usize, 16, 128, 1024, 4096] {
        let stacks: Vec<Vec<LayerCoeffs<f64>>> = (0..n_stacks)
            .map(|i| graded_stack(100, 2.0, 1.0 + 0.1 * i as f64 / n_stacks as f64))
            .collect();
        let n = stacks[0].len();
        let flat: Vec<LayerCoeffs<f64>> = stacks.concat();
        let flat32 = to_f32(&flat);
        let k0s = vec![k0; n_stacks];
        let k0s32 = vec![k0 as f32; n_stacks];
        let t_tmm = best_of(5, || {
            solve_flat(&q, &k0s, &flat, n, true).unwrap();
        });
        let t_rec = best_of(5, || {
            solve_flat_recursive(&q, &k0s, &flat, n, true).unwrap();
        });
        let t_rmx = best_of(5, || {
            solve_flat_rmatrix(&q, &k0s, &flat, n, true).unwrap();
        });
        let t_rec32 = best_of(5, || {
            solve_flat_recursive(&q32, &k0s32, &flat32, n, true).unwrap();
        });
        gpu.reflectivity_flat(&q32, &k0s32, &flat32, n).unwrap();
        let t_gpu = best_of(5, || {
            gpu.reflectivity_flat(&q32, &k0s32, &flat32, n).unwrap();
        });
        println!(
            "{n_stacks:>8} {t_tmm:>12.4e} {t_rmx:>12.4e} {t_rec:>12.4e} {t_rec32:>12.4e} {t_gpu:>12.4e} {:>8.1}x",
            t_tmm / t_gpu
        );
    }
}
