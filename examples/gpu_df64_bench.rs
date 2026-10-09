//! Accuracy and cost of the df64 GPU recursion against GPU f32 and CPU f64.
//!
//! Run with
//! `cargo run --profile perf --no-default-features --features gpu --example gpu_df64_bench`.

use std::time::Instant;

use num_complex::Complex;
use refloxide::gpu::GpuContext;
use refloxide::kernel::{solve_flat_recursive, LayerCoeffs};

const HC_EV_ANGSTROM: f64 = 12398.4193;

fn chi(d: f64, b: f64) -> Complex<f64> {
    Complex::new(-2.0 * d, 2.0 * b)
}

fn layer(o: (f64, f64), e: (f64, f64), thickness: f64, sigma: f64) -> LayerCoeffs<f64> {
    LayerCoeffs {
        chi_o: chi(o.0, o.1),
        chi_e: chi(e.0, e.1),
        thickness,
        sigma,
    }
}

fn stack(film: Vec<LayerCoeffs<f64>>) -> Vec<LayerCoeffs<f64>> {
    let mut s = vec![layer((0.0, 0.0), (0.0, 0.0), 0.0, 0.0)];
    s.extend(film);
    s.push(layer((1.5e-3, 2e-4), (1.5e-3, 2e-4), 15.0, 3.0));
    s.push(layer((1.2e-3, 1.5e-4), (1.2e-3, 1.5e-4), 0.0, 2.0));
    s
}

fn graded(n: usize, dz: f64, scale: f64) -> Vec<LayerCoeffs<f64>> {
    (0..n)
        .map(|j| {
            let f = j as f64 / n as f64;
            layer(
                (2e-3 * (1.0 - 0.3 * f) * scale, 1.2e-3 * (1.0 + 0.5 * f)),
                (1e-3 * (1.0 + f), 2.5e-3 * (1.0 - 0.4 * f) * scale),
                dz,
                if j == 0 { 4.0 } else { 0.0 },
            )
        })
        .collect()
}

fn narrow(l: &LayerCoeffs<f64>) -> LayerCoeffs<f32> {
    LayerCoeffs {
        chi_o: Complex::new(l.chi_o.re as f32, l.chi_o.im as f32),
        chi_e: Complex::new(l.chi_e.re as f32, l.chi_e.im as f32),
        thickness: l.thickness as f32,
        sigma: l.sigma as f32,
    }
}

fn max_rel(reference: &[[f64; 2]], test: &[[f64; 2]]) -> f64 {
    reference
        .iter()
        .zip(test)
        .flat_map(|(a, b)| (0..2).map(move |c| ((b[c] - a[c]) / a[c]).abs()))
        .fold(0.0, f64::max)
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
    let k0 = 2.0 * std::f64::consts::PI / (HC_EV_ANGSTROM / 285.0);
    let q: Vec<f64> = (0..500)
        .map(|i| 0.005 + (0.98 * 2.0 * k0 - 0.005) * i as f64 / 499.0)
        .collect();

    println!("\naccuracy vs CPU f64 recursion (max relative error, R_pp and R_ss)");
    let films = [
        (
            "uniform 200 A",
            vec![layer((2e-3, 1.2e-3), (1e-3, 2.5e-3), 200.0, 4.0)],
        ),
        (
            "uniform 3000 A",
            vec![layer((2e-3, 1.2e-3), (1e-3, 2.5e-3), 3000.0, 4.0)],
        ),
        ("graded 100 x 2 A", graded(100, 2.0, 1.0)),
    ];
    for (name, film) in films {
        let s = stack(film);
        let n = s.len();
        let reference = solve_flat_recursive(&q, &[k0], &s, n, false).unwrap();
        let q32: Vec<f32> = q.iter().map(|&v| v as f32).collect();
        let s32: Vec<LayerCoeffs<f32>> = s.iter().map(narrow).collect();
        let stack_of = vec![0u32; q.len()];
        let f32_out: Vec<[f64; 2]> = gpu
            .reflectivity_points(&q32, &stack_of, &[k0 as f32], &s32, n)
            .unwrap()
            .iter()
            .map(|r| [f64::from(r[0]), f64::from(r[1])])
            .collect();
        let df64_out = gpu
            .reflectivity_points_df64(&q, &stack_of, &[k0], &s, n)
            .unwrap();
        println!(
            "  {name:<18} gpu f32 {:.1e}   gpu df64 {:.1e}",
            max_rel(&reference, &f32_out),
            max_rel(&reference, &df64_out)
        );
    }

    println!("\nthroughput, 500 q x 104-layer stacks (best of 5, seconds)");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "stacks", "cpu f64", "gpu f32", "gpu df64", "df64/f32", "cpu/df64"
    );
    for n_stacks in [1usize, 16, 128, 1024, 4096] {
        let flat: Vec<LayerCoeffs<f64>> = (0..n_stacks)
            .flat_map(|i| stack(graded(100, 2.0, 1.0 + 0.1 * i as f64 / n_stacks as f64)))
            .collect();
        let n = flat.len() / n_stacks;
        let flat32: Vec<LayerCoeffs<f32>> = flat.iter().map(narrow).collect();
        let k0s = vec![k0; n_stacks];
        let k0s32 = vec![k0 as f32; n_stacks];
        let q_all: Vec<f64> = (0..n_stacks).flat_map(|_| q.iter().copied()).collect();
        let q_all32: Vec<f32> = q_all.iter().map(|&v| v as f32).collect();
        let stack_of: Vec<u32> = (0..n_stacks as u32)
            .flat_map(|s| std::iter::repeat_n(s, q.len()))
            .collect();
        let t_cpu = best_of(5, || {
            solve_flat_recursive(&q, &k0s, &flat, n, true).unwrap();
        });
        gpu.reflectivity_points(&q_all32, &stack_of, &k0s32, &flat32, n)
            .unwrap();
        let t_f32 = best_of(5, || {
            gpu.reflectivity_points(&q_all32, &stack_of, &k0s32, &flat32, n)
                .unwrap();
        });
        gpu.reflectivity_points_df64(&q_all, &stack_of, &k0s, &flat, n)
            .unwrap();
        let t_df = best_of(5, || {
            gpu.reflectivity_points_df64(&q_all, &stack_of, &k0s, &flat, n)
                .unwrap();
        });
        println!(
            "{n_stacks:>8} {t_cpu:>12.4e} {t_f32:>12.4e} {t_df:>12.4e} {:>9.1}x {:>9.1}x",
            t_df / t_f32,
            t_cpu / t_df
        );
    }
}
