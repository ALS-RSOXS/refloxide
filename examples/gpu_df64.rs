//! Device self-test of the df64 (double-single) arithmetic used by f64 GPU kernels.
//!
//! Checks the reassociation canary (the f32 two-sum error term must be
//! nonzero), then compares every df64 operation against `f64` on the host
//! over the argument ranges reflectivity kernels use. Run with
//! `cargo run --profile perf --no-default-features --features gpu --example gpu_df64`.

use refloxide::gpu::{Df64Op, GpuContext};

type Df64Case = (
    &'static str,
    Df64Op,
    Vec<f64>,
    Vec<f64>,
    fn(f64, f64) -> f64,
    bool,
);

fn grid(n: usize, lo: f64, hi: f64, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            lo + (hi - lo) * u
        })
        .collect()
}

fn main() {
    let gpu = GpuContext::new().expect("gpu context");
    println!("adapter: {}", gpu.describe());

    let canary = gpu
        .df64_selftest(&[1.0, 3.0], &[1e-8, 7e-9], Df64Op::TwoSumError)
        .expect("canary");
    let intact = canary
        .iter()
        .zip([1.0 + 1e-8f64, 3.0 + 7e-9])
        .all(|(&v, exact)| ((v - exact) / exact).abs() < 1e-14);
    println!(
        "two-sum error terms {canary:?}: {}",
        if intact {
            "error-free transforms intact"
        } else {
            "REASSOCIATED - df64 invalid on this device"
        }
    );

    let n = 100_000;
    let mixed = |seed| -> Vec<f64> {
        grid(n, -1.0, 1.0, seed)
            .iter()
            .zip(grid(n, -12.0, 3.0, seed + 1))
            .map(|(s, e)| s.signum() * 10f64.powf(e) * (1.0 + s.abs()))
            .collect()
    };
    let (a, b) = (mixed(1), mixed(3));
    let positive = |lo: f64, hi: f64, seed| -> Vec<f64> {
        grid(n, lo.log10(), hi.log10(), seed)
            .iter()
            .map(|e| 10f64.powf(*e))
            .collect()
    };
    let cases: Vec<Df64Case> = vec![
        (
            "add",
            Df64Op::Add,
            a.clone(),
            b.clone(),
            |x, y| x + y,
            false,
        ),
        (
            "mul",
            Df64Op::Mul,
            a.clone(),
            b.clone(),
            |x, y| x * y,
            false,
        ),
        (
            "div",
            Df64Op::Div,
            a.clone(),
            b.clone(),
            |x, y| x / y,
            false,
        ),
        (
            "sqrt",
            Df64Op::Sqrt,
            positive(1e-12, 1e3, 5),
            vec![0.0; n],
            |x, _| x.sqrt(),
            false,
        ),
        (
            "exp",
            Df64Op::Exp,
            grid(n, -60.0, 20.0, 7),
            vec![0.0; n],
            |x, _| x.exp(),
            false,
        ),
        (
            "sin",
            Df64Op::Sin,
            grid(n, -900.0, 900.0, 9),
            vec![0.0; n],
            |x, _| x.sin(),
            true,
        ),
        (
            "cos",
            Df64Op::Cos,
            grid(n, -900.0, 900.0, 11),
            vec![0.0; n],
            |x, _| x.cos(),
            true,
        ),
        (
            "log",
            Df64Op::Log,
            positive(1e-10, 1e10, 13),
            vec![0.0; n],
            |x, _| x.ln(),
            true,
        ),
    ];
    println!(
        "{:<5} {:>12} {:>12}   (vs f64; sin/cos/log absolute error; exp for results > 1e-26)",
        "op", "max err", "median err"
    );
    for (name, op, x, y, f, absolute) in cases {
        let got = gpu.df64_selftest(&x, &y, op).expect("selftest");
        let mut errs: Vec<f64> = x
            .iter()
            .zip(&y)
            .zip(&got)
            .map(|((&xi, &yi), &g)| {
                let xi = f64::from(xi as f32) + f64::from((xi - f64::from(xi as f32)) as f32);
                let yi = f64::from(yi as f32) + f64::from((yi - f64::from(yi as f32)) as f32);
                let exact = f(xi, yi);
                if absolute {
                    (g - exact).abs()
                } else {
                    ((g - exact) / exact).abs()
                }
            })
            .filter(|e| e.is_finite())
            .collect();
        errs.sort_by(f64::total_cmp);
        println!(
            "{name:<5} {:>12.2e} {:>12.2e}",
            errs[errs.len() - 1],
            errs[errs.len() / 2]
        );
    }
}
