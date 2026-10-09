//! Single- vs double-precision error study for the uniaxial kernel.
//!
//! Evaluates representative soft X-ray stacks through
//! `refloxide::kernel::solve_flat` in `f64` and `f32` and reports relative
//! error of `R_ss` and `R_pp` against the `f64` result. Run with
//! `cargo run --release --no-default-features --example kernel_precision`.

use num_complex::Complex;
use refloxide::kernel::{solve_flat, solve_point_recursive, solve_point_rmatrix, LayerCoeffs};

const HC_EV_ANGSTROM: f64 = 12398.4193;

fn chi_from_delta_beta(delta: f64, beta: f64) -> Complex<f64> {
    Complex::new(-2.0 * delta, 2.0 * beta)
}

fn slab(o: (f64, f64), e: (f64, f64), thickness: f64, sigma: f64) -> LayerCoeffs<f64> {
    LayerCoeffs {
        chi_o: chi_from_delta_beta(o.0, o.1),
        chi_e: chi_from_delta_beta(e.0, e.1),
        thickness,
        sigma,
    }
}

fn substrate(film: Vec<LayerCoeffs<f64>>) -> Vec<LayerCoeffs<f64>> {
    let vac = slab((0.0, 0.0), (0.0, 0.0), 0.0, 0.0);
    let sio2 = slab((1.5e-3, 2.0e-4), (1.5e-3, 2.0e-4), 15.0, 3.0);
    let si = slab((1.2e-3, 1.5e-4), (1.2e-3, 1.5e-4), 0.0, 2.0);
    let mut stack = vec![vac];
    stack.extend(film);
    stack.push(sio2);
    stack.push(si);
    stack
}

fn uniform_film(thickness: f64) -> Vec<LayerCoeffs<f64>> {
    vec![slab((2.0e-3, 1.2e-3), (1.0e-3, 2.5e-3), thickness, 4.0)]
}

fn graded_film(n: usize, dz: f64) -> Vec<LayerCoeffs<f64>> {
    (0..n)
        .map(|j| {
            let f = j as f64 / n as f64;
            slab(
                (2.0e-3 * (1.0 - 0.3 * f), 1.2e-3 * (1.0 + 0.5 * f)),
                (1.0e-3 * (1.0 + f), 2.5e-3 * (1.0 - 0.4 * f)),
                dz,
                if j == 0 { 4.0 } else { 0.0 },
            )
        })
        .collect()
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

fn rel_err(reference: &[[f64; 2]], test: &[[f64; 2]], ch: usize) -> (f64, f64) {
    let mut rel: Vec<f64> = reference
        .iter()
        .zip(test)
        .map(|(a, b)| ((b[ch] - a[ch]) / a[ch]).abs())
        .map(|v| if v.is_finite() { v } else { f64::INFINITY })
        .collect();
    rel.sort_by(|a, b| a.total_cmp(b));
    (rel[(rel.len() * 99) / 100], rel[rel.len() - 1])
}

fn diag<T: Copy + Into<f64>>(r: &[[T; 2]; 2]) -> [f64; 2] {
    [r[0][0].into(), r[1][1].into()]
}

fn main() {
    let energies: Vec<f64> = vec![250.0, 283.0, 285.0, 290.0, 320.0];
    let cases: Vec<(&str, Vec<LayerCoeffs<f64>>)> = vec![
        ("uniform 200 A", substrate(uniform_film(200.0))),
        ("uniform 1000 A", substrate(uniform_film(1000.0))),
        ("uniform 3000 A", substrate(uniform_film(3000.0))),
        ("graded 100 x 2 A", substrate(graded_film(100, 2.0))),
        ("graded 400 x 2.5 A", substrate(graded_film(400, 2.5))),
    ];

    println!("relative error of R vs f64 4x4 reference: p99 / max over 2000 q-points");
    println!(
        "{:<19} {:>6} {:>3} {:>9} {:>19} {:>19} {:>19} {:>19} {:>19}",
        "stack",
        "E(eV)",
        "pol",
        "min R",
        "f64 recursive",
        "f64 4x4 rmatrix",
        "f32 4x4",
        "f32 recursive",
        "f32 4x4 rmatrix"
    );
    for (name, stack) in &cases {
        let n_layers = stack.len();
        let stack32 = to_f32(stack);
        for &energy in &energies {
            let k0 = 2.0 * std::f64::consts::PI / (HC_EV_ANGSTROM / energy);
            let q_max = 0.98 * 2.0 * k0;
            let q: Vec<f64> = (0..2000)
                .map(|i| 0.005 + (q_max - 0.005) * i as f64 / 1999.0)
                .collect();
            let q32: Vec<f32> = q.iter().map(|&v| v as f32).collect();
            let reference: Vec<[f64; 2]> = solve_flat(&q, &[k0], stack, n_layers, false)
                .expect("f64 4x4 solve")
                .iter()
                .map(|r| diag(&r.0))
                .collect();
            let rec64: Vec<[f64; 2]> = q
                .iter()
                .map(|&qi| diag(&solve_point_recursive(qi, k0, stack)))
                .collect();
            let tmm32: Vec<[f64; 2]> =
                match solve_flat(&q32, &[k0 as f32], &stack32, n_layers, false) {
                    Ok(r) => r.iter().map(|r| diag(&r.0)).collect(),
                    Err(_) => vec![[f64::NAN; 2]; q.len()],
                };
            let rmx64: Vec<[f64; 2]> = q
                .iter()
                .map(|&qi| match solve_point_rmatrix(qi, k0, stack) {
                    Ok(r) => diag(&r),
                    Err(_) => [f64::NAN; 2],
                })
                .collect();
            let rmx32: Vec<[f64; 2]> = q32
                .iter()
                .map(|&qi| match solve_point_rmatrix(qi, k0 as f32, &stack32) {
                    Ok(r) => diag(&r),
                    Err(_) => [f64::NAN; 2],
                })
                .collect();
            let rec32: Vec<[f64; 2]> = q32
                .iter()
                .map(|&qi| diag(&solve_point_recursive(qi, k0 as f32, &stack32)))
                .collect();
            for (pol, ch) in [("pp", 0), ("ss", 1)] {
                let r_min = reference
                    .iter()
                    .map(|r| r[ch])
                    .fold(f64::INFINITY, f64::min);
                let cols: Vec<String> = [&rec64, &rmx64, &tmm32, &rec32, &rmx32]
                    .iter()
                    .map(|t| {
                        let (p99, max) = rel_err(&reference, t, ch);
                        format!("{p99:>9.1e} {max:>9.1e}")
                    })
                    .collect();
                println!(
                    "{name:<19} {energy:>6.1} {pol:>3} {r_min:>9.1e} {}",
                    cols.join(" ")
                );
            }
        }
    }
}
