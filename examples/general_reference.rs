//! Reference dump for validating the general-tensor 4x4 engine.
//!
//! Prints tilted-uniaxial, rotated, biaxial, and graded-tilt stacks (full
//! susceptibility tensors) with `solve_point_general` reflectance in `f64`
//! and `f32` at 40 q-points each. Pipe the output into
//! `examples/general_reference_mpmath.py`, which recomputes every point
//! with a 50-digit transfer-matrix product (matrix exponentials, no mode
//! sorting) and reports the maximum relative error per channel:
//!
//! ```text
//! cargo run --profile perf --no-default-features --example general_reference > ref.txt
//! uv run --with mpmath python examples/general_reference_mpmath.py ref.txt
//! ```

use num_complex::Complex;
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
fn iso(c: C, t: f64) -> GeneralLayer<f64> {
    let z = C::new(0.0, 0.0);
    GeneralLayer {
        chi: [[c, z, z], [z, c, z], [z, z, c]],
        thickness: t,
        sigma: 0.0,
    }
}
fn main() {
    let vac = iso(C::new(0.0, 0.0), 0.0);
    let sio2 = iso(chi(1.5e-3, 2e-4), 15.0);
    let si = iso(chi(1.2e-3, 1.5e-4), 0.0);
    let uni = [chi(2e-3, 1.2e-3), chi(2e-3, 1.2e-3), chi(1e-3, 2.5e-3)];
    let bi = [
        chi(2.4e-3, 0.9e-3),
        chi(1.6e-3, 1.6e-3),
        chi(0.8e-3, 2.8e-3),
    ];
    let stacks: Vec<(&str, Vec<GeneralLayer<f64>>)> = vec![
        (
            "tilted uniaxial 40deg, phi=0 (xz plane)",
            vec![
                vac,
                GeneralLayer {
                    chi: tensor(uni, rot(0.7, 0.0, 0.0)),
                    thickness: 250.0,
                    sigma: 0.0,
                },
                sio2,
                si,
            ],
        ),
        (
            "tilted uniaxial 55deg, phi=35deg",
            vec![
                vac,
                GeneralLayer {
                    chi: tensor(uni, rot(0.96, 0.61, 0.0)),
                    thickness: 250.0,
                    sigma: 0.0,
                },
                sio2,
                si,
            ],
        ),
        (
            "biaxial, euler (0.4,0.9,1.3)",
            vec![
                vac,
                GeneralLayer {
                    chi: tensor(bi, rot(0.9, 0.4, 1.3)),
                    thickness: 180.0,
                    sigma: 0.0,
                },
                sio2,
                si,
            ],
        ),
        ("graded tilt 20 x 8 A + uniaxial-z", {
            let mut v = vec![vac];
            for j in 0..20 {
                let f = j as f64 / 19.0;
                v.push(GeneralLayer {
                    chi: tensor(uni, rot(0.2 + 1.1 * f, 0.3 * f, 0.0)),
                    thickness: 8.0,
                    sigma: 0.0,
                });
            }
            v.push(GeneralLayer {
                chi: tensor(uni, rot(0.0, 0.0, 0.0)),
                thickness: 30.0,
                sigma: 0.0,
            });
            v.push(sio2);
            v.push(si);
            v
        }),
    ];
    let k0 = 2.0 * std::f64::consts::PI / (12398.4193 / 285.0);
    println!("K {k0:e}");
    for (si_, (name, st)) in stacks.iter().enumerate() {
        println!("N {si_} {name}");
        for l in st {
            let mut row = format!("L {si_} {:e}", l.thickness);
            for i in 0..3 {
                for j in 0..3 {
                    row += &format!(" {:e} {:e}", l.chi[i][j].re, l.chi[i][j].im);
                }
            }
            println!("{row}");
        }
        let st32: Vec<GeneralLayer<f32>> = st
            .iter()
            .map(|l| GeneralLayer {
                chi: l
                    .chi
                    .map(|r| r.map(|c| Complex::new(c.re as f32, c.im as f32))),
                thickness: l.thickness as f32,
                sigma: 0.0,
            })
            .collect();
        for i in 0..40 {
            let q = 0.01 + (0.98 * 2.0 * k0 - 0.01) * i as f64 / 39.0;
            let r = solve_point_general(q, k0, st).unwrap_or([[f64::NAN; 2]; 2]);
            let r32 = solve_point_general(q as f32, k0 as f32, &st32).unwrap_or([[f32::NAN; 2]; 2]);
            println!(
                "Q {si_} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e}",
                q, r[0][0], r[1][1], r[0][1], r[1][0], r32[0][0], r32[1][1], r32[0][1], r32[1][0]
            );
        }
    }
}
