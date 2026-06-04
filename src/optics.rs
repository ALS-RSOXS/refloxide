//! Energy-dependent optical constants and laboratory-frame tensor assembly.
//!
//! Supports deferred evaluation: tabulated OOC curves are interpolated at
//! query energy, then combined with orientation profiles to form diagonal
//! dielectric tensors used by the uniaxial transfer-matrix kernel.

use num_complex::Complex;

type C = Complex<f64>;

/// Piecewise-linear interpolation on a sorted energy grid.
///
/// Returns `[n_xx, n_ixx, n_zz, n_izz]` at `query_ev`. Out-of-range queries
/// clamp to the grid endpoints.
pub fn interpolate_ooc_linear(
    energy_ev: &[f64],
    n_xx: &[f64],
    n_ixx: &[f64],
    n_zz: &[f64],
    n_izz: &[f64],
    query_ev: f64,
) -> [f64; 4] {
    let n = energy_ev.len();
    if n == 0 {
        return [0.0; 4];
    }
    if n == 1 {
        return [n_xx[0], n_ixx[0], n_zz[0], n_izz[0]];
    }
    if query_ev <= energy_ev[0] {
        return [n_xx[0], n_ixx[0], n_zz[0], n_izz[0]];
    }
    if query_ev >= energy_ev[n - 1] {
        return [
            n_xx[n - 1],
            n_ixx[n - 1],
            n_zz[n - 1],
            n_izz[n - 1],
        ];
    }
    let mut hi = 1usize;
    while hi < n && energy_ev[hi] < query_ev {
        hi += 1;
    }
    let lo = hi - 1;
    let t = (query_ev - energy_ev[lo]) / (energy_ev[hi] - energy_ev[lo]);
    [
        n_xx[lo] + t * (n_xx[hi] - n_xx[lo]),
        n_ixx[lo] + t * (n_ixx[hi] - n_ixx[lo]),
        n_zz[lo] + t * (n_zz[hi] - n_zz[lo]),
        n_izz[lo] + t * (n_izz[hi] - n_izz[lo]),
    ]
}

/// Builds laboratory-frame diagonal `(n_o, n_o, n_e)` from uniaxial molecular axes.
///
/// `n_mol_xx` and `n_mol_zz` are complex index values along principal axes;
/// `orientation_rad` is the polar rotation in radians.
pub fn lab_diagonal_uniaxial(n_mol_xx: C, n_mol_zz: C, orientation_rad: f64) -> [C; 3] {
    let c = orientation_rad.cos();
    let cos2 = c * c;
    let sin2 = 1.0 - cos2;
    let n_o = (n_mol_xx * (1.0 + cos2) + n_mol_zz * sin2) / 2.0;
    let n_e = n_mol_xx * sin2 + n_mol_zz * cos2;
    [n_o, n_o, n_e]
}

/// Batch laboratory tensors as length-`n` vectors of diagonal `(n_o, n_o, n_e)`.
pub fn lab_diagonal_uniaxial_batch(
    n_mol_xx: C,
    n_mol_zz: C,
    orientations_rad: &[f64],
) -> Vec<[C; 3]> {
    orientations_rad
        .iter()
        .map(|&theta| lab_diagonal_uniaxial(n_mol_xx, n_mol_zz, theta))
        .collect()
}

/// Packs diagonal laboratory indices into `(3, 3)` tensors with off-diagonals zero.
pub fn pack_diagonal_tensors(diagonals: &[[C; 3]]) -> Vec<[[C; 3]; 3]> {
    diagonals
        .iter()
        .map(|[n_o, _, n_e]| {
            [
                [*n_o, C::new(0.0, 0.0), C::new(0.0, 0.0)],
                [C::new(0.0, 0.0), *n_o, C::new(0.0, 0.0)],
                [C::new(0.0, 0.0), C::new(0.0, 0.0), *n_e],
            ]
        })
        .collect()
}

/// Isotropic `(3, 3)` tensor from a scalar laboratory index `n`.
pub fn isotropic_tensor(n: C) -> [[C; 3]; 3] {
    [
        [n, C::new(0.0, 0.0), C::new(0.0, 0.0)],
        [C::new(0.0, 0.0), n, C::new(0.0, 0.0)],
        [C::new(0.0, 0.0), C::new(0.0, 0.0), n],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ooc_linear_interp_midpoint() {
        let e = [250.0, 300.0];
        let xx = [1.0, 3.0];
        let ix = [0.0, 0.0];
        let zz = [2.0, 4.0];
        let iz = [0.0, 0.0];
        let v = interpolate_ooc_linear(&e, &xx, &ix, &zz, &iz, 275.0);
        assert_relative_eq!(v[0], 2.0);
        assert_relative_eq!(v[2], 3.0);
    }
}
