//! Composable SLD primitives for deferred-energy uniaxial stacks.
//!
//! Owns OOC lookup, density scaling, laboratory-frame rotation, and refnx slab
//! row packing so Python scatterers stay thin parameter wrappers.

use num_complex::Complex;

use crate::optics::{interpolate_ooc_linear, isotropic_tensor, lab_diagonal_uniaxial};

type C = Complex<f64>;

/// Density-scaled uniaxial molecular indices from tabulated OOC components.
pub fn molecular_index(density: f64, n_xx: f64, n_ixx: f64, n_zz: f64, n_izz: f64) -> (C, C) {
    (
        density * C::new(n_xx, n_ixx),
        density * C::new(n_zz, n_izz),
    )
}

/// Linear OOC lookup followed by :func:`molecular_index` at ``query_ev``.
pub fn molecular_index_at_ooc(
    energy_ev: &[f64],
    n_xx: &[f64],
    n_ixx: &[f64],
    n_zz: &[f64],
    n_izz: &[f64],
    query_ev: f64,
    density: f64,
) -> (C, C) {
    let v = interpolate_ooc_linear(energy_ev, n_xx, n_ixx, n_zz, n_izz, query_ev);
    molecular_index(density, v[0], v[1], v[2], v[3])
}

/// Laboratory ``(3, 3)`` tensor for one uniaxial orientation (radians).
pub fn uniaxial_lab_tensor(n_mol_xx: C, n_mol_zz: C, orientation_rad: f64) -> [[C; 3]; 3] {
    let [n_o, _, n_e] = lab_diagonal_uniaxial(n_mol_xx, n_mol_zz, orientation_rad);
    [
        [n_o, C::new(0.0, 0.0), C::new(0.0, 0.0)],
        [C::new(0.0, 0.0), n_o, C::new(0.0, 0.0)],
        [C::new(0.0, 0.0), C::new(0.0, 0.0), n_e],
    ]
}

/// Isotropic laboratory ``(3, 3)`` tensor from scalar index ``n``.
pub fn isotropic_lab_tensor(n: C) -> [[C; 3]; 3] {
    isotropic_tensor(n)
}

/// Pack ``[thickness, delta, beta, roughness]`` from a diagonal laboratory tensor.
///
/// The tensor already holds ``delta + i*beta`` directly (the crate-wide
/// convention: `berreman_dielectric` builds `eps = conj(I - 2*tensor)` from
/// it, and `uniaxial_lab_tensor`/`molecular_index_at_ooc` produce it straight
/// from density-scaled OOC delta/beta values). The mean diagonal index is
/// therefore the packed `(delta, beta)` as-is — no `1 - n_avg` correction.
pub fn tensor_to_slab_row(thickness: f64, roughness: f64, tensor: &[[C; 3]; 3]) -> [f64; 4] {
    let n_avg = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
    [thickness, n_avg.re, n_avg.im, roughness]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn molecular_index_scales_density() {
        let (xx, zz) = molecular_index(2.0, 0.01, 0.001, 0.02, 0.002);
        assert_relative_eq!(xx.re, 0.02);
        assert_relative_eq!(zz.re, 0.04);
    }

    #[test]
    fn tensor_to_slab_row_treats_tensor_as_delta_beta_directly() {
        // delta/beta at real-material scale (~1e-6), not a refractive index near 1 —
        // this is the actual convention every caller (uniaxial_lab_tensor,
        // molecular_index_at_ooc) produces.
        let delta = 5.0e-6;
        let beta = 1.0e-7;
        let tensor = isotropic_lab_tensor(C::new(delta, beta));
        let row = tensor_to_slab_row(100.0, 2.0, &tensor);
        assert_relative_eq!(row[0], 100.0);
        assert_relative_eq!(row[1], delta);
        assert_relative_eq!(row[2], beta);
        assert_relative_eq!(row[3], 2.0);
    }

    #[test]
    fn tensor_to_slab_row_accepts_delta_beta_tensor_from_ooc_pipeline() {
        // exercises the real call path: OOC lookup + density scaling ->
        // uniaxial_lab_tensor -> tensor_to_slab_row, confirming delta/beta
        // survive the full pipeline at the right order of magnitude.
        let (n_mol_xx, n_mol_zz) = molecular_index_at_ooc(
            &[500.0, 700.0],
            &[4.0e-6, 4.0e-6],
            &[1.0e-7, 1.0e-7],
            &[8.0e-6, 8.0e-6],
            &[2.0e-7, 2.0e-7],
            700.0,
            1.0,
        );
        let tensor = uniaxial_lab_tensor(n_mol_xx, n_mol_zz, 0.0);
        let row = tensor_to_slab_row(150.0, 3.0, &tensor);
        assert!(
            row[1].abs() < 1e-3,
            "delta={} looks like it went through `1 - n`, not delta+i*beta directly",
            row[1]
        );
        assert!(row[1] > 0.0, "expected a positive delta on the order of 1e-6, got {}", row[1]);
    }
}
