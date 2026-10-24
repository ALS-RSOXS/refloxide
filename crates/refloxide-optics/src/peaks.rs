//! Peak-sum molecular optical response from `lineshapes` line profiles.
//!
//! Each peak is a unit-amplitude lineshape (result `delta + i beta`) scaled per
//! slice by `height * f_xx` (or `height * f_zz`). A peak with a single shape row
//! is evaluated once per energy and reused for every slice; a peak with one shape
//! row per slice is evaluated slice by slice.

use lineshapes::{Component, Model};
use num_complex::Complex;
use refloxide_core::{RefloxideError, Result};

/// Lineshape family for a peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    /// Gaussian with shape `[e0, width]`, `width` the 1/e half-width.
    Gaussian,
    /// Lorentzian with shape `[e0, gamma]`, `gamma` the HWHM.
    Lorentzian,
    /// Voigt with shape `[e0, sigma, gamma]`.
    Voigt,
    /// Pseudo-Voigt with shape `[e0, fwhm, eta]`.
    PseudoVoigt,
}

impl LineKind {
    /// Parse a lineshape name: `"gaussian"`, `"lorentzian"`, `"voigt"` or `"pseudo_voigt"`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "gaussian" => Some(Self::Gaussian),
            "lorentzian" => Some(Self::Lorentzian),
            "voigt" => Some(Self::Voigt),
            "pseudo_voigt" => Some(Self::PseudoVoigt),
            _ => None,
        }
    }

    /// Number of shape parameters for one slice of this lineshape.
    pub fn n_shape(self) -> usize {
        match self {
            Self::Gaussian | Self::Lorentzian => 2,
            Self::Voigt | Self::PseudoVoigt => 3,
        }
    }

    /// Unit-amplitude `lineshapes` component for one shape row.
    fn component(self, shape: &[f64]) -> Component {
        match self {
            Self::Gaussian => Component::Gaussian {
                e0: shape[0],
                width: shape[1],
                amp: 1.0,
            },
            Self::Lorentzian => Component::Lorentzian {
                e0: shape[0],
                gamma: shape[1],
                amp: 1.0,
            },
            Self::Voigt => Component::Voigt {
                e0: shape[0],
                sigma: shape[1],
                gamma: shape[2],
                amp: 1.0,
            },
            Self::PseudoVoigt => Component::PseudoVoigt {
                e0: shape[0],
                fwhm: shape[1],
                eta: shape[2],
                amp: 1.0,
            },
        }
    }
}

/// One peak, with per-slice or broadcast parameters.
#[derive(Clone, Copy, Debug)]
pub struct PeakSpec<'a> {
    /// Lineshape family.
    pub kind: LineKind,
    /// Shape parameters: `n_shape` values (broadcast to all slices) or `n_slices * n_shape`
    /// values in row-major (slice, parameter) order.
    pub shape: &'a [f64],
    /// Peak height: length 1 (broadcast) or `n_slices`.
    pub height: &'a [f64],
    /// Molecular `xx` projection: length 1 or `n_slices`.
    pub f_xx: &'a [f64],
    /// Molecular `zz` projection: length 1 or `n_slices`.
    pub f_zz: &'a [f64],
}

fn invalid_param(name: &str, value: f64) -> RefloxideError {
    RefloxideError::InvalidParameter(format!("{name} must be finite, got {value}"))
}

/// Read `values[s]` for a broadcast-or-per-slice array of length 1 or `n_slices`.
fn slice_value(values: &[f64], s: usize) -> f64 {
    if values.len() == 1 {
        values[0]
    } else {
        values[s]
    }
}

/// Check that a broadcast-or-per-slice array has length 1 or `n_slices` and that
/// every value is finite.
fn check_slice_array(name: &str, values: &[f64], n_slices: usize) -> Result<()> {
    if values.len() != 1 && values.len() != n_slices {
        return Err(RefloxideError::InvalidShape(format!(
            "{name} has length {}, expected 1 or {n_slices}",
            values.len()
        )));
    }
    for &v in values {
        if !v.is_finite() {
            return Err(invalid_param(name, v));
        }
    }
    Ok(())
}

/// Validate one shape row with the `lineshapes` domain rules and build its unit model.
fn unit_model(kind: LineKind, shape: &[f64]) -> Result<Model> {
    let component = kind.component(shape);
    component
        .validate()
        .map_err(RefloxideError::InvalidParameter)?;
    let mut model = Model::new();
    model
        .try_add(component)
        .map_err(RefloxideError::InvalidParameter)?;
    Ok(model)
}

/// Accumulate the molecular peak sum into slab-major `(slice, energy)` buffers.
///
/// Each peak contributes `height[s] * f_xx[s] * line(E)` to `out_xx` and
/// `height[s] * f_zz[s] * line(E)` to `out_zz`, where `line` is the unit-amplitude
/// `lineshapes` profile (`delta + i beta`). A peak with one shape row evaluates
/// its line once per energy; a peak with one row per slice evaluates per slice.
///
/// # Errors
///
/// Returns [`RefloxideError::InvalidParameter`] naming the lineshape when a shape
/// parameter is non-finite, has width or gamma `<= 0`, or has eta outside `[0, 1]`,
/// and when a height or projection is non-finite. Returns
/// [`RefloxideError::InvalidShape`] on length mismatches.
pub fn peak_sum_molecular(
    peaks: &[PeakSpec],
    n_slices: usize,
    energies: &[f64],
    out_xx: &mut [Complex<f64>],
    out_zz: &mut [Complex<f64>],
) -> Result<()> {
    let n_e = energies.len();
    let expected = n_slices * n_e;
    if out_xx.len() != expected || out_zz.len() != expected {
        return Err(RefloxideError::InvalidShape(format!(
            "output buffers have lengths {} and {}, expected {expected} (n_slices * energies)",
            out_xx.len(),
            out_zz.len()
        )));
    }
    out_xx.fill(Complex::new(0.0, 0.0));
    out_zz.fill(Complex::new(0.0, 0.0));

    let mut line = vec![Complex::new(0.0, 0.0); n_e];
    for peak in peaks {
        let n_shape = peak.kind.n_shape();
        let broadcast = if peak.shape.len() == n_shape {
            true
        } else if peak.shape.len() == n_slices * n_shape {
            false
        } else {
            return Err(RefloxideError::InvalidShape(format!(
                "{:?} shape has length {}, expected {n_shape} or n_slices * {n_shape}",
                peak.kind,
                peak.shape.len()
            )));
        };
        check_slice_array("height", peak.height, n_slices)?;
        check_slice_array("f_xx", peak.f_xx, n_slices)?;
        check_slice_array("f_zz", peak.f_zz, n_slices)?;

        if broadcast {
            // Validate the shared row once and evaluate it once for every slice.
            let model = unit_model(peak.kind, peak.shape)?;
            if n_slices > 0 {
                model.eval_into(energies, &mut line);
            }
        }
        for s in 0..n_slices {
            if !broadcast {
                let row = &peak.shape[s * n_shape..(s + 1) * n_shape];
                unit_model(peak.kind, row)?.eval_into(energies, &mut line);
            }
            let h = slice_value(peak.height, s);
            let w_xx = h * slice_value(peak.f_xx, s);
            let w_zz = h * slice_value(peak.f_zz, s);
            let base = s * n_e;
            for (e, &l) in line.iter().enumerate() {
                out_xx[base + e] += l * w_xx;
                out_zz[base + e] += l * w_zz;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lineshapes::Model;

    type C = Complex<f64>;

    fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect()
    }

    #[test]
    fn gaussian_matches_lineshapes_model() {
        let energies = linspace(283.0, 286.0, 50);
        let shape = [284.5, 0.4];
        let peak = PeakSpec {
            kind: LineKind::Gaussian,
            shape: &shape,
            height: &[2.0],
            f_xx: &[1.5],
            f_zz: &[0.0],
        };
        let mut out_xx = vec![C::new(0.0, 0.0); energies.len()];
        let mut out_zz = vec![C::new(0.0, 0.0); energies.len()];
        peak_sum_molecular(&[peak], 1, &energies, &mut out_xx, &mut out_zz).unwrap();

        let expected = Model::new().gaussian(284.5, 0.4, 3.0).eval(&energies);
        for ((got, want), zz) in out_xx.iter().zip(&expected).zip(&out_zz) {
            let err = (*got - *want).norm();
            assert!(err <= 1e-12 * want.norm(), "got {got}, want {want}");
            assert_eq!(*zz, C::new(0.0, 0.0));
        }
    }

    #[test]
    fn broadcast_equals_per_slice() {
        let energies = linspace(283.0, 286.0, 50);
        let n_slices = 4;
        let row = [284.5, 0.4];
        let repeated: Vec<f64> = row.iter().copied().cycle().take(n_slices * 2).collect();
        let broadcast = PeakSpec {
            kind: LineKind::Gaussian,
            shape: &row,
            height: &[2.0],
            f_xx: &[1.5],
            f_zz: &[0.3],
        };
        let per_slice = PeakSpec {
            kind: LineKind::Gaussian,
            shape: &repeated,
            height: &[2.0; 4],
            f_xx: &[1.5; 4],
            f_zz: &[0.3; 4],
        };
        let n = n_slices * energies.len();
        let mut a_xx = vec![C::new(0.0, 0.0); n];
        let mut a_zz = vec![C::new(0.0, 0.0); n];
        let mut b_xx = vec![C::new(0.0, 0.0); n];
        let mut b_zz = vec![C::new(0.0, 0.0); n];
        peak_sum_molecular(&[broadcast], n_slices, &energies, &mut a_xx, &mut a_zz).unwrap();
        peak_sum_molecular(&[per_slice], n_slices, &energies, &mut b_xx, &mut b_zz).unwrap();
        assert_eq!(a_xx, b_xx);
        assert_eq!(a_zz, b_zz);
    }
}
