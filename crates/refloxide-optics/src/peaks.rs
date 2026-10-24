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
    let mut model = Model::new();
    model
        .try_add(kind.component(shape))
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
/// parameter is non-finite, has width, gamma, sigma or fwhm `<= 0`, or has eta outside `[0, 1]`,
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

    /// Unit-amplitude-times-`amp` reference line from an independent `lineshapes` builder.
    fn model_eval(kind: LineKind, shape: &[f64], amp: f64, energies: &[f64]) -> Vec<C> {
        let model = match kind {
            LineKind::Gaussian => Model::new().gaussian(shape[0], shape[1], amp),
            LineKind::Lorentzian => Model::new().lorentzian(shape[0], shape[1], amp),
            LineKind::Voigt => Model::new().voigt(shape[0], shape[1], shape[2], amp),
            LineKind::PseudoVoigt => Model::new().pseudo_voigt(shape[0], shape[1], shape[2], amp),
        };
        model.eval(energies)
    }

    fn assert_close(got: &[C], want: &[C]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            let err = (*g - *w).norm();
            assert!(err <= 1e-12 * w.norm() + 1e-300, "got {g}, want {w}");
        }
    }

    fn run(peaks: &[PeakSpec], n_slices: usize, energies: &[f64]) -> (Vec<C>, Vec<C>) {
        let n = n_slices * energies.len();
        let mut xx = vec![C::new(0.0, 0.0); n];
        let mut zz = vec![C::new(0.0, 0.0); n];
        peak_sum_molecular(peaks, n_slices, energies, &mut xx, &mut zz).unwrap();
        (xx, zz)
    }

    #[test]
    fn distinct_per_slice_rows_match_single_slice_calls_and_model() {
        let energies = linspace(283.0, 286.0, 40);
        let n_e = energies.len();
        let e0 = [284.0, 284.5, 285.0, 285.5];
        let width = [0.3, 0.4, 0.5, 0.6];
        let height = [1.0, 2.0, 0.5, 3.0];
        let f_xx = [1.5, 0.2, 1.0, 0.7];
        let f_zz = [0.0, 0.3, 0.9, 0.4];
        let shape: Vec<f64> = (0..4).flat_map(|s| [e0[s], width[s]]).collect();
        let peak = PeakSpec {
            kind: LineKind::Gaussian,
            shape: &shape,
            height: &height,
            f_xx: &f_xx,
            f_zz: &f_zz,
        };
        let (xx, zz) = run(&[peak], 4, &energies);

        for s in 0..4 {
            let row = [e0[s], width[s]];
            let single = PeakSpec {
                kind: LineKind::Gaussian,
                shape: &row,
                height: &height[s..=s],
                f_xx: &f_xx[s..=s],
                f_zz: &f_zz[s..=s],
            };
            let (sx, sz) = run(&[single], 1, &energies);
            assert_eq!(&xx[s * n_e..(s + 1) * n_e], &sx[..]);
            assert_eq!(&zz[s * n_e..(s + 1) * n_e], &sz[..]);

            let want_xx = model_eval(LineKind::Gaussian, &row, height[s] * f_xx[s], &energies);
            let want_zz = model_eval(LineKind::Gaussian, &row, height[s] * f_zz[s], &energies);
            assert_close(&xx[s * n_e..(s + 1) * n_e], &want_xx);
            assert_close(&zz[s * n_e..(s + 1) * n_e], &want_zz);
        }
    }

    #[test]
    fn mixed_broadcast_and_per_slice_peaks_match_single_slice_calls_and_model() {
        let energies = linspace(283.0, 286.0, 40);
        let n_e = energies.len();
        let n_slices = 3;

        // Peak A: one Lorentzian row shared by every slice (broadcast).
        let a_row = [284.8, 0.2];
        let a_height = [0.7];
        let a_xx = [1.0];
        let a_zz = [0.5];

        // Peak B: one pseudo-Voigt row per slice with distinct scalars.
        let b_rows = [[284.2, 0.5, 0.1], [284.6, 0.4, 0.6], [285.1, 0.6, 0.9]];
        let b_shape: Vec<f64> = b_rows.iter().flatten().copied().collect();
        let b_height = [1.2, 0.4, 2.5];
        let b_xx = [0.3, 1.1, 0.8];
        let b_zz = [0.9, 0.0, 0.2];

        let peaks = [
            PeakSpec {
                kind: LineKind::Lorentzian,
                shape: &a_row,
                height: &a_height,
                f_xx: &a_xx,
                f_zz: &a_zz,
            },
            PeakSpec {
                kind: LineKind::PseudoVoigt,
                shape: &b_shape,
                height: &b_height,
                f_xx: &b_xx,
                f_zz: &b_zz,
            },
        ];
        let (xx, zz) = run(&peaks, n_slices, &energies);

        for s in 0..n_slices {
            let single = [
                PeakSpec {
                    kind: LineKind::Lorentzian,
                    shape: &a_row,
                    height: &a_height,
                    f_xx: &a_xx,
                    f_zz: &a_zz,
                },
                PeakSpec {
                    kind: LineKind::PseudoVoigt,
                    shape: &b_rows[s],
                    height: &b_height[s..=s],
                    f_xx: &b_xx[s..=s],
                    f_zz: &b_zz[s..=s],
                },
            ];
            let (sx, sz) = run(&single, 1, &energies);
            assert_eq!(&xx[s * n_e..(s + 1) * n_e], &sx[..]);
            assert_eq!(&zz[s * n_e..(s + 1) * n_e], &sz[..]);

            let ref_a = model_eval(
                LineKind::Lorentzian,
                &a_row,
                a_height[0] * a_xx[0],
                &energies,
            );
            let ref_b = model_eval(
                LineKind::PseudoVoigt,
                &b_rows[s],
                b_height[s] * b_xx[s],
                &energies,
            );
            let want_xx: Vec<C> = ref_a.iter().zip(&ref_b).map(|(p, q)| p + q).collect();
            assert_close(&xx[s * n_e..(s + 1) * n_e], &want_xx);

            let ref_a = model_eval(
                LineKind::Lorentzian,
                &a_row,
                a_height[0] * a_zz[0],
                &energies,
            );
            let ref_b = model_eval(
                LineKind::PseudoVoigt,
                &b_rows[s],
                b_height[s] * b_zz[s],
                &energies,
            );
            let want_zz: Vec<C> = ref_a.iter().zip(&ref_b).map(|(p, q)| p + q).collect();
            assert_close(&zz[s * n_e..(s + 1) * n_e], &want_zz);
        }
    }
}
