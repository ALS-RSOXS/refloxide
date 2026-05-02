//! Complex scalar alias and physical constants.

/// The kernel's working complex scalar type. Aliasing
/// `num_complex::Complex64` keeps the type surface terse and gives
/// a single point of control for any future precision change.
pub type C64 = num_complex::Complex64;

/// Classical electron radius in meters, used by the scattering-
/// factor parameterization in [`crate::types::parameterization`].
pub const CLASSICAL_ELECTRON_RADIUS_M: f64 = 2.8179403262e-15;

/// Speed of light in vacuum, in meters per second.
pub const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;

/// Vacuum permittivity in farads per meter.
pub const VACUUM_PERMITTIVITY_F_M: f64 = 8.854_187_812_8e-12;

/// Convenience constructor for purely real `C64`.
#[inline]
pub fn real(x: f64) -> C64 {
    C64::new(x, 0.0)
}

/// Convenience constructor for purely imaginary `C64`.
#[inline]
pub fn imag(x: f64) -> C64 {
    C64::new(0.0, x)
}
