//! Complex scalar alias and physical constants.

/// The kernel's working complex scalar type. Aliasing
/// `num_complex::Complex64` keeps the type surface terse and gives
/// a single point of control for any future precision change.
pub type C64 = num_complex::Complex64;

/// Classical electron radius in meters (CODATA 2018 via
/// [`physical_constants`]), used by the scattering-factor
/// parameterization in [`crate::types::parameterization`].
pub const CLASSICAL_ELECTRON_RADIUS_M: f64 =
    physical_constants::CLASSICAL_ELECTRON_RADIUS;

/// Speed of light in vacuum, in meters per second (CODATA 2018).
pub const SPEED_OF_LIGHT_M_S: f64 = physical_constants::SPEED_OF_LIGHT_IN_VACUUM;

/// Vacuum wavelength in meters from angular frequency ``omega_rad_per_s``.
#[inline]
pub fn wavelength_m_from_omega(omega_rad_per_s: f64) -> f64 {
    2.0 * std::f64::consts::PI * SPEED_OF_LIGHT_M_S / omega_rad_per_s
}

/// Vacuum electric permittivity in farads per meter (CODATA 2018).
pub const VACUUM_PERMITTIVITY_F_M: f64 =
    physical_constants::VACUUM_ELECTRIC_PERMITTIVITY;

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
