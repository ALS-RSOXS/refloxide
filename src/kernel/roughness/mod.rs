//! Roughness models and dispatcher.
//!
//! Three models are implemented here, all conforming to the
//! [`traits::RoughnessModel`] trait. The dispatcher in
//! [`dispatch`] consumes a [`crate::stack::RoughnessSpec`] per
//! interface and selects the appropriate model.

pub mod debye_waller;
pub mod dispatch;
pub mod graded;
pub mod nevot_croce;
pub mod traits;

pub use debye_waller::DebyeWaller;
pub use dispatch::{auto_select, RoughnessDispatchResult, RoughnessInterfaceLog};
pub use graded::Graded;
pub use nevot_croce::NevotCroce;
pub use traits::RoughnessModel;
