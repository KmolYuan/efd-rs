//! This crate implements Elliptical Fourier Descriptor (EFD) and its related
//! functions.
//!
//! ```
//! use efd::Efd2;
//!
//! let curve = vec![[0.; 2], [1.; 2], [2.; 2], [3.; 2]];
//! let new_curve = Efd2::from_curve_gate(curve, None).unwrap().generate(20);
//! # assert_eq!(new_curve.len(), 20);
//! ```
//!
//! # Features
//!
//! This crate support no-std solution via using "libm", a crate provide
//! pure-rust math functions. Disable the "std" feature will automatic enable
//! it.
#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
#[cfg(not(feature = "std"))]
extern crate core as std; // for `ndarray::s!` macro

pub use crate::{efd::*, error::*, geo_info::*};

/// Copy-on-write curve type.
///
/// This crate support both `Vec<[f64; 2]>` and `&[[f64; 2]]` as input type.
pub type CowCurve<'a> = alloc::borrow::Cow<'a, [[f64; 2]]>;

mod efd;
mod error;
mod geo_info;
pub mod tests;
