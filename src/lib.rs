//! This crate implements Elliptical Fourier Descriptor (EFD) and its related
//! functions.
//!
//! This crate support both `Vec<[f64; 2]>` and `&[[f64; 2]]` as input type via
//! `AsRef<[[f64; 2]]>`.
//!
//! ```
//! let curve = vec![[0.; 2], [1.; 2], [2.; 2], [3.; 2]];
//! let new_curve = efd::Efd2::from_curve_gate(curve, None)
//!     .unwrap()
//!     .generate(20);
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

pub use crate::{efd::*, error::*, transform::*};

mod efd;
mod error;
pub mod tests;
mod transform;
