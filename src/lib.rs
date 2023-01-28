//! This crate implements 2D/3D Elliptical Fourier Descriptor (EFD) and its
//! related functions.
//!
//! This crate support both `Vec<[f64; D]>` and `&[[f64; D]]` as input type via
//! `Cow<[[f64; D]]>` ([`CowCurve`]), and the first coordinate must be close to
//! the last with [`closed_curve()`].
//!
//! ```
//! let curve = vec![
//!     [0., 0.],
//!     [1., 1.],
//!     [2., 2.],
//!     [3., 3.],
//!     [2., 2.],
//!     [1., 1.],
//!     [0., 0.],
//! ];
//! let described_curve = efd::Efd2::from_curve(curve).unwrap().generate(20);
//! # assert_eq!(described_curve.len(), 20);
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
pub extern crate nalgebra as na;
pub extern crate ndarray;

pub use crate::{efd::*, efd_dim::*, error::*, transform::*, utility::*};

mod efd;
mod efd_dim;
mod error;
pub mod tests;
mod transform;
mod utility;
