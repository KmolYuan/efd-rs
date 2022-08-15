//! This crate implements Elliptical Fourier Descriptor (EFD) and its related functions.
//!
//! Simple usage of resampling circle:
//!
//! ```
//! use efd::Efd2;
//!
//! let curve = vec![[0.; 2], [1.; 2], [2.; 2], [3.; 2]];
//! let new_curve = Efd2::from_curve(&curve, None).generate(20);
//! # assert_eq!(new_curve.len(), 20);
//! ```
//!
//! # Features
//!
//! This crate support no-std solution via using "libm", a crate provide pure-rust math functions.
//! Disable the "std" feature will automatic enable it.
#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
#[cfg(not(feature = "std"))]
extern crate core as std; // for `ndarray::s!` macro

pub use crate::{efd::*, error::*, geo_info::*};

mod efd;
mod error;
mod geo_info;
pub mod tests;
