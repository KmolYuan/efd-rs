//! This crate implements Elliptical Fourier Descriptor (EFD) and its related functions.
//!
//! Simple usage of resampling circle:
//!
//! ```
//! use efd::Efd;
//! use ndarray::{stack, Array1, Axis};
//! use std::f64::consts::TAU;
//!
//! const N: usize = 10;
//! let circle = stack![
//!     Axis(1),
//!     Array1::linspace(0., TAU, N).mapv(f64::cos),
//!     Array1::linspace(0., TAU, N).mapv(f64::sin)
//! ];
//! # assert_eq!(circle.shape(), &[10, 2]);
//! let curve = circle
//!     .axis_iter(Axis(0))
//!     .map(|c| [c[0], c[1]])
//!     .collect::<Vec<_>>();
//! let new_curve = Efd::from_curve(&curve, None).generate(20);
//! # assert_eq!(new_curve.len(), 20);
//! ```
#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("please enable math function from either \"std\" or \"libm\"");

pub use crate::{efd::*, geo_info::*};

mod efd;
mod geo_info;
mod math;
pub mod tests;
