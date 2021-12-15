//! This crate implements Elliptical Fourier Descriptor (EFD) and its related functions.
//!
//! Reference: Kuhl, FP and Giardina, CR (1982). Elliptic Fourier features of
//! a closed contour. Computer graphics and image processing, 18(3), 236-258.
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
//! assert_eq!(circle.shape(), &[10, 2]);
//! let curve = circle
//!     .axis_iter(Axis(0))
//!     .map(|c| [c[0], c[1]])
//!     .collect::<Vec<_>>();
//! let new_curve = Efd::from_curve(&curve, None).generate(20);
//! assert_eq!(new_curve.len(), 20);
//! ```
//!
//! Arrays have "owned" and "view" two data types, all functions are compatible.
#![warn(missing_docs)]
pub use crate::efd::*;

mod efd;
mod element_opt;
pub mod tests;
