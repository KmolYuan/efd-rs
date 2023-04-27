#![doc = include_str!("../README.md")]
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

pub use crate::{curve::*, efd::*, efd_dim::*, error::*, transform::*, utility::*};

mod curve;
mod efd;
mod efd_dim;
mod error;
pub mod tests;
mod transform;
mod utility;
