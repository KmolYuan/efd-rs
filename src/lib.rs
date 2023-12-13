#![doc = include_str!("../README.md")]
//!
//! # Features
//!
//! This crate supports no-std solution. Disable the "std" feature will
//! enable it.
//!
//! ```toml
//! default-features = false
//! ```
//!
//! An extra `curve_diff` features provides some functions to calculate the
//! difference between two curves. Requires the `interp` crate.
//!
//! ```toml
//! features = ["curve_diff"]
//! ```
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
pub extern crate nalgebra as na;

pub use crate::efd::*;
#[doc(no_inline)]
pub use crate::{curve::*, dim::*, geo::*};
pub use dist::Distance;

pub mod curve;
pub mod dim;
mod dist;
mod efd;
pub mod geo;
pub mod tests;
pub mod util;
