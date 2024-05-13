#![doc = include_str!("../README.md")]
//!
//! # Features
//! This crate supports no-std solution. Disable the "std" feature will
//! enable it.
//! ```toml
//! default-features = false
//! ```
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
pub extern crate nalgebra as na;

#[doc(inline)]
pub use crate::posed::{MotionSig, PosedEfd, PosedEfd1, PosedEfd2, PosedEfd3};
pub use crate::{curve::*, dim::*, dist::*, efd::*, geo::*};

mod curve;
mod dim;
mod dist;
mod efd;
mod geo;
pub mod posed;
pub mod tests;
pub mod util;
