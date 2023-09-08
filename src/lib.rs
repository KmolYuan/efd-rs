#![doc = include_str!("../README.md")]
//!
//! # Features
//!
//! This crate supports no-std solution via using "libm", a crate provides
//! pure-rust math functions. Disable the "std" feature will automatically
//! enable it.
//!
//! ```toml
//! default-features = false
//! ```
#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
pub extern crate nalgebra as na;

pub use crate::efd::*;
#[doc(no_inline)]
pub use crate::{curve::*, dim::*, distance::*, transform::*, utility::*};

pub mod curve;
pub mod dim;
pub mod distance;
mod efd;
pub mod tests;
pub mod transform;
pub mod utility;
