#![doc = include_str!("../README.md")]
//! # Features
//!
//! This crate support no-std solution via using "libm", a crate provide
//! pure-rust math functions. Disable the "std" feature will automatic enable
//! it.
#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
pub extern crate nalgebra as na;

pub use crate::efd::*;
#[doc(no_inline)]
pub use crate::{curve::*, dim::*, transform::*, utility::*};

pub mod curve;
pub mod dim;
mod efd;
pub mod tests;
pub mod transform;
pub mod utility;
