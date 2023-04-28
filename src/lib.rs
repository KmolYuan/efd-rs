#![doc = include_str!("../README.md")]
//! # Features
//!
//! This crate support no-std solution via using "libm", a crate provide
//! pure-rust math functions. Disable the "std" feature will automatic enable
//! it.
#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
pub extern crate nalgebra as na;

pub use crate::{curve::*, efd::*, efd_dim::*, transform::*, utility::*};

mod curve;
mod efd;
mod efd_dim;
pub mod tests;
mod transform;
mod utility;
