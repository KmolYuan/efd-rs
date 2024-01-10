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
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
pub extern crate nalgebra as na;

#[doc(no_inline)]
pub use crate::{curve::*, dim::*, geo::*};
pub use crate::{efd::*, posed::*};
pub use dist::Distance;

pub mod curve;
pub mod dim;
mod dist;
mod efd;
pub mod geo;
mod posed;
pub mod tests;
pub mod util;
