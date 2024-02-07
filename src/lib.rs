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

pub use crate::{curve::*, dim::*, dist::*, efd::*, geo::*, posed::*};

/// Calculate the number of harmonics.
///
/// The number of harmonics is calculated by the minimum length of the curves.
/// And if the curve is open, the number is doubled.
///
/// ```
/// use efd::harmonic;
///
/// let is_open = true;
/// assert_eq!(harmonic!(is_open, [0.; 2], [0.; 3]), 10);
/// ```
#[macro_export]
macro_rules! harmonic {
    ($is_open:expr, $curve1:expr $(, $curve2:expr)*) => {{
        let len = $curve1.len()$( + $curve2.len())*;
        if $is_open { len * 2 } else { len }
    }};
}

/// Calculate the number of harmonics with the Nyquist frequency.
///
/// This macro is similar to [`harmonic!`], but the number of harmonics is half
/// if the given curve meets the Nyquist–Shannon sampling theorem.
#[macro_export]
macro_rules! harmonic_nyquist {
    ($is_open:expr, $curve1:expr $(, $curve2:expr)*) => {{
        let len = $curve1.len()$( + $curve2.len())*;
        if $is_open { len } else { len / 2 }
    }};
}

mod curve;
mod dim;
mod dist;
mod efd;
mod geo;
mod posed;
pub mod tests;
pub mod util;
