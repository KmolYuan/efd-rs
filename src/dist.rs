//! Distance trait.
use crate::{Efd, EfdDim};
use alloc::vec::Vec;

/// Be able to calculate the distance between two instances.
///
/// Each data number has the same weight by default, but you can change it by
/// implementing [`Self::err_buf()`].
pub trait Distance: Sized {
    /// Calculate the error between each pair of datas.
    fn err_buf(&self, rhs: &Self) -> Vec<f64>;

    /// Calculate the square error.
    #[must_use]
    fn square_err(&self, rhs: &Self) -> f64 {
        self.err_buf(rhs).into_iter().map(|x| x * x).sum()
    }

    /// Calculate the L0 norm of the error.
    #[must_use]
    fn l0_norm(&self, rhs: &Self) -> f64 {
        let err_buf = self.err_buf(rhs);
        let len = err_buf.len();
        err_buf
            .into_iter()
            .filter(|x| x.abs() < f64::EPSILON)
            .count() as f64
            / len as f64
    }

    /// Calculate the L1 norm of the error.
    #[must_use]
    fn l1_norm(&self, rhs: &Self) -> f64 {
        self.err_buf(rhs).into_iter().map(|x| x.abs()).sum()
    }

    /// Calculate the L2 norm of the error.
    #[must_use]
    fn l2_norm(&self, rhs: &Self) -> f64 {
        self.square_err(rhs).sqrt()
    }

    /// Calculate the Lp norm of the error.
    #[must_use]
    fn lp_norm(&self, rhs: &Self, p: i32) -> f64 {
        self.err_buf(rhs)
            .into_iter()
            .map(|x| x.abs().powi(p))
            .sum::<f64>()
            .powf(1. / p as f64)
    }
}

impl<const N: usize> Distance for [f64; N] {
    fn err_buf(&self, rhs: &Self) -> Vec<f64> {
        self.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect()
    }
}

impl<D: EfdDim> Distance for Efd<D> {
    fn err_buf(&self, rhs: &Self) -> Vec<f64> {
        use core::cmp::Ordering::*;
        let a = self.coeffs();
        let b = rhs.coeffs();
        let padding = core::iter::repeat(&0.);
        match a.len().cmp(&b.len()) {
            Less => b
                .iter()
                .zip(a.iter().chain(padding))
                .map(|(b, a)| a - b)
                .collect(),
            Equal => a.iter().zip(b.iter()).map(|(a, b)| a - b).collect(),
            Greater => a
                .iter()
                .zip(b.iter().chain(padding))
                .map(|(a, b)| a - b)
                .collect(),
        }
    }
}
