use crate::*;
use ndarray::{s, Array2};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// 3D EFD implementation.
#[derive(Clone, Debug)]
pub struct Efd3 {
    coeffs: Array2<f64>,
    trans: Transform3,
}

impl Efd3 {
    /// Builder method for adding transform type.
    pub fn trans(self, trans: Transform3) -> Self {
        Self { trans, ..self }
    }

    /// Consume self and return raw array.
    pub fn unwrap(self) -> Array2<f64> {
        self.coeffs
    }

    /// Get the array view of the coefficients.
    pub fn coeffs(&self) -> ndarray::ArrayView2<f64> {
        self.coeffs.view()
    }

    /// Get the reference of transform type.
    pub fn as_trans(&self) -> &Transform3 {
        self
    }

    /// Get the mutable reference of transform type.
    pub fn as_trans_mut(&mut self) -> &mut Transform3 {
        self
    }

    /// Get the harmonic of the coefficients.
    pub fn harmonic(&self) -> usize {
        self.coeffs.nrows()
    }

    /// Square error.
    pub fn square_err(&self, rhs: &Self) -> f64 {
        (&self.coeffs - &rhs.coeffs).mapv(pow2).sum()
    }

    /// L1 norm error, aka Manhattan distance.
    pub fn l1_norm(&self, rhs: &Self) -> f64 {
        (&self.coeffs - &rhs.coeffs).mapv(f64::abs).sum()
    }

    /// L2 norm error, aka Euclidean distance.
    pub fn l2_norm(&self, rhs: &Self) -> f64 {
        (&self.coeffs - &rhs.coeffs).mapv(pow2).sum().sqrt()
    }

    /// Lp norm error, slower than [`Self::l1_norm()`] and [`Self::l2_norm()`].
    pub fn lp_norm(&self, rhs: &Self, p: i32) -> f64 {
        (&self.coeffs - &rhs.coeffs)
            .mapv(|x| x.abs().powi(p))
            .sum()
            .powf(1. / p as f64)
    }

    /// Reverse the order of described curve then return a mutable reference.
    pub fn reverse(&mut self) -> &mut Self {
        let mut s = self.coeffs.slice_mut(s![.., 1]);
        s *= -1.;
        let mut s = self.coeffs.slice_mut(s![.., 3]);
        s *= -1.;
        self
    }

    /// Consume and return a reversed version of the coefficients. This method
    /// can avoid mutable require.
    ///
    /// Please clone the object if you want to do self-comparison.
    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }
}

impl std::ops::Deref for Efd3 {
    type Target = Transform3;

    fn deref(&self) -> &Self::Target {
        &self.trans
    }
}

impl std::ops::DerefMut for Efd3 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.trans
    }
}
