//! Distance trait.
use crate::*;
use core::iter::zip;
#[cfg(not(feature = "std"))]
use num_traits::*;

#[inline]
fn cmp((a, b): (&f64, &f64)) -> f64 {
    a - b
}

/// Be able to calculate the distance between two instances.
///
/// Each data number has the same weight by default, but you can change it by
/// implementing [`Distance::err_buf()`].
pub trait Distance {
    /// Calculate the error between each pair of data.
    fn err_buf<'a>(&'a self, rhs: &'a Self) -> impl Iterator<Item = f64> + 'a;

    /// Calculate the square error.
    fn square_err(&self, rhs: &Self) -> f64 {
        self.err_buf(rhs).map(util::pow2).sum()
    }

    /// Calculate the L0 norm of the error.
    ///
    /// This method is also called Hamming distance, which counts the number of
    /// errors that are less than the epsilon.
    fn l0_norm(&self, rhs: &Self) -> f64 {
        self.err_buf(rhs).filter(|x| x.abs() < f64::EPSILON).count() as f64
    }

    /// Calculate the L1 norm of the error.
    ///
    /// This method is also called Manhattan distance.
    fn l1_norm(&self, rhs: &Self) -> f64 {
        self.err_buf(rhs).map(|x| x.abs()).sum()
    }

    /// Calculate the L2 norm of the error.
    ///
    /// This method is also called Euclidean distance.
    fn l2_norm(&self, rhs: &Self) -> f64 {
        self.square_err(rhs).sqrt()
    }

    /// Calculate the Lp norm of the error.
    ///
    /// This method is also called Minkowski distance.
    fn lp_norm(&self, rhs: &Self, p: f64) -> f64 {
        let err = self.err_buf(rhs).map(|x| x.abs().powf(p)).sum::<f64>();
        err.powf(p.recip())
    }

    /// Calculate the Linf norm of the error.
    ///
    /// This method is also called Chebyshev distance.
    fn linf_norm(&self, rhs: &Self) -> f64 {
        self.err_buf(rhs).map(|x| x.abs()).fold(0., f64::max)
    }
}

impl<S: AsRef<[f64]>> Distance for S {
    fn err_buf<'a>(&'a self, rhs: &'a Self) -> impl Iterator<Item = f64> + 'a {
        zip(self.as_ref(), rhs.as_ref()).map(cmp)
    }
}

impl<const D: usize> Distance for Efd<D>
where
    U<D>: EfdDim<D>,
{
    fn err_buf<'a>(&'a self, rhs: &'a Self) -> impl Iterator<Item = f64> + 'a {
        let a = self.coeffs().iter().flat_map(na::Matrix::iter);
        let b = rhs.coeffs().iter().flat_map(na::Matrix::iter);
        let padding = core::iter::repeat(&0.);
        match self.harmonic() >= rhs.harmonic() {
            true => zip(a, b.chain(padding)).map(cmp),
            false => zip(b, a.chain(padding)).map(cmp),
        }
    }
}
