//! Distance trait.
use crate::*;
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use num_traits::*;

#[inline]
fn cmp((a, b): (&f64, &f64)) -> f64 {
    a - b
}

/// Be able to calculate the distance between two instances.
///
/// Each data number has the same weight by default, but you can change it by
/// implementing [`Self::err_buf()`].
pub trait Distance {
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
        err_buf.iter().filter(|x| x.abs() < f64::EPSILON).count() as f64 / err_buf.len() as f64
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

impl<S: AsRef<[f64]>> Distance for S {
    fn err_buf(&self, rhs: &Self) -> Vec<f64> {
        self.as_ref().iter().zip(rhs.as_ref()).map(cmp).collect()
    }
}

impl<const D: usize> Distance for Efd<D>
where
    U<D>: EfdDim<D>,
    na::Const<D>: na::DimNameMul<na::U2>,
{
    fn err_buf(&self, rhs: &Self) -> Vec<f64> {
        let a = self.coeffs().iter();
        let b = rhs.coeffs().iter();
        let padding = core::iter::repeat(&0.);
        match self.harmonic() >= rhs.harmonic() {
            true => a.zip(b.chain(padding)).map(cmp).collect(),
            false => a.chain(padding).zip(b).map(cmp).collect(),
        }
    }
}

impl<const D: usize> Distance for PosedEfd<D>
where
    U<D>: EfdDim<D>,
    na::Const<D>: na::DimNameMul<na::U2>,
{
    fn err_buf(&self, rhs: &Self) -> Vec<f64> {
        let mut buf = self.curve_efd().err_buf(rhs.curve_efd());
        buf.extend(self.pose_efd().err_buf(rhs.pose_efd()));
        buf
    }
}
