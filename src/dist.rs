//! Distance trait.
use crate::*;
use core::iter::{repeat, zip};
#[cfg(not(feature = "std"))]
use num_traits::*;

macro_rules! impl_norm {
    ($(($name:ident, $err:ident)),+) => {$(
        /// Calculate the norm (vector length) according to the origin (zeros).
        ///
        /// See also
        #[doc = concat![" [`Distance::", stringify!($err), "()`]."]]
        #[inline]
        fn $name(&self) -> f64 {
            self.$err(&Origin)
        }
    )+};
}

#[inline]
fn sum(x: f64, y: f64) -> f64 {
    x + y
}

fn impl_err<A, B, F1, F2>(a: &A, b: &B, map: F1, fold: F2) -> f64
where
    A: Distance + ?Sized,
    B: Distance + ?Sized,
    F1: Fn(f64) -> f64,
    F2: Fn(f64, f64) -> f64,
{
    macro_rules! err_calc {
        ($iter:expr) => {
            $iter.map(|x| map(cmp(x))).fold(0., fold)
        };
    }
    #[inline]
    fn cmp((a, b): (&f64, &f64)) -> f64 {
        a - b
    }
    let a = a.as_components();
    let b = b.as_components();
    match (a.size_hint().1, b.size_hint().1) {
        (None, None) => panic!("The size of the data is unknown"),
        (Some(n), Some(m)) if n != m => {
            if n > m {
                err_calc!(zip(a, b.chain(repeat(&0.))))
            } else {
                err_calc!(zip(b, a.chain(repeat(&0.))))
            }
        }
        _ => err_calc!(zip(a, b)),
    }
}

/// Be able to calculate the distance between two instances.
///
/// Each data number has the same weight by default, but you can change it by
/// implementing [`Distance::as_components()`].
pub trait Distance {
    /// Get the components of the data.
    ///
    /// The [`Iterator::size_hint()`] method is used to determine the size of
    /// the data.
    fn as_components(&self) -> impl Iterator<Item = &f64>;

    /// Calculate the square error.
    fn square_err(&self, rhs: &impl Distance) -> f64 {
        impl_err(self, rhs, util::pow2, sum)
    }

    /// Calculate the L0 norm of the error.
    ///
    /// This method is also called Hamming distance, which counts the number of
    /// errors that are less than the epsilon.
    fn l0_err(&self, rhs: &impl Distance) -> f64 {
        let bin = |x: f64| (x.abs() < f64::EPSILON) as u8 as f64;
        impl_err(self, rhs, bin, sum)
    }

    /// Calculate the L1 norm of the error.
    ///
    /// This method is also called Manhattan distance.
    fn l1_err(&self, rhs: &impl Distance) -> f64 {
        impl_err(self, rhs, |x| x.abs(), sum)
    }

    /// Calculate the L2 norm of the error.
    ///
    /// This method is also called Euclidean distance.
    fn l2_err(&self, rhs: &impl Distance) -> f64 {
        self.square_err(rhs).sqrt()
    }

    /// Calculate the Lp norm of the error.
    ///
    /// This method is also called Minkowski distance.
    fn lp_err(&self, rhs: &impl Distance, p: f64) -> f64 {
        impl_err(self, rhs, |x| x.abs().powf(p), sum).powf(p.recip())
    }

    /// Calculate the Linf norm of the error.
    ///
    /// This method is also called Chebyshev distance.
    fn linf_err(&self, rhs: &impl Distance) -> f64 {
        impl_err(self, rhs, |x| x.abs(), f64::max)
    }

    impl_norm!(
        (l0_norm, l0_err),
        (l1_norm, l1_err),
        (l2_norm, l2_err),
        (linf_norm, linf_err)
    );

    /// Calculate the norm (vector length) according to the origin (zeros).
    ///
    /// See also [`Distance::lp_err()`].
    fn lp_norm(&self, p: f64) -> f64 {
        self.lp_err(&Origin, p)
    }
}

struct Origin;

impl Distance for Origin {
    fn as_components(&self) -> impl Iterator<Item = &f64> {
        repeat(&0.)
    }
}

impl Distance for [f64] {
    fn as_components(&self) -> impl Iterator<Item = &f64> {
        self.iter()
    }
}

impl<const D: usize> Distance for [f64; D] {
    fn as_components(&self) -> impl Iterator<Item = &f64> {
        self.iter()
    }
}

impl<const D: usize> Distance for Efd<D>
where
    U<D>: EfdDim<D>,
{
    fn as_components(&self) -> impl Iterator<Item = &f64> {
        EfdComponents {
            iter: self.coeffs_iter().flatten(),
            size: self.harmonic() * D * 2,
        }
    }
}

struct EfdComponents<I> {
    iter: I,
    size: usize,
}

impl<'a, I> Iterator for EfdComponents<I>
where
    I: Iterator<Item = &'a f64>,
{
    type Item = &'a f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}
