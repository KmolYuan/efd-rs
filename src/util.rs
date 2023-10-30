//! Utility functions for the library.
use crate::*;

// Only used in "map" methods
#[inline]
pub(crate) fn pow2(x: f64) -> f64 {
    x * x
}

/// Differentiate an array.
pub fn diff<R, C, S>(arr: na::Matrix<f64, R, C, S>) -> na::OMatrix<f64, R, na::Dyn>
where
    R: na::DimName,
    C: na::Dim,
    S: na::Storage<f64, R, C>,
    na::DefaultAllocator: na::allocator::Allocator<f64, R, C>,
{
    let head = arr.columns_range(..arr.ncols() - 1);
    let tail = arr.columns_range(1..);
    tail - head
}

/// Cumulative sum of an array. (Integral)
pub fn cumsum<R, C, S>(arr: na::Matrix<f64, R, C, S>) -> na::OMatrix<f64, R, C>
where
    R: na::Dim,
    C: na::Dim,
    S: na::Storage<f64, R, C>,
    na::DefaultAllocator: na::allocator::Allocator<f64, R, C>,
{
    let mut arr = arr.into_owned();
    arr.column_iter_mut().reduce(|prev, mut next| {
        next += prev;
        next
    });
    arr
}

/// Check if the curve is valid.
pub fn valid_curve<C, const DIM: usize>(curve: C) -> Option<C>
where
    C: Curve<[f64; DIM]>,
{
    let c = curve.as_curve();
    (c.len() > 2 && c.iter().flatten().all(|x| x.is_finite())).then_some(curve)
}
