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
///
/// See also [`is_valid_curve()`].
pub fn valid_curve<C, const D: usize>(curve: C) -> Option<C>
where
    C: Curve<D>,
{
    is_valid_curve(curve.as_curve()).then_some(curve)
}

/// Return true if the curve is valid.
#[inline]
pub fn is_valid_curve<C, const D: usize>(curve: C) -> bool
where
    C: Curve<D>,
{
    let c = curve.as_curve();
    c.len() > 2 && c.iter().flatten().all(|x| x.is_finite())
}

/// Return the zipped average distance error between two curves.
///
/// Returns 0 if either curve is empty.
///
/// See also [`dist_err()`] for a more general case where the curves are not
/// corresponded. (`curve1[i] !== curve2[i]`)
pub fn dist_err_zipped<const D: usize>(curve1: impl Curve<D>, curve2: impl Curve<D>) -> f64 {
    let len = curve1.len().min(curve2.len());
    if len == 0 {
        0.
    } else {
        core::iter::zip(curve1.as_curve(), curve2.as_curve())
            .map(|(a, b)| a.l2_err(b))
            .sum::<f64>()
            / len as f64
    }
}

/// Return the average distance error between two curves.
///
/// In this algorithm, a curve is assumed to be longer or equal to another, and
/// the distance error is mapped to the nearest point in the shorter curve.
///
/// Returns 0 if either curve is empty.
///
/// See also [`dist_err_zipped()`] for faster computation if the curve points
/// are corresponded.
pub fn dist_err<const D: usize>(curve1: impl Curve<D>, curve2: impl Curve<D>) -> f64 {
    if curve1.is_empty() || curve2.is_empty() {
        return 0.;
    }
    let (mut iter1, iter2, len) = {
        let iter1 = curve1.as_curve().iter();
        let iter2 = curve2.as_curve().iter();
        if curve1.len() >= curve2.len() {
            (iter1, iter2, curve2.len())
        } else {
            (iter2, iter1, curve1.len())
        }
    };
    let mut total = 0.;
    let mut prev_err = None;
    let mut last_pt1 = None; // a pointer indicating the last point of curve1
    for pt2 in iter2 {
        loop {
            if let Some(pt1) = iter1.next() {
                last_pt1 = Some(pt1);
                let err = pt1.l2_err(pt2);
                match prev_err {
                    // The previous error is the nearest
                    Some(prev_err) if err > prev_err => {
                        total += prev_err;
                        break;
                    }
                    // The previous error is not the nearest or unset
                    _ => prev_err = Some(err),
                }
            } else {
                // curve1 is exhausted, compare the last point
                total += last_pt1.unwrap().l2_err(pt2);
                break;
            }
        }
    }
    total / len as f64
}
