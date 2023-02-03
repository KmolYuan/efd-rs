use crate::Trans;
use alloc::vec::Vec;
use ndarray::{arr2, Array, Axis, CowArray, Dimension, FixedInitializer};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Owned curve type.
pub type Curve<D> = Vec<<D as Trans>::Coord>;
/// Copy-on-write curve type.
pub type CowCurve<'a, D> = alloc::borrow::Cow<'a, [<D as Trans>::Coord]>;

#[inline(always)]
pub(crate) fn pow2(x: f64) -> f64 {
    x * x
}

pub(crate) fn diff<'a, D, A>(arr: A, axis: Option<Axis>) -> Array<f64, D>
where
    D: Dimension,
    A: Into<CowArray<'a, f64, D>>,
{
    let arr = arr.into();
    let axis = axis.unwrap_or_else(|| Axis(arr.ndim() - 1));
    let head = arr.slice_axis(axis, (..-1).into());
    let tail = arr.slice_axis(axis, (1..).into());
    &tail - &head
}

pub(crate) fn cumsum<'a, D, A>(arr: A, axis: Option<Axis>) -> Array<f64, D>
where
    D: Dimension + ndarray::RemoveAxis,
    A: Into<CowArray<'a, f64, D>>,
{
    let mut arr = arr.into().to_owned();
    let axis = axis.unwrap_or(Axis(0));
    arr.axis_iter_mut(axis).reduce(|prev, mut next| {
        next += &prev;
        next
    });
    arr
}

/// The maximum coordinate difference between two curves.
#[must_use]
pub fn curve_diff<A, B>(a: &[A], b: &[B]) -> f64
where
    A: FixedInitializer<Elem = f64> + Clone,
    B: FixedInitializer<Elem = f64> + Clone,
{
    let a = arr2(a);
    let b = arr2(b);
    a.axis_iter(Axis(0))
        .zip(b.axis_iter(Axis(0)))
        .map(|(a, b)| (&a - &b).mapv(f64::abs).sum())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

/// Close the curve by the first coordinate.
#[must_use]
pub fn closed_curve<'a, A, C>(curve: C) -> Vec<A>
where
    A: Clone + 'a,
    C: Into<alloc::borrow::Cow<'a, [A]>>,
{
    let mut c = curve.into().into_owned();
    c.push(c[0].clone());
    c
}
