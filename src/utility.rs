use crate::Trans;
use alloc::{borrow::Cow, vec::Vec};
use ndarray::{arr2, Array, Axis, CowArray, Dimension, FixedInitializer};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Owned curve type.
pub type Curve<D> = Vec<<D as Trans>::Coord>;
/// Copy-on-write curve type.
pub type CowCurve<'a, D> = Cow<'a, [<D as Trans>::Coord]>;

#[inline(always)]
pub(crate) fn pow2(x: f64) -> f64 {
    x * x
}

pub(crate) fn diff<'a, D, A>(arr: A, axis: Option<Axis>) -> Array<f64, D>
where
    D: Dimension,
    CowArray<'a, f64, D>: From<A>,
{
    let arr = CowArray::from(arr);
    let axis = axis.unwrap_or_else(|| Axis(arr.ndim() - 1));
    let head = arr.slice_axis(axis, (..-1).into());
    let tail = arr.slice_axis(axis, (1..).into());
    &tail - &head
}

pub(crate) fn cumsum<'a, D, A>(arr: A, axis: Option<Axis>) -> Array<f64, D>
where
    D: Dimension + ndarray::RemoveAxis,
    CowArray<'a, f64, D>: From<A>,
{
    let mut arr = CowArray::from(arr).into_owned();
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

/// Check if a curve's first and end points are the same.
pub fn is_closed<A: PartialEq>(curve: &[A]) -> bool {
    match (curve.first(), curve.last()) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// Close the curve by the first coordinate.
#[must_use]
#[deprecated = "function renamed, please use `closed_lin` instead"]
pub fn closed_curve<'a, A, C>(curve: C) -> Vec<A>
where
    A: Clone + 'a,
    Cow<'a, [A]>: From<C>,
{
    closed_lin(curve)
}

/// Close the curve by the first coordinate.
#[must_use]
pub fn closed_lin<'a, A, C>(curve: C) -> Vec<A>
where
    A: Clone + 'a,
    Cow<'a, [A]>: From<C>,
{
    let mut c = Cow::from(curve).into_owned();
    c.push(c[0].clone());
    c
}

/// Close the open curve with its direction-inverted part.
///
/// Panic with empty curve.
#[must_use]
pub fn closed_rev<'a, A, C>(curve: C) -> Vec<A>
where
    A: Clone + 'a,
    Cow<'a, [A]>: From<C>,
{
    let mut curve = Cow::from(curve).into_owned();
    let curve2 = curve.iter().rev().skip(1).cloned().collect::<Vec<_>>();
    curve.extend(curve2);
    curve
}
