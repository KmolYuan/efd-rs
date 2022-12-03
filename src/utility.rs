use ndarray::{arr2, Array, Axis, CowArray, Dimension, FixedInitializer};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

pub(crate) type CowCurve2<'a> = alloc::borrow::Cow<'a, [[f64; 2]]>;
pub(crate) type CowCurve3<'a> = alloc::borrow::Cow<'a, [[f64; 3]]>;

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

/// Check the difference between two curves.
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
        .sum()
}
