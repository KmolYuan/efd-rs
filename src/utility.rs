use ndarray::{Array, Axis, CowArray, Dimension};

pub(crate) type CowCurve<'a> = alloc::borrow::Cow<'a, [[f64; 2]]>;

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
pub fn curve_diff<C1, C2>(a: C1, b: C2) -> f64
where
    C1: AsRef<[[f64; 2]]>,
    C2: AsRef<[[f64; 2]]>,
{
    a.as_ref()
        .iter()
        .zip(b.as_ref())
        .map(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs())
        .sum()
}
