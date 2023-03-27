use alloc::vec;
use ndarray::{arr2, Array, Axis, CowArray, Dimension, FixedInitializer};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

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

/// Coordinate difference between two curves using interpolation.
#[must_use]
pub fn curve_diff<A, B>(a: &[A], b: &[B], res: usize) -> f64
where
    A: FixedInitializer<Elem = f64> + Clone,
    B: FixedInitializer<Elem = f64> + Clone,
{
    assert!(res > 0);
    fn timing(curve: &ndarray::Array2<f64>) -> ndarray::Array1<f64> {
        let dxy = diff(curve, Some(Axis(0)));
        let dt = dxy.mapv(pow2).sum_axis(Axis(1)).mapv(f64::sqrt);
        let t = ndarray::concatenate![Axis(0), ndarray::array![0.], cumsum(&dt, None)];
        let zt = *t.last().unwrap();
        t / zt
    }

    let a = arr2(a);
    let at = timing(&a);
    let b = arr2(b);
    let bt = timing(&b).to_vec();
    (0..res)
        .map(|v| v as f64 / res as f64)
        .map(|shift| {
            let t = (&at + shift) % 1.;
            let t = t.as_slice().unwrap();
            (0..b.ncols())
                .map(|i| {
                    let ax = a.index_axis(Axis(1), i);
                    let bx = b.index_axis(Axis(1), i);
                    let bx = bx.as_standard_layout();
                    let bx = ndarray::arr1(&interp::interp_slice(&bt, bx.as_slice().unwrap(), t));
                    (&ax - bx).mapv(f64::abs).sum()
                })
                .sum::<f64>()
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}
