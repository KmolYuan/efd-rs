use alloc::{vec, vec::Vec};
use ndarray::{arr2, s, Array, Axis, CowArray, Dimension, FixedInitializer};

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

macro_rules! impl_curve_diff {
    ($($(#[$meta:meta])+ fn $name:ident($($arg:ident:$ty:ty),*) { ($($expr:expr),+) })+) => {$(
        $(#[$meta])+
        #[must_use]
        pub fn $name<A, B>(a: &[A], b: &[B], $($arg:$ty),*) -> f64
        where
            A: FixedInitializer<Elem = f64> + Clone,
            B: FixedInitializer<Elem = f64> + Clone,
        {
            curve_diff_res_norm(a, b, $($expr),+)
        }
    )+};
}

impl_curve_diff! {
    /// Curve difference between two curves using interpolation.
    ///
    /// Two curves must be periodic.
    fn curve_diff() { (crate::tests::RES, true) }

    /// Custom resolution of [`curve_diff`] function.
    fn curve_diff_res(res: usize) { (res, true) }

    /// Coordinate difference between two curves using interpolation.
    ///
    /// The fist curve is a part of the second curve.
    fn partial_curve_diff() { (crate::tests::RES, false) }

    /// Custom resolution of [`partial_curve_diff`] function.
    fn partial_curve_diff_res(res: usize) { (res, false) }
}

fn curve_diff_res_norm<A, B>(a: &[A], b: &[B], res: usize, norm: bool) -> f64
where
    A: FixedInitializer<Elem = f64> + Clone,
    B: FixedInitializer<Elem = f64> + Clone,
{
    assert!(a.len() >= 2 && b.len() >= 2 && res > 0);
    let a = arr2(a);
    let b = arr2(b);
    let (at, ac) = get_time_center(&a, norm);
    let (bt, bc) = get_time_center(&b, norm);
    let a = &a - ndarray::Array1::from(ac).insert_axis(Axis(0));
    let b = &b - ndarray::Array1::from(bc).insert_axis(Axis(0));
    let bt = bt.to_vec();
    let bzt = if norm { 1. } else { *bt.last().unwrap() };
    let err = (0..res)
        .map(|v| (&at + v as f64 / res as f64) % bzt)
        .map(|t| {
            let t = t.as_slice().unwrap();
            (0..b.ncols())
                .map(|i| {
                    let ax = a.slice(s![.., i]);
                    let bx = b.slice(s![.., i]);
                    let bx = bx.as_standard_layout();
                    let bx = interp::interp_slice(&bt, bx.as_slice().unwrap(), t);
                    (&ax - ndarray::Array1::from(bx)).mapv(pow2)
                })
                .reduce(|a, b| a + b)
                .unwrap()
                .mapv(f64::sqrt)
                .sum()
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    err / res as f64
}

fn get_time_center(curve: &ndarray::Array2<f64>, norm: bool) -> (ndarray::Array1<f64>, Vec<f64>) {
    let dxyz = diff(curve, Some(Axis(0)));
    let dt = dxyz.mapv(pow2).sum_axis(Axis(1)).mapv(f64::sqrt);
    let t = ndarray::concatenate![Axis(0), ndarray::array![0.], cumsum(&dt, None)];
    let zt = *t.last().unwrap();
    let center = {
        let tdt = &t.slice(s![1..]) / &dt;
        let c = diff(t.mapv(pow2), None) * 0.5 / &dt;
        (0..curve.ncols())
            .map(|i| {
                let xi = cumsum(dxyz.slice(s![.., i]), None) - &dxyz.slice(s![.., i]) * &tdt;
                let a0 = (&dxyz.slice(s![.., i]) * &c + xi * &dt).sum() / zt;
                curve[[0, i]] + a0
            })
            .collect()
    };
    (if norm { t / zt } else { t }, center)
}
