//! Utility functions for the library.
use crate::*;
use alloc::vec::Vec;

#[inline(always)]
pub(crate) fn pow2(x: f64) -> f64 {
    x * x
}

pub(crate) fn diff<R, C, S>(arr: na::Matrix<f64, R, C, S>) -> na::OMatrix<f64, R, na::Dyn>
where
    R: na::DimName,
    C: na::Dim,
    S: na::Storage<f64, R, C>,
    na::DefaultAllocator: na::allocator::Allocator<f64, R, C>,
{
    let arr = arr.into_owned();
    let head = arr.columns_range(..arr.ncols() - 1);
    let tail = arr.columns_range(1..);
    tail - head
}

pub(crate) fn cumsum<R, C, S>(arr: na::Matrix<f64, R, C, S>) -> na::OMatrix<f64, R, C>
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
    let c_ref = curve.as_curve();
    (c_ref.len() > 2 && c_ref.iter().flatten().all(|x| x.is_finite())).then_some(curve)
}

macro_rules! impl_curve_diff {
    ($($(#[$meta:meta])+ fn $name:ident($($arg:ident:$ty:ty),*) { ($($expr:expr),+) })+) => {$(
        $(#[$meta])+
        #[must_use]
        pub fn $name<A: CoordHint>( a: &[A], b: &[A], $($arg:$ty),*) -> f64 {
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
    /// The first curve is a part of the second curve, the second curve is periodic.
    fn partial_curve_diff() { (crate::tests::RES, false) }

    /// Custom resolution of [`partial_curve_diff`] function.
    fn partial_curve_diff_res(res: usize) { (res, false) }
}

fn curve_diff_res_norm<A: CoordHint>(a: &[A], b: &[A], res: usize, norm: bool) -> f64 {
    assert!(a.len() >= 2 && b.len() >= 2 && res > 0);
    let a = to_mat(a.closed_lin());
    let b = to_mat(b.closed_lin());
    let at = get_time(&a, norm);
    let bt = get_time(&b, norm).iter().copied().collect::<Vec<_>>();
    let bzt = *bt.last().unwrap();
    let err = (0..res)
        .map(|v| at.add_scalar(v as f64 / res as f64).map(|x| x % bzt))
        .map(|t| {
            let t = t.as_slice();
            (0..b.nrows())
                .map(|i| {
                    let ax = a.row(i);
                    let bx = b.row(i).iter().copied().collect::<Vec<_>>();
                    let bx = interp::interp_slice(&bt, &bx, t);
                    (ax - na::Matrix1xX::from_vec(bx)).map(pow2)
                })
                .reduce(|a, b| a + b)
                .unwrap()
                .map(f64::sqrt)
                .sum()
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    err / at.len() as f64
}

fn get_time<D: na::DimName>(curve: &MatrixRxX<D>, norm: bool) -> na::Matrix1xX<f64> {
    let dxyz = diff(curve.clone());
    let dt = dxyz.map(pow2).row_sum().map(f64::sqrt);
    let t = cumsum(dt).insert_column(0, 0.);
    if norm {
        let zt = t[t.len() - 1];
        t / zt
    } else {
        t
    }
}
