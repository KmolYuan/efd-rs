use crate::*;
use alloc::vec::Vec;

#[inline(always)]
pub(crate) fn pow2(x: f64) -> f64 {
    x * x
}

pub(crate) fn diff<R, C, S>(arr: na::Matrix<f64, R, C, S>) -> na::OMatrix<f64, na::Dyn, C>
where
    R: na::Dim,
    C: na::Dim,
    S: na::Storage<f64, R, C>,
    na::DefaultAllocator: na::allocator::Allocator<f64, R, C>,
{
    let arr = arr.into_owned();
    let head = arr.rows_range(..arr.nrows() - 1);
    let tail = arr.rows_range(1..);
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
    arr.row_iter_mut().reduce(|prev, mut next| {
        next += &prev;
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
    (c_ref.len() > 1 && !c_ref.iter().flatten().any(|x| x.is_infinite())).then_some(curve)
}

macro_rules! impl_curve_diff {
    ($($(#[$meta:meta])+ fn $name:ident($($arg:ident:$ty:ty),*) { ($($expr:expr),+) })+) => {$(
        $(#[$meta])+
        #[must_use]
        pub fn $name<const DIM: usize>( a: &[[f64; DIM]], b: &[[f64; DIM]], $($arg:$ty),*) -> f64 {
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

fn curve_diff_res_norm<const DIM: usize>(
    a: &[[f64; DIM]],
    b: &[[f64; DIM]],
    res: usize,
    norm: bool,
) -> f64 {
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
            (0..b.ncols())
                .map(|i| {
                    let ax = a.column(i);
                    let bx = b.column(i);
                    let bx = interp::interp_slice(&bt, bx.as_slice(), t);
                    (ax - na::MatrixXx1::from_vec(bx)).map(pow2)
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

fn get_time<D: na::DimName>(curve: &MatrixXxC<D>, norm: bool) -> na::MatrixXx1<f64> {
    let dxyz = diff(curve.clone());
    let dt = dxyz.map(pow2).column_sum().map(f64::sqrt);
    let t = cumsum(dt).insert_row(0, 0.);
    if norm {
        let zt = t[t.len() - 1];
        t / zt
    } else {
        t
    }
}
