//! Dimension specific implementation.
use crate::{util::*, *};
use core::f64::consts::{PI, TAU};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::*;

/// 1D EFD dimension marker.
pub enum D1 {}
/// 2D EFD dimension marker.
pub enum D2 {}
/// 3D EFD dimension marker.
pub enum D3 {}

/// 2D Coefficient type.
pub type Coeff2 = Coeff<D2>;
/// 3D Coefficient type.
pub type Coeff3 = Coeff<D3>;
/// Coefficient type.
pub type Coeff<D> = na::OMatrix<f64, CDim<D>, na::Dyn>;
/// Coordinate view used in the conversion method.
pub type CoordView<'a, D> = na::MatrixView<'a, f64, Dim<D>, na::U1>;
/// Alias for the dimension.
pub type Dim<D> = <Coord<D> as CoordHint>::Dim;
/// Alias for the coefficient number. (DIM * 2)
pub type CDim<D> = TCDim<<D as EfdDim>::Trans>;
/// A matrix view of specific coefficients. (DIM * 2)
pub type CKernel<'a, D> = na::MatrixView<'a, f64, Dim<D>, na::U2>;
/// A mutable matrix view of specific coefficients. (DIM * 2)
pub type CKernelMut<'a, D> = na::MatrixViewMut<'a, f64, Dim<D>, na::U2>;

type TCoeff<T> = MatrixRxX<TCDim<T>>;
type TCKernel<'a, T> = na::MatrixView<'a, f64, TDim<T>, na::U2>;
type TCKernelMut<'a, T> = na::MatrixViewMut<'a, f64, TDim<T>, na::U2>;
type TCDim<T> = na::DimNameProd<TDim<T>, na::U2>;
type TDim<T> = <<T as Transform>::Coord as CoordHint>::Dim;

macro_rules! impl_rot {
    ($m:ident, $rot:expr) => {{
        let rot = $rot;
        for mut c in $m.column_iter_mut() {
            for mut v in CKernelMut::<Self>::from_slice(c.as_mut_slice()).column_iter_mut() {
                let rotated = rot.inverse() * &v;
                v.copy_from(&rotated);
            }
        }
        rot
    }};
}

/// Trait for EFD dimension.
pub trait EfdDim {
    /// Transformation type of similarity matrix.
    type Trans: Transform;

    /// Generate coefficients and similarity matrix.
    fn get_coeff(
        curve: &[Coord<Self>],
        harmonic: usize,
        is_open: bool,
    ) -> (Coeff<Self>, GeoVar<Self::Trans>) {
        let (mut coeffs, trans1) = Self::get_coeff_unnorm(curve, harmonic, is_open);
        let trans2 = Self::coeff_norm(&mut coeffs);
        (coeffs, trans1 * trans2)
    }

    /// Generate coefficients and similarity matrix **without** normalization.
    fn get_coeff_unnorm(
        curve: &[Coord<Self>],
        harmonic: usize,
        is_open: bool,
    ) -> (Coeff<Self>, GeoVar<Self::Trans>) {
        impl_coeff(curve, harmonic, is_open)
    }

    /// Normalize coefficients.
    fn coeff_norm(coeffs: &mut Coeff<Self>) -> GeoVar<Self::Trans>;
}

impl EfdDim for D1 {
    type Trans = T1;

    fn coeff_norm(coeffs: &mut Coeff<Self>) -> GeoVar<Self::Trans> {
        impl_norm(coeffs, |_| {
            na::Rotation::from_matrix_unchecked(na::matrix![1.])
        })
    }
}

impl EfdDim for D2 {
    type Trans = T2;

    fn coeff_norm(coeffs: &mut Coeff<Self>) -> GeoVar<Self::Trans> {
        impl_norm::<Self::Trans>(coeffs, |m| {
            impl_rot!(m, na::UnitComplex::new(m[(1, 0)].atan2(m[(0, 0)])))
        })
    }
}

impl EfdDim for D3 {
    type Trans = T3;

    fn coeff_norm(coeffs: &mut Coeff<Self>) -> GeoVar<Self::Trans> {
        impl_norm::<Self::Trans>(coeffs, |m| {
            let m1 = CKernel::<Self>::from_slice(m.column(0).data.into_slice());
            let u = m1.column(0).normalize();
            let rot = if let Some(v) = m1.column(1).try_normalize(f64::EPSILON) {
                // Closed curve, use `u` and `v` plane as basis
                let w = u.cross(&v);
                na::UnitQuaternion::from_basis_unchecked(&[u, v, w])
            } else if m.ncols() > 1 {
                // Open curve, `v` is zero vector, use `u1` and `u2` plane as basis
                let m2 = CKernel::<Self>::from_slice(m.column(1).data.into_slice());
                // `w` is orthogonal to `u` and `u2`
                let w = u.cross(&m2.column(0)).normalize();
                let u2 = w.cross(&u);
                na::UnitQuaternion::from_basis_unchecked(&[u, u2, w])
            } else {
                // Open curve, one harmonic, just rotate `u` to x-axis
                let (u, v) = (na::Vector3::x(), u);
                na::UnitQuaternion::from_scaled_axis(u.cross(&v).normalize() * u.dot(&v).acos())
            };
            impl_rot!(m, rot)
        })
    }
}

fn impl_coeff<T: Transform>(
    curve: &[T::Coord],
    harmonic: usize,
    is_open: bool,
) -> (TCoeff<T>, GeoVar<T>) {
    let dxyz = diff(if is_open || curve.first() == curve.last() {
        to_mat(curve)
    } else {
        to_mat(curve.closed_lin())
    });
    let dt = dxyz.map(pow2).row_sum().map(f64::sqrt);
    let t = cumsum(dt.clone()).insert_column(0, 0.);
    let zt = t[t.len() - 1];
    let scalar = zt / (PI * PI) * if is_open { 2. } else { 0.5 };
    let phi = &t * TAU / zt * if is_open { 0.5 } else { 1. };
    // Coefficients (2dim * N)
    // [x_cos, y_cos, z_cos, x_sin, y_sin, z_sin]'
    let mut n = 0.;
    let mut coeffs = MatrixRxX::<TCDim<T>>::zeros(harmonic);
    for mut c in coeffs.column_iter_mut() {
        n += 1.;
        let phi = &phi * n;
        let scalar = scalar / (n * n);
        let cos_phi = diff(phi.map(f64::cos)).component_div(&dt);
        dxyz.row_iter()
            .zip(c.iter_mut().take(dxyz.nrows()))
            .for_each(|(d, c)| *c = scalar * d.component_mul(&cos_phi).sum());
        if is_open {
            continue;
        }
        let sin_phi = diff(phi.map(f64::sin)).component_div(&dt);
        dxyz.row_iter()
            .zip(c.iter_mut().skip(dxyz.nrows()))
            .for_each(|(d, c)| *c = scalar * d.component_mul(&sin_phi).sum());
    }
    let tdt = t.columns_range(1..).component_div(&dt);
    let c = 0.5 * diff(t.map(pow2)).component_div(&dt);
    let mut center = curve[0].clone();
    dxyz.row_iter()
        .zip(center.flat_mut())
        .for_each(|(dxyz, oxyz)| {
            let xi = cumsum(dxyz) - dxyz.component_mul(&tdt);
            *oxyz += (dxyz.component_mul(&c) + xi.component_mul(&dt)).sum() / zt;
        });
    (coeffs, GeoVar::new(center, Default::default(), 1.))
}

fn impl_norm<T: Transform>(
    coeffs: &mut TCoeff<T>,
    get_and_rot: impl FnOnce(&mut TCoeff<T>) -> T::Rot,
) -> GeoVar<T>
where
    na::DefaultAllocator:
        na::allocator::Allocator<f64, TDim<T>> + na::allocator::Allocator<f64, TDim<T>, na::U2>,
{
    // Angle of starting point
    // m = m * theta
    let theta = {
        let c = TCKernel::<T>::from_slice(coeffs.column(0).data.into_slice());
        let dy = 2. * c.column_product().sum();
        let dx = c.map(pow2).row_sum();
        0.5 * dy.atan2(dx[0] - dx[1])
    };
    for (i, mut c) in coeffs.column_iter_mut().enumerate() {
        let theta = na::Rotation2::new((i + 1) as f64 * theta);
        let mut m = TCKernelMut::<T>::from_slice(c.as_mut_slice());
        m.copy_from(&(&m * theta));
    }
    // Normalize coefficients sign
    if coeffs.ncols() > 1 && coeffs[(0, 0)] * coeffs[(0, 1)] < 0. {
        coeffs
            .column_iter_mut()
            .step_by(2)
            .for_each(|mut s| s *= -1.);
    }
    // Rotation angle
    // m = psi' * m
    let psi = get_and_rot(coeffs);
    // Scale factor
    // |u1| == a1 (after rotation)
    let scale = coeffs[(0, 0)].abs();
    *coeffs /= scale;
    GeoVar::new(Default::default(), psi, scale)
}
