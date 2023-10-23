//! Dimension specific implementation.
use crate::*;
use core::f64::consts::{PI, TAU};

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
pub type CDim<D> = CCDim<<D as EfdDim>::Trans>;
/// A matrix view of specific coefficients. (DIM * 2)
pub type CKernel<'a, D> = na::MatrixView<'a, f64, Dim<D>, na::U2>;
/// A mutable matrix view of specific coefficients. (DIM * 2)
pub type CKernelMut<'a, D> = na::MatrixViewMut<'a, f64, Dim<D>, na::U2>;

type CCoeff<const DIM: usize> = na::OMatrix<f64, na::Const<DIM>, na::Dyn>;
type CCKernel<'a, const DIM: usize> = na::MatrixView<'a, f64, na::Const<DIM>, na::U2>;
type CCKernelMut<'a, const DIM: usize> = na::MatrixViewMut<'a, f64, na::Const<DIM>, na::U2>;
type CCDim<T> = na::DimNameProd<<<T as Trans>::Coord as CoordHint>::Dim, na::U2>;

/// Trait for EFD dimension.
pub trait EfdDim {
    /// Transformation type of similarity matrix.
    type Trans: Trans;

    /// Generate coefficients and similarity matrix.
    fn get_coeff(
        curve: &[Coord<Self>],
        harmonic: usize,
        is_open: bool,
    ) -> (Coeff<Self>, Transform<Self::Trans>) {
        let (mut coeffs, trans1) = Self::get_coeff_unnorm(curve, harmonic, is_open);
        let trans2 = Self::coeff_norm(&mut coeffs);
        (coeffs, trans1 * trans2)
    }

    /// Generate coefficients and similarity matrix **without** normalization.
    fn get_coeff_unnorm(
        curve: &[Coord<Self>],
        harmonic: usize,
        is_open: bool,
    ) -> (Coeff<Self>, Transform<Self::Trans>) {
        impl_coeff(curve, harmonic, is_open)
    }

    /// Normalize coefficients.
    fn coeff_norm(coeffs: &mut Coeff<Self>) -> Transform<Self::Trans>;
}

impl EfdDim for D2 {
    type Trans = T2;

    fn coeff_norm(coeffs: &mut Coeff<Self>) -> Transform<Self::Trans> {
        impl_norm(coeffs, |m| na::Rotation2::new(m[(1, 0)].atan2(m[(0, 0)])))
    }
}

impl EfdDim for D3 {
    type Trans = T3;

    fn coeff_norm(coeffs: &mut Coeff<Self>) -> Transform<Self::Trans> {
        impl_norm(coeffs, |coeffs| {
            let m1 = CCKernel::<3>::from_slice(coeffs.column(0).data.into_slice());
            let u = m1.column(0).normalize();
            if let Some(v) = m1.column(1).try_normalize(f64::EPSILON) {
                // Closed curve, use `u` and `v` plane as basis
                let w = u.cross(&v);
                na::Rotation3::from_basis_unchecked(&[u, v, w])
            } else if coeffs.ncols() > 1 {
                // Open curve, `v` is zero vector, use `u1` and `u2` plane as basis
                let m2 = CCKernel::<3>::from_slice(coeffs.column(1).data.into_slice());
                // `w` is orthogonal to `u` and `u2`
                let w = u.cross(&m2.column(0)).normalize();
                let u2 = w.cross(&u);
                na::Rotation3::from_basis_unchecked(&[u, u2, w])
            } else {
                // Open curve, one harmonic, just rotate `u` to x-axis
                let (u, v) = (na::Vector3::x(), u);
                na::Rotation3::from_scaled_axis(u.cross(&v).normalize() * u.dot(&v).acos())
            }
        })
    }
}

fn impl_coeff<T: Trans>(
    curve: &[T::Coord],
    harmonic: usize,
    is_open: bool,
) -> (MatrixRxX<CCDim<T>>, Transform<T>) {
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
    let mut coeffs = MatrixRxX::<CCDim<T>>::zeros(harmonic);
    for (i, mut c) in coeffs.column_iter_mut().enumerate() {
        let n = (i + 1) as f64;
        let phi = &phi * n;
        let phi_front = phi.columns_range(..phi.len() - 1);
        let phi_back = phi.columns_range(1..);
        let scalar = scalar / (n * n);
        let cos_phi = (phi_back.map(f64::cos) - phi_front.map(f64::cos)).component_div(&dt);
        dxyz.row_iter()
            .zip(c.iter_mut().take(dxyz.nrows()))
            .for_each(|(d, c)| *c = scalar * d.component_mul(&cos_phi).sum());
        if is_open {
            continue;
        }
        let sin_phi = (phi_back.map(f64::sin) - phi_front.map(f64::sin)).component_div(&dt);
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
    (coeffs, Transform::new(center, Default::default(), 1.))
}

fn impl_norm<T, const DIM: usize, const CDIM: usize>(
    coeffs: &mut CCoeff<CDIM>,
    get_psi: impl FnOnce(&CCoeff<CDIM>) -> na::Rotation<f64, DIM>,
) -> Transform<T>
where
    // const-generic assertion
    na::Const<CDIM>: na::DimNameDiv<na::U2, Output = na::Const<DIM>>,
    T: Trans,
    T::Rot: From<na::Rotation<f64, DIM>>,
{
    // Angle of starting point
    // m = m * theta
    let theta = {
        let c = CCKernel::<DIM>::from_slice(coeffs.column(0).data.into_slice());
        let dy = 2. * c.column_product().sum();
        let dx = c.map(pow2).row_sum();
        0.5 * dy.atan2(dx[0] - dx[1])
    };
    for (i, mut c) in coeffs.column_iter_mut().enumerate() {
        let theta = na::Rotation2::new((i + 1) as f64 * theta);
        let mut m = CCKernelMut::<DIM>::from_slice(c.as_mut_slice());
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
    let psi = get_psi(coeffs);
    for mut c in coeffs.column_iter_mut() {
        let mut m = CCKernelMut::<DIM>::from_slice(c.as_mut_slice());
        m.tr_mul(psi.matrix()).transpose_to(&mut m);
    }
    // Scale factor
    // |u1| == a1 (after rotation)
    let scale = coeffs[(0, 0)].abs();
    *coeffs /= scale;
    Transform::new(Default::default(), psi.into(), scale)
}
