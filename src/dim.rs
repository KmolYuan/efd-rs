//! Dimension specific implementation.
use crate::{util::*, *};
use alloc::vec::Vec;
use core::f64::consts::{PI, TAU};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::*;

/// EFD dimension marker.
pub enum U<const N: usize> {}

/// 2D Coefficient type.
pub type Coeff2 = Coeff<2>;
/// 3D Coefficient type.
pub type Coeff3 = Coeff<3>;
/// Coefficient type.
pub type Coeff<const D: usize> = na::OMatrix<f64, CDim<D>, na::Dyn>;
/// Coordinate view used in the conversion method.
pub type CoordView<'a, const D: usize> = na::MatrixView<'a, f64, CDim<D>, na::U1>;
/// Alias for the coefficient number. (DIM * 2)
pub type CDim<const D: usize> = na::DimNameProd<na::Const<D>, na::U2>;
/// A matrix view of specific coefficients. (DIM * 2)
pub type CKernel<'a, const D: usize> = na::MatrixView<'a, f64, na::Const<D>, na::U2>;
/// A mutable matrix view of specific coefficients. (DIM * 2)
pub type CKernelMut<'a, const D: usize> = na::MatrixViewMut<'a, f64, na::Const<D>, na::U2>;
/// Rotation type of the EFD.
pub type Rot<const D: usize> = <U<D> as EfdDim<D>>::Rot;

trait Sealed {}
impl<const D: usize> Sealed for U<D> {}

/// Trait for EFD dimension.
///
/// **This trait is sealed and cannot be implemented outside of this crate.**
#[allow(private_bounds)]
pub trait EfdDim<const D: usize>: Sealed
where
    na::Const<D>: na::DimNameMul<na::U2>,
{
    /// Rotation type.
    ///
    /// For the memory efficiency, the generic rotation matrix [`na::Rotation`]
    /// is not used.
    type Rot: RotHint<D>;

    /// Get the rotation matrix from the coefficients.
    ///
    /// In different dimensions, the rotation needs to be calculated
    /// differently. So only the specific dimension can implement this trait.
    fn get_rot(coeffs: &Coeff<D>) -> Self::Rot;

    /// Generate coefficients and similarity matrix **without** normalization.
    fn get_coeff_unnorm(
        curve: &[Coord<D>],
        is_open: bool,
        harmonic: usize,
    ) -> (Coeff<D>, GeoVar<Self::Rot, D>) {
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
        let mut coeffs = Coeff::<D>::zeros(harmonic);
        for mut c in coeffs.column_iter_mut() {
            n += 1.;
            let phi = &phi * n;
            let scalar = scalar / (n * n);
            let cos_phi = diff(phi.map(f64::cos)).component_div(&dt);
            dxyz.row_iter()
                .zip(&mut c.rows_range_mut(..dxyz.nrows()))
                .for_each(|(d, c)| *c = scalar * d.component_mul(&cos_phi).sum());
            if is_open {
                continue;
            }
            let sin_phi = diff(phi.map(f64::sin)).component_div(&dt);
            dxyz.row_iter()
                .zip(&mut c.rows_range_mut(dxyz.nrows()..))
                .for_each(|(d, c)| *c = scalar * d.component_mul(&sin_phi).sum());
        }
        let tdt = t.columns_range(1..).component_div(&dt);
        let c = 0.5 * diff(t.map(pow2)).component_div(&dt);
        let mut center = curve[0];
        dxyz.row_iter().zip(&mut center).for_each(|(dxyz, oxyz)| {
            let xi = cumsum(dxyz) - dxyz.component_mul(&tdt);
            *oxyz += (dxyz.component_mul(&c) + xi.component_mul(&dt)).sum() / zt;
        });
        let rot = na::AbstractRotation::identity();
        (coeffs, GeoVar::new(center, rot, 1.))
    }

    /// Get the posed coefficients.
    #[allow(unused_variables)]
    fn get_posed(
        curve: &[Coord<D>],
        vectors: MatrixRxX<D>,
        is_open: bool,
        harmonic: usize,
    ) -> (Coeff<D>, Coeff<D>, GeoVar<Self::Rot, D>) {
        // TODO
        todo!()
    }

    /// Normalize coefficients.
    fn coeff_norm(coeffs: &mut Coeff<D>) -> GeoVar<Self::Rot, D> {
        // Angle of starting point
        // m = m * theta
        let theta = {
            let c = coeffs.column(0).reshape_generic(na::Const::<D>, na::U2);
            let dy = 2. * c.column_product().sum();
            let dx = c.map(pow2).row_sum();
            0.5 * dy.atan2(dx[0] - dx[1])
        };
        for (i, c) in coeffs.column_iter_mut().enumerate() {
            let theta = na::Rotation2::new((i + 1) as f64 * theta);
            let mut m = c.reshape_generic(na::Const::<D>, na::U2);
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
        let psi = Self::get_rot(coeffs);
        let psi_mat = psi.clone().matrix();
        for c in coeffs.column_iter_mut() {
            let mut m = c.reshape_generic(na::Const::<D>, na::U2);
            m.tr_mul(&psi_mat).transpose_to(&mut m);
        }
        // Scale factor
        // |u1| == a1 (after rotation)
        let scale = coeffs[(0, 0)].abs();
        *coeffs /= scale;
        GeoVar::new([0.; D], psi, scale)
    }

    /// Reconstruct the curve from the coefficients.
    fn reconstruct(coeffs: &Coeff<D>, n: usize, theta: f64) -> Vec<Coord<D>> {
        assert!(n > 1, "n ({n}) must larger than 1");
        let t = na::Matrix1xX::from_fn(n, |_, i| i as f64 / (n - 1) as f64 * theta);
        coeffs
            .column_iter()
            .enumerate()
            .map(|(i, c)| {
                let t = &t * (i + 1) as f64;
                let t = na::Matrix2xX::from_rows(&[t.map(f64::cos), t.map(f64::sin)]);
                c.reshape_generic(na::Const::<D>, na::U2) * t
            })
            .reduce(|a, b| a + b)
            .unwrap_or_else(|| MatrixRxX::<D>::from_vec(Vec::new()))
            .column_iter()
            .map(|row| core::array::from_fn(|i| row[i]))
            .collect()
    }
}

impl EfdDim<1> for U<1> {
    type Rot = na::Rotation<f64, 1>;

    fn get_rot(_coeffs: &Coeff<1>) -> Self::Rot {
        na::Rotation::from_matrix_unchecked(na::matrix![1.])
    }
}

impl EfdDim<2> for U<2> {
    type Rot = na::UnitComplex<f64>;

    fn get_rot(m: &Coeff<2>) -> Self::Rot {
        na::UnitComplex::new(m[(1, 0)].atan2(m[(0, 0)]))
    }
}

impl EfdDim<3> for U<3> {
    type Rot = na::UnitQuaternion<f64>;

    fn get_rot(m: &Coeff<3>) -> Self::Rot {
        let m1 = m.column(0).reshape_generic(na::U3, na::U2);
        let u = m1.column(0).normalize();
        if let Some(v) = m1.column(1).try_normalize(f64::EPSILON) {
            // Closed curve, use `u` and `v` plane as basis
            let w = u.cross(&v);
            na::UnitQuaternion::from_basis_unchecked(&[u, v, w])
        } else if m.ncols() > 1 {
            // Open curve, `v` is zero vector, use `u1` and `u2` plane as basis
            let m2 = m.column(1).reshape_generic(na::U3, na::U2);
            // `w` is orthogonal to `u` and `u2`
            let w = u.cross(&m2.column(0)).normalize();
            let u2 = w.cross(&u);
            na::UnitQuaternion::from_basis_unchecked(&[u, u2, w])
        } else {
            // Open curve, one harmonic, just rotate `u` to x-axis
            let (u, v) = (na::Vector3::x(), u);
            na::UnitQuaternion::from_scaled_axis(u.cross(&v).normalize() * u.dot(&v).acos())
        }
    }
}
