use crate::{util::*, *};
use alloc::{vec, vec::Vec};
use core::{
    f64::consts::{PI, TAU},
    iter::zip,
};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::*;

/// EFD dimension marker.
pub enum U<const D: usize> {}

/// Coefficients type.
pub type Coeffs<const D: usize> = Vec<Kernel<D>>;
/// An owned matrix of specific coefficients. (Dx2)
pub type Kernel<const D: usize> = na::SMatrix<f64, D, 2>;
/// Rotation type of the EFD.
pub type Rot<const D: usize> = <U<D> as EfdDim<D>>::Rot;

trait Sealed {}
impl<const D: usize> Sealed for U<D> {}

/// Trait for the dimension [`U<D>`] of EFD.
///
/// Use `where U<D>: EfdDim<D>` bound to constraint the dimension `D` that
/// implements this trait. Please see the [implementors section](#implementors)
/// for the supported dimensions.
///
/// **This trait is sealed and cannot be implemented outside of this crate.**
/// The API of this trait is not public and may change in the future.
#[allow(private_bounds)]
pub trait EfdDim<const D: usize>: Sealed {
    /// Rotation type of the dimension `D`.
    ///
    /// For the memory efficiency, the generic rotation matrix [`na::Rotation`]
    /// is not used.
    type Rot: RotHint<D>;

    #[doc(hidden)]
    fn get_rot(m: &[Kernel<D>]) -> Self::Rot;

    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    fn get_coeff(
        curve: &[[f64; D]],
        is_open: bool,
        harmonic: usize,
        guide: Option<&[f64]>,
    ) -> (Vec<f64>, Coeffs<D>, GeoVar<Self::Rot, D>) {
        let is_closed = !is_open && curve.first() != curve.last();
        // Differential of the components
        let dxyz = diff(if is_closed {
            to_mat(curve.closed_lin())
        } else {
            to_mat(curve)
        });
        // Length of curve between points
        // from: (1) provided guide, or
        //       (2) differential of the components
        let dt = if let Some(guide) = guide {
            debug_assert_eq!(guide.len(), dxyz.ncols());
            na::Matrix1xX::from_row_slice(guide)
        } else {
            dxyz.map(pow2).row_sum().map(f64::sqrt)
        };
        // Length of curve from start to each point
        let t = cumsum(dt.clone()).insert_column(0, 0.);
        // Total length of curve
        let zt = t[t.len() - 1];
        // Length to angle
        let phi = &t * TAU / zt * if is_open { 0.5 } else { 1. };
        // Scalar for coefficients
        let scalar = zt / pow2(PI) * if is_open { 2. } else { 0.5 };
        // Coefficients (2dim * N)
        // [x_cos, y_cos, z_cos, x_sin, y_sin, z_sin]'
        let mut coeff = vec![Kernel::<D>::zeros(); harmonic];
        for (n, c) in coeff.iter_mut().enumerate() {
            let n = (n + 1) as f64;
            let phi = &phi * n;
            let scalar = scalar / pow2(n);
            let cos_phi = diff(phi.map(f64::cos)).component_div(&dt);
            zip(dxyz.row_iter(), &mut c.column_mut(0))
                .for_each(|(d, c)| *c = scalar * d.component_mul(&cos_phi).sum());
            if is_open {
                continue;
            }
            let sin_phi = diff(phi.map(f64::sin)).component_div(&dt);
            zip(dxyz.row_iter(), &mut c.column_mut(1))
                .for_each(|(d, c)| *c = scalar * d.component_mul(&sin_phi).sum());
        }
        // Percentage of total stroke versus current stroke
        let tdt = t.columns_range(1..).component_div(&dt);
        // Scalar for the shape center
        let scalar = 0.5 * diff(t.map(pow2)).component_div(&dt);
        // Shape center
        let mut center = curve[0];
        for (dxyz, oxyz) in zip(dxyz.row_iter(), &mut center) {
            let xi = cumsum(dxyz) - dxyz.component_mul(&tdt);
            *oxyz += (dxyz.component_mul(&scalar) + xi.component_mul(&dt)).sum() / zt;
        }
        // Keep the t number the same as the input curve
        let mut t = Vec::from(phi.data);
        if is_closed {
            t.pop();
        }
        (t, coeff, GeoVar::from_trans(center))
    }

    #[doc(hidden)]
    fn norm_coeff(coeffs: &mut [Kernel<D>], mut t: Option<&mut [f64]>) -> GeoVar<Self::Rot, D> {
        // Angle of starting point (theta)
        // theta = atan2(2 * sum(m[:, 0] * m[:, 1]), sum(m[:, 0]^2) - sum(m[:, 1]^2))
        // theta = 0 if is open curve
        // m = m * theta
        if coeffs[0].column(1).sum() != 0. {
            let theta = {
                let m1 = &coeffs[0];
                let dy = 2. * m1.column_product().sum();
                let dx = m1.map(pow2).row_sum();
                0.5 * dy.atan2(dx[0] - dx[1])
            };
            for (i, m) in coeffs.iter_mut().enumerate() {
                let theta = na::Rotation2::new((i + 1) as f64 * theta);
                m.copy_from(&(*m * theta));
            }
            if let Some(t) = &mut t {
                t.iter_mut().for_each(|v| *v -= theta);
            }
        }
        // Normalize coefficients sign (zeta)
        // - Check 1st and 2nd harmonics if two local coordinate systems are the closest
        // - Plus PI to time parameters if zeta is -1
        Self::norm_zeta(coeffs, t);
        // Rotation angle (psi)
        // m = psi' * m
        let psi = Self::get_rot(coeffs);
        let psi_mat = psi.clone().matrix();
        for m in coeffs.iter_mut() {
            m.tr_mul(&psi_mat).transpose_to(m);
        }
        // Scaling factor
        // |u1| == a1 (after rotation normalized)
        let scale = coeffs[0][0];
        coeffs.iter_mut().for_each(|m| *m /= scale);
        debug_assert!(scale.is_sign_positive());
        GeoVar::new([0.; D], psi, scale)
    }

    #[doc(hidden)]
    fn norm_zeta(coeffs: &mut [Kernel<D>], t: Option<&mut [f64]>) {
        if coeffs.len() > 1 && {
            let [u1, v1] = [coeffs[0].column(0), coeffs[0].column(1)];
            let [u2, v2] = [coeffs[1].column(0), coeffs[1].column(1)];
            (u1 - u2).norm() + (v1 - v2).norm() > (u1 + u2).norm() + (v1 + v2).norm()
        } {
            coeffs.iter_mut().step_by(2).for_each(|s| *s *= -1.);
            if let Some(t) = t {
                t.iter_mut().for_each(|v| *v += PI);
            }
        }
    }

    #[doc(hidden)]
    fn reconstruct(
        coeffs: &[Kernel<D>],
        t_iter: impl ExactSizeIterator<Item = f64>,
    ) -> Vec<[f64; D]> {
        let t = na::Matrix1xX::from_iterator(t_iter.len(), t_iter);
        coeffs
            .iter()
            .enumerate()
            .map(|(n, m)| {
                let t = (n + 1) as f64 * &t;
                m * na::Matrix2xX::from_rows(&[t.map(f64::cos), t.map(f64::sin)])
            })
            .reduce(|a, b| a + b)
            .unwrap_or_else(|| MatrixRxX::from_vec(Vec::new())) // empty coeffs
            .column_iter()
            .map(|row| row.into())
            .collect()
    }
}

impl EfdDim<1> for U<1> {
    type Rot = na::Rotation<f64, 1>;

    fn get_rot(m: &[Kernel<1>]) -> Self::Rot {
        na::Rotation::from_matrix_unchecked(na::matrix![m[0][0].signum()])
    }
}

impl EfdDim<2> for U<2> {
    type Rot = na::UnitComplex<f64>;

    fn get_rot(m: &[Kernel<2>]) -> Self::Rot {
        na::UnitComplex::new(m[0][1].atan2(m[0][0]))
    }
}

impl EfdDim<3> for U<3> {
    type Rot = na::UnitQuaternion<f64>;

    fn get_rot(m: &[Kernel<3>]) -> Self::Rot {
        let m1 = &m[0];
        let u = m1.column(0).normalize();
        if let Some(v) = m1.column(1).try_normalize(f64::EPSILON) {
            // Closed curve, use `u` and `v` plane as basis
            let w = u.cross(&v);
            na::UnitQuaternion::from_basis_unchecked(&[u, v, w])
        } else if m.len() > 1 {
            // Open curve, `v` is zero vector, use `u1` and `u2` plane as basis
            let u2 = m[1].column(0);
            // `w` is orthogonal to `u` and `u2`
            let w = u.cross(&u2).normalize();
            // A new `v` is orthogonal to `w` and `u`
            let v = w.cross(&u);
            na::UnitQuaternion::from_basis_unchecked(&[u, v, w])
        } else {
            // Open curve, one harmonic, just rotate `u` to x-axis
            let [u, x] = [na::Unit::new_unchecked(u), na::Vector3::x_axis()];
            na::UnitQuaternion::rotation_between_axis(&u, &x).unwrap_or_default()
        }
    }
}
