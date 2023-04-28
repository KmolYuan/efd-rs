use crate::*;
use alloc::vec::Vec;
use core::f64::consts::{PI, TAU};

/// 2D EFD dimension type.
pub type D2 = [f64; 2];
/// 3D EFD dimension type.
pub type D3 = [f64; 3];
/// 2D Coefficient type.
pub type Coeff2 = Coeff<D2>;
/// 3D Coefficient type.
pub type Coeff3 = Coeff<D3>;
/// Coefficient type.
pub type Coeff<D> = na::OMatrix<f64, na::Dyn, <D as EfdDim>::CDim>;
/// Coordinate view used in the conversion method.
pub type CoordView<'a, D> = na::MatrixView<'a, f64, na::U1, <D as EfdDim>::Dim, na::U1, na::Dyn>;

/// Trait for EFD dimension.
pub trait EfdDim {
    /// Transform type.
    type Trans: Trans;
    /// Dimension. Is a constant width.
    type Dim: na::base::DimName;
    /// Coefficient number per harmonic. Is a constant width.
    type CDim: na::base::DimName;

    /// Generate coefficients and transform.
    fn from_curve_harmonic(
        curve: &[Coord<Self>],
        harmonic: usize,
        is_open: bool,
    ) -> (Coeff<Self>, Transform<Self::Trans>);

    /// Transform array slice to coordinate type.
    fn to_coord(c: CoordView<Self>) -> Coord<Self>;
}

impl EfdDim for D2 {
    type Trans = T2;
    type Dim = na::U2;
    type CDim = na::U4;

    fn from_curve_harmonic(
        curve: &[Coord<Self>],
        harmonic: usize,
        is_open: bool,
    ) -> (Coeff<Self>, Transform<Self::Trans>) {
        let (mut coeffs, center) = get_coeff_center(curve, harmonic, is_open);
        // Angle of starting point
        let theta = {
            let c = na::Matrix2::from_iterator(coeffs.row(0).iter().copied());
            let dy = 2. * c.row_product().sum();
            let dx = c.map(pow2).column_sum();
            0.5 * dy.atan2(dx[0] - dx[1])
        };
        for (i, mut c) in coeffs.row_iter_mut().enumerate() {
            let theta = na::Rotation2::new((i + 1) as f64 * -theta);
            let m = theta * na::Matrix2::from_iterator(c.iter().copied());
            c.copy_from_slice(m.as_slice());
        }
        // Normalize coefficients sign
        if harmonic > 1 && coeffs[(0, 0)] * coeffs[(1, 0)] < 0. {
            coeffs.row_iter_mut().step_by(2).for_each(|mut s| s *= -1.);
        }
        // Angle of semi-major axis
        // Case in 2D are equivalent to:
        //
        // let u = na::Vector3::new(coeffs[[0, 0]], coeffs[[0, 2]], 0.).normalize();
        // na::UnitComplex::new(u.dot(&na::Vector3::x()).acos())
        let psi = na::UnitComplex::new(coeffs[(0, 2)].atan2(coeffs[(0, 0)]));
        let psi_inv = psi.to_rotation_matrix();
        for mut c in coeffs.row_iter_mut() {
            let m = na::Matrix2::from_iterator(c.iter().copied()) * psi_inv;
            c.copy_from_slice(m.as_slice());
        }
        let scale = coeffs[(0, 0)].hypot(coeffs[(0, 2)]);
        coeffs /= scale;
        let trans = Transform::new(center, psi, scale);
        (coeffs, trans)
    }

    fn to_coord(c: CoordView<Self>) -> Coord<Self> {
        [c[0], c[1]]
    }
}

impl EfdDim for D3 {
    type Trans = T3;
    type Dim = na::U3;
    type CDim = na::U6;

    fn from_curve_harmonic(
        curve: &[Coord<Self>],
        harmonic: usize,
        is_open: bool,
    ) -> (Coeff<Self>, Transform<Self::Trans>) {
        let (mut coeffs, center) = get_coeff_center(curve, harmonic, is_open);
        // Angle of starting point
        let theta = {
            let c = na::Matrix2x3::from_iterator(coeffs.row(0).iter().copied());
            let dy = 2. * c.row_product().sum();
            let dx = c.map(pow2).column_sum();
            0.5 * dy.atan2(dx[0] - dx[1])
        };
        for (i, mut c) in coeffs.row_iter_mut().enumerate() {
            let theta = na::Rotation2::new((i + 1) as f64 * -theta);
            let m = theta * na::Matrix2x3::from_iterator(c.iter().copied());
            c.copy_from_slice(m.as_slice());
        }
        // Normalize coefficients sign
        if harmonic > 1 && coeffs[(0, 0)] * coeffs[(1, 0)] < 0. {
            coeffs.row_iter_mut().step_by(2).for_each(|mut s| s *= -1.);
        }
        // Angle of semi-major axis
        let psi = {
            let c = na::Matrix2x3::from_iterator(coeffs.row(0).iter().copied());
            let u = c.row(0).transpose().normalize();
            let v = c.row(1).transpose().normalize();
            na::Rotation3::from_basis_unchecked(&[u, v, u.cross(&v)])
        };
        let psi_inv = psi;
        for mut c in coeffs.row_iter_mut() {
            let m = na::Matrix2x3::from_iterator(c.iter().copied()) * psi_inv;
            c.copy_from_slice(m.as_slice());
        }
        let scale = coeffs
            .row(0)
            .fixed_columns_with_step::<3>(0, 1)
            .map(pow2)
            .sum()
            .sqrt();
        coeffs /= scale;
        let trans = Transform::new(center, psi.into(), scale);
        (coeffs, trans)
    }

    fn to_coord(c: CoordView<Self>) -> Coord<Self> {
        [c[0], c[1], c[2]]
    }
}

fn get_coeff_center<D: na::DimName, const DIM: usize>(
    curve: &[[f64; DIM]],
    harmonic: usize,
    is_open: bool,
) -> (na::OMatrix<f64, na::Dyn, D>, [f64; DIM]) {
    let curve_arr = if is_open {
        to_mat(curve)
    } else {
        to_mat(curve.closed_lin())
    };
    let dxyz = diff(curve_arr);
    let dt = dxyz.map(pow2).column_sum().map(f64::sqrt);
    let t = cumsum(dt.clone()).insert_row(0, 0.);
    let zt = t[t.len() - 1] * if is_open { 2. } else { 1. };
    let scalar = zt / (PI * PI) * if is_open { 1. } else { 0.5 };
    let phi = &t * TAU / zt;
    let mut coeffs = MatrixXxC::<D>::zeros(harmonic);
    for (i, mut c) in coeffs.row_iter_mut().enumerate() {
        let n = i as f64 + 1.;
        let phi = &phi * n;
        let phi_front = phi.rows_range(..phi.nrows() - 1);
        let phi_back = phi.rows_range(1..);
        let scalar = scalar / (n * n);
        let cos_phi = (phi_back.map(f64::cos) - phi_front.map(f64::cos)).component_div(&dt);
        dxyz.column_iter()
            .zip(c.iter_mut().step_by(2))
            .for_each(|(d, c)| *c = (scalar * d.component_mul(&cos_phi)).sum());
        if !is_open {
            let sin_phi = (phi_back.map(f64::sin) - phi_front.map(f64::sin)).component_div(&dt);
            dxyz.column_iter()
                .zip(c.iter_mut().skip(1).step_by(2))
                .for_each(|(d, c)| *c = (scalar * d.component_mul(&sin_phi)).sum());
        }
    }
    let center = {
        let tdt = t.rows_range(1..).component_div(&dt);
        let c = 0.5 * diff(t.map(pow2)).component_div(&dt);
        dxyz.column_iter()
            .zip(curve[0])
            .map(|(dxyz, oxyz)| {
                let xi = cumsum(dxyz) - dxyz.component_mul(&tdt);
                oxyz + (dxyz.component_mul(&c) + xi.component_mul(&dt)).sum() / zt
                    * if is_open { 2. } else { 1. }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    };
    (coeffs, center)
}
